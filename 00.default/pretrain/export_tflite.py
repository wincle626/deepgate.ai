"""Export the pretrained `wakeword` model to a full-integer (int8) .tflite file.

Dedicated to the artifact in ``pretrain/wakeword/`` (saved by
``wakeword_test.py`` via ``model.save_pretrained``); writes
``pretrain/wakeword/wakeword.tflite``. Paths are resolved relative to this
file, so it can be run from anywhere:

    ./pyenv/bin/python pretrain/export_tflite.py

Route: rebuild an equivalent float Keras model from the dg layers' BatchNorm-
folded float weights, then run TFLite full-integer post-training quantization
(tf.lite.TFLiteConverter). There is no dataset alongside the pretrained model,
so activations are calibrated on synthetic samples drawn from dg's Norm
statistics. TFLite recomputes its own scales during PTQ, so the int8 result is
a faithful int8 model of the same network, not bit-identical to dg's own
calibration.
"""
import os

import numpy as np
import tensorflow as tf
import torch

import dg
from dg.layer import (
    Norm, Flatten, QuantConv2d, QuantDepthwiseConv2d, QuantAvgPool2d, QuantLinear,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "wakeword")                 # save_pretrained artifact
OUT_PATH = os.path.join(SRC_DIR, "wakeword.tflite")
INPUT_HW = (49, 10, 1)            # H, W, C  (dg uses channels-last / HWC)


def load_torch_model():
    # local save_pretrained artifact -> strict load
    model = dg.from_pretrained(SRC_DIR)
    model.eval()
    return model


def build_keras(model):
    """Construct a float tf.keras model mirroring model.model_graph and load
    BN-folded float weights pulled directly from the torch layers."""
    inp = tf.keras.Input(shape=INPUT_HW, name="features")
    x = inp
    pending_weights = []  # (keras_layer, [arrays...]) applied after build

    for layer in model.model_graph:
        if isinstance(layer, Norm):
            # (x - mean) / std  ==  x * (1/std) + (-mean/std)
            std = float(layer.std) if hasattr(layer, "std") else model.NORM_STD
            mean = float(layer.mean) if hasattr(layer, "mean") else model.NORM_MEAN
            x = tf.keras.layers.Rescaling(1.0 / std, offset=-mean / std, name="norm")(x)

        elif isinstance(layer, QuantConv2d):
            w, b = layer._get_fused_weight_bias()        # torch [out, in, kh, kw]
            kh, kw = layer.kernel_size
            sh, sw = layer.stride
            L, R, T, B = layer.padding
            if (L, R, T, B) != (0, 0, 0, 0):
                x = tf.keras.layers.ZeroPadding2D(padding=((T, B), (L, R)))(x)
            kl = tf.keras.layers.Conv2D(
                filters=layer.out_channels, kernel_size=(kh, kw), strides=(sh, sw),
                padding="valid", use_bias=b is not None,
                activation="relu" if layer.act_func == "relu" else None,
            )
            x = kl(x)
            kernel = w.numpy().transpose(2, 3, 1, 0)     # -> [kh, kw, in, out]
            pending_weights.append((kl, [kernel] + ([b.numpy()] if b is not None else [])))

        elif isinstance(layer, QuantDepthwiseConv2d):
            w, b = layer._get_fused_weight_bias()        # torch [ch, 1, kh, kw]
            kh, kw = layer.kernel_size
            sh, sw = layer.stride
            L, R, T, B = layer.padding
            if (L, R, T, B) != (0, 0, 0, 0):
                x = tf.keras.layers.ZeroPadding2D(padding=((T, B), (L, R)))(x)
            kl = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(kh, kw), strides=(sh, sw), padding="valid",
                use_bias=b is not None,
                activation="relu" if layer.act_func == "relu" else None,
            )
            x = kl(x)
            dk = w.numpy().transpose(2, 3, 0, 1)         # -> [kh, kw, ch, 1]
            pending_weights.append((kl, [dk] + ([b.numpy()] if b is not None else [])))

        elif isinstance(layer, QuantAvgPool2d):
            kh, kw = layer.kernel_size
            sh, sw = layer.stride
            x = tf.keras.layers.AveragePooling2D(
                pool_size=(kh, kw), strides=(sh, sw), padding="valid")(x)

        elif isinstance(layer, Flatten):
            x = tf.keras.layers.Flatten()(x)

        elif isinstance(layer, QuantLinear):
            w, b = layer._get_fused_weight_bias()        # torch [out, in]
            kl = tf.keras.layers.Dense(
                units=layer.out_features, use_bias=b is not None,
                activation="relu" if layer.act_func == "relu" else None,
            )
            x = kl(x)
            kernel = w.numpy().transpose(1, 0)           # -> [in, out]
            pending_weights.append((kl, [kernel] + ([b.numpy()] if b is not None else [])))

        else:
            raise TypeError(f"Unhandled dg layer: {type(layer).__name__}")

    keras_model = tf.keras.Model(inp, x, name="wakeword")
    for kl, arrs in pending_weights:
        kl.set_weights(arrs)
    return keras_model


def representative_dataset():
    # no dataset ships with the pretrained model; dg's Norm constants imply raw
    # inputs ~ N(mean, std), which sets plausible activation ranges for PTQ
    rng = np.random.default_rng(0)
    for _ in range(300):
        yield [rng.normal(-1.6787, 7.6337, size=(1, *INPUT_HW)).astype(np.float32)]


def convert(keras_model):
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def validate(keras_model, tflite_bytes, torch_model):
    x = np.random.default_rng(1).normal(
        -1.6787, 7.6337, size=(1, *INPUT_HW)).astype(np.float32)
    with torch.no_grad():
        t_out = torch_model(torch.from_numpy(x.transpose(0, 3, 1, 2))).numpy()
    k_out = keras_model.predict(x, verbose=0)
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    s, zp = inp_d["quantization"]
    xq = np.clip(np.round(x / s + zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp_d["index"], xq)
    interp.invoke()
    raw = interp.get_tensor(out_d["index"]).astype(np.float32)
    so, zpo = out_d["quantization"]
    lite_out = (raw - zpo) * so
    print("torch argmax :", int(t_out.argmax()), t_out.round(3))
    print("keras argmax :", int(k_out.argmax()), k_out.round(3))
    print("tflite argmax:", int(lite_out.argmax()), lite_out.round(3))
    print("max|keras-torch| =", float(np.abs(k_out - t_out).max()))
    print("max|tflite-torch|=", float(np.abs(lite_out - t_out).max()))


def main():
    torch_model = load_torch_model()
    keras_model = build_keras(torch_model)
    tflite_bytes = convert(keras_model)
    with open(OUT_PATH, "wb") as f:
        f.write(tflite_bytes)
    print(f"wrote {OUT_PATH} ({len(tflite_bytes)} bytes)")
    validate(keras_model, tflite_bytes, torch_model)


if __name__ == "__main__":
    main()
