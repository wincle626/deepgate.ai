"""Export the saved AlexNet dg bundle to a full-integer (int8) .tflite file.

Dedicated to the artifact in ./alexnet/ (alexnet.pt, alexnet.py, config.json);
writes ./alexnet/alexnet.tflite. Activations are calibrated on real CIFAR-10
images. Paths are resolved relative to this file:

    ../pyenv/bin/python export_tflite.py

Route: rebuild an equivalent float Keras model from the dg layers' BatchNorm-
folded float weights, then run TFLite full-integer post-training quantization
(tf.lite.TFLiteConverter). TFLite recomputes its own scales during PTQ, so the
int8 result is a faithful int8 model of the same network, not bit-identical to
dg's own calibration.
"""
import importlib.util
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import torch

from dg.layer import (
    Norm, Flatten, MaxPool2d, QuantConv2d, QuantDepthwiseConv2d,
    QuantAvgPool2d, QuantLinear,
)

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "alexnet")
DATA_DIR = os.path.join(HERE, "data")
NAME = "alexnet"
OUT_PATH = os.path.join(BUNDLE, NAME + ".tflite")
INPUT_HW = (32, 32, 3)            # H, W, C  (dg uses channels-last / HWC)


def load_torch_model():
    """Reload the model from the renamed bundle (alexnet.py + alexnet.pt)."""
    src = os.path.join(BUNDLE, NAME + ".py")
    spec = importlib.util.spec_from_file_location("_dg_bundle", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(open(os.path.join(BUNDLE, "config.json")).read())
    model = module.Model(**config)
    state = torch.load(os.path.join(BUNDLE, NAME + ".pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def build_keras(model):
    """Construct a float tf.keras model mirroring model.model_graph and load
    BN-folded float weights pulled directly from the torch layers."""
    inp = tf.keras.Input(shape=INPUT_HW, name="image")
    x = inp
    pending_weights = []  # (keras_layer, [arrays...]) applied after build

    for layer in model.model_graph:
        if isinstance(layer, Norm):
            std = float(layer.std)
            mean = float(layer.mean)
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

        elif isinstance(layer, MaxPool2d):
            kh, kw = layer.kernel
            sh, sw = layer.stride
            x = tf.keras.layers.MaxPooling2D(pool_size=(kh, kw), strides=(sh, sw), padding="valid")(x)

        elif isinstance(layer, QuantAvgPool2d):
            kh, kw = layer.kernel_size
            sh, sw = layer.stride
            L, R, T, B = layer.padding
            if (L, R, T, B) != (0, 0, 0, 0):
                x = tf.keras.layers.ZeroPadding2D(padding=((T, B), (L, R)))(x)
            x = tf.keras.layers.AveragePooling2D(pool_size=(kh, kw), strides=(sh, sw), padding="valid")(x)

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

    keras_model = tf.keras.Model(inp, x, name=NAME)
    for kl, arrs in pending_weights:
        kl.set_weights(arrs)
    return keras_model


def _cifar_test():
    # Read CIFAR-10 test batch directly (torchvision can't be imported next to
    # tensorflow without a native segfault). Run the *_data.py helper first.
    path = os.path.join(DATA_DIR, "cifar-10-batches-py", "test_batch")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    x = data.transpose(0, 2, 3, 1)                        # (N, 32, 32, 3) NHWC [0,1]
    y = np.array(d[b"labels"])
    return x, y


def representative_dataset():
    x, _ = _cifar_test()
    idx = np.random.default_rng(0).permutation(len(x))[:300]
    for i in idx:
        yield [x[i:i + 1].astype(np.float32)]


def convert(keras_model):
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def validate(keras_model, tflite_bytes, torch_model):
    x, _ = _cifar_test()
    x = x[:1].astype(np.float32)                          # a real sample
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
    print("torch argmax :", int(t_out.argmax()))
    print("keras argmax :", int(k_out.argmax()))
    print("tflite argmax:", int(lite_out.argmax()))
    print("max|keras-torch| =", float(np.abs(k_out - t_out).max()))
    print("max|tflite-torch|=", float(np.abs(lite_out - t_out).max()))


def tflite_accuracy(tflite_bytes, n=2000):
    x, y = _cifar_test()
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    s, zp = inp_d["quantization"]
    n = min(n, len(x))
    correct = 0
    for i in range(n):
        xq = np.clip(np.round(x[i:i + 1] / s + zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp_d["index"], xq)
        interp.invoke()
        if int(interp.get_tensor(out_d["index"])[0].argmax()) == int(y[i]):
            correct += 1
    print(f"tflite top-1 over {n} CIFAR-10 test images: {100 * correct / n:.2f}%")


def main():
    torch_model = load_torch_model()
    keras_model = build_keras(torch_model)
    tflite_bytes = convert(keras_model)
    with open(OUT_PATH, "wb") as f:
        f.write(tflite_bytes)
    print(f"wrote {OUT_PATH} ({len(tflite_bytes)} bytes)")
    validate(keras_model, tflite_bytes, torch_model)
    tflite_accuracy(tflite_bytes)


if __name__ == "__main__":
    main()
