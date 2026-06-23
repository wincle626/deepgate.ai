# DeepGate SDK Examples

This repository contains worked examples of building, training, validating, and
exporting **int8-quantized neural networks** with the DeepGate (`dg`) SDK, and
converting the results to **TensorFlow Lite** for deployment.

Each example follows the same lifecycle:

```
data  →  train / generate  →  validate  →  export to .tflite
```

---

## Environment

All scripts run against the pre-built virtual environment in `pyenv/` (using Python 3.12 under WSL2). Invoke Python through it rather than your system Python:

```bash
# from the deepgate root
./pyenv/bin/python 00.default/train/train.py

# from inside a case folder
cd 01.alexnet && ../pyenv/bin/python alexnet_cifar10.py
```

---

## Repository layout

```
deepgate/
├── README.md                  ← this file
├── pyenv/                     ← Python virtual environment (ignore)
│
├── 00.default/                ← keyword spotting ("wakeword"), Speech Commands v2
│   ├── pretrain/              ← use a model pretrained by DeepGate
│   │   ├── wakeword_test.py
│   │   ├── export_tflite.py
│   │   └── wakeword/          ← dg bundle (model.pt, model.py, config.json,
│   │                            schema.json) + wakeword.tflite
│   └── train/                 ← train a wakeword model from scratch
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── validate.py
│       ├── export_tflite.py
│       └── trained/           ← dg bundle + model.tflite
│
├── 01.alexnet/                ← AlexNet on CIFAR-10
│   ├── alexnet_cifar10.py
│   ├── alexnet_data.py
│   ├── alexnet_validate.py
│   ├── export_tflite.py
│   ├── data/                  ← CIFAR-10 (downloaded)
│   └── alexnet/               ← dg bundle (alexnet.pt, alexnet.py, config.json,
│                                schema.json) + alexnet.tflite
│
└── 02.resnet/                 ← ResNet-18 on CIFAR-10
    ├── resnet18_cifar10.py
    ├── resnet_data.py
    ├── resnet_validate.py
    ├── export_tflite.py
    ├── data/                  ← CIFAR-10 (downloaded)
    └── resnet/                ← dg bundle (resnet18.pt, resnet18.py,
                                 config.json, schema.json) + resnet18.tflite
```

---

## The dg artifact bundle

`model.save_pretrained(<dir>)` writes a **bundle** of four files that fully
describe a trained model:

| File | What it is |
|------|------------|
| `*.pt` | the model weights (PyTorch `state_dict`) |
| `*.py` | the self-contained model definition (`Model(DLGModel)` subclass) |
| `config.json` | the constructor kwargs (e.g. `{"num_classes": 10}`) |
| `schema.json` | the compiler-ready graph spec (int8 weights, scales, shapes); **only written after a forward pass has recorded the input shape** |

- In `00.default`, the bundle keeps dg's default names `model.pt` / `model.py`,
  so it can be reloaded directly with `dg.from_pretrained("<dir>")`.
- In `01.alexnet` / `02.resnet`, the two model files are renamed to
  `alexnet.*` / `resnet18.*`. Because `dg.from_pretrained` expects
  `model.pt`/`model.py`, the matching `*_validate.py` and `export_tflite.py`
  scripts reload the renamed files manually.

---

## 00.default — Wakeword (Speech Commands v2)

### `pretrain/` — use a DeepGate-pretrained model

| File | Usage |
|------|-------|
| `wakeword_test.py` | Download the pretrained `wakeword` model, run one forward pass, and `save_pretrained` it into `wakeword/` (producing `schema.json`). |
| `export_tflite.py` | Convert the `wakeword/` bundle to `wakeword/wakeword.tflite` (int8). Calibrated on synthetic samples — no dataset ships with the pretrained model. |

```bash
cd 00.default/pretrain
../../pyenv/bin/python wakeword_test.py
../../pyenv/bin/python export_tflite.py
```

### `train/` — train from scratch

| File | Usage |
|------|-------|
| `data.py` | Dataset loader: downloads Speech Commands v2 and builds/caches MFCC features. Imported by the others (not run directly). |
| `model.py` | The dg wakeword model definition (`Model(DLGModel)`). |
| `train.py` | Train the model (dg `Trainer`), then `save_pretrained` into `trained/`. |
| `validate.py` | Load `trained/` with `dg.from_pretrained` and print a prediction on one test sample. |
| `export_tflite.py` | Convert `trained/` to `trained/model.tflite` (int8), calibrated on **real** cached MFCC features, and report int8 top-1 accuracy. |

```bash
cd 00.default/train
../../pyenv/bin/python train.py
../../pyenv/bin/python validate.py
../../pyenv/bin/python export_tflite.py
```

---

## 01.alexnet — AlexNet on CIFAR-10

| File | Usage |
|------|-------|
| `alexnet_data.py` | Download CIFAR-10 into `data/` (idempotent). |
| `alexnet_cifar10.py` | Define an AlexNet-style `Model(DLGModel)`, train on CIFAR-10, and write the bundle to `alexnet/` (files renamed to `alexnet.pt` / `alexnet.py`). |
| `alexnet_validate.py` | Reload the `alexnet/` bundle and report CIFAR-10 test accuracy. |
| `export_tflite.py` | Convert the bundle to `alexnet/alexnet.tflite` (int8), calibrated on real CIFAR-10 images, and report int8 top-1 accuracy. |

```bash
cd 01.alexnet
../pyenv/bin/python alexnet_data.py        # 1. get data
../pyenv/bin/python alexnet_cifar10.py     # 2. train + generate bundle
../pyenv/bin/python alexnet_validate.py    # 3. validate
../pyenv/bin/python export_tflite.py       # 4. export tflite
```

Quick smoke runs: prefix training/validation with `MAX_STEPS=10` (and
`EPOCHS=N` for the trainer) to cap the number of batches.

---

## 02.resnet — ResNet-18 on CIFAR-10

Same structure and commands as `01.alexnet`, with `resnet`-prefixed names. The
model is a residual-free, ResNet-18-like topology (skip connections were dropped
for hls4ml compatibility; dg supports true residuals via `QuantResidualAdd` if a
genuine ResNet is wanted).

| File | Usage |
|------|-------|
| `resnet_data.py` | Download CIFAR-10 into `data/` (idempotent). |
| `resnet18_cifar10.py` | Define the ResNet-18-like `Model(DLGModel)`, train, and write the bundle to `resnet/` (files renamed to `resnet18.pt` / `resnet18.py`). |
| `resnet_validate.py` | Reload the `resnet/` bundle and report CIFAR-10 test accuracy. |
| `export_tflite.py` | Convert the bundle to `resnet/resnet18.tflite` (int8), calibrated on real CIFAR-10 images, and report int8 top-1 accuracy. |

```bash
cd 02.resnet
../pyenv/bin/python resnet_data.py
../pyenv/bin/python resnet18_cifar10.py
../pyenv/bin/python resnet_validate.py
../pyenv/bin/python export_tflite.py
```

---

## How the TFLite export works

`export_tflite.py` does **not** rely on a dg→TFLite converter (there isn't one).
Instead it:

1. Loads the trained dg model and rebuilds an **equivalent float Keras model**
   from the dg layers' BatchNorm-folded weights.
2. Runs `tf.lite.TFLiteConverter` **full-integer post-training quantization**
   (int8 weights *and* activations, int8 input/output), calibrated on a
   representative dataset.
3. Validates the result by comparing the torch, Keras, and TFLite outputs, and
   reports int8 accuracy.

TFLite recomputes its own activation scales during PTQ, so the int8 model is a
*faithful* int8 version of the network, not bit-identical to dg's calibration.
The biggest accuracy lever is the representative dataset — these scripts use real
data, so run the corresponding `*_data.py` (or the trainer) first.

---

## Notes & gotchas

- **Run the trainer before validating/exporting.** A freshly generated bundle is
  only as good as the training run. The `MAX_STEPS` smoke runs produce
  near-chance accuracy — do a full run (no env vars) for real numbers.
- **Do not `import torchvision` in a script that also imports `tensorflow`** in
  this environment — it causes a native segfault. The CIFAR `export_tflite.py`
  scripts therefore read the dataset directly from the `data/cifar-10-batches-py`
  pickle instead of via torchvision.
- **`schema.json` can be large** when a model has wide fully-connected layers
  (AlexNet's is ~100 MB because of its 4096-wide FC head; ResNet's is ~48 MB).
  The weights are serialized as JSON arrays.

## Upload to DeepGate Project

1. Need an account first and create a project for specific devices.
2. Currently, only the testcases by DeepGate works in `00.default` folder, by using the schema.json not the .tflite. Need to understand why.
3. `01.alexnet` and `02.resnet` are not working properly when uploading, which has hint of "failed to fetch files". Might because of the large size of model .json. 

## Some Thoughts

1. Could it support ONNX ?
2. Could it automatically convert or quantize the model ?
3. Could it specify the actual available hardware platform for hobby users ?
4. Could it include more detail instructions across different OS ?
5. Could it provide more detail about building up custom models ?
6. Could it support actual hardware generation rather than existing IoT platforms ?
7. Could it consider constomized arithmetic and quantization rather int8 ? 
