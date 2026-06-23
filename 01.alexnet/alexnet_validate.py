# alexnet_validate.py
# Validate the saved dg bundle in ./alexnet/ (alexnet.pt, alexnet.py,
# config.json, schema.json) by reloading the model and measuring its accuracy
# on the CIFAR-10 test set.
#
# The bundle uses alexnet.* names (not dg's default model.pt/model.py), so this
# reproduces what dg.from_pretrained does, against the renamed files.
#
#     ../pyenv/bin/python alexnet_validate.py        # full test set
#     MAX_STEPS=20 ../pyenv/bin/python alexnet_validate.py   # quick check

import importlib.util
import json
import os

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "alexnet")
DATA_DIR = os.path.join(HERE, "data")


def load_bundle(bundle):
    """Reload a model from a dg bundle that uses alexnet.* file names."""
    src = os.path.join(bundle, "alexnet.py")
    weights = os.path.join(bundle, "alexnet.pt")
    config_path = os.path.join(bundle, "config.json")
    for p in (src, weights, config_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing bundle file: {p}")

    # import alexnet.py as a throwaway module to get the Model class
    spec = importlib.util.spec_from_file_location("_alexnet_bundle", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = json.loads(open(config_path).read())
    model = module.Model(**config)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)          # strict: fresh graph matches saved keys
    model.eval()
    return model, config


def main():
    max_steps = int(os.environ.get("MAX_STEPS", "0")) or None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_bundle(BUNDLE)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded bundle: {BUNDLE}")
    print(f"  config={config}  params={n_params:,}  layers={len(model.model_graph)}")

    # cross-check schema.json against the live graph
    schema = json.loads(open(os.path.join(BUNDLE, "schema.json")).read())
    print(f"  schema dg_version={schema['dg_version']} "
          f"schema_layers={len(schema['model_graph'])}")
    assert len(schema["model_graph"]) == len(model.model_graph), "schema/graph layer mismatch"

    test_set = torchvision.datasets.CIFAR10(
        DATA_DIR, train=False, download=True, transform=T.Compose([T.ToTensor()]))
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    correct, total = 0, 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loader):
            if max_steps and step >= max_steps:
                break
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    print(f"CIFAR-10 test accuracy: {100.0 * correct / max(total, 1):.2f}%  "
          f"({correct}/{total})")


if __name__ == "__main__":
    main()
