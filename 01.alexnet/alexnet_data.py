# alexnet_data.py
# Download the CIFAR-10 dataset into ./data (relative to this file).
# Idempotent: torchvision skips the download if the files are already present.
#
#     ../pyenv/bin/python alexnet_data.py

import os

import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
    print(f"CIFAR-10 ready in {DATA_DIR}")
    print(f"  train samples: {len(train_set)}")
    print(f"  test samples:  {len(test_set)}")


if __name__ == "__main__":
    main()
