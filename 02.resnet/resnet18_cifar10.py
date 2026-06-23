# resnet18_cifar10.py
# ResNet-18-like CIFAR-10 model expressed with the DeepGate (dg) SDK.
#
# Running this script trains the model and writes a dg artifact bundle to
# ./resnet/ :  config.json, resnet18.pt, resnet18.py, schema.json
# (save_pretrained emits model.pt/model.py; they are renamed to resnet18.*).
#
# This mirrors the original residual-free ResNet-18 topology (the skip-adds
# were dropped for hls4ml). Each conv keeps its exact channels/kernel/stride/
# padding; BatchNorm is folded for schema export and ReLU is fused via
# act_func="relu". dg supports true residuals via QuantResidualAdd if a
# genuine ResNet is wanted later.

import torch
import torch.nn as nn

from dg.base_model import DLGModel
from dg.layer import Flatten, QuantAvgPool2d, QuantConv2d, QuantLinear


class Model(DLGModel):
    """ResNet-18-like int8-quantized classifier for 32x32x3 CIFAR-10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _conv(in_ch, out_ch, stride):
        # 3x3 "same" conv (pad 1) + BN + ReLU, no bias (folded into BN)
        return QuantConv2d(in_ch, out_ch, 3, stride, 1, bias=False, act_func="relu", bn=True)

    def _basic_block(self, in_ch, out_ch, stride):
        # two 3x3 conv-bn-relu; first conv carries the stride (residual-free)
        return [self._conv(in_ch, out_ch, stride), self._conv(out_ch, out_ch, 1)]

    def build_model_graph(self):
        layers = [self._conv(3, 64, 1)]                      # stem        -> 32x32x64
        layers += self._basic_block(64, 64, 1)              # stage1      -> 32x32x64
        layers += self._basic_block(64, 64, 1)
        layers += self._basic_block(64, 128, 2)             # stage2 (s2) -> 16x16x128
        layers += self._basic_block(128, 128, 1)
        layers += self._basic_block(128, 256, 2)            # stage3 (s2) -> 8x8x256
        layers += self._basic_block(256, 256, 1)
        layers += self._basic_block(256, 512, 2)            # stage4 (s2) -> 4x4x512
        layers += self._basic_block(512, 512, 1)
        layers += [
            QuantAvgPool2d(4, 4),                            # -> 1x1x512
            Flatten(),                                       # -> 512
            QuantLinear(512, self.num_classes, bias=True),  # logits (no act)
        ]
        return layers


def main():
    import os

    import torch.optim as optim
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "resnet")
    data_dir = os.path.join(here, "data")

    # EPOCHS / MAX_STEPS are overridable via env for quick smoke runs.
    epochs = int(os.environ.get("EPOCHS", "1"))
    max_steps = int(os.environ.get("MAX_STEPS", "0")) or None  # 0 -> full epoch

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 (ToTensor scales to [0,1], matching the divide-by-255 behavior)
    transform = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = Model(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())  # mirrors the original

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for step, (images, labels) in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for step, (images, labels) in enumerate(test_loader):
                if max_steps and step >= max_steps:
                    break
                images, labels = images.to(device), labels.to(device)
                test_correct += model(images).argmax(1).eq(labels).sum().item()
                test_total += labels.size(0)
        test_acc = 100.0 * test_correct / max(test_total, 1)
        print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # Record input shape (needed for schema.json) then export the dg bundle.
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=device))
    model.save_pretrained(out_dir)  # writes model.pt/model.py/config.json/schema.json

    # Rename to the requested resnet18.* names.
    os.replace(os.path.join(out_dir, "model.pt"), os.path.join(out_dir, "resnet18.pt"))
    os.replace(os.path.join(out_dir, "model.py"), os.path.join(out_dir, "resnet18.py"))
    print("Wrote dg bundle to", out_dir, "->", sorted(os.listdir(out_dir)))


if __name__ == "__main__":
    main()
