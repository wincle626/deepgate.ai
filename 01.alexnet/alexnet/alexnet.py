# alexnet_cifar10.py
# AlexNet-style CIFAR-10 model expressed with the DeepGate (dg) SDK.
#
# Running this script trains the model and writes a dg artifact bundle to
# ./alexnet/ :  config.json, alexnet.pt, alexnet.py, schema.json
# (save_pretrained emits model.pt/model.py; they are renamed to alexnet.*).
#
# Layer mapping vs. the original PyTorch AlexNet:
#   - Conv layers keep their exact channels / kernel / stride / padding (dg
#     QuantConv2d supports padding); bn is folded for schema export.
#   - dg MaxPool2d has no padding, so the padded 3x3 pools are replaced with
#     no-pad pools that produce the same output sizes on 32x32 input.
#   - ReLU is fused into each Quant layer via act_func="relu".

import torch
import torch.nn as nn

from dg.base_model import DLGModel
from dg.layer import Flatten, MaxPool2d, QuantConv2d, QuantLinear


class Model(DLGModel):
    """AlexNet-style int8-quantized classifier for 32x32x3 CIFAR-10."""

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

    def build_model_graph(self):
        return [
            # Conv1: 96 @ 11x11 s4, no pad, + BN + ReLU      -> 6x6x96
            QuantConv2d(3, 96, 11, 4, 0, bias=True, act_func="relu", bn=True),
            MaxPool2d(3, 2),                                 # -> 2x2x96
            # Conv2: 256 @ 5x5 s1 "same" (pad2), + BN + ReLU -> 2x2x256
            QuantConv2d(96, 256, 5, 1, 2, bias=True, act_func="relu", bn=True),
            MaxPool2d(2, 2),                                 # -> 1x1x256
            # Conv3/4/5: 384 @ 3x3 s1 "same" (pad1), + ReLU  -> 1x1x384
            QuantConv2d(256, 384, 3, 1, 1, bias=True, act_func="relu"),
            QuantConv2d(384, 384, 3, 1, 1, bias=True, act_func="relu"),
            QuantConv2d(384, 384, 3, 1, 1, bias=True, act_func="relu"),
            Flatten(),                                       # -> 384
            QuantLinear(384, 4096, bias=True, act_func="relu"),
            QuantLinear(4096, 4096, bias=True, act_func="relu"),
            QuantLinear(4096, self.num_classes, bias=True),  # logits (no act)
        ]


def main():
    import os

    import torch.optim as optim
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "alexnet")
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
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

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

    # Rename to the requested alexnet.* names.
    os.replace(os.path.join(out_dir, "model.pt"), os.path.join(out_dir, "alexnet.pt"))
    os.replace(os.path.join(out_dir, "model.py"), os.path.join(out_dir, "alexnet.py"))
    print("Wrote dg bundle to", out_dir, "->", sorted(os.listdir(out_dir)))


if __name__ == "__main__":
    main()
