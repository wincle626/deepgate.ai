import torch

from dg import eval as dg_eval, train
from model import Model
from data import load_speech_commands

device = "cpu"   # set to "cuda" or "mps" if available

train_dataset = load_speech_commands(subset="train+val", device=device)
test_dataset = load_speech_commands(subset="testing", device=device)

torch.manual_seed(42)
model = Model()

train.Trainer(
    model, train_dataset, test_dataset,
    val_epoch=1, epochs=30, batch_size=128, lr=0.001,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"weight_decay": 0.01},
    device=device,
).train()

train_acc = dg_eval.get_acc(model, train_dataset, batch_size=128, device=device)
test_acc = dg_eval.get_acc(model, test_dataset, batch_size=128, device=device)
print(f"Train acc: {100 * train_acc:.2f}%")
print(f"Test acc:  {100 * test_acc:.2f}%")

model.save_pretrained("./trained")
