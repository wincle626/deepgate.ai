import dg
from data import load_speech_commands

device = "cpu"   # set to "cuda" or "mps" if available

test_dataset = load_speech_commands(subset="testing", device=device)
model = dg.from_pretrained("./trained", device=device)

x, y = test_dataset[0]
pred = model(x.unsqueeze(0)).argmax(dim=1).item()
print(f"Predicted: {pred}, Actual: {int(y)}")
