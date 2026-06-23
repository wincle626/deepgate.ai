import dg
import torch

# Download the model
model = dg.from_pretrained("wakeword", num_classes=12)

# Run one forward pass so the model records its input shape (required
# before save_pretrained can write schema.json).
model(torch.zeros(1, 1, 49, 10))

# Export the schema JSON
model.save_pretrained("./wakeword")
