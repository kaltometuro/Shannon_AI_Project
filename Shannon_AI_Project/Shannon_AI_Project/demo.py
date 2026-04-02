import torch
from model import AdaptiveECC, K

# Load trained model
model = AdaptiveECC()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Generate one sample message
bits = torch.randint(0, 2, (1, K)).float()
noise = torch.tensor([[0.1]])

# Run through system
output = model(bits, noise)
pred = (torch.sigmoid(output) > 0.5).float()

print("\n--- DEMO ---")
print("Original Bits : ", bits)
print("Recovered Bits:", pred)