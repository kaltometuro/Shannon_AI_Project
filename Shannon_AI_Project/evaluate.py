import torch
from model import AdaptiveECC, K

# Load trained model
model = AdaptiveECC()
model.load_state_dict(torch.load("model.pth"))
model.eval()

noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]

print("\nEvaluation Results (BER):\n")

for n in noise_levels:

    bits = torch.randint(0, 2, (2000, K)).float()
    noise = torch.full((2000, 1), n)

    with torch.no_grad():
        output = model(bits, noise)

    pred = (torch.sigmoid(output) > 0.5).float()
    ber = (pred != bits).float().mean().item()

    print(f"Noise {n:.2f} → BER {ber:.4f}")