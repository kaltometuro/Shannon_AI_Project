from model import AdaptiveECC
import torch

model = AdaptiveECC()

bits = torch.randint(0, 2, (1, 16)).float()
noise = torch.tensor([[0.1]])

output = model(bits, noise)

print("Output shape:", output.shape)
input("Press Enter to close...")