import torch
import torch.nn.functional as F
from model import AdaptiveECC, K

# Create model
model = AdaptiveECC()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

EPOCHS = 120

for epoch in range(EPOCHS):

    # Generate random binary data
    bits = torch.randint(0, 2, (1024, K)).float()

    # Random noise between 0 and 0.2
    noise = torch.rand(512, 1) * 0.2

    # Forward pass
    output = model(bits, noise)

    # Loss
    loss = F.binary_cross_entropy_with_logits(output, bits)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 5 == 0:
        pred = (torch.sigmoid(output) > 0.5).float()
        ber = (pred != bits).float().mean().item()

        print(f"Epoch {epoch}, Loss {loss.item():.4f}, BER {ber:.4f}")

# Save trained model
torch.save(model.state_dict(), "model.pth")
print("Training complete. Model saved as model.pth")