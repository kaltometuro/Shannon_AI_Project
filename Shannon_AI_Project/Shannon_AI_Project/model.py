import torch
import torch.nn as nn
import torch.nn.functional as F

K = 16
N = 24


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(K + 1, 64)
        self.fc2 = nn.Linear(64, N)
        self.gate1 = nn.Linear(1, 16)
        self.gate2 = nn.Linear(16, N)

    def forward(self, bits, noise):
        x = torch.cat([bits, noise], dim=1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        gate = F.relu(self.gate1(noise))
        gate = torch.sigmoid(self.gate2(gate))

        encoded = x * gate
        return encoded


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N + 1, 64)
        self.fc2 = nn.Linear(64, K)

    def forward(self, received, noise):
        x = torch.cat([received, noise], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AdaptiveECC(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def channel(self, encoded, noise):
        flips = (torch.rand_like(encoded) < noise).float()
        corrupted = encoded * (1 - 2 * flips)
        return corrupted

    def forward(self, bits, noise):
        encoded = self.encoder(bits, noise)
        corrupted = self.channel(encoded, noise)
        decoded = self.decoder(corrupted, noise)
        return decoded