"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""
# Import statements
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Function definitions
class MultiHeadAttention(nn.Module):
    def __init__(self, h, dm, dk, dv):
        """Paper: h = 8, dm = 512, dk = 64, dv = 64"""
        super().__init__()
        self.dk = dk
        self.h = h
        # Projection matrices with learnable parameters
        self.W_q = [torch.nn.Parameter(torch.randn(dm, dk)) for _ in range(h)]
        self.W_k = [torch.nn.Parameter(torch.randn(dm, dk)) for _ in range(h)]
        self.W_v = [torch.nn.Parameter(torch.randn(dm, dv)) for _ in range(h)]
        self.W_o = torch.nn.Parameter(torch.randn(h*dv, dm))

    def attention(self, q, k, v) -> torch.Tensor:
        numerator = torch.matmul(q, torch.transpose(k, 0, 1))
        denominator = self.dk ** 0.5
        softmax = F.softmax(torch.divide(numerator, denominator), dim=-1)
        return torch.matmul(softmax, v)

    def forward(self, q, k, v) -> torch.Tensor:
        heads_output = []
        for head in range(self.h):
            # Linearly project q, k, v for each head
            q_proj = torch.matmul(q, self.W_q[head])
            k_proj = torch.matmul(k, self.W_k[head])
            v_proj = torch.matmul(v, self.W_v[head])
            # Calculate attention for each linear projection
            heads_output.append(self.attention(q_proj, k_proj, v_proj))
        return torch.matmul(torch.hstack(heads_output), self.W_o)


class PFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 2048), nn.ReLU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x) -> torch.Tensor:
        return self.layer(x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.feedforward = PFFN()
        self.norm = nn.LayerNorm([512])

    def forward(self, x) -> torch.Tensor:
        x = self.norm(x + self.attn(x, x, x))
        x = self.norm(x + self.feedforward(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.feedforward = PFFN()
        self.norm = nn.LayerNorm([512])

    def forward(self, x, enc_key, enc_val) -> torch.Tensor:
        x = self.norm(x + self.attn(x, x, x))
        x = self.norm(x + self.attn(enc_key, x, enc_val))
        x = self.norm(x + self.feedforward(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_stack = [EncoderLayer() for _ in range(6)]
        self.decoder_stack = [DecoderLayer() for _ in range(6)]

    def forward(self, x, y):
        for i in self.encoder_stack:
            x = i(x)
        for i in self.decoder_stack:
            x = i(y, x, x)


def main():
    pass


if __name__ == "__main__":
    main()
