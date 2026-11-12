"""Inference-only S3 architecture with the original state_dict keys.

Keep the training Policy and its imports untouched. All parameters and scale
buffers are loaded strictly from S3; no simulator or training modules are loaded.
"""

from __future__ import annotations

import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        layers = []
        for size in hidden:
            layers.extend((nn.Linear(inputs, size), nn.SiLU()))
            inputs = size
        layers.append(nn.Linear(inputs, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, value):
        return self.net(value)


class _Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = _MLP(689, cfg["encoder_hidden"], cfg["latent_dim"])
        frame = torch.cat(
            [
                torch.full((size,), 1.0 / divisor)
                for size, divisor in zip((29, 29, 3, 3, 3, 1), cfg["goal_term_scale"], strict=True)
            ]
        )
        self.register_buffer("scale", torch.cat((frame.repeat(10), torch.ones(9))))

    def forward(self, goal):
        return torch.tanh(self.mlp(goal * self.scale))


class _Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = _MLP(cfg["latent_dim"] + 930, cfg["decoder_hidden"], 29)
        scale = torch.cat(
            [
                torch.full((10 * size,), 1.0 / divisor)
                for size, divisor in zip((29, 29, 3, 3, 29), cfg["obs_term_scale"], strict=True)
            ]
        )
        self.register_buffer("scale", scale)

    def forward(self, latent, obs):
        return self.mlp(torch.cat((latent, obs * self.scale), dim=-1))


class FrozenS3Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoders = nn.ModuleDict({"motion": _Encoder(cfg)})
        self.decoder = _Decoder(cfg)

    def forward(self, obs, goal):
        return self.decoder(self.encoders["motion"](goal), obs)
