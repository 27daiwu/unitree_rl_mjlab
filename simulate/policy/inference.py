"""Inference-only TorchScript boundary: raw observations in, raw actions out."""
from pathlib import Path
import numpy as np
import torch

class Policy:
    def __init__(self, artifact: str | Path):
        self.module=torch.jit.load(str(artifact),map_location='cpu').eval()

    @torch.inference_mode()
    def __call__(self, observation):
        obs=np.asarray(observation,dtype=np.float32)
        if obs.shape!=(274,) or not np.isfinite(obs).all():
            raise ValueError('Expected finite raw 274D observation')
        action=self.module(torch.from_numpy(obs.copy()).unsqueeze(0)).squeeze(0).numpy()
        if action.shape!=(29,) or not np.isfinite(action).all():
            raise ValueError('Policy must return finite raw 29D action')
        return action.copy()
