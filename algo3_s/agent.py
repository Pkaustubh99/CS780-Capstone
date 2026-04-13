"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""
"""
Submission template (USES trained weights from Algo3 SAC).
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    # ===== SAME POLICY ARCHITECTURE AS Algo3 =====
    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(18, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 5),
            )

        def forward(self, x):
            logits = self.net(x)
            return torch.softmax(logits, dim=-1)

    model = PolicyNet()

    checkpoint = torch.load(wpath, map_location="cpu")
    model.load_state_dict(checkpoint["policy"])   # IMPORTANT

    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        probs = _MODEL(x).squeeze(0).numpy()

    action = int(np.argmax(probs))
    return ACTIONS[action]