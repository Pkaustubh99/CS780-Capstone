"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""
"""
Submission template (USES trained weights from Algo2 PPO).
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

    # ===== SAME ARCHITECTURE AS Algo2 =====
    class ActorCritic(nn.Module):
        def __init__(self, stateDim, actionDim, hiddenDim=64):
            super().__init__()

            self.shared = nn.Sequential(
                nn.Linear(stateDim, hiddenDim),
                nn.ReLU(),
                nn.Linear(hiddenDim, hiddenDim),
                nn.ReLU()
            )

            self.actor = nn.Linear(hiddenDim, actionDim)
            self.critic = nn.Linear(hiddenDim, 1)

        def forward(self, state):
            x = self.shared(state)
            return self.actor(x), self.critic(x)

    model = ActorCritic(stateDim=18, actionDim=5)

    checkpoint = torch.load(wpath, map_location="cpu")
    model.load_state_dict(checkpoint["model"])   # IMPORTANT

    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits, _ = _MODEL(x)
        logits = logits.squeeze(0).numpy()

    action = int(np.argmax(logits))
    return ACTIONS[action]