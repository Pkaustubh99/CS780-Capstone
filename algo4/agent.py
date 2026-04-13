"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""
"""
Submission template (USES trained weights from Algo4 PPO-LSTM).
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_HIDDEN = None


def _load_once():
    global _MODEL, _HIDDEN
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    # ===== SAME ARCHITECTURE AS Algo4 =====
    class ActorCritic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()

            self.fc = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
            )

            self.lstm = nn.LSTM(128, 128, batch_first=True)

            self.policy = nn.Linear(128, action_dim)
            self.value = nn.Linear(128, 1)

        def forward(self, x, hidden):
            x = self.fc(x)
            x = x.unsqueeze(1)
            x, hidden = self.lstm(x, hidden)
            x = x.squeeze(1)

            logits = self.policy(x)
            value = self.value(x)
            return logits, value, hidden

    model = ActorCritic(state_dim=18, action_dim=5)

    checkpoint = torch.load(wpath, map_location="cpu")
    model.load_state_dict(checkpoint["model"])  # IMPORTANT

    model.eval()

    _MODEL = model

    # init hidden state
    _HIDDEN = (
        torch.zeros(1, 1, 128),
        torch.zeros(1, 1, 128),
    )


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HIDDEN
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits, _, _HIDDEN = _MODEL(x, _HIDDEN)
        logits = logits.squeeze(0).numpy()

    action = int(np.argmax(logits))
    return ACTIONS[action]