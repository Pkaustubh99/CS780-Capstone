import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_STATE_BUFFER = []
_ACTION_BUFFER = []

MAX_SEQ = 256
START_TOKEN = 5  # action_dim


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    # ===== EXACT MODEL FROM Algo7 =====
    class Attention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.headDim = dim // heads
            self.qkv = nn.Linear(dim, dim * 3)
            self.out = nn.Linear(dim, dim)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, T, self.heads, self.headDim).transpose(1, 2)
            k = k.view(B, T, self.heads, self.headDim).transpose(1, 2)
            v = v.view(B, T, self.heads, self.headDim).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) / (self.headDim ** 0.5)

            mask = torch.tril(torch.ones(T, T, device=x.device))
            att = att.masked_fill(mask == 0, float('-inf'))

            att = torch.softmax(att, dim=-1)
            out = att @ v

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out(out)

    class Block(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, heads)
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class PPOTransformer(nn.Module):
        def __init__(self, stateDim, actionDim, dim=64, depth=2, heads=2):
            super().__init__()
            self.stateEmbed = nn.Linear(stateDim, dim)
            self.actionEmbed = nn.Embedding(actionDim + 1, dim)

            self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
            self.ln = nn.LayerNorm(dim)

            self.policy = nn.Linear(dim, actionDim)
            self.value = nn.Linear(dim, 1)

        def forward(self, states, actions):
            s = self.stateEmbed(states)
            a = self.actionEmbed(actions)
            x = s + a

            for b in self.blocks:
                x = b(x)

            x = self.ln(x)
            return self.policy(x), self.value(x).squeeze(-1)

    import torch

    model = PPOTransformer(stateDim=18, actionDim=5)

    checkpoint = torch.load(wpath, map_location="cpu")
    model.load_state_dict(checkpoint["model"])  # ✅ correct

    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STATE_BUFFER, _ACTION_BUFFER

    _load_once()

    import torch

    # ===== BUILD SEQUENCE =====
    _STATE_BUFFER.append(torch.tensor(obs, dtype=torch.float32))
    if len(_ACTION_BUFFER) == 0:
        _ACTION_BUFFER.append(START_TOKEN)

    stateSeq = torch.stack(_STATE_BUFFER[-MAX_SEQ:]).unsqueeze(0)
    actionSeq = torch.tensor(_ACTION_BUFFER[-MAX_SEQ:], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits, _ = _MODEL(stateSeq, actionSeq)
        logits = logits[:, -1]  # last step
        action = int(torch.argmax(logits, dim=-1).item())

    _ACTION_BUFFER.append(action)

    return ACTIONS[action]