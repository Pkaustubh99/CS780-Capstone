import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_STATE_BUF = []
_ACTION_BUF = []
_REWARD_BUF = []

MAX_SEQ = 256
START_TOKEN = 5


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "algo9/weights.pth")

    # -----------------------
    # RoPE
    # -----------------------
    class RotaryEmbedding:
        def __init__(self, dim, maxSeqLen=2048):
            invFreq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(maxSeqLen).float()
            freqs = torch.einsum("i,j->ij", t, invFreq)
            self.cos = torch.cos(freqs)
            self.sin = torch.sin(freqs)

        def apply(self, x, T):
            cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0).unsqueeze(0)

            x1, x2 = x[..., ::2], x[..., 1::2]
            xRot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

            return x * cos.repeat_interleave(2, dim=-1) + \
                   xRot * sin.repeat_interleave(2, dim=-1)

    # -----------------------
    # Transformer
    # -----------------------
    class Attention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.headDim = dim // heads

            self.qkv = nn.Linear(dim, dim * 3)
            self.out = nn.Linear(dim, dim)
            self.rope = RotaryEmbedding(self.headDim)

        def forward(self, x):
            B, T, C = x.shape

            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, T, self.heads, self.headDim).transpose(1, 2)
            k = k.view(B, T, self.heads, self.headDim).transpose(1, 2)
            v = v.view(B, T, self.heads, self.headDim).transpose(1, 2)

            q = self.rope.apply(q, T)
            k = self.rope.apply(k, T)

            att = (q @ k.transpose(-2, -1)) / (self.headDim ** 0.5)

            mask = torch.tril(torch.ones(T, T))
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

    # -----------------------
    # MODEL
    # -----------------------
    class PPOTransformer(nn.Module):
        def __init__(self, stateDim, actionDim, dim=64):
            super().__init__()

            self.stateEmbed = nn.Linear(stateDim, dim)
            self.actionEmbed = nn.Embedding(actionDim + 1, dim)
            self.rewardEmbed = nn.Linear(1, dim)

            self.blocks = nn.ModuleList([Block(dim, 2) for _ in range(2)])
            self.ln = nn.LayerNorm(dim)

            self.policy = nn.Linear(dim, actionDim)
            self.value = nn.Linear(dim, 1)

        def forward(self, states, actions, rewards):
            s = self.stateEmbed(states)
            a = self.actionEmbed(actions)
            r = self.rewardEmbed(rewards.unsqueeze(-1))

            x = s + a + r

            for b in self.blocks:
                x = b(x)

            x = self.ln(x)

            return self.policy(x), self.value(x).squeeze(-1)

    import torch

    model = PPOTransformer(stateDim=18, actionDim=5)

    checkpoint = torch.load(wpath, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _STATE_BUF, _ACTION_BUF, _REWARD_BUF

    _load_once()

    import torch

    # init
    if len(_ACTION_BUF) == 0:
        _ACTION_BUF.append(START_TOKEN)
        _REWARD_BUF.append(0.0)

    # append state
    _STATE_BUF.append(torch.tensor(obs, dtype=torch.float32))

    # build sequence
    stateSeq = torch.stack(_STATE_BUF[-MAX_SEQ:]).unsqueeze(0)
    actionSeq = torch.tensor(_ACTION_BUF[-MAX_SEQ:], dtype=torch.long).unsqueeze(0)
    rewardSeq = torch.tensor(_REWARD_BUF[-MAX_SEQ:], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, _ = _MODEL(stateSeq, actionSeq, rewardSeq)  # ✅ FIX
        logits = logits[:, -1]
        action = int(torch.argmax(logits, dim=-1).item())

    # update buffers
    _ACTION_BUF.append(action)
    _REWARD_BUF.append(_REWARD_BUF[-1] if len(_REWARD_BUF) > 0 else 0.0)

    # safety reset
    if len(_STATE_BUF) > 1000:
        _STATE_BUF.clear()
        _ACTION_BUF.clear()
        _REWARD_BUF.clear()

    return ACTIONS[action]