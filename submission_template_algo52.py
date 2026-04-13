"""
Submission template (USES trained Transformer from Algo5).
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_SEQ_BUFFER = []   # stores past states
MAX_SEQ_LEN = 2000


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ===== SAME ARCHITECTURE AS Algo5 =====

    class RotaryEmbedding:
        def __init__(self, dim, maxSeqLen=4096):
            invFreq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(maxSeqLen).float()
            freqs = torch.einsum("i,j->ij", t, invFreq)
            self.cos = torch.cos(freqs)
            self.sin = torch.sin(freqs)

        def apply(self, x, seqLen):
            cos = self.cos[:seqLen].unsqueeze(0).unsqueeze(0)
            sin = self.sin[:seqLen].unsqueeze(0).unsqueeze(0)

            x1, x2 = x[..., ::2], x[..., 1::2]
            xRot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

            return x * cos.repeat_interleave(2, dim=-1) + \
                   xRot * sin.repeat_interleave(2, dim=-1)


    class CausalSelfAttention(nn.Module):
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

            mask = torch.tril(torch.ones(T, T, device=x.device))
            att = att.masked_fill(mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            out = att @ v

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out(out)


    class Block(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = CausalSelfAttention(dim, heads)
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


    class TransformerPolicy(nn.Module):
        def __init__(self):
            super().__init__()

            self.embed = nn.Linear(18, 128)
            self.blocks = nn.ModuleList([Block(128, 4) for _ in range(4)])
            self.ln = nn.LayerNorm(128)
            self.policyHead = nn.Linear(128, 5)

        def forward(self, x):
            x = self.embed(x)

            for block in self.blocks:
                x = block(x)

            x = self.ln(x)
            return self.policyHead(x[:, -1])


    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "algo52/weights.pth")

    import torch
    model = TransformerPolicy()

    model.load_state_dict(torch.load(wpath, map_location="cpu"))  # IMPORTANT
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _SEQ_BUFFER

    _load_once()

    import torch

    # append current state
    _SEQ_BUFFER.append(obs.astype(np.float32))

    # truncate sequence
    if len(_SEQ_BUFFER) > MAX_SEQ_LEN:
        _SEQ_BUFFER = _SEQ_BUFFER[-MAX_SEQ_LEN:]

    seq = torch.tensor(_SEQ_BUFFER).unsqueeze(0)  # (1, T, 18)

    with torch.no_grad():
        logits = _MODEL(seq).squeeze(0).numpy()

    action = int(np.argmax(logits))
    return ACTIONS[action]