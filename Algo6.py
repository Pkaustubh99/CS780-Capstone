import torch
import torch.nn as nn
import torch.nn.functional as F





from obelix import OBELIX



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------
# RoPE
# -----------------------
class RotaryEmbedding:
    def __init__(self, dim, maxSeqLen=2048):
        invFreq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(maxSeqLen).float()
        freqs = torch.einsum("i,j->ij", t, invFreq)
        self.cos = torch.cos(freqs).to(DEVICE)
        self.sin = torch.sin(freqs).to(DEVICE)

    def apply(self, x, T):
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., ::2], x[..., 1::2]
        xRot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

        return x * cos.repeat_interleave(2, dim=-1) + xRot * sin.repeat_interleave(2, dim=-1)


# -----------------------
# Attention
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

        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# -----------------------
# Block
# -----------------------
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
# PPO Model
# -----------------------
class PPOTransformer(nn.Module):
    def __init__(self, stateDim, actionDim, dim=64, depth=2, heads=2):
        super().__init__()

        self.embed = nn.Linear(stateDim, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)

        self.policy = nn.Linear(dim, actionDim)
        self.value = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)

        last = x[:, -1]

        return self.policy(last), self.value(last).squeeze(-1)

    def act(self, x, temperature=1.5):
        logits, value = self.forward(x)
        probs = torch.softmax(logits / temperature, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value





DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
CLIP = 0.2
LR = 3e-4
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5
MAX_SEQ = 256


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]



import os
import torch

WEIGHT_PATH = "algo6/weights.pth"


def saveModel(model, optimizer=None, path=WEIGHT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    torch.save(state, path)


def loadModel(model, optimizer=None, path=WEIGHT_PATH, device="cpu"):
    if not os.path.exists(path):
        print("No weights found, training from scratch")
        return

    checkpoint = torch.load(path, map_location=device)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    print("Loaded weights from", path)


def computeGAE(rewards, values, dones):
    returns = []
    G = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0
        G = r + GAMMA * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=DEVICE)


def train(env, episodes=10000):
    model = PPOTransformer(stateDim=18, actionDim=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loadModel(model, optimizer,WEIGHT_PATH,device=DEVICE)

    for ep in range(episodes):
        state = env.reset()

        states, actions, logProbs, rewards, values, dones = [], [], [], [], [], []

        done = False

        while not done:
            stateTensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            states.append(stateTensor)

            seq = torch.stack(states[-MAX_SEQ:]).unsqueeze(0)

            action, logProb, entropy, value = model.act(seq)

            actionIdx = action.item()
            nextState, reward, done = env.step(ACTIONS[actionIdx])

            actions.append(actionIdx)
            logProbs.append(logProb)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            state = nextState

        returns = computeGAE(rewards, values, dones)
        values = torch.stack(values)

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        seq = torch.stack(states).unsqueeze(0)
        logits, newValues = model(seq)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        newLogProbs = dist.log_prob(torch.tensor(actions, device=DEVICE))
        entropy = dist.entropy().mean()

        oldLogProbs = torch.stack(logProbs).detach()

        ratio = torch.exp(newLogProbs - oldLogProbs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantages

        policyLoss = -torch.min(surr1, surr2).mean()
        valueLoss = (returns - newValues).pow(2).mean()

        loss = policyLoss + VALUE_COEF * valueLoss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            print(f"Ep {ep} | Loss {loss.item():.3f}")
            saveModel(model,optimizer,WEIGHT_PATH)
env = OBELIX(scaling_factor=5, difficulty=1, wall_obstacles=False, arena_size=500)
train(env)

