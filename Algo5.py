import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTROPY_COEF = 0.01

# -----------------------
# RoPE
# -----------------------
class RotaryEmbedding:
    def __init__(self, dim, maxSeqLen=4096):
        self.dim = dim
        invFreq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(maxSeqLen).float()
        freqs = torch.einsum("i,j->ij", t, invFreq)
        self.cos = torch.cos(freqs).to(DEVICE)
        self.sin = torch.sin(freqs).to(DEVICE)

    def apply(self, x, seqLen):
        cos = self.cos[:seqLen].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seqLen].unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., ::2], x[..., 1::2]
        xRot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

        return x * cos.repeat_interleave(2, dim=-1) + xRot * sin.repeat_interleave(2, dim=-1)


# -----------------------
# Attention
# -----------------------
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


# -----------------------
# Transformer Block
# -----------------------
class Block(nn.Module):
    def __init__(self, dim, heads, mlpRatio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlpRatio),
            nn.GELU(),
            nn.Linear(dim * mlpRatio, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# -----------------------
# Policy Model
# -----------------------
class TransformerPolicy(nn.Module):
    def __init__(self, stateDim, actionDim, dim=128, depth=4, heads=4):
        super().__init__()

        self.embed = nn.Linear(stateDim, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)

        self.policyHead = nn.Linear(dim, actionDim)

    def forward(self, x):
        # x: (B, T, stateDim)
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.policyHead(x[:, -1])  # last timestep
        return logits

    def act(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()


# -----------------------
# Save / Load
# -----------------------
def saveModel(model, path="algo5/weights.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def loadModel(model, path="algo5/weights.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print("Loaded weights from", path)
    else:
        print("No weights found, training from scratch")




# model = TransformerPolicy(stateDim, actionDim).to("cuda" if torch.cuda.is_available() else "cpu")
#
# loadModel(model)
#
# # dummy sequence (batch=1, seq=2000)
# x = torch.randn(1, 2000, stateDim).to(next(model.parameters()).device)
#
# action = model.act(x)
# print("Action:", action)
#
# saveModel(model)


import torch
import torch.nn.functional as F

from obelix import  OBELIX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Hyperparams
# -----------------------
STATE_DIM = 18    # change to your env
ACTION_DIM = 5
MAX_SEQ_LEN = 2000
GAMMA = 0.99
LR = 3e-4


# -----------------------
# Utils
# -----------------------
def computeReturns(rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=DEVICE)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
# -----------------------
# Training Loop
# -----------------------

import csv

base_path = "algo5/base_path.csv"
edited_path = "algo5/edited_path.csv"


def appendRow(path = base_path, rowData = []):

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rowData)


def train(env, episodes=1000):

    push_enabled = False
    model = TransformerPolicy(18, 5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loadModel(model)

    for ep in range(episodes):
        state = env.reset()

        states = []
        actions = []
        rewards = []
        base_rewards = []

        done = False

        while not done:
            stateTensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)

            states.append(stateTensor)

            # build sequence (truncate if too long)
            seq = torch.stack(states[-MAX_SEQ_LEN:]).unsqueeze(0)

            logits = model(seq)
            probs = F.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)




            epsilon = 0.6 - (0.4 * (ep/1000))  # 20% random
            if torch.rand(1).item() < epsilon:
                actionIndex = torch.randint(0, ACTION_DIM, (1,)).item()
            else:
                actionIndex = dist.sample().item()




            nextState, reward, done = env.step(ACTIONS[actionIndex],render= True)

            base_rewards.append(reward)


            if reward >= 90:
                print("push enabled")
                push_enabled = True

            # we are removing step penalty in push state. (reward shaping) to stop penalise movement in push step that causes it to keep rotating and avoid the box

            if push_enabled:
                reward += 1
                if reward < 0 and reward > -100:
                    print("reward is negative", reward)




            actions.append(actionIndex)
            rewards.append(reward)

            state = nextState

        # -----------------------
        # Compute loss
        # -----------------------
        returns = computeReturns(rewards)

        appendRow(edited_path, rewards)
        appendRow(base_path, base_rewards)

        seq = torch.stack(states).unsqueeze(0)  # (1, T, stateDim)

        



        # logits = model(seq)  # ONLY ONCE
        # logProbs = torch.log_softmax(logits, dim=-1)
        #
        # selectedLogProbs = logProbs[0, actions]  # gather all at once
        #
        # loss = -(selectedLogProbs * returns).mean()
        #
        # loss = loss / len(actions)
        logits = model(seq)

        temperature = 1.5  # try 1.5 → 2.5

        probs = torch.softmax(logits / temperature, dim=-1)
        # probs = torch.softmax(logits, dim=-1)
        logProbs = torch.log_softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)

        selectedLogProbs = logProbs[0, actions]

        entropy = dist.entropy().mean()

        loss = -(selectedLogProbs * returns).mean() - 0.01 * entropy




        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            print(f"Episode {ep}, Loss: {loss.item():.4f}")

            saveModel(model)


env = OBELIX(scaling_factor=5, difficulty=1, wall_obstacles = False, arena_size= 500)
train(env)

