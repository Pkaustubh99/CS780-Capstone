import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from obelix import OBELIX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_DIM = len(ACTIONS)
START_TOKEN = ACTION_DIM

GAMMA = 0.99
LR = 3e-4
CLIP = 0.2
ENTROPY_COEF = 0.03
VALUE_COEF = 0.5
MAX_SEQ = 256
WEIGHT_PATH = "algo7/weights.pth"


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

        att = F.softmax(att, dim=-1)
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

    def act(self, states, actions):
        logits, values = self.forward(states, actions)

        logits = logits[:, -1]
        value = values[:, -1]

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


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


def saveModel(model, optimizer=None):
    os.makedirs("algo7", exist_ok=True)
    state = {"model": model.state_dict()}
    if optimizer:
        state["opt"] = optimizer.state_dict()
    torch.save(state, WEIGHT_PATH)


def loadModel(model, optimizer=None):
    if not os.path.exists(WEIGHT_PATH):
        print("No weights found, training from scratch")
        return
    ckpt = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    if optimizer and "opt" in ckpt:
        optimizer.load_state_dict(ckpt["opt"])
    print("Loaded weights")


def train(env, episodes=1000):
    stateDim = len(env.reset())

    model = PPOTransformer(stateDim, ACTION_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loadModel(model, optimizer)

    for ep in range(episodes):

        state = env.reset()

        states, actions = [], []
        logProbs, rewards, values = [], [], []

        prevAction = START_TOKEN
        done = False

        # -------- ROLLOUT --------
        while not done:
            s = torch.tensor(state, dtype=torch.float32).to(DEVICE)

            states.append(s)
            actions.append(prevAction)

            stateSeq = torch.stack(states[-MAX_SEQ:]).unsqueeze(0)
            actionSeq = torch.tensor(actions[-MAX_SEQ:], device=DEVICE).unsqueeze(0)

            action, logProb, entropy, value = model.act(stateSeq, actionSeq)

            actionIdx = action.item()
            nextState, reward, done = env.step(ACTIONS[actionIdx])

            logProbs.append(logProb)
            rewards.append(reward)
            values.append(value)

            prevAction = actionIdx
            state = nextState

        # -------- TRAJECTORY --------

        values = torch.stack(values)

        # 🔥 ensure same base length BEFORE anything else
        T_raw = min(len(rewards), values.shape[0])

        rewards = rewards[:T_raw]
        values = values[:T_raw]
        returns = computeReturns(rewards)
        oldLogProbs = torch.stack(logProbs).detach()

        # truncate consistently
        returns = returns[-MAX_SEQ:]
        values = values[-MAX_SEQ:]
        oldLogProbs = oldLogProbs[-MAX_SEQ:]

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -------- MODEL FORWARD --------
        # IMPORTANT: use SAME window as trajectory
        states_trunc = states[-(len(returns)+1):]
        actions_trunc = actions[-(len(returns)+1):]

        stateSeq = torch.stack(states_trunc).unsqueeze(0)
        actionSeq = torch.tensor(actions_trunc, device=DEVICE).unsqueeze(0)

        logits, newValues = model(stateSeq, actionSeq)

        # NOW align naturally: drop LAST instead of FIRST
        logits = logits[:, :-1]        # (T, A)
        newValues = newValues[:, :-1]  # (T)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        actionsTensor = torch.tensor(actions_trunc[1:], device=DEVICE).unsqueeze(0)

        newLogProbs = dist.log_prob(actionsTensor).squeeze(0)
        newValues = newValues.squeeze(0) if newValues.dim() > 1 else newValues
        newLogProbs = newLogProbs.squeeze(0) if newLogProbs.dim() > 1 else newLogProbs

        # -----------------------
        # FINAL SAFE ALIGNMENT (PERMANENT FIX)
        # -----------------------
        # T = min(
        #     newLogProbs.shape[0],
        #     oldLogProbs.shape[0],
        #     advantages.shape[0],
        #     returns.shape[0],
        #     newValues.shape[0],
        # )
        #
        #
        #
        # newLogProbs = newLogProbs[:T]
        # oldLogProbs = oldLogProbs[:T]
        # advantages = advantages[:T]
        # returns = returns[:T]
        # newValues = newValues[:T]

        # -----------------------
        # FINAL SAFE ALIGNMENT (SHAPE FIX)
        # -----------------------

        advantages = advantages.view(-1)
        newLogProbs = newLogProbs.view(-1)
        oldLogProbs = oldLogProbs.view(-1)
        returns = returns.view(-1)
        newValues = newValues.view(-1)

        T = min(

            advantages.shape[0],
            newLogProbs.shape[0],
            oldLogProbs.shape[0],
            returns.shape[0],
            newValues.shape[0],
        )


        advantages = advantages[:T]
        newLogProbs = newLogProbs[:T]
        oldLogProbs = oldLogProbs[:T]
        returns = returns[:T]
        newValues = newValues[:T]

        # -------- PPO LOSS --------

        ratio = torch.exp(newLogProbs - oldLogProbs)
        ratio = ratio.view(-1)
        ratio = ratio[:T]

        print(len(ratio))
        print(len(advantages))




        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantages

        policyLoss = -torch.min(surr1, surr2).mean()
        valueLoss = (returns - newValues).pow(2).mean()
        entropy = dist.entropy().mean()

        loss = policyLoss + VALUE_COEF * valueLoss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            print(f"Ep {ep} | Loss {loss.item():.3f}")
            saveModel(model, optimizer)


# -----------------------
# RUN
# -----------------------
for i in range(16, 0, -1):
    for j in range(1, 7):
        env = OBELIX(scaling_factor=i, difficulty=1, wall_obstacles= False if j % 2 == 0 else True, arena_size=500)
        train(env, episodes= 30)


for i in range(16, 0, -1):
    for j in range(1, 7):
        env = OBELIX(scaling_factor=i, difficulty=2, wall_obstacles=False if j % 2 == 0 else True, arena_size=500)
        train(env, episodes=30)


for i in range(16, 0, -1):
    for j in range(1, 7):
        env = OBELIX(scaling_factor=i, difficulty=3, wall_obstacles=False if j % 2 == 0 else True, arena_size=500)
        train(env, episodes=30)

