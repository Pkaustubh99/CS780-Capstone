import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from obelix import OBELIX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Hyperparameters
# -----------------------
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
LR = 3e-4
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5

MAX_SEQ = 256

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_DIM = len(ACTIONS)
START_TOKEN = ACTION_DIM

WEIGHT_PATH = "algo9/weights.pth"
dim = 64
depth = 2
heads = 2
MAX_SEQ = 256

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


# -----------------------
# PPO Model
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

    def act(self, states, actions, rewards):
        logits, values = self.forward(states, actions, rewards)

        probs = torch.softmax(logits[:, -1] / 1.5, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), values[:, -1]


# -----------------------
# GAE
# -----------------------
def computeGAE(rewards, values, dones):
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

    T = len(rewards)
    advantages = torch.zeros(T, device=DEVICE)

    lastAdv = 0

    for t in reversed(range(T)):
        nextValue = values[t + 1] if t < T - 1 else 0

        delta = rewards[t] + GAMMA * nextValue * (1 - dones[t]) - values[t]
        lastAdv = delta + GAMMA * LAMBDA * (1 - dones[t]) * lastAdv

        advantages[t] = lastAdv

    returns = advantages + values

    return advantages, returns



class RND(nn.Module):
    def __init__(self, stateDim, hidden=128):
        super().__init__()

        self.target = nn.Sequential(
            nn.Linear(stateDim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.predictor = nn.Sequential(
            nn.Linear(stateDim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        # freeze target
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x):
        target = self.target(x)
        pred = self.predictor(x)

        return pred, target

# -----------------------
# Save / Load
# -----------------------
def saveModel(model, optimizer):
    os.makedirs("algo9", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": optimizer.state_dict()
    }, WEIGHT_PATH)


def loadModel(model, optimizer):
    if not os.path.exists(WEIGHT_PATH):
        print("No weights found, training from scratch")
        return
    ckpt = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])


# -----------------------
# Train
# -----------------------
def train(env, episodes=10000):

    state = env.reset()
    stateDim = len(state)

    model = PPOTransformer(stateDim, ACTION_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rnd = RND(stateDim).to(DEVICE)
    rndOptimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=LR)

    loadModel(model, optimizer)

    for ep in range(episodes):

        state = env.reset()

        states, actions, rewardsHist = [], [], []
        logProbs, rewards, values, dones = [], [], [], []

        prevAction = START_TOKEN
        prevReward = 0.0

        done = False

        while not done:
            # --- inside loop ---
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE)

            states.append(s)
            actions.append(prevAction)
            rewardsHist.append(prevReward)

            # RND (fixed shape)
            with torch.no_grad():
                sTensor = s.unsqueeze(0)
                pred, target = rnd(sTensor)
                intrinsic = ((pred - target) ** 2).mean().item()

            # model input
            stateSeq = torch.stack(states[-MAX_SEQ:]).unsqueeze(0)
            actionSeq = torch.tensor(actions[-MAX_SEQ:], device=DEVICE).unsqueeze(0)
            rewardSeq = torch.tensor(rewardsHist[-MAX_SEQ:], device=DEVICE).unsqueeze(0)

            action, logProb, entropy, value = model.act(stateSeq, actionSeq, rewardSeq)

            actionIdx = action.item()
            nextState, reward, done = env.step(ACTIONS[actionIdx], render=False)

            totalReward = reward + 0.1 * intrinsic

            logProbs.append(logProb)
            rewards.append(totalReward)
            values.append(value.squeeze())  # 🔥 FIX
            dones.append(done)

            prevAction = actionIdx
            prevReward = totalReward  # 🔥 FIX
            state = nextState

        values = torch.stack(values)
        statesTensor = torch.stack(states).detach()

        pred, target = rnd(statesTensor)

        rndLoss = (pred - target).pow(2).mean()

        rndOptimizer.zero_grad()
        rndLoss.backward()
        rndOptimizer.step()

        advantages, returns = computeGAE(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO forward
        stateSeq = torch.stack(states).unsqueeze(0)
        actionSeq = torch.tensor(actions, device=DEVICE).unsqueeze(0)
        rewardSeq = torch.tensor(rewardsHist, device=DEVICE).unsqueeze(0)

        logits, newValues = model(stateSeq, actionSeq, rewardSeq)

        logits = logits.squeeze(0)
        newValues = newValues.squeeze(0)

        actionsTensor = torch.tensor(actions, device=DEVICE)
        oldLogProbs = torch.stack(logProbs).detach()

        # 🔥 DROP FIRST STEP EVERYWHERE (alignment fix)
        actionsTensor = actionsTensor[1:]
        oldLogProbs = oldLogProbs[1:]
        advantages = advantages[1:]
        returns = returns[1:]
        newValues = newValues[1:]
        logits = logits[1:]

        # Distribution
        dist = torch.distributions.Categorical(torch.softmax(logits, dim=-1))
        newLogProbs = dist.log_prob(actionsTensor)

        # PPO loss
        ratio = torch.exp(newLogProbs - oldLogProbs)

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

