# import multiprocessing as mp
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
# ACTION_DIM = len(ACTIONS)
# START_TOKEN = ACTION_DIM
#
# GAMMA = 0.99
# LR = 3e-4
# CLIP = 0.2
# ENTROPY_COEF = 0.03
# VALUE_COEF = 0.5
# MAX_SEQ = 256
# WEIGHT_PATH = "algo8/weights.pth"
#
# # -----------------------
# # RoPE
# # -----------------------
# class RotaryEmbedding:
#     def __init__(self, dim, maxSeqLen=2048):
#         invFreq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         t = torch.arange(maxSeqLen).float()
#         freqs = torch.einsum("i,j->ij", t, invFreq)
#         self.cos = torch.cos(freqs).to(DEVICE)
#         self.sin = torch.sin(freqs).to(DEVICE)
#
#     def apply(self, x, T):
#         cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
#         sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
#
#         x1, x2 = x[..., ::2], x[..., 1::2]
#         xRot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
#
#         return x * cos.repeat_interleave(2, dim=-1) + xRot * sin.repeat_interleave(2, dim=-1)
#
#
# # -----------------------
# # Transformer
# # -----------------------
# class Attention(nn.Module):
#     def __init__(self, dim, heads):
#         super().__init__()
#         self.heads = heads
#         self.headDim = dim // heads
#
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.out = nn.Linear(dim, dim)
#
#         self.rope = RotaryEmbedding(self.headDim)
#
#     def forward(self, x):
#         B, T, C = x.shape
#
#         qkv = self.qkv(x)
#         q, k, v = qkv.chunk(3, dim=-1)
#
#         q = q.view(B, T, self.heads, self.headDim).transpose(1, 2)
#         k = k.view(B, T, self.heads, self.headDim).transpose(1, 2)
#         v = v.view(B, T, self.heads, self.headDim).transpose(1, 2)
#
#         q = self.rope.apply(q, T)
#         k = self.rope.apply(k, T)
#
#         att = (q @ k.transpose(-2, -1)) / (self.headDim ** 0.5)
#
#         mask = torch.tril(torch.ones(T, T, device=x.device))
#         att = att.masked_fill(mask == 0, float('-inf'))
#
#         att = F.softmax(att, dim=-1)
#         out = att @ v
#
#         out = out.transpose(1, 2).contiguous().view(B, T, C)
#         return self.out(out)
#
#
# class Block(nn.Module):
#     def __init__(self, dim, heads):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(dim)
#         self.attn = Attention(dim, heads)
#         self.ln2 = nn.LayerNorm(dim)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim)
#         )
#
#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x
#
#
# # -----------------------
# # PPO Model
# # -----------------------
# class PPOTransformer(nn.Module):
#     def __init__(self, stateDim, actionDim, dim=64, depth=2, heads=2):
#         super().__init__()
#
#         self.stateEmbed = nn.Linear(stateDim, dim)
#         self.actionEmbed = nn.Embedding(actionDim + 1, dim)
#
#         self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
#         self.ln = nn.LayerNorm(dim)
#
#         self.policy = nn.Linear(dim, actionDim)
#         self.value = nn.Linear(dim, 1)
#
#     def forward(self, states, actions):
#         s = self.stateEmbed(states)
#         a = self.actionEmbed(actions)
#
#         x = s + a
#
#         for b in self.blocks:
#             x = b(x)
#
#         x = self.ln(x)
#         last = x[:, -1]
#
#         return self.policy(last), self.value(last).squeeze(-1)
#
#     def act(self, states, actions, temperature=1.5):
#         logits, value = self.forward(states, actions)
#
#         probs = torch.softmax(logits / temperature, dim=-1)
#         dist = torch.distributions.Categorical(probs)
#
#         action = dist.sample()
#         return action, dist.log_prob(action), dist.entropy(), value
#
#
# # -----------------------
# # Save / Load
# # -----------------------
# def saveModel(model, optimizer=None):
#     os.makedirs("algo5", exist_ok=True)
#
#     state = {
#         "model": model.state_dict()
#     }
#
#     if optimizer:
#         state["opt"] = optimizer.state_dict()
#
#     torch.save(state, WEIGHT_PATH)
#
#
# def loadModel(model, optimizer=None):
#     if not os.path.exists(WEIGHT_PATH):
#         print("No weights found, training from scratch")
#         return
#
#     ckpt = torch.load(WEIGHT_PATH, map_location=DEVICE)
#     model.load_state_dict(ckpt["model"])
#
#     if optimizer and "opt" in ckpt:
#         optimizer.load_state_dict(ckpt["opt"])
#
#     print("Loaded weights")
#
#
#
# class AsyncVecEnv:
#     def __init__(self, makeEnv, n):
#         self.n = n
#         self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(n)])
#         self.ps = [mp.Process(target=self.worker, args=(child, makeEnv)) for child in self.child_conns]
#
#         for p in self.ps:
#             p.start()
#
#     def worker(self, conn, makeEnv):
#         env = makeEnv()
#         while True:
#             cmd, data = conn.recv()
#             if cmd == "reset":
#                 conn.send(env.reset())
#             elif cmd == "step":
#                 conn.send(env.step(data))
#             elif cmd == "close":
#                 conn.close()
#                 break
#
#     def reset(self):
#         for conn in self.parent_conns:
#             conn.send(("reset", None))
#         return [conn.recv() for conn in self.parent_conns]
#
#     def step(self, actions):
#         for conn, a in zip(self.parent_conns, actions):
#             conn.send(("step", a))
#         results = [conn.recv() for conn in self.parent_conns]
#         s, r, d, i = zip(*results)
#         return list(s), list(r), list(d), list(i)
#
#
# class RolloutBuffer:
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.logProbs = []
#         self.rewards = []
#         self.values = []
#         self.dones = []
#
#     def add(self, s, a, lp, r, v, d):
#         self.states.append(s)
#         self.actions.append(a)
#         self.logProbs.append(lp)
#         self.rewards.append(r)
#         self.values.append(v)
#         self.dones.append(d)
#
#     def stack(self):
#         return (
#             torch.stack(self.states),
#             torch.tensor(self.actions),
#             torch.stack(self.logProbs),
#             torch.tensor(self.rewards, dtype=torch.float32),
#             torch.stack(self.values),
#             torch.tensor(self.dones)
#         )
# def computeReturnsBatch(rewards):
#     rewards = rewards.to(DEVICE)  # (B, T)
#
#     T = rewards.shape[1]
#
#     discounts = GAMMA ** torch.arange(T, device=DEVICE)
#     discounted = rewards * discounts
#
#     returns = torch.flip(
#         torch.cumsum(torch.flip(discounted, dims=[1]), dim=1),
#         dims=[1]
#     )
#
#     returns = returns / discounts
#     return returns
#
#
# def computeReturns(rewards, dones):
#     rewards = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
#     dones = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE)
#
#     if rewards.dim() == 1:
#         rewards = rewards.unsqueeze(0)
#         dones = dones.unsqueeze(0)
#
#     B, T = rewards.shape
#     returns = torch.zeros_like(rewards)
#
#     G = torch.zeros(B, device=DEVICE)
#
#     for t in reversed(range(T)):
#         G = rewards[:, t] + GAMMA * G * (1 - dones[:, t])
#         returns[:, t] = G
#
#     return returns.squeeze(0) if B == 1 else returns
#
# def train(makeEnv, numEnvs=8, rolloutSteps=512, epochs=4, batchSize=64):
#
#     envs = AsyncVecEnv(makeEnv, numEnvs)
#     states = envs.reset()
#
#     stateDim = len(states[0])
#
#     model = PPOTransformer(stateDim, ACTION_DIM).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#
#     loadModel(model, optimizer)
#
#     prevActions = [START_TOKEN] * numEnvs
#
#     for update in range(10000):
#
#         buffer = RolloutBuffer()
#
#         seqStates = [[] for _ in range(numEnvs)]
#         seqActions = [[] for _ in range(numEnvs)]
#
#         # -----------------------
#         # Rollout
#         # -----------------------
#         for step in range(rolloutSteps):
#
#             batchStateSeq = []
#             batchActionSeq = []
#
#             for i in range(numEnvs):
#                 s = torch.tensor(states[i], dtype=torch.float32).to(DEVICE)
#
#                 seqStates[i].append(s)
#                 seqActions[i].append(prevActions[i])
#
#                 stateSeq = torch.stack(seqStates[i][-MAX_SEQ:])
#                 actionSeq = torch.tensor(seqActions[i][-MAX_SEQ:], device=DEVICE)
#
#                 batchStateSeq.append(stateSeq)
#                 batchActionSeq.append(actionSeq)
#
#             stateSeq = torch.stack(batchStateSeq)
#             actionSeq = torch.stack(batchActionSeq)
#
#             actions, logProbs, _, values = model.act(stateSeq, actionSeq)
#
#             actionList = actions.tolist()
#             envActions = [ACTIONS[a] for a in actionList]
#
#             nextStates, rewards, dones, _ = envs.step(envActions)
#
#             for i in range(numEnvs):
#                 buffer.add(
#                     states[i],
#                     actionList[i],
#                     logProbs[i],
#                     rewards[i],
#                     values[i],
#                     dones[i]
#                 )
#
#                 prevActions[i] = actionList[i]
#
#                 if dones[i]:
#                     prevActions[i] = START_TOKEN
#                     seqStates[i] = []
#                     seqActions[i] = []
#
#             states = nextStates
#
#         # -----------------------
#         # Prepare batch
#         # -----------------------
#         states, actions, oldLogProbs, rewards, values, dones = buffer.stack()
#
#         states = states.to(DEVICE)
#         actions = actions.to(DEVICE)
#         oldLogProbs = oldLogProbs.to(DEVICE)
#         values = values.to(DEVICE)
#
#         returns = computeReturns(rewards.tolist(), dones.tolist()).to(DEVICE)
#
#         advantages = returns - values.detach()
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
#
#         N = len(states)
#
#         # -----------------------
#         # PPO Minibatch updates
#         # -----------------------
#         for _ in range(epochs):
#             idx = torch.randperm(N)
#
#             for start in range(0, N, batchSize):
#                 end = start + batchSize
#                 batchIdx = idx[start:end]
#
#                 batchStates = states[batchIdx]
#                 batchActions = actions[batchIdx]
#                 batchOldLog = oldLogProbs[batchIdx]
#                 batchAdv = advantages[batchIdx]
#                 batchRet = returns[batchIdx]
#
#                 logits, newValues = model(
#                     batchStates.unsqueeze(1),
#                     batchActions.unsqueeze(1)
#                 )
#
#                 probs = torch.softmax(logits, dim=-1)
#                 dist = torch.distributions.Categorical(probs)
#
#                 newLog = dist.log_prob(batchActions)
#
#                 ratio = torch.exp(newLog - batchOldLog)
#
#                 s1 = ratio * batchAdv
#                 s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * batchAdv
#
#                 policyLoss = -torch.min(s1, s2).mean()
#                 valueLoss = (batchRet - newValues).pow(2).mean()
#                 entropy = dist.entropy().mean()
#
#                 loss = policyLoss + VALUE_COEF * valueLoss - ENTROPY_COEF * entropy
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#         if update % 10 == 0:
#             print(f"Update {update} | Loss {loss.item():.3f}")
#             saveModel(model, optimizer)



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

# -----------------------
# Config
# -----------------------
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_DIM = len(ACTIONS)
START_TOKEN = ACTION_DIM

GAMMA = 0.99
LR = 3e-4
CLIP = 0.2
ENTROPY_COEF = 0.03
VALUE_COEF = 0.5

MAX_SEQ = 128
ROLLOUT_STEPS = 256
EPOCHS = 4
BATCH_SIZE = 64

WEIGHT_PATH = "algo8/weights.pth"


# -----------------------
# DDP Setup
# -----------------------
def setup(rank, worldSize):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=worldSize,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# -----------------------
# Async Env
# -----------------------
class AsyncVecEnv:
    def __init__(self, makeEnv, n):
        import multiprocessing as mp
        self.parent, self.child = zip(*[mp.Pipe() for _ in range(n)])
        self.ps = [mp.Process(target=self.worker, args=(c, makeEnv)) for c in self.child]

        for p in self.ps:
            p.start()

    def worker(self, conn, makeEnv):
        env = makeEnv()
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(env.reset())
            elif cmd == "step":
                conn.send(env.step(data))
            elif cmd == "close":
                break

    def reset(self):
        for p in self.parent:
            p.send(("reset", None))
        return [p.recv() for p in self.parent]

    def step(self, actions):
        for p, a in zip(self.parent, actions):
            p.send(("step", a))
        results = [p.recv() for p in self.parent]
        s, r, d, i = zip(*results)
        return list(s), list(r), list(d), list(i)


# -----------------------
# RoPE
# -----------------------
class RotaryEmbedding:
    def __init__(self, dim, maxSeqLen=2048, device="cpu"):
        invFreq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(maxSeqLen).float()
        freqs = torch.einsum("i,j->ij", t, invFreq)
        self.cos = torch.cos(freqs).to(device)
        self.sin = torch.sin(freqs).to(device)

    def apply(self, x, T):
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., ::2], x[..., 1::2]
        xRot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

        return x * cos.repeat_interleave(2, -1) + xRot * sin.repeat_interleave(2, -1)


# -----------------------
# Transformer
# -----------------------
class Attention(nn.Module):
    def __init__(self, dim, heads, device):
        super().__init__()
        self.heads = heads
        self.headDim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

        self.rope = RotaryEmbedding(self.headDim, device=device)

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
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, device):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, device)
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
    def __init__(self, stateDim, actionDim, device, dim=64, depth=2, heads=2):
        super().__init__()

        self.stateEmbed = nn.Linear(stateDim, dim)
        self.actionEmbed = nn.Embedding(actionDim + 1, dim)

        self.blocks = nn.ModuleList([Block(dim, heads, device) for _ in range(depth)])
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
        last = x[:, -1]

        return self.policy(last), self.value(last).squeeze(-1)

    def act(self, states, actions):
        logits, value = self.forward(states, actions)
        probs = torch.softmax(logits / 1.5, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


# -----------------------
# Utils
# -----------------------
def computeReturns(rewards, device):
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]

    for t in reversed(range(len(rewards) - 1)):
        returns[t] = rewards[t] + GAMMA * returns[t + 1]

    return returns


def saveModel(model, optimizer):
    os.makedirs("algo8", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": optimizer.state_dict()
    }, WEIGHT_PATH)


def loadModel(model, optimizer, device):
    if not os.path.exists(WEIGHT_PATH):
        print("No weights found")
        return
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    print("Loaded weights")

from obelix import OBELIX
# -----------------------
# Training
# -----------------------
def train(rank, worldSize):

    setup(rank, worldSize)
    device = torch.device(f"cuda:{rank}")


    def makeEnv():
        return OBELIX

    numEnvs = 4
    envs = AsyncVecEnv(makeEnv, numEnvs)

    states = envs.reset()
    stateDim = len(states[0])

    model = PPOTransformer(stateDim, ACTION_DIM, device).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if rank == 0:
        loadModel(model.module, optimizer, device)
    dist.barrier()

    prevActions = [START_TOKEN] * numEnvs

    try:
        for update in range(10000):

            buffer = []

            seqStates = [[] for _ in range(numEnvs)]
            seqActions = [[] for _ in range(numEnvs)]

            for _ in range(ROLLOUT_STEPS):

                batchS, batchA = [], []

                for i in range(numEnvs):
                    s = torch.tensor(states[i], dtype=torch.float32).to(device)

                    seqStates[i].append(s)
                    seqActions[i].append(prevActions[i])

                    batchS.append(torch.stack(seqStates[i][-MAX_SEQ:]))
                    batchA.append(torch.tensor(seqActions[i][-MAX_SEQ:], device=device))

                stateSeq = torch.stack(batchS)
                actionSeq = torch.stack(batchA)

                actions, logProbs, _, values = model.module.act(stateSeq, actionSeq)

                actList = actions.tolist()
                envActs = [ACTIONS[a] for a in actList]

                nextStates, rewards, dones, _ = envs.step(envActs)

                for i in range(numEnvs):
                    buffer.append((
                        states[i], actList[i], logProbs[i],
                        rewards[i], values[i], dones[i]
                    ))

                    prevActions[i] = actList[i]

                    if dones[i]:
                        prevActions[i] = START_TOKEN
                        seqStates[i] = []
                        seqActions[i] = []

                states = nextStates

            # -----------------------
            # Prepare batch
            # -----------------------
            states, actions, logp, rewards, values, _ = zip(*buffer)

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions).to(device)
            oldLog = torch.stack(logp).to(device)
            values = torch.stack(values).to(device)

            returns = computeReturns(rewards, device)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            N = len(states)

            for _ in range(EPOCHS):
                idx = torch.randperm(N)

                for start in range(0, N, BATCH_SIZE):
                    batch = idx[start:start+BATCH_SIZE]

                    s = states[batch].unsqueeze(1)
                    a = actions[batch].unsqueeze(1)

                    logits, v = model(s, a)

                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)

                    newLog = dist.log_prob(actions[batch])

                    ratio = torch.exp(newLog - oldLog[batch])

                    s1 = ratio * advantages[batch]
                    s2 = torch.clamp(ratio, 1-CLIP, 1+CLIP) * advantages[batch]

                    policyLoss = -torch.min(s1, s2).mean()
                    valueLoss = (returns[batch] - v).pow(2).mean()
                    entropy = dist.entropy().mean()

                    loss = policyLoss + VALUE_COEF * valueLoss - ENTROPY_COEF * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if rank == 0 and update % 10 == 0:
                print(f"Update {update} | Loss {loss.item():.3f}")
                saveModel(model.module, optimizer)


    finally:
        cleanup()


# -----------------------
# Launch
# -----------------------
def main():
    worldSize = torch.cuda.device_count()
    mp.spawn(train, args=(worldSize,), nprocs=worldSize)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()