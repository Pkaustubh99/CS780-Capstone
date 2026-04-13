import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_DIM = len(ACTIONS)
STATE_DIM = 18


# -----------------------
# Utils
# -----------------------
def toTensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


def softUpdate(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1 - tau) * t.data)


# -----------------------
# Replay Buffer
# -----------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batchSize):
        batch = random.sample(self.buffer, batchSize)
        s, a, r, s2, d = zip(*batch)

        return (
            toTensor(np.array(s)),
            torch.LongTensor(a).to(DEVICE),
            toTensor(np.array(r)).unsqueeze(1),
            toTensor(np.array(s2)),
            toTensor(np.array(d)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# -----------------------
# Networks
# -----------------------
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, x):
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


# -----------------------
# SAC Agent (Discrete)
# -----------------------
class SACAgent:
    def __init__(self):
        self.q1 = QNet().to(DEVICE)
        self.q2 = QNet().to(DEVICE)
        self.q1Target = QNet().to(DEVICE)
        self.q2Target = QNet().to(DEVICE)

        self.policy = PolicyNet().to(DEVICE)

        self.q1Target.load_state_dict(self.q1.state_dict())
        self.q2Target.load_state_dict(self.q2.state_dict())

        self.qOpt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.piOpt = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.logAlpha = torch.tensor(0.0, requires_grad=True, device=DEVICE)
        self.alphaOpt = optim.Adam([self.logAlpha], lr=3e-4)

        self.targetEntropy = -np.log(1.0 / ACTION_DIM)

        self.gamma = 0.99
        self.tau = 0.005

    def getAction(self, state, eval=False):
        state = toTensor(state).unsqueeze(0)
        probs = self.policy(state).detach().cpu().numpy()[0]
        if eval:
            return np.argmax(probs)
        return np.random.choice(ACTION_DIM, p=probs)

    def update(self, buffer, batchSize=64):
        if len(buffer) < batchSize:
            return

        s, a, r, s2, d = buffer.sample(batchSize)

        with torch.no_grad():
            nextProbs = self.policy(s2)
            nextLogProbs = torch.log(nextProbs + 1e-8)

            q1Next = self.q1Target(s2)
            q2Next = self.q2Target(s2)
            minQNext = torch.min(q1Next, q2Next)

            alpha = self.logAlpha.exp()

            vNext = (nextProbs * (minQNext - alpha * nextLogProbs)).sum(dim=1, keepdim=True)
            targetQ = r + (1 - d) * self.gamma * vNext

        q1 = self.q1(s).gather(1, a.unsqueeze(1))
        q2 = self.q2(s).gather(1, a.unsqueeze(1))

        qLoss = ((q1 - targetQ).pow(2) + (q2 - targetQ).pow(2)).mean()

        self.qOpt.zero_grad()
        qLoss.backward()
        self.qOpt.step()

        probs = self.policy(s)
        logProbs = torch.log(probs + 1e-8)

        q1Vals = self.q1(s)
        q2Vals = self.q2(s)
        minQ = torch.min(q1Vals, q2Vals)

        alpha = self.logAlpha.exp()

        piLoss = (probs * (alpha * logProbs - minQ)).sum(dim=1).mean()

        self.piOpt.zero_grad()
        piLoss.backward()
        self.piOpt.step()

        alphaLoss = -(self.logAlpha * (logProbs + self.targetEntropy).detach()).mean()

        self.alphaOpt.zero_grad()
        alphaLoss.backward()
        self.alphaOpt.step()

        softUpdate(self.q1Target, self.q1, self.tau)
        softUpdate(self.q2Target, self.q2, self.tau)

    def save(self, path):
        torch.save(
            {
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "policy": self.policy.state_dict(),
            },
            path,
        )

def train(env, episodes=500):
    agent = SACAgent()
    buffer = ReplayBuffer()

    saveDir = "algo3"
    os.makedirs(saveDir, exist_ok=True)
    savePath = os.path.join(saveDir, "weights.pth")

    for ep in range(episodes):
        state = env.reset()
        totalReward = 0

        for t in range(2000):
            action = agent.getAction(state)
            nextState, reward, done = env.step(ACTIONS[action], render = False)

            buffer.add(state, action, reward, nextState, done)
            agent.update(buffer)

            state = nextState
            totalReward += reward

            if done:
                break

        if ep % 10 == 0:
            agent.save(savePath)

        print(f"Episode {ep} Reward {totalReward}")

    agent.save(savePath)


# -----------------------
# Policy for Submission
# -----------------------
_agent = None


def loadAgent():
    global _agent
    if _agent is None:
        _agent = SACAgent()
        path = "algo3/update.pth"
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu")
            _agent.policy.load_state_dict(checkpoint["policy"])
    return _agent


def policy(obs):
    agent = loadAgent()
    action = agent.getAction(obs, eval=True)
    return ACTIONS[action]

if __name__ == "__main__":
    from obelix import OBELIX

    env = OBELIX(scaling_factor=1, difficulty=3)
    train(env, episodes=500)
