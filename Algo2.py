
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from obelix import OBELIX

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Model
# =====================
class ActorCritic(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(stateDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU()
        )

        self.actor = nn.Linear(hiddenDim, actionDim)
        self.critic = nn.Linear(hiddenDim, 1)

    def forward(self, state):
        x = self.shared(state)
        return self.actor(x), self.critic(x)


# =====================
# Parallel Environment
# =====================
class ParallelEnv:
    def __init__(self, num_envs):
        self.envs = [OBELIX(scaling_factor=1, difficulty=3) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        next_states, rewards, dones = [], [], []

        for i, env in enumerate(self.envs):
            s, r, d = env.step(self.ACTIONS[actions[i]], render=False)

            if d:
                s = env.reset()

            next_states.append(s)
            rewards.append(r / 100.0)
            dones.append(d)

        return next_states, rewards, dones


# =====================
# Rollout Buffer
# =====================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.__init__()


# =====================
# Rollout Collection
# =====================
def collect_rollout(envs, model, buffer, horizon):
    states = envs.reset()

    for _ in range(horizon):
        stateTensor = torch.from_numpy(np.array(states)).float().to(device)

        with torch.no_grad():
            logits, values = model(stateTensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            actions = dist.sample()
            logprobs = dist.log_prob(actions)

        next_states, rewards, dones = envs.step(actions.cpu().numpy())

        # ✅ Correct storage
        buffer.states.append(stateTensor)
        buffer.actions.append(actions.unsqueeze(1))
        buffer.logprobs.append(logprobs.unsqueeze(1))

        buffer.values.extend(values.squeeze().cpu().numpy())
        buffer.rewards.extend(rewards)
        buffer.dones.extend(dones)

        states = next_states


# =====================
# PPO Update
# =====================
def ppo_update(model, optimizer, buffer, epochs=10, batch_size=64):

    states = torch.cat(buffer.states).to(device)
    actions = torch.cat(buffer.actions, dim=0).squeeze().to(device)
    old_logprobs = torch.cat(buffer.logprobs, dim=0).squeeze().to(device)

    rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
    values = torch.tensor(buffer.values, dtype=torch.float32)
    dones = torch.tensor(buffer.dones, dtype=torch.float32)

    # ===== GAE =====
    advantages = torch.zeros_like(rewards)
    gae = 0

    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(values) else 0
        delta = rewards[t] + 0.99 * next_value * (1 - dones[t]) - values[t]
        gae = delta + 0.99 * 0.95 * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    advantages = advantages.to(device)
    returns = returns.to(device)

    dataset_size = states.size(0)

    for _ in range(epochs):
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_logprobs = old_logprobs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            # safety
            if batch_states.dim() == 1:
                batch_states = batch_states.unsqueeze(0)

            logits, values_pred = model(batch_states)

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_logprobs = dist.log_prob(batch_actions)

            ratio = torch.exp(new_logprobs - batch_old_logprobs)

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values_pred.squeeze(), batch_returns)
            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# =====================
# Training Loop
# =====================
def train():

    stateDim = 18
    actionDim = 5

    model = ActorCritic(stateDim, actionDim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    num_envs = 8
    horizon = 256

    envs = ParallelEnv(num_envs)
    buffer = RolloutBuffer()

    # ===== Checkpointing =====
    savePath = "algo2"
    os.makedirs(savePath, exist_ok=True)
    weightsFile = os.path.join(savePath, "weights.pth")

    start_update = 0

    if os.path.exists(weightsFile):
        checkpoint = torch.load(weightsFile, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = checkpoint["update"]
        print("Resumed from checkpoint")

    for update in range(start_update, 500):

        collect_rollout(envs, model, buffer, horizon)

        # debug once if needed
        # print(torch.cat(buffer.states).shape)

        ppo_update(model, optimizer, buffer)

        buffer.clear()

        if update % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update
            }, weightsFile)

        if update % 10 == 0:
            print(f"Update {update}")


if __name__ == "__main__":
    train()

