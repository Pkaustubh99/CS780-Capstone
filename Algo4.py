import os
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from obelix import OBELIX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CONFIG
# =========================
CONFIG = {
    "gamma": 0.99,
    "lambda": 0.95,
    "clip": 0.2,
    "lr": 1e-4,
    "entropy_coef": 0.02,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "rollout_steps": 256,
    "epochs": 4,
    "batch_size": 64,
    "hidden_size": 128,
}

# =========================
# MODEL (PPO + LSTM)
# =========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(128, CONFIG["hidden_size"], batch_first=True)

        self.policy = nn.Linear(CONFIG["hidden_size"], action_dim)
        self.value = nn.Linear(CONFIG["hidden_size"], 1)

    def forward(self, x, hidden):
        x = self.fc(x)
        x = x.unsqueeze(1)  # (B, 1, F)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(1)

        logits = self.policy(x)
        value = self.value(x)

        return logits, value, hidden

# =========================
# MEMORY
# =========================
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

# =========================
# AGENT
# =========================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["lr"])

        self.buffer = RolloutBuffer()
        self.action_dim = action_dim

        self.hidden = None

    def reset_hidden(self):
        self.hidden = (
            torch.zeros(1, 1, CONFIG["hidden_size"]).to(device),
            torch.zeros(1, 1, CONFIG["hidden_size"]).to(device),
        )

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        logits, value, self.hidden = self.model(state, self.hidden)

        logits = torch.clamp(logits, -20, 20)
        probs = torch.softmax(logits, dim=-1)
        if torch.isnan(probs).any():
            print("NaN detected in probs")
            probs = torch.ones_like(probs) / probs.shape[-1]
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.detach()

    def store(self, state, action, reward, done, log_prob, value):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)

    def compute_gae(self, next_value):
        rewards = self.buffer.rewards
        values = self.buffer.values + [next_value]
        dones = self.buffer.dones

        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + CONFIG["gamma"] * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + CONFIG["gamma"] * CONFIG["lambda"] * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])

        return returns

    def update(self, next_state):
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        _, next_value, _ = self.model(next_state, self.hidden)

        returns = self.compute_gae(next_value.detach())

        states = torch.FloatTensor(np.array(self.buffer.states)).to(device)
        actions = torch.LongTensor(self.buffer.actions).to(device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        values = torch.cat(self.buffer.values).squeeze().to(device)

        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages * 0.0
        for _ in range(CONFIG["epochs"]):
            h = torch.zeros(1, states.size(0), CONFIG["hidden_size"]).to(device)
            c = torch.zeros(1, states.size(0), CONFIG["hidden_size"]).to(device)

            logits, new_values, _ = self.model(states, (h, c))

            logits = torch.clamp(logits, -20, 20)
            probs = torch.softmax(logits, dim=-1)
            if torch.isnan(probs).any():
                print("NaN detected in probs")
                probs = torch.ones_like(probs) / probs.shape[-1]
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CONFIG["clip"], 1 + CONFIG["clip"]) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - new_values.squeeze()).pow(2).mean()

            loss = policy_loss + CONFIG["value_coef"] * value_loss - CONFIG["entropy_coef"] * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self.buffer.clear()

# =========================
# TRAIN LOOP
# =========================
def train(env, state_dim, action_dim, max_episodes=1000):
    agent = PPOAgent(state_dim, action_dim)

    os.makedirs("algo4", exist_ok=True)

    visited = set()
    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    for ep in range(max_episodes):

        state = env.reset( wall_obstacles = True if max_episodes % 2 == 0 else False)
        agent.reset_hidden()

        total_reward = 0

        while True:
            action, log_prob, value = agent.select_action(state)

            env_action = ACTIONS[action]
            next_state, reward, done = env.step(env_action, render= True)

            # ===== Reward Normalization =====
            reward = reward / 100.0
            reward = np.clip(reward, -10, 10)

            # ===== Exploration Bonus =====
            state_key = tuple(state)
            if state_key not in visited:
                reward += 0.2
                visited.add(state_key)

            agent.store(state, action, reward, done, log_prob, value)

            state = next_state
            total_reward += reward


            if len(agent.buffer.states) >= CONFIG["rollout_steps"]:
                agent.update(state)

            if done:
                break

        print(f"Episode {ep}, Reward: {total_reward:.2f}")

        # ===== Save Model =====
        if ep % 50 == 0:
            torch.save({
                "model": agent.model.state_dict(),
                "optimizer": agent.optimizer.state_dict()
            }, "algo4/weights.pth")

    print("Training complete!")

# =========================
# USAGE
# =========================

env = OBELIX(scaling_factor=5, difficulty=1, wall_obstacles = True, arena_size= 500)
state_dim = 18
action_dim = 5

train(env, state_dim, action_dim)
