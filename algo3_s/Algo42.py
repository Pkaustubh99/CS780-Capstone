import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from obelix import OBELIX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    "hidden_size": 128,
}

# ================= MODEL =================
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
        x = x.unsqueeze(1) if x.dim() == 2 else x  # (B,1,F) or (1,T,F)

        x, hidden = self.lstm(x, hidden)

        x = x.squeeze(1) if x.size(1) == 1 else x

        logits = self.policy(x)
        value = self.value(x)

        return logits, value, hidden


# ================= BUFFER =================
class RolloutBuffer:
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.hiddens = []

    def __init__(self):
        self.clear()


# ================= AGENT =================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["lr"])

        self.buffer = RolloutBuffer()
        self.hidden = None
        self.action_dim = action_dim

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
            probs = torch.ones_like(probs) / probs.shape[-1]

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.detach(), self.hidden

    def store(self, state, action, reward, done, log_prob, value, hidden):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)
        self.buffer.hiddens.append((hidden[0].detach(), hidden[1].detach()))

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

        # ===== SEQUENCE TRAINING =====
        states = states.unsqueeze(0)  # (1, T, F)

        h0, c0 = self.buffer.hiddens[0]
        h0, c0 = h0.detach(), c0.detach()

        for _ in range(CONFIG["epochs"]):
            logits, new_values, _ = self.model(states, (h0, c0))

            logits = logits.squeeze(0)
            new_values = new_values.squeeze(0)

            logits = torch.clamp(logits, -20, 20)
            probs = torch.softmax(logits, dim=-1)

            if torch.isnan(probs).any():
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
            nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG["max_grad_norm"])
            self.optimizer.step()

        self.buffer.clear()


# ================= TRAIN =================
def train(env, state_dim, action_dim, max_episodes=1000):
    agent = PPOAgent(state_dim, action_dim)

    os.makedirs("algo4", exist_ok=True)

    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    visited = set()

    for ep in range(max_episodes):
        state = env.reset(wall_obstacles=True if ep % 2 == 0 else False)
        agent.reset_hidden()

        total_reward = 0

        while True:
            action, log_prob, value, hidden = agent.select_action(state)

            next_state, reward, done = env.step(ACTIONS[action], render=True)

            reward = np.clip(reward / 100.0, -10, 10)

            key = tuple(state)
            if key not in visited:
                reward += 0.2
                visited.add(key)

            agent.store(state, action, reward, done, log_prob, value, hidden)

            state = next_state
            total_reward += reward

            if len(agent.buffer.states) >= CONFIG["rollout_steps"]:
                agent.update(state)

            if done:
                break

        print(f"Episode {ep}, Reward: {total_reward:.2f}")

        if ep % 50 == 0:
            torch.save({
                "model": agent.model.state_dict(),
                "optimizer": agent.optimizer.state_dict()
            }, "algo4/weights.pth")

    print("Training complete!")


# ================= RUN =================
env = OBELIX(scaling_factor=1, difficulty=1, wall_obstacles=True)

train(env, state_dim=18, action_dim=5)