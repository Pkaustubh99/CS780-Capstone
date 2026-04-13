import numpy as np
from collections import deque
import argparse
import cv2
import matplotlib.pyplot as plt




from obelix import OBELIX

class RelativeStateObelix:
    def __init__(self, history_len=10):
        self.history_len = history_len
        self.reset()

    def reset(self):
        self.theta = 0.0  # orientation (degrees)
        self.pos = np.array([0.0, 0.0], dtype=np.float32)

        self.action_hist = deque(maxlen=self.history_len)
        self.reward_hist = deque(maxlen=self.history_len)

    def _update_orientation(self, action):
        if action == 0:   # L45
            self.theta += 45
        elif action == 1: # L22
            self.theta += 22.5
        elif action == 3: # R22
            self.theta -= 22.5
        elif action == 4: # R45
            self.theta -= 45

        self.theta = self.theta % 360
        self.theta = round(self.theta, 1)

    def _forward_delta(self):
        rad = np.deg2rad(self.theta)
        step = 5
        delta = step * np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)
        return np.round(delta)

    def update(self, action, reward, obs):
        stuck_flag = obs[-1]

        prev_pos = self.pos.copy()

        # update orientation first
        self._update_orientation(action)

        if action == 2:  # FW
            delta = self._forward_delta()
            new_pos = self.pos + delta

            if stuck_flag == 0:
                self.pos = new_pos
            # else: do nothing (stay in place)

        self.action_hist.append(action)
        self.reward_hist.append(reward)

    def update(self, action, reward, obs):
        stuck_flag = obs[-1]

        prev_pos = self.pos.copy()

        # update orientation first
        self._update_orientation(action)

        if action == 2:  # FW
            delta = self._forward_delta()
            new_pos = self.pos + delta

            if stuck_flag == 0:
                self.pos = new_pos
            # else: do nothing (stay in place)

        self.action_hist.append(action)
        self.reward_hist.append(reward)

    def get_state(self, obs):
        action_hist = np.array(self.action_hist, dtype=np.float32)
        reward_hist = np.array(self.reward_hist, dtype=np.float32)

        if len(action_hist) < self.history_len:
            pad = self.history_len - len(action_hist)
            action_hist = np.pad(action_hist, (pad, 0))
            reward_hist = np.pad(reward_hist, (pad, 0))

        # normalize angle
        theta_norm = self.theta / 360.0

        return np.concatenate([
            obs,
            [theta_norm],
            self.pos,
            action_hist,
            reward_hist
        ])

def to_plot_coords(p, arena_size):
    x, y = p
    return int(x), int(arena_size - y)

def get_true_state(env):
    return {
        "agent_pos": np.array([env.bot_center_x, env.bot_center_y], dtype=np.float32),
        "agent_theta": float(env.facing_angle),
        "box_pos": np.array([env.box_center_x, env.box_center_y], dtype=np.float32)
    }

env = OBELIX(scaling_factor=10, difficulty=1, wall_obstacles = False, arena_size= 500)

true = get_true_state(env)

rel_state = RelativeStateObelix(history_len=10)

print("TRUE:", true["agent_pos"], true["agent_theta"])
print("EST :", rel_state.pos, rel_state.theta)


pos_error = np.linalg.norm(true["agent_pos"] - rel_state.pos)

theta_error = abs(true["agent_theta"] - rel_state.theta)
theta_error = min(theta_error, 360 - theta_error)

errors = []

errors.append({
    "pos_error": pos_error,
    "theta_error": theta_error
})

true = get_true_state(env)

rel_state.reset()

rel_state.pos = true["agent_pos"].copy()
rel_state.theta = true["agent_theta"]



# manual play
move_choice = ["L45", "L22", "FW", "R22", "R45"]

user_input_choice = [ord("q"), ord("a"), ord("w"), ord("d"), ord("e")]
env.render_frame()
episode_reward = 0
errors = []
steps = []

env.render_frame()
episode_reward = 0

true_traj = []
est_traj = []

def plot_trajectories(true_traj, est_traj, arena_size=500):
    canvas = np.zeros((arena_size, arena_size, 3), dtype=np.uint8)



    # draw true trajectory (GREEN)
    for i in range(1, len(true_traj)):
        p1 = to_plot_coords(true_traj[i - 1], arena_size)
        p2 = to_plot_coords(true_traj[i], arena_size)
        cv2.line(canvas, p1, p2, (0, 255, 0), 3)

    # draw estimated trajectory (RED)
    for i in range(1, len(est_traj)):
        p1 = to_plot_coords(est_traj[i - 1], arena_size)
        p2 = to_plot_coords(est_traj[i], arena_size)
        cv2.line(canvas, p1, p2, (0, 0, 255), 1)

    # mark start
    if len(true_traj) > 0:
        cv2.circle(canvas, tuple(true_traj[0].astype(int)), 5, (255, 255, 255), -1)

    # mark end
    if len(true_traj) > 0:
        cv2.circle(canvas, tuple(true_traj[-1].astype(int)), 5, (0, 255, 255), -1)

    cv2.imshow("Trajectory Overlay (Green=True, Red=Estimated)", canvas)
    cv2.imwrite("canvas.jpg", canvas)
    cv2.waitKey(0)
    cv2.waitKey(0)


for step in range(1, 2000):
    x = cv2.waitKey(0)

    if x in user_input_choice:
        action_str = move_choice[user_input_choice.index(x)]

        # map to action index (IMPORTANT)
        action_idx = move_choice.index(action_str)

        sensor_feedback, reward, done = env.step(action_str)
        episode_reward += reward

        # ✅ UPDATE RELATIVE STATE
        rel_state.update(action_idx, reward, sensor_feedback)

        # ✅ GET TRUE STATE
        true = get_true_state(env)
        # ✅ COMPUTE ERRORS
        pos_error = np.linalg.norm(true["agent_pos"] - rel_state.pos)

        theta_error = abs(true["agent_theta"] - rel_state.theta)
        theta_error = min(theta_error, 360 - theta_error)

        # ✅ STORE
        errors.append((pos_error, theta_error))
        steps.append(step)

        true_traj.append(true["agent_pos"].copy())
        est_traj.append(rel_state.pos.copy())

        print(f"{step} | PosErr: {pos_error:.2f} | ThetaErr: {theta_error:.2f}")

        if done:
            print("Episode done. Total score:", episode_reward)
            break



# cv2.waitKey(0)

plot_trajectories(true_traj, est_traj, arena_size=env.arena_size)

# errors = np.array(errors)
#
# plt.figure()
# plt.plot(steps, errors[:, 0])
# plt.xlabel("Steps")
# plt.ylabel("Position Error")
# plt.title("Position Drift")
# plt.grid()
#
# plt.figure()
# plt.plot(steps, errors[:, 1])
# plt.xlabel("Steps")
# plt.ylabel("Theta Error")
# plt.title("Orientation Drift")
# plt.grid()

# plt.show()
#
#
#






# exit()

