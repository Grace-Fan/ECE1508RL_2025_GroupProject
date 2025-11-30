import gymnasium as gym
import highway_env
import os
import torch
import imageio
import matplotlib.pyplot as plt
import numpy as np

from gymnasium.wrappers import RecordVideo  
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

print("CUDA available:", torch.cuda.is_available())

# -----------------------------
# Global configuration
# -----------------------------
TRAINED_VEHICLES = 15  # change to 15 if using a 15-vehicle trained model

# -----------------------------
# Environment setup function
# -----------------------------
def make_env(record_video=False, video_dir="dqn_videos", config_mod=None, n_vehicles=None):
    render_mode = "rgb_array"
    env = gym.make("intersection-v1", render_mode=render_mode)

    vehicles_count = n_vehicles if n_vehicles is not None else TRAINED_VEHICLES

    config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True,
        },
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "vehicles_count": vehicles_count,
        },
        "destination": "o1",
        "offroad_terminal": True,
        "reward": {
            "collision_penalty": -5,
            "reward_speed": False,
            "reward_lanes": False,
            "high_speed_reward": 0.1
        },
        "stochastic": True
    }

    if config_mod is not None:
        for key, value in config_mod.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    try:
        env.unwrapped.configure(config)
    except Exception as e:
        print("Failed to configure environment:", e)

    env.reset()

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=video_dir,
                          episode_trigger=lambda ep_id: True, name_prefix="dqn_agent")
        print(f"Video recording enabled, saving to {video_dir}")

    return env

# -----------------------------
# Training logger callback
# -----------------------------
class TrainingLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.rewards = []
        self.steps = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
                self.steps.append(self.num_timesteps)
        return True
 
# -----------------------------
# Training DQN
# -----------------------------   
def train_agent(total_timesteps=1_000_000):
    env = make_env()
    env.reset(seed=42)
    logger = TrainingLogger()  # <-- Added callback for convergence & stability

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        target_update_interval=100,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs_intersection_dqn/",
        device="cuda",
    )

    model.learn(total_timesteps=total_timesteps, callback=logger)
    model.save("dqn_intersection_agent")
    env.close()

    # Plot convergence & stability curve
    plt.figure(figsize=(10,5))
    plt.plot(logger.steps, logger.rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Training Curve: Convergence & Stability")
    plt.grid(True)
    plt.show()

    print("Training complete. Model saved as dqn_intersection_agent")
    return model
# -----------------------------
# Observation padding
# -----------------------------
def pad_observation(obs, target_vehicles=None):
    if target_vehicles is None:
        target_vehicles = TRAINED_VEHICLES
    n = obs.shape[0]
    if n < target_vehicles:
        padding = np.zeros((target_vehicles - n, obs.shape[1]))
        obs = np.vstack([obs, padding])
    elif n > target_vehicles:
        obs = obs[:target_vehicles, :]  # truncate extra vehicles
    return obs

# -----------------------------
# Evaluation functions
# -----------------------------
def evaluate_metrics(model, n_episodes=100, config_mod=None, trained_vehicles=None):
    if trained_vehicles is None:
        trained_vehicles = TRAINED_VEHICLES

    total_time = 0
    total_speed = 0
    success_count = 0

    for ep in range(n_episodes):
        env = make_env(config_mod=config_mod, n_vehicles=trained_vehicles)
        obs, info = env.reset()
        done = truncated = False
        ep_steps = 0
        ep_speed_sum = 0

        while not (done or truncated):
            obs_padded = pad_observation(obs, target_vehicles=trained_vehicles)
            action, _ = model.predict(obs_padded, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            ep_steps += 1
            ep_speed_sum += info.get("speed", 0)

        total_time += ep_steps
        total_speed += ep_speed_sum / ep_steps

        success = 1 if info.get("arrived", True) and not info.get("crashed", False) else 0
        success_count += success
        env.close()

    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"Success Rate: {success_count/n_episodes:.2f}")
    print(f"Average Time Steps to Goal: {total_time/n_episodes:.2f}")
    print(f"Average Speed: {total_speed/n_episodes:.2f}")

def evaluate_robustness(model, trained_vehicles=None):
    if trained_vehicles is None:
        trained_vehicles = TRAINED_VEHICLES

    modifications = [
        {"observation": {"vehicles_count": 10}},  # fewer vehicles
        {"observation": {"vehicles_count": 15}},  # same as training
    ]

    for i, mod in enumerate(modifications):
        print(f"\n--- Evaluation under modification {i+1} ---")
        evaluate_metrics(model, n_episodes=100, config_mod=mod, trained_vehicles=trained_vehicles)

def evaluate_ablation(model, trained_vehicles=None):
    if trained_vehicles is None:
        trained_vehicles = TRAINED_VEHICLES

    ablations = [
        {"reward": {"collision_penalty": 0}},                 
        {"reward": {"reward_speed": True, "high_speed_reward": 0.5}},  
    ]

    for i, mod in enumerate(ablations):
        print(f"\n--- Ablation experiment {i+1} ---")
        evaluate_metrics(model, n_episodes=100, config_mod=mod, trained_vehicles=trained_vehicles)
        
def record_episodes_to_one_video(model, make_env_fn, n_episodes=100, out_path="evaluation_100episodes.mp4"):
    env = make_env_fn()
    frames = []

    print(f"Recording {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = truncated = False

        while not (done or truncated):
            frame = env.render()
            if frame is None or len(frame.shape) < 2:
                continue
            frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        print(f"Episode {ep+1}/{n_episodes} finished")

    env.close()
    print(f"Saving video to {out_path} ...")
    imageio.mimsave(out_path, frames, fps=30)
    print("Video saved.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load the model (change path if you have different models for 10 vs 15 vehicles)
    
    model = train_agent(total_timesteps=1_000_000)
    # model = DQN.load("dqn_intersection_agent")  
    
    # Standard evaluation
    evaluate_metrics(model, n_episodes=100)

    # Robustness evaluation
    evaluate_robustness(model)

    # Ablation experiments
    evaluate_ablation(model)

    # Record video for demonstration
    record_episodes_to_one_video(model, make_env_fn=make_env, n_episodes=100)