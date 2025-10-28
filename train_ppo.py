from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Callable, Optional
import numpy as np
import gymnasium as gym
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecTransposeImage


def make_env_fn(obs_type: str, rank: int, seed: int = 0, render_mode: Optional[str] = None) -> Callable[[], Any]:
    """Create a callable that builds a single `intersection-v1` env and configures it.

    If `render_mode` is provided, it will be passed to gym.make (useful for evaluation).
    """

    def _init():
        # Pass render_mode only for evaluation (training envs should not render)
        if render_mode:
            env = gym.make("intersection-v1", render_mode=render_mode)
        else:
            env = gym.make("intersection-v1")
        # configure observation type if supported
        try:
            if obs_type.lower() == "kinematics":
                env.configure({"observation": {"type": "Kinematics"}})
            else:
                env.configure({"observation": {"type": "Grayscale"}})
        except Exception:
            pass
        env = Monitor(env)
        try:
            env.reset(seed=seed + rank)
        except Exception:
            try:
                env.seed(seed + rank)
            except Exception:
                pass
        return env

    return _init


def choose_policy_from_env(env) -> str:
    """Return 'CnnPolicy' if observation looks like an image, else 'MlpPolicy'."""
    try:
        shape = env.observation_space.shape
        if shape is not None and len(shape) == 3:
            return "CnnPolicy"
    except Exception:
        pass
    return "MlpPolicy"


def find_latest_checkpoint(save_path: str) -> Optional[str]:
    """Return the latest checkpoint file path (ppo_checkpoint_*.zip) or None."""
    pattern = os.path.join(save_path, "ppo_checkpoint_*.zip")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-type", type=str, default="kinematics", choices=["kinematics", "rgb", "grayscale"], help="Observation type to request from HighwayEnv")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models/ppo_intersection")
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--render-mode", type=str, choices=["none", "human", "rgb_array"], default="none", help="Render mode for evaluation (none/human/rgb_array)")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Build vectorized envs
    env_fns = [make_env_fn(args.obs_type, i, seed=args.seed) for i in range(args.n_envs)]
    if args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Normalize observations (do not normalize rewards for episodic tasks)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # Determine policy type using a single env instance
    probe_env = gym.make("intersection-v1")
    try:
        # apply same observation configuration to probe as training envs
        if args.obs_type.lower() == "kinematics":
            probe_env.configure({"observation": {"type": "Kinematics"}})
        else:
            probe_env.configure({"observation": {"type": "Grayscale"}})
    except Exception:
        pass
    policy = choose_policy_from_env(probe_env)
    probe_env.close()

    # If image observations, transpose to channel-first
    if policy == "CnnPolicy":
        try:
            vec_env = VecTransposeImage(vec_env)
        except Exception:
            pass

    tensorboard_log = args.save_path if args.tensorboard else None

    # Prepare callbacks
    checkpoint_callback = CheckpointCallback(save_freq=max(1000, args.total_timesteps // 10), save_path=args.save_path, name_prefix="ppo_checkpoint") # Save every 1000 steps or 10% of total training
    
    # Create and wrap eval env the same way as training env (with optional rendering)
    eval_render = None if args.render_mode == "none" else args.render_mode
    eval_env = DummyVecEnv([lambda: Monitor(make_env_fn(args.obs_type, 0, args.seed, render_mode=eval_render)())])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # training=False for evaluation
    
    # Add image wrapper if needed
    if policy == "CnnPolicy":
        try:
            eval_env = VecTransposeImage(eval_env)
        except Exception:
            pass
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.save_path, log_path=args.save_path, eval_freq=max(1000, args.total_timesteps // 10), n_eval_episodes=3, deterministic=True)

    # Use CPU for MlpPolicy (faster) and GPU for CnnPolicy (if available)
    device = 'cpu' if policy == "MlpPolicy" else 'auto'

    model = None
    if args.resume:
        latest = find_latest_checkpoint(args.save_path)
        if latest is not None:
            print(f"Resuming from checkpoint: {latest}")
            try:
                # load model and attach vec_env
                model = PPO.load(latest, env=vec_env)
                # Try to load VecNormalize statistics if saved
                vecnorm_path = os.path.join(args.save_path, "vecnormalize.pkl")
                if os.path.exists(vecnorm_path):
                    try:
                        VecNormalize.load(vecnorm_path, vec_env)
                        print("Loaded VecNormalize stats.")
                    except Exception:
                        print("Failed to load VecNormalize stats; continuing without them.")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    if model is None:
        model = PPO(
            policy,
            vec_env,
            n_steps=2048,        # Collect 2048 steps per update
            batch_size=64,       # 64 steps per minibatch
            n_epochs=10,         # Number of updates on the same batch
            learning_rate=3e-4,
            gamma=0.99,          # Discount factor for long-term planning
            verbose=1,
            seed=args.seed,
            device=device,       # CPU for MLP, GPU for CNN
            tensorboard_log=tensorboard_log,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])) if policy == "MlpPolicy" else None,
        )

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_callback, eval_callback])

    # Save final model and VecNormalize stats
    final_model_path = os.path.join(args.save_path, "ppo_intersection")
    model.save(final_model_path)
    try:
        vec_env.save(os.path.join(args.save_path, "vecnormalize.pkl"))
    except Exception:
        pass

    # Final evaluation using same wrapped env setup as training (with optional rendering)
    eval_env = DummyVecEnv([lambda: Monitor(make_env_fn(args.obs_type, 0, args.seed, render_mode=eval_render)())])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    if policy == "CnnPolicy":
        try:
            eval_env = VecTransposeImage(eval_env)
        except Exception:
            pass

    # Run multiple evaluation episodes
    n_eval_episodes = 10
    success_count = collision_count = timeout_count = 0
    total_rewards = []
    progress_rewards = []
    collision_penalties = []
    episode_lengths = []

    print("\nRunning evaluation episodes ...")
    for episode in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0.0
        episode_progress = 0.0
        episode_collision = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            reward = float(rewards[0])
            info = infos[0]

            episode_reward += reward
            # Expect info['rewards'] to be a dict with keys: collision_reward, high_speed_reward, arrived_reward, on_road_reward
            rw = info.get('rewards', {}) if isinstance(info.get('rewards', {}), dict) else {}
            # `rw['collision_reward']` is a per-objective flag (True/1.0) when a collision happened.
            # The actual scalar penalty applied per agent is available in `info['agents_rewards']` (can be negative).
            agents_rewards = info.get('agents_rewards', ())
            # Accumulate collision penalty from agents_rewards if present (negative values indicate penalty)
            if isinstance(agents_rewards, (list, tuple)) and len(agents_rewards) > 0:
                episode_collision += sum(min(0.0, float(x)) for x in agents_rewards)

            episode_progress += float(rw.get('high_speed_reward', 0.0)) + float(rw.get('on_road_reward', 0.0)) + float(rw.get('arrived_reward', 0.0))

            done = dones[0]
            steps += 1
            try:
                eval_env.render()
            except Exception:
                pass

        rw = info.get('rewards', {}) if isinstance(info.get('rewards', {}), dict) else {}
        agents_rewards = info.get('agents_rewards', ())

        # Collision if explicit crashed flag OR per-objective collision flag (rw['collision_reward']) is truthy
        collision_flag = info.get('crashed', False) or bool(rw.get('collision_reward'))

        if collision_flag:
            collision_count += 1
            outcome = 'Collision'
        elif float(rw.get('arrived_reward', 0.0)) > 0.0:
            success_count += 1
            outcome = 'Success'
        elif info.get('TimeLimit.truncated', False):
            timeout_count += 1
            outcome = 'Timeout'
        else:
            timeout_count += 1
            outcome = 'Timeout'

        total_rewards.append(episode_reward)
        progress_rewards.append(episode_progress)
        collision_penalties.append(episode_collision)
        episode_lengths.append(steps)

        print(f"Episode {episode+1}/{n_eval_episodes}: Reward={episode_reward:.2f}, Length={steps}, {outcome}")

    # Print evaluation statistics
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_count/n_eval_episodes*100:.1f}% ({success_count}/{n_eval_episodes} episodes)")
    print(f"Collision Rate: {collision_count/n_eval_episodes*100:.1f}% ({collision_count}/{n_eval_episodes} episodes)")
    print(f"Timeout Rate: {timeout_count/n_eval_episodes*100:.1f}% ({timeout_count}/{n_eval_episodes} episodes)")
    print(f"\nReward Breakdown (mean ± std):")
    print(f"Total Reward: {sum(total_rewards)/len(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Progress Reward: {sum(progress_rewards)/len(progress_rewards):.2f} ± {np.std(progress_rewards):.2f}")
    print(f"Collision Penalty: {sum(collision_penalties)/len(collision_penalties):.2f} ± {np.std(collision_penalties):.2f}")
    print(f"Episode Length: {sum(episode_lengths)/len(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")


if __name__ == "__main__":
    main()
