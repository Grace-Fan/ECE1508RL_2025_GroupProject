from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Callable, List, Optional
import numpy as np
import gymnasium as gym
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecTransposeImage


def make_env_fn(rank: int, seed: int = 0, render_mode: Optional[str] = None, curriculum_level: int = 0) -> Callable[[], Any]:
    """Create a callable that builds a single `intersection-v1` env and configures it.

    If `render_mode` is provided, it will be passed to gym.make (useful for evaluation).
    curriculum_level mapping:
        0 = EASY    (2 vehicles,  0.2 spawn, duration 30)
        1 = MEDIUM  (6 vehicles,  0.4 spawn, duration 35)
        2 = TRANSIT (8 vehicles,  0.5 spawn, duration 40)
        3 = NORMAL  (10 vehicles, 0.6 spawn, duration 40)
    """

    def _init():
        # Pass render_mode only for evaluation (training envs should not render)
        if render_mode:
            env = gym.make("intersection-v1", render_mode=render_mode)
        else:
            env = gym.make("intersection-v1")
        try:
            # Observation configuration
            obs_cfg = {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": [
                    "presence",
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "cos_h",
                    "sin_h",
                ],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False,
            }

            # Curriculum: easy -> medium -> transit -> normal
            if curriculum_level == 0:
                duration = 30
                initial_vehicle_count = 2
                spawn_probability = 0.2
            elif curriculum_level == 1:
                duration = 35
                initial_vehicle_count = 6
                spawn_probability = 0.4
            elif curriculum_level == 2:
                duration = 40
                initial_vehicle_count = 8
                spawn_probability = 0.5
            else:  # level 3 or higher
                duration = 40
                initial_vehicle_count = 10
                spawn_probability = 0.6

            cfg = {
                "observation": obs_cfg,
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                },
                "duration": duration,
                "destination": "o1",
                "initial_vehicle_count": initial_vehicle_count,
                "spawn_probability": spawn_probability,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "normalize_reward": False,
                "offroad_terminal": True,
            }
            env.unwrapped.configure(cfg)
        except Exception as e:
            print("Env Configuration failed:", e)
        # Ensure collision penalty surfaces in scalar reward when missing
        env = CollisionAugmentWrapper(env)
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


def find_latest_checkpoint(save_path: str) -> Optional[str]:
    """Return the latest checkpoint file path (ppo_checkpoint_*.zip) or None."""
    pattern = os.path.join(save_path, "ppo_checkpoint_*.zip")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


class CollisionAugmentWrapper(gym.Wrapper):
    """Inject collision penalty into scalar reward when env reports crash but reward is non-negative.

    Uses env.unwrapped.config['collision_reward'] (fallback -30.0) and tags info['injected_collision_penalty'].
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            rw = info.get('rewards', {}) if isinstance(info.get('rewards', {}), dict) else {}
            crashed = info.get('crashed', False) or bool(rw.get('collision_reward'))
            if crashed and float(reward) >= 0.0:
                collision_val = float(getattr(self.env.unwrapped, 'config', {}).get('collision_reward', -5.0))
                reward = collision_val
        except Exception:
            print("CollisionAugmentWrapper: Exception during step reward adjustment.")
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class CurriculumCallback(BaseCallback):
    """Callback to gradually increase environment difficulty during training."""

    def __init__(self, vec_env, n_envs: int, seed: int,
                 steps_per_level: int = 100000, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.n_envs = n_envs
        self.seed = seed
        self.steps_per_level = steps_per_level
        self.current_level = 0

    def _on_step(self) -> bool:
        # Log current curriculum level to TensorBoard
        self.logger.record("curriculum/level", self.current_level)

        # Check if we should advance to next level
        new_level = min(self.num_timesteps // self.steps_per_level, 3)

        if new_level > self.current_level:
            self.current_level = new_level
            level_names = ["EASY", "MEDIUM", "TRANSIT", "NORMAL"]
            print(f"\n[Curriculum] Advancing to {level_names[self.current_level]} (level {self.current_level}) at {self.num_timesteps} steps")

            # Log level transition to TensorBoard
            self.logger.record("curriculum/level_transition", self.current_level)

            # Recreate environments with new difficulty
            env_fns = [make_env_fn(i, seed=self.seed, curriculum_level=self.current_level)
                      for i in range(self.n_envs)]

            # Get the base VecEnv from the wrapper stack
            base_env = self.vec_env
            # Unwrap to find the actual vectorized environment
            while hasattr(base_env, 'venv'):
                base_env = base_env.venv

            # Update environments
            if hasattr(base_env, 'envs'):
                for i, env_fn in enumerate(env_fns):
                    try:
                        base_env.envs[i].close()
                        base_env.envs[i] = env_fn()
                    except Exception as e:
                        print(f"Warning: Could not update env {i}: {e}")

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models_version2_1m/ppo_intersection")
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--render-mode", type=str, choices=["none", "human", "rgb_array"], default="none", help="Render mode for evaluation (none/human/rgb_array)")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (easy->medium->transit->normal)")
    parser.add_argument("--curriculum-steps", type=int, default=100000, help="Steps per curriculum level")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Determine initial curriculum level
    initial_curriculum_level = 0 if args.curriculum else 3  # Start easy if curriculum enabled, else normal (3)

    # Build base vectorized envs (no wrappers yet)
    env_fns = [make_env_fn(i, seed=args.seed, curriculum_level=initial_curriculum_level) for i in range(args.n_envs)]
    if args.n_envs > 1:
        base_env = SubprocVecEnv(env_fns)
    else:
        base_env = DummyVecEnv(env_fns)

    # Build normalized env, loading stats if resuming; apply image transpose after normalization for consistency
    vec_env = None
    vecnorm_path = os.path.join(args.save_path, "vecnormalize.pkl")
    if args.resume and os.path.exists(vecnorm_path):
        try:
            vec_env = VecNormalize.load(vecnorm_path, base_env)
            vec_env.training = True
            vec_env.norm_reward = False
            print("Loaded VecNormalize stats for training env.")
        except Exception:
            print("Failed to load VecNormalize stats; creating new VecNormalize.")
            vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=False)
    else:
        vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=False)

    tensorboard_log = args.save_path if args.tensorboard else None

    # Prepare callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=args.save_path, name_prefix="ppo_checkpoint") # Save every 10000 steps

    # Add curriculum callback if enabled
    callbacks = [checkpoint_callback]
    if args.curriculum:
        curriculum_callback = CurriculumCallback(
            vec_env=vec_env,
            n_envs=args.n_envs,
            seed=args.seed,
            steps_per_level=args.curriculum_steps,
            verbose=1
        )
        callbacks.append(curriculum_callback)
    print(f"Curriculum learning enabled: {args.curriculum_steps} steps per level (easy->medium->transit->normal)")

    # Determine render mode for evaluation based on user input
    eval_render_mode = None if args.render_mode == "none" else args.render_mode

    # Create and wrap eval env the same way as training env (with optional rendering)
    # Always use normal difficulty (level 3) for evaluation
    eval_env = DummyVecEnv([lambda: Monitor(make_env_fn(0, args.seed, render_mode=eval_render_mode, curriculum_level=3)())])
    # Load VecNormalize stats for eval callback if available to match training normalization
    vecnorm_path = os.path.join(args.save_path, "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        try:
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        except Exception:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # training=False for evaluation

    eval_callback = EvalCallback(eval_env, best_model_save_path=args.save_path, log_path=args.save_path, eval_freq=10000, n_eval_episodes=3, deterministic=True)
    callbacks.append(eval_callback)

    policy = "MlpPolicy"
    device = "cpu"

    model = None
    if args.resume:
        latest = find_latest_checkpoint(args.save_path)
        if latest is not None:
            print(f"Resuming from checkpoint: {latest}")
            try:
                # load model and attach already-prepared vec_env
                model = PPO.load(latest, env=vec_env, device=device)
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    if model is None:
        model = PPO(
            policy,
            vec_env,
            n_steps=1024,        # Collect 1024 steps per update
            batch_size=128,      # 128 steps per minibatch
            n_epochs=20,         # Number of updates on the same batch
            learning_rate=3e-4,
            ent_coef=0.02,
            gamma=0.99,          # Discount factor for long-term planning
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=tensorboard_log,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # Save final model and VecNormalize stats
    final_model_path = os.path.join(args.save_path, "ppo_intersection")
    model.save(final_model_path)
    try:
        vec_env.save(os.path.join(args.save_path, "vecnormalize.pkl"))
    except Exception:
        pass

    # Final evaluation using same wrapped env setup as training (with optional rendering)
    # Use normal difficulty (level 3) for final evaluation
    eval_env = DummyVecEnv([lambda: Monitor(make_env_fn(0, args.seed, render_mode=eval_render_mode, curriculum_level=3)())])
    vecnorm_path = os.path.join(args.save_path, "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        try:
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        except Exception:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Run multiple evaluation episodes
    n_eval_episodes = 100
    success_count = collision_count = timeout_count = 0
    ep_rewards: List[float] = []
    ep_lengths: List[int] = []

    print("\nRunning evaluation episodes ...")
    for episode in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            print(f"Episode {episode}, step {steps}: action = {action[0]}")
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            steps += 1
            last_info = info[0]
            try:
                eval_env.render()
            except Exception:
                pass

        ep_rewards.append(ep_reward)
        ep_lengths.append(steps)

        rw = last_info.get('rewards', {}) if isinstance(last_info.get('rewards', {}), dict) else {}
        crashed = bool(last_info.get("crashed", False))
        arrived = bool(float(rw.get('arrived_reward', 0.0)) > 0.0)

        if arrived:
            success_count += 1
        elif crashed:
            collision_count += 1
        else:
            timeout_count += 1

    # Print evaluation statistics
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_count/n_eval_episodes*100:.1f}% ({success_count}/{n_eval_episodes} episodes)")
    print(f"Collision Rate: {collision_count/n_eval_episodes*100:.1f}% ({collision_count}/{n_eval_episodes} episodes)")
    print(f"Timeout Rate: {timeout_count/n_eval_episodes*100:.1f}% ({timeout_count}/{n_eval_episodes} episodes)")


if __name__ == "__main__":
    main()
