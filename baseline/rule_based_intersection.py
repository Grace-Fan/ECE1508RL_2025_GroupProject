import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import gymnasium as gym
import highway_env


IDLE, LANE_LEFT, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


@dataclass
class AgentConfig:
    desired_speed: float = 20.0  # m/s
    min_headway: float = 6.0     # m
    ttc_brake: float = 2.0       # s
    lateral_tol: float = 4.0     # m, classify ahead vs side
    close_radius: float = 12.0   # m, generic proximity for yielding
    accel_hysteresis: float = 0.5  # m/s tolerance to avoid chattering


class RuleBasedIntersectionAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    def act(self, obs: np.ndarray) -> int:
        """
        Choose a DiscreteMetaAction based on simple, explainable traffic rules.

        High-level policy:
        1) Parse the kinematics grid to get ego pose/velocity and nearby vehicles.
        2) Identify the closest lead vehicle in a forward corridor (same heading lane region).
        3) Apply two independent safety checks:
           - Cross-traffic proximity: yield if a vehicle is close and approximately perpendicular.
           - Car-following: yield/slow if time-to-collision (TTC) is small or headway is short.
        4) Enforce no-backward motion: if longitudinal speed along heading is negative, accelerate.
        5) Track target speed when safe: accelerate towards desired_speed or slow down if overspeeding.

        Returns one of {IDLE=0, LANE_LEFT=1, LANE_RIGHT=2, FASTER=3, SLOWER=4}.
        This baseline only uses longitudinal actions (IDLE/FASTER/SLOWER).
        """
        # 1) Parse observation: expect first row as ego with [presence, x, y, vx, vy, heading]
        ego, others = self._parse_kinematics(obs)
        if ego is None:
            return IDLE

        # Ego state and basic kinematics
        ex, ey, evx, evy, epsi = ego
        v_ego = math.hypot(evx, evy)
        # Ego forward unit vector (heading) and forward speed component
        fx, fy = math.cos(epsi), math.sin(epsi)
        v_forward = evx * fx + evy * fy

        # 2) Lead vehicle search: project each neighbor into ego's Frenet-like frame
        # fwd = distance along heading; lat = signed lateral offset. Keep smallest positive fwd within lateral corridor.
        lead = None
        lead_forward_dist = float("inf")
        lead_rel_speed = 0.0

        for (x, y, vx, vy) in others:
            dx, dy = x - ex, y - ey
            fwd = dx * fx + dy * fy  # forward distance along ego heading
            lat = -dx * fy + dy * fx  # lateral offset
            if fwd <= 0:
                continue
            if abs(lat) > self.cfg.lateral_tol:
                continue
            # Candidate lead vehicle in same corridor
            if fwd < lead_forward_dist:
                lead_forward_dist = fwd
                v_lead_along = vx * fx + vy * fy
                lead_rel_speed = v_ego - v_lead_along
                lead = (x, y, vx, vy)

        # 3a) Cross-traffic yield: if a vehicle is nearby and roughly perpendicular to ego, brake to yield
        yield_brake = False
        for (x, y, vx, vy) in others:
            dx, dy = x - ex, y - ey
            dist = math.hypot(dx, dy)
            if dist < self.cfg.close_radius:
                # Approximate cross-traffic by checking if relative bearing is roughly perpendicular
                # bearing angle between ego forward and vector to other
                dot = (dx * fx + dy * fy) / (dist + 1e-6)
                # cos(theta) ~ 0 => |dot| small
                if abs(dot) < 0.5:  # ~60-120 degrees
                    yield_brake = True
                    break

        # 3b) Car-following safety: compute TTC and headway relative to the nearest lead vehicle
        need_brake = False
        if lead is not None:
            # time-to-collision if ego is faster than lead
            if lead_rel_speed > 0.1:
                ttc = lead_forward_dist / (lead_rel_speed + 1e-6)
            else:
                ttc = float("inf")
            if ttc < self.cfg.ttc_brake or lead_forward_dist < self.cfg.min_headway:
                need_brake = True

        # 4) Decision logic
        # No-backward safety: if moving backward along heading, command acceleration immediately
        if v_forward < -0.1:
            return FASTER

        if yield_brake or need_brake:
            # Avoid commanding reverse: if almost stopped, prefer coasting over braking
            if v_forward < 0.2:
                return IDLE
            return SLOWER

        # 5) Speed tracking towards desired speed when safe
        if v_ego < (self.cfg.desired_speed - self.cfg.accel_hysteresis):
            return FASTER
        elif v_ego > (self.cfg.desired_speed + self.cfg.accel_hysteresis):
            # Avoid reverse when nearly stopped
            if v_forward < 0.2:
                return IDLE
            return SLOWER
        return IDLE

    def _parse_kinematics(self, obs: np.ndarray):
        """
        Parse Kinematics observation into ego (x,y,vx,vy,heading) and list of others [(x,y,vx,vy)].
        Supports both flat array and dict-observation variants used by highway_env.
        Assumes features order: [presence, x, y, vx, vy, heading].
        """
        # If observation is a dict-like containing 'observation'
        if isinstance(obs, dict) and "observation" in obs:
            arr = np.asarray(obs["observation"])  # shape: (N, F)
        else:
            arr = np.asarray(obs)

        if arr.ndim == 1:
            # Unexpected shape, cannot parse
            return None, []

        # Expect shape (N, F) or (F,) per vehicle; first row is ego in highway_env
        ego_row = arr[0]
        if ego_row.shape[0] < 6:
            return None, []

        presence = ego_row[0]
        if presence <= 0:
            return None, []

        ego = (float(ego_row[1]), float(ego_row[2]), float(ego_row[3]), float(ego_row[4]), float(ego_row[5]))
        others: List[Tuple[float, float, float, float]] = []

        for i in range(1, arr.shape[0]):
            row = arr[i]
            if row.shape[0] < 6:
                continue
            if row[0] <= 0:
                continue
            others.append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))

        return ego, others


def make_env(render: bool = False, seed: int = 0):
    render_mode = "human" if render else None
    env = gym.make(
        "intersection-v1",
        render_mode=render_mode,
    )
    try:
        env.unwrapped.configure(
            {
                "action": {"type": "DiscreteMetaAction"},
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy", "heading"],
                    "vehicles_count": 15,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True,
                },
                "destination": "o1",
                "offroad_terminal": True,
            }
        )
    except Exception as e:
        print("Env Configuration failed:", e)

    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
    return env


def evaluate(agent: RuleBasedIntersectionAgent, env, episodes: int = 10, render: bool = False) -> Dict[str, float]:
    stats = {
        "episodes": episodes,
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "mean_reward": 0.0,
        "mean_length": 0.0,
    }

    ep_rewards: List[float] = []
    ep_lengths: List[int] = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        last_info = {}

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            steps += 1
            last_info = info
            if render and hasattr(env, "render"):
                env.render()

        ep_rewards.append(ep_reward)
        ep_lengths.append(steps)

        rw = last_info.get('rewards', {}) if isinstance(last_info.get('rewards', {}), dict) else {}
        crashed = bool(last_info.get("crashed", False))
        arrived = bool(float(rw.get('arrived_reward', 0.0)) > 0.0)

        if arrived:
            stats["success"] += 1
        elif crashed:
            stats["collision"] += 1
        else:
            stats["timeout"] += 1

    stats["mean_reward"] = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    stats["mean_length"] = float(np.mean(ep_lengths)) if ep_lengths else 0.0
    stats["success_rate"] = stats["success"] / max(1, episodes)
    stats["collision_rate"] = stats["collision"] / max(1, episodes)
    stats["timeout_rate"] = stats["timeout"] / max(1, episodes)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Rule-based baseline for highway_env intersection-v1"
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", type=int, default=0, help="Render the environment (1=yes, 0=no)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", type=str, default="", help="Optional path to save JSON metrics")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = AgentConfig()
    agent = RuleBasedIntersectionAgent(cfg)

    env = make_env(render=bool(args.render), seed=args.seed)

    try:
        stats = evaluate(agent, env, episodes=args.episodes, render=bool(args.render))
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("Rule-based baseline results:")
    print(json.dumps(stats, indent=2))

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print(f"Saved metrics to {args.out}")
        except Exception as e:
            print(f"Failed to write metrics: {e}")


if __name__ == "__main__":
    main()
