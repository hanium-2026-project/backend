"""MaskablePPO training + evaluation script for parking slot allocation.

Requirements
------------
    pip install stable-baselines3 sb3-contrib gymnasium

Usage (CLI)
-----------
    python -m rl.train_sb3                 # train then evaluate
    python -m rl.train_sb3 --eval-only    # skip training, evaluate saved model

Usage (Python)
--------------
    from rl.train_sb3 import train, evaluate_all
    path = train(total_timesteps=100_000)
    evaluate_all(model_path=path, n_episodes=100)
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from typing import Any

import numpy as np

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH: str = "models/sb3_parking_policy.zip"
DEFAULT_TB_LOG: str = "./tb_logs/"
DEFAULT_TIMESTEPS: int = 100_000
N_EVAL_EPISODES: int = 100

POLICY_TYPES = ["random", "heuristic", "ppo"]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    output_path: str = DEFAULT_MODEL_PATH,
    tensorboard_log: str = DEFAULT_TB_LOG,
) -> str:
    """Train a MaskablePPO agent on ParkingRoutingEnv and save the model.

    Parameters
    ----------
    total_timesteps : Total environment steps for training.
    output_path     : Destination path for the saved .zip model.
    tensorboard_log : Directory for TensorBoard logs.
                      Launch viewer with: tensorboard --logdir ./tb_logs/

    Returns
    -------
    str — Absolute path to the saved model file.
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError as exc:
        raise ImportError(
            "sb3-contrib is required for training. "
            "Install with: pip install stable-baselines3 sb3-contrib"
        ) from exc

    from .parking_env import ParkingRoutingEnv

    env = ActionMasker(ParkingRoutingEnv(), lambda e: e.action_masks())

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log=tensorboard_log,
    )

    print(f"\n[train] Starting MaskablePPO — {total_timesteps:,} timesteps")
    print(f"[train] TensorBoard logs → {os.path.abspath(tensorboard_log)}")
    print(f"[train] Run: tensorboard --logdir {tensorboard_log}\n")

    model.learn(total_timesteps=total_timesteps)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model.save(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"\n[train] Model saved → {abs_path}")
    return abs_path


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION — single episode
# ═══════════════════════════════════════════════════════════════════════════════

def _run_episode(
    policy_type: str,
    env,                        # ParkingRoutingEnv instance
    model: Any | None = None,
) -> dict[str, Any]:
    """Run one episode and return per-episode metrics."""
    from .parking_env import SLOT_ROUTES, NODE_TRAVEL_TIME

    obs, _ = env.reset()

    total_reward: float = 0.0
    n_conflicts: int = 0
    n_steps: int = 0
    slots_assigned: list[str] = []          # successfully reserved slots
    travel_times: list[float] = []          # route traversal time per slot

    while True:
        masks: np.ndarray = env.action_masks()

        # Safety: all masked → episode should have terminated already
        if not masks.any():
            break

        # ── select action ────────────────────────────────────────────────────
        if policy_type == "random":
            valid = [i for i, ok in enumerate(masks) if ok]
            action = random.choice(valid)

        elif policy_type == "heuristic":
            from .inference import heuristic_policy
            action = heuristic_policy(masks)

        else:  # ppo
            if model is None:
                raise ValueError("model must be provided for ppo policy")
            obs_batch = obs.reshape(1, -1)
            action_arr, _ = model.predict(
                obs_batch,
                action_masks=masks,
                deterministic=True,
            )
            action = int(np.asarray(action_arr).flat[0])

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        n_steps += 1

        if info.get("conflict"):
            n_conflicts += 1
        elif info.get("reason") != "slot_already_taken":
            # Successful assignment
            slot = info["slot"]
            slots_assigned.append(slot)
            travel_times.append(len(SLOT_ROUTES[slot]) * NODE_TRAVEL_TIME)

        if terminated or truncated:
            break

    throughput = len(slots_assigned)
    avg_travel = float(np.mean(travel_times)) if travel_times else 0.0
    # efficiency: ratio of successful steps vs total steps
    efficiency = throughput / max(n_steps, 1)

    return {
        "total_reward": total_reward,
        "n_conflicts": n_conflicts,
        "n_steps": n_steps,
        "throughput": throughput,
        "avg_travel_time": avg_travel,
        "efficiency": efficiency,
        "slots_assigned": slots_assigned,
        "n_fallback": env.n_fallback_triggers,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION — aggregate over n episodes
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(
    policy_type: str,
    model: Any | None = None,
    n_episodes: int = N_EVAL_EPISODES,
) -> dict[str, Any]:
    """Evaluate a single policy over n_episodes, return aggregated metrics."""
    from .parking_env import ParkingRoutingEnv, SLOT_NAMES

    env = ParkingRoutingEnv()

    rewards: list[float] = []
    conflicts: list[int] = []
    throughputs: list[int] = []
    travel_times: list[float] = []
    efficiencies: list[float] = []
    slot_counts: dict[str, int] = defaultdict(int)
    n_fallbacks: list[int] = []

    for _ in range(n_episodes):
        m = _run_episode(policy_type, env, model)
        rewards.append(m["total_reward"])
        conflicts.append(m["n_conflicts"])
        throughputs.append(m["throughput"])
        travel_times.append(m["avg_travel_time"])
        efficiencies.append(m["efficiency"])
        n_fallbacks.append(m["n_fallback"])
        for s in m["slots_assigned"]:
            slot_counts[s] += 1

    return {
        "avg_reward":       float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "avg_conflicts":    float(np.mean(conflicts)),
        "avg_throughput":   float(np.mean(throughputs)),
        "avg_travel_time":  float(np.mean(travel_times)),
        "avg_efficiency":   float(np.mean(efficiencies)),
        "avg_fallbacks":    float(np.mean(n_fallbacks)),
        "slot_distribution": dict(slot_counts),
        # raw lists for stability analysis
        "_rewards":  rewards,
        "_conflicts": conflicts,
        "_fallbacks": n_fallbacks,
    }


def evaluate_all(
    model_path: str = DEFAULT_MODEL_PATH,
    n_episodes: int = N_EVAL_EPISODES,
) -> dict[str, dict]:
    """Compare Random, Heuristic, PPO policies over n_episodes each.

    Returns
    -------
    dict mapping policy name → aggregated metrics.
    """
    # ── load PPO model ────────────────────────────────────────────────────────
    ppo_model: Any | None = None
    active_policies = ["random", "heuristic"]

    try:
        from sb3_contrib import MaskablePPO
        zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
        load_path = model_path if os.path.exists(model_path) else (
            zip_path if os.path.exists(zip_path) else None
        )
        if load_path:
            ppo_model = MaskablePPO.load(load_path)
            active_policies.append("ppo")
            print(f"[eval] Loaded PPO model ← {load_path}")
        else:
            print(f"[eval] No model found at '{model_path}' — skipping PPO eval")
    except ImportError:
        print("[eval] sb3-contrib not available — skipping PPO eval")

    # ── run evaluations ───────────────────────────────────────────────────────
    print(f"\nEvaluating {n_episodes} episodes × {len(active_policies)} policies …\n")

    results: dict[str, dict] = {}
    for ptype in active_policies:
        print(f"  [{ptype:>9}] …", end=" ", flush=True)
        results[ptype] = evaluate_policy(
            ptype,
            model=ppo_model if ptype == "ppo" else None,
            n_episodes=n_episodes,
        )
        print("done")

    # ── reports ───────────────────────────────────────────────────────────────
    _print_results(results)
    _check_stability(results)
    _print_slot_histogram(results)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_COL = 12   # column width for table


def _fmt(val: Any, width: int = _COL) -> str:
    if isinstance(val, float):
        return f"{val:>{width}.3f}"
    return f"{str(val):>{width}}"


def _print_results(results: dict[str, dict]) -> None:
    """Print side-by-side comparison table."""
    policies = list(results.keys())
    sep = "─" * (28 + _COL * len(policies))

    print("\n" + "═" * len(sep))
    print("EVALUATION RESULTS")
    print("═" * len(sep))

    header = f"  {'Metric':<26}" + "".join(f"{p.upper():>{_COL}}" for p in policies)
    print(header)
    print("  " + sep)

    rows = [
        ("avg_reward",      "Avg Reward"),
        ("std_reward",      "Reward Std"),
        ("avg_conflicts",   "Avg Conflicts / ep"),
        ("avg_throughput",  "Avg Throughput (slots)"),
        ("avg_travel_time", "Avg Travel Time (s)"),
        ("avg_efficiency",  "Efficiency (slots/step)"),
        ("avg_fallbacks",   "Avg Fallback Masks / ep"),
    ]

    for key, label in rows:
        row = f"  {label:<26}"
        for p in policies:
            row += _fmt(results[p].get(key, float("nan")))
        print(row)

    print()
    # One-liner summary
    for ptype in policies:
        r = results[ptype]
        print(
            f"  {ptype.capitalize():<10} avg reward: {r['avg_reward']:+.3f} ± {r['std_reward']:.3f}"
            f"   conflicts: {r['avg_conflicts']:.2f}"
            f"   throughput: {r['avg_throughput']:.2f}"
        )


def _check_stability(results: dict[str, dict]) -> None:
    """Check for training / inference instability signals."""
    print("\n" + "═" * 60)
    print("STABILITY CHECKS")
    print("═" * 60)

    for ptype, r in results.items():
        rewards   = r["_rewards"]
        conflicts = r["_conflicts"]
        fallbacks = r["_fallbacks"]

        # NaN / Inf in rewards
        nan_eps = sum(1 for x in rewards if math.isnan(x) or math.isinf(x))

        # Reward std — very low std can indicate collapse
        reward_std = float(np.std(rewards))

        # Action entropy from slot distribution
        dist  = r["slot_distribution"]
        total = sum(dist.values()) or 1
        probs = [v / total for v in dist.values()]
        entropy     = -sum(p * math.log(p + 1e-9) for p in probs)
        max_entropy = math.log(8)              # uniform over 8 slots
        entropy_pct = entropy / max_entropy * 100

        # All-False fallback rate
        avg_fb   = float(np.mean(fallbacks))
        n_eps    = len(rewards)
        fb_eps   = sum(1 for f in fallbacks if f > 0)  # episodes with ≥1 fallback

        # Conflict rate (conflicts / total steps estimated)
        avg_conf = float(np.mean(conflicts))

        issues = []
        if nan_eps > 0:
            issues.append(f"⚠  {nan_eps} episodes with NaN/Inf reward")
        if reward_std < 1e-3:
            issues.append("⚠  reward std ≈ 0 — possible policy collapse")
        if entropy_pct < 30:
            issues.append(f"⚠  low action entropy ({entropy_pct:.0f}%) — policy is overspecialised")
        if avg_fb > 2:
            issues.append(f"⚠  high fallback rate ({avg_fb:.1f}/ep) — conflicts dominate")

        print(f"\n  {ptype.upper()}")
        print(f"    NaN / Inf rewards    : {nan_eps} / {n_eps} episodes")
        print(f"    Reward std           : {reward_std:.4f}")
        print(f"    Action entropy       : {entropy:.3f} / {max_entropy:.3f}  ({entropy_pct:.1f}%)")
        print(f"    Fallback triggers    : {avg_fb:.2f} avg / ep   ({fb_eps}/{n_eps} eps affected)")
        print(f"    Avg conflicts / ep   : {avg_conf:.2f}")
        if issues:
            for msg in issues:
                print(f"    {msg}")
        else:
            print("    ✓ No stability issues detected")


def _print_slot_histogram(results: dict[str, dict]) -> None:
    """ASCII histogram of slot assignment frequency per policy."""
    from .parking_env import SLOT_NAMES

    print("\n" + "═" * 60)
    print("SLOT USAGE HISTOGRAM")
    print("═" * 60)

    for ptype, r in results.items():
        dist  = r["slot_distribution"]
        total = sum(dist.values()) or 1

        print(f"\n  {ptype.upper()}")
        print(f"  {'Slot':<6} {'Count':>6}  {'%':>6}  {'Bar':<30}")
        print(f"  {'─'*55}")

        for name in SLOT_NAMES:
            cnt  = dist.get(name, 0)
            pct  = cnt / total * 100
            bar  = "█" * int(pct / 2.5)       # 40 chars = 100%
            flag = "  ← bias?" if pct > 20 else ""
            print(f"  {name:<6} {cnt:>6}  {pct:>5.1f}%  {bar:<30}{flag}")

    # Cross-policy bias summary (slots with >20% share in any policy)
    all_biased: set[str] = set()
    for r in results.values():
        dist  = r["slot_distribution"]
        total = sum(dist.values()) or 1
        for name in SLOT_NAMES:
            if dist.get(name, 0) / total * 100 > 20:
                all_biased.add(name)

    if all_biased:
        print(f"\n  ⚠  Heavily-used slots (>20% share in ≥1 policy): {sorted(all_biased)}")
    else:
        print("\n  ✓ No slot shows >20% usage bias across any policy")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and/or evaluate MaskablePPO for parking slot allocation"
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate the saved model only",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        metavar="N",
        help=f"Training timesteps (default: {DEFAULT_TIMESTEPS:,})",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=N_EVAL_EPISODES,
        metavar="N",
        help=f"Evaluation episodes per policy (default: {N_EVAL_EPISODES})",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"Model path (default: {DEFAULT_MODEL_PATH})",
    )
    p.add_argument(
        "--tb-log",
        default=DEFAULT_TB_LOG,
        metavar="DIR",
        help=f"TensorBoard log directory (default: {DEFAULT_TB_LOG})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.eval_only:
        train(
            total_timesteps=args.timesteps,
            output_path=args.model,
            tensorboard_log=args.tb_log,
        )

    evaluate_all(model_path=args.model, n_episodes=args.episodes)
