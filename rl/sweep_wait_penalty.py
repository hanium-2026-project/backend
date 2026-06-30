"""WAIT-penalty hyperparameter sweep — Pareto curve for conflict vs throughput.

Runs train + eval for several WAIT_PENALTY_BASE values and prints a side-by-side
comparison table.  The goal is to locate the knee point in the conflict-vs-
throughput tradeoff curve.

Reasoning (recap)
-----------------
With WAIT_PENALTY_BASE = -0.2 and conflict penalty = -10:
    5 consecutive WAITs   = -0.2 - 0.3 - 0.4 - 0.5 - 0.6 = -2.0
    avoiding one conflict = +10.0
→ The agent always prefers WAITing.  Throughput collapses.

By increasing WAIT_PENALTY_BASE the breakeven point moves:
    base = -1.0 → 5-WAIT cost = -1.0 - 1.1 - 1.2 - 1.3 - 1.4 = -6.0
    base = -2.0 → 5-WAIT cost = -2.0 - 2.1 - 2.2 - 2.3 - 2.4 = -11.0  (~ conflict)
    base = -3.0 → 5-WAIT cost = -3.0 - 3.1 - 3.2 - 3.3 - 3.4 = -16.0  (> conflict)

We sweep these values and pick the one that best balances metrics.

Usage
-----
    python -m rl.sweep_wait_penalty                       # defaults
    python -m rl.sweep_wait_penalty --timesteps 100000    # longer training
    python -m rl.sweep_wait_penalty --penalties -0.2 -1 -2 -3 --episodes 30
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np

from .train_sb3 import train, evaluate_policy

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_PENALTIES: list[float] = [-0.2, -1.0, -2.0, -3.0]
DEFAULT_TIMESTEPS: int         = 100_000
DEFAULT_EPISODES:  int         = 30
DEFAULT_N_ENVS:    int         = 4
SWEEP_MODEL_DIR:   str         = "models/sweep"


def _fmt(v: float, w: int = 10) -> str:
    return f"{v:>{w}.3f}" if isinstance(v, float) else f"{str(v):>{w}}"


def run_sweep(
    penalties:       list[float] = DEFAULT_PENALTIES,
    total_timesteps: int         = DEFAULT_TIMESTEPS,
    episodes:        int         = DEFAULT_EPISODES,
    n_envs:          int         = DEFAULT_N_ENVS,
    device:          str | None  = None,
) -> dict[float, dict[str, Any]]:
    """Train + evaluate PPO at each WAIT_PENALTY_BASE value; return metrics.

    Heuristic and Random baselines are NOT re-trained — only PPO results are
    affected by the env hyperparameter, but baseline metrics shift slightly
    too because they share the env's WAIT mechanics.  We re-evaluate each
    baseline at every penalty so the table is fully self-consistent.

    Returns
    -------
    dict mapping penalty → {ppo_metrics, heuristic_metrics, random_metrics}
    """
    os.makedirs(SWEEP_MODEL_DIR, exist_ok=True)

    sweep_results: dict[float, dict[str, Any]] = {}

    for i, pen in enumerate(penalties):
        print("\n" + "═" * 70)
        print(f"[sweep {i+1}/{len(penalties)}] WAIT_PENALTY_BASE = {pen:+.2f}")
        print("═" * 70)

        env_kwargs = {"wait_penalty_base": pen}
        model_path = os.path.join(SWEEP_MODEL_DIR, f"ppo_wp{pen:+.2f}.zip")

        # ── train ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        train(
            total_timesteps=total_timesteps,
            output_path=model_path,
            tensorboard_log=f"./tb_logs/sweep_wp{pen:+.2f}/",
            n_envs=n_envs,
            run_bench=False,
            device=device,
            env_kwargs=env_kwargs,
        )
        train_elapsed = time.perf_counter() - t0

        # ── evaluate all 3 policies at this penalty setting ──────────────────
        try:
            from sb3_contrib import MaskablePPO
            ppo_model = MaskablePPO.load(model_path)
        except Exception as exc:
            print(f"  [warn] PPO load failed: {exc}")
            ppo_model = None

        print(f"  [eval]  {episodes} episodes × 3 policies …")
        ppo_m       = evaluate_policy("ppo",       model=ppo_model,
                                       n_episodes=episodes, env_kwargs=env_kwargs)
        heuristic_m = evaluate_policy("heuristic", model=None,
                                       n_episodes=episodes, env_kwargs=env_kwargs)
        random_m    = evaluate_policy("random",    model=None,
                                       n_episodes=episodes, env_kwargs=env_kwargs)

        sweep_results[pen] = {
            "ppo":          ppo_m,
            "heuristic":    heuristic_m,
            "random":       random_m,
            "train_elapsed": train_elapsed,
        }

    return sweep_results


def print_pareto_table(sweep_results: dict[float, dict[str, Any]]) -> None:
    """Print a Pareto-style table: rows = penalties, columns = key metrics."""
    print("\n" + "═" * 90)
    print("PARETO TABLE — conflict vs throughput vs WAIT rate (PPO policy)")
    print("═" * 90)
    print(
        f"  {'WAIT_pen':>10}  "
        f"{'Reward':>10}  {'Conflicts':>10}  {'Throughput':>10}  "
        f"{'WAITs':>8}  {'WAIT %':>7}  {'Departs':>8}  {'OccUtil':>8}"
    )
    print("  " + "─" * 86)
    for pen, res in sweep_results.items():
        m = res["ppo"]
        print(
            f"  {pen:>+10.2f}  "
            f"{m['avg_reward']:>+10.2f}  {m['avg_conflicts']:>10.2f}  "
            f"{m['avg_throughput']:>10.2f}  {m['avg_waits']:>8.1f}  "
            f"{m['avg_wait_rate']*100:>6.1f}%  "
            f"{m['avg_departures']:>8.1f}  {m['avg_occ_util']*100:>7.1f}%"
        )

    # ── secondary table: heuristic comparison (sanity check) ─────────────────
    print("\n  Heuristic baseline (for comparison)")
    print(
        f"  {'WAIT_pen':>10}  "
        f"{'Reward':>10}  {'Conflicts':>10}  {'Throughput':>10}  "
        f"{'WAITs':>8}  {'WAIT %':>7}"
    )
    print("  " + "─" * 70)
    for pen, res in sweep_results.items():
        m = res["heuristic"]
        print(
            f"  {pen:>+10.2f}  "
            f"{m['avg_reward']:>+10.2f}  {m['avg_conflicts']:>10.2f}  "
            f"{m['avg_throughput']:>10.2f}  {m['avg_waits']:>8.1f}  "
            f"{m['avg_wait_rate']*100:>6.1f}%"
        )

    # ── PPO vs Heuristic delta ───────────────────────────────────────────────
    print("\n  PPO – Heuristic Δ  (positive = PPO better; negative throughput = PPO sacrifices throughput)")
    print(
        f"  {'WAIT_pen':>10}  {'Δreward':>10}  {'Δconflicts':>10}  {'Δthroughput':>12}"
    )
    print("  " + "─" * 50)
    for pen, res in sweep_results.items():
        dr = res["ppo"]["avg_reward"]     - res["heuristic"]["avg_reward"]
        dc = res["ppo"]["avg_conflicts"]  - res["heuristic"]["avg_conflicts"]
        dt = res["ppo"]["avg_throughput"] - res["heuristic"]["avg_throughput"]
        print(f"  {pen:>+10.2f}  {dr:>+10.2f}  {dc:>+10.2f}  {dt:>+12.2f}")

    # ── knee-point heuristic ─────────────────────────────────────────────────
    print("\n  Knee-point analysis (PPO):")
    rows = [(pen, r["ppo"]) for pen, r in sweep_results.items()]
    # Score: normalised (throughput – conflict_weight × conflicts)
    confs = np.array([m["avg_conflicts"] for _, m in rows])
    thps  = np.array([m["avg_throughput"] for _, m in rows])
    if confs.max() > confs.min() and thps.max() > thps.min():
        n_conf = (confs - confs.min()) / (confs.max() - confs.min())  # 0=best
        n_thp  = (thps  - thps.min())  / (thps.max() - thps.min())    # 1=best
        score  = n_thp - n_conf   # higher = better
        best_i = int(np.argmax(score))
        print(f"    Recommended: WAIT_PENALTY_BASE = {rows[best_i][0]:+.2f}")
        print(f"      → reward={rows[best_i][1]['avg_reward']:+.2f}, "
              f"conflicts={rows[best_i][1]['avg_conflicts']:.2f}, "
              f"throughput={rows[best_i][1]['avg_throughput']:.2f}")
    else:
        print("    (insufficient variation across penalties)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep WAIT_PENALTY_BASE — Pareto curve for conflict vs throughput"
    )
    p.add_argument("--penalties", type=float, nargs="+",
                   default=DEFAULT_PENALTIES,
                   help=f"WAIT_PENALTY_BASE values to sweep (default: {DEFAULT_PENALTIES})")
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                   help=f"Training timesteps per penalty (default: {DEFAULT_TIMESTEPS:,})")
    p.add_argument("--episodes",  type=int, default=DEFAULT_EPISODES,
                   help=f"Eval episodes per policy (default: {DEFAULT_EPISODES})")
    p.add_argument("--n-envs",    type=int, default=DEFAULT_N_ENVS,
                   help=f"Parallel envs (default: {DEFAULT_N_ENVS})")
    p.add_argument("--device",    default=None,
                   help="torch device: cuda / mps / cpu (auto if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"\n[sweep] penalties = {args.penalties}")
    print(f"[sweep] timesteps = {args.timesteps:,}  episodes = {args.episodes}")
    print(f"[sweep] n_envs    = {args.n_envs}")
    print(f"[sweep] total runs= {len(args.penalties)}")
    print(f"[sweep] estimated time per run ≈ {args.timesteps / 1500:.0f}s "
          f"(plus eval)")

    sweep_results = run_sweep(
        penalties=args.penalties,
        total_timesteps=args.timesteps,
        episodes=args.episodes,
        n_envs=args.n_envs,
        device=args.device,
    )
    print_pareto_table(sweep_results)
