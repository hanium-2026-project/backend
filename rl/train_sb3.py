"""MaskablePPO training + evaluation script for parking slot allocation.

Design notes for the departure-augmented environment (MAX_STEPS=64)
--------------------------------------------------------------------
* Device detection  : CUDA → MPS (Apple Silicon) → CPU, auto-selected.
* VecEnv strategy   : DummyVecEnv beats SubprocVecEnv for this env because
                      env.step() is ~0.05 ms — far below the IPC round-trip
                      overhead of SubprocVecEnv (~1 ms).  We run n_envs
                      workers inside DummyVecEnv for data diversity.
* Hyperparameters   : tuned for 64-step episodes with dynamic departures.
    n_steps    = 4096   (was 2048; covers ~64 full episodes per update)
    batch_size = 256    (was  64;  stable gradients for larger rollouts)
    gamma      = 0.995  (was 0.99; longer horizon)
    ent_coef   = 0.005  (small entropy bonus; departure widens action space)
    net_arch   = [128, 128]  (was default [64,64]; slightly more capacity)
* Extended metrics  : departure_count, slot_reuse, occupancy_utilization,
                      fast_forward_count tracked alongside reward/conflicts.

Requirements
------------
    pip install stable-baselines3 sb3-contrib gymnasium

Usage (CLI)
-----------
    python -m rl.train_sb3                                   # train + eval
    python -m rl.train_sb3 --eval-only                       # eval only
    python -m rl.train_sb3 --timesteps 500000 --n-envs 4    # custom

Usage (Python)
--------------
    from rl.train_sb3 import train, evaluate_all
    path = train(total_timesteps=500_000, n_envs=4)
    evaluate_all(model_path=path, n_episodes=100)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import defaultdict
from typing import Any

import numpy as np

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH: str  = "models/sb3_parking_policy.zip"
DEFAULT_TB_LOG: str      = "./tb_logs/"
DEFAULT_TIMESTEPS: int   = 300_000
DEFAULT_N_ENVS: int      = 4
N_EVAL_EPISODES: int     = 100

POLICY_TYPES = ["random", "heuristic", "heuristic_v2", "heuristic_v3", "heuristic_v4", "ppo"]


# ═══════════════════════════════════════════════════════════════════════════════
# Device detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_device() -> str:
    """Return the best available torch device string.

    Priority: CUDA → MPS (Apple Silicon) → CPU.
    For the tiny MLP used here, CPU is often competitive with GPU because
    the policy forward/backward pass is not the training bottleneck —
    rollout collection (env.step) is.  GPU is still used if available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[device] CUDA  — {name} ({vram:.1f} GB VRAM)")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[device] MPS   — Apple Silicon GPU (Metal Performance Shaders)")
            return "mps"
    except Exception:
        pass
    print("[device] CPU   — no GPU detected (or torch not installed)")
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# VecEnv benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def _benchmark_vec_env(n_envs: int = 4, n_steps: int = 500) -> str:
    """Return 'dummy' or 'subproc' based on measured throughput.

    For environments where env.step() < IPC round-trip time (~1 ms),
    DummyVecEnv is faster.  This function measures both and picks the winner.

    Returns 'dummy' or 'subproc'.
    """
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .parking_env import ParkingRoutingEnv

    def _make():
        return ActionMasker(ParkingRoutingEnv(), lambda e: e.action_masks())

    timings: dict[str, float] = {}

    for cls_name, cls in [("dummy", DummyVecEnv), ("subproc", SubprocVecEnv)]:
        try:
            venv = make_vec_env(_make, n_envs=n_envs, vec_env_cls=cls)
            obs = venv.reset()
            t0 = time.perf_counter()
            for _ in range(n_steps):
                actions = [venv.action_space.sample() for _ in range(n_envs)]
                venv.step(actions)
            timings[cls_name] = time.perf_counter() - t0
            venv.close()
        except Exception as exc:
            timings[cls_name] = float("inf")
            print(f"  [bench] {cls_name} failed: {exc}")

    winner = min(timings, key=timings.get)
    for k, v in timings.items():
        steps_per_sec = n_envs * n_steps / v if v < float("inf") else 0
        flag = " ← winner" if k == winner else ""
        print(f"  [bench] {k:<8} {steps_per_sec:>7.0f} steps/s{flag}")

    return winner


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    output_path: str     = DEFAULT_MODEL_PATH,
    tensorboard_log: str = DEFAULT_TB_LOG,
    n_envs: int          = DEFAULT_N_ENVS,
    run_bench: bool      = False,
    device: str | None   = None,
    env_kwargs: dict | None = None,
) -> str:
    """Train a MaskablePPO agent on ParkingRoutingEnv and save the model.

    Parameters
    ----------
    total_timesteps : Total environment interaction steps.
    output_path     : Path to save the .zip model.
    tensorboard_log : TensorBoard log directory.
    n_envs          : Number of parallel environments (DummyVecEnv).
    run_bench       : If True, benchmark DummyVecEnv vs SubprocVecEnv first.
    device          : torch device ('cuda'/'mps'/'cpu').  Auto-detected if None.

    Returns
    -------
    str — Absolute path to the saved model.
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "sb3-contrib is required.  "
            "Install: pip install stable-baselines3 sb3-contrib"
        ) from exc

    from .parking_env import ParkingRoutingEnv

    # ── device ────────────────────────────────────────────────────────────────
    if device is None:
        device = _detect_device()

    # ── VecEnv benchmark (optional) ───────────────────────────────────────────
    if run_bench and n_envs > 1:
        print(f"\n[bench] Benchmarking VecEnv strategies (n_envs={n_envs}) …")
        bench_winner = _benchmark_vec_env(n_envs=n_envs)
        print(f"[bench] Using: {bench_winner}\n")
    else:
        bench_winner = "dummy"

    # ── build vectorised env ──────────────────────────────────────────────────
    ekw = env_kwargs or {}
    def _make_env():
        return ActionMasker(ParkingRoutingEnv(**ekw), lambda e: e.action_masks())

    if bench_winner == "subproc" and n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        venv = make_vec_env(_make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    else:
        venv = make_vec_env(_make_env, n_envs=n_envs, vec_env_cls=DummyVecEnv)

    # ── model ──────────────────────────────────────────────────────────────────
    # Hyperparameters tuned for MAX_STEPS=64 departure environment.
    #
    # n_steps=4096 : One update covers ~64 full episodes (64 steps × ~64 envs
    #                equivalent).  Larger rollout → stable value estimates.
    # batch_size=256: Minibatch large enough for stable gradients.
    # gamma=0.995  : Longer horizon; departure rewards arrive 8–24 s later.
    # ent_coef=0.005: Tiny entropy bonus prevents early slot-ordering collapse.
    # net_arch=[128,128]: Modest capacity increase over default [64,64].
    model = MaskablePPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.005,
        clip_range=0.2,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=tensorboard_log,
        device=device,
    )

    print(f"\n[train] MaskablePPO — {total_timesteps:,} timesteps")
    print(f"[train] device={device}  n_envs={n_envs}  "
          f"n_steps={model.n_steps}  batch={model.batch_size}")
    print(f"[train] gamma={model.gamma}  ent_coef={model.ent_coef}")
    print(f"[train] net_arch: {model.policy.net_arch}")
    print(f"[train] TensorBoard → tensorboard --logdir {tensorboard_log}\n")

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - t0
    fps = total_timesteps / elapsed
    print(f"\n[train] Finished in {elapsed:.0f}s  ({fps:.0f} steps/s)")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    model.save(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"[train] Model saved → {abs_path}")
    return abs_path


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION — single episode
# ═══════════════════════════════════════════════════════════════════════════════

def _run_episode(
    policy_type: str,
    env,
    model: Any | None = None,
) -> dict[str, Any]:
    """Run one episode; return per-episode metrics (extended for departure env)."""
    from .parking_env import SLOT_ROUTES, NODE_TRAVEL_TIME

    obs, _ = env.reset()

    total_reward:    float = 0.0
    n_conflicts:     int   = 0
    n_waits:         int   = 0
    n_steps:         int   = 0
    slots_assigned:  list[str]   = []
    travel_times:    list[float] = []
    free_slot_sum:   float = 0.0    # accumulated free slots each step (→ avg)
    departure_count: int   = 0
    fast_fwd_count:  int   = 0
    max_consec_wait: int   = 0

    while True:
        masks: np.ndarray = env.action_masks()
        if not masks.any():
            break

        # ── select action ────────────────────────────────────────────────────
        if policy_type == "random":
            valid  = [i for i, ok in enumerate(masks) if ok]
            action = random.choice(valid)

        elif policy_type == "heuristic":
            from .inference import heuristic_policy
            action = heuristic_policy(masks)

        elif policy_type == "heuristic_v2":
            from .inference import heuristic_policy_v2
            action = heuristic_policy_v2(masks, env)

        elif policy_type == "heuristic_v3":
            from .inference import heuristic_policy_v3
            action = heuristic_policy_v3(masks, env)

        elif policy_type == "heuristic_v4":
            from .inference import heuristic_policy_v4
            action = heuristic_policy_v4(masks, env)

        else:  # ppo
            if model is None:
                raise ValueError("model required for ppo policy")
            obs_b = obs.reshape(1, -1)
            a, _  = model.predict(obs_b, action_masks=masks, deterministic=True)
            action = int(np.asarray(a).flat[0])

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        n_steps      += 1
        free_slot_sum += float(info.get("free_slot_count", 0))

        if info.get("wait"):
            n_waits += 1
            cw = int(info.get("consecutive_waits", 0))
            if cw > max_consec_wait:
                max_consec_wait = cw
        elif info.get("conflict"):
            n_conflicts += 1
        elif info.get("reason") != "slot_already_taken":
            slot = info.get("slot")
            if slot is not None:
                slots_assigned.append(slot)
                travel_times.append(len(SLOT_ROUTES[slot]) * NODE_TRAVEL_TIME)

        # Capture latest episode-level counters from info
        departure_count = int(info.get("departure_count",    departure_count))
        fast_fwd_count  = int(info.get("fast_forward_count", fast_fwd_count))

        if terminated or truncated:
            break

    throughput = len(slots_assigned)
    avg_travel = float(np.mean(travel_times)) if travel_times else 0.0
    # efficiency = assignments per total step (WAIT steps count as steps)
    efficiency = throughput / max(n_steps, 1)
    # Slot reuse: assignments beyond the initial 8 (only meaningful if > 8)
    slot_reuse = max(0, throughput - 8)
    # Occupancy utilisation: average fraction of slots that were occupied
    avg_free      = free_slot_sum / max(n_steps, 1)
    occ_util      = 1.0 - avg_free / 8.0
    wait_rate     = n_waits / max(n_steps, 1)

    return {
        "total_reward":     total_reward,
        "n_conflicts":      n_conflicts,
        "n_waits":          n_waits,
        "wait_rate":        wait_rate,
        "max_consec_wait":  max_consec_wait,
        "n_steps":          n_steps,
        "throughput":       throughput,
        "avg_travel_time":  avg_travel,
        "efficiency":       efficiency,
        "slot_reuse":       slot_reuse,
        "departure_count":  departure_count,
        "fast_fwd_count":   fast_fwd_count,
        "occ_utilization":  occ_util,
        "slots_assigned":   slots_assigned,
        "n_fallback":       env.n_fallback_triggers,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION — aggregate over n episodes
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(
    policy_type: str,
    model: Any | None = None,
    n_episodes: int   = N_EVAL_EPISODES,
    env_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run n_episodes and aggregate metrics."""
    from .parking_env import ParkingRoutingEnv, SLOT_NAMES

    env = ParkingRoutingEnv(**(env_kwargs or {}))

    rewards:        list[float] = []
    conflicts:      list[int]   = []
    waits:          list[int]   = []
    wait_rates:     list[float] = []
    max_consecs:    list[int]   = []
    throughputs:    list[int]   = []
    travel_times:   list[float] = []
    efficiencies:   list[float] = []
    slot_reuses:    list[int]   = []
    departures:     list[int]   = []
    fast_fwds:      list[int]   = []
    occ_utils:      list[float] = []
    n_fallbacks:    list[int]   = []
    slot_counts:    dict[str, int] = defaultdict(int)

    for _ in range(n_episodes):
        m = _run_episode(policy_type, env, model)
        rewards.append(m["total_reward"])
        conflicts.append(m["n_conflicts"])
        waits.append(m["n_waits"])
        wait_rates.append(m["wait_rate"])
        max_consecs.append(m["max_consec_wait"])
        throughputs.append(m["throughput"])
        travel_times.append(m["avg_travel_time"])
        efficiencies.append(m["efficiency"])
        slot_reuses.append(m["slot_reuse"])
        departures.append(m["departure_count"])
        fast_fwds.append(m["fast_fwd_count"])
        occ_utils.append(m["occ_utilization"])
        n_fallbacks.append(m["n_fallback"])
        for s in m["slots_assigned"]:
            slot_counts[s] += 1

    return {
        "avg_reward":        float(np.mean(rewards)),
        "std_reward":        float(np.std(rewards)),
        "avg_conflicts":     float(np.mean(conflicts)),
        "avg_waits":         float(np.mean(waits)),
        "avg_wait_rate":     float(np.mean(wait_rates)),
        "avg_max_consec":    float(np.mean(max_consecs)),
        "avg_throughput":    float(np.mean(throughputs)),
        "avg_travel_time":   float(np.mean(travel_times)),
        "avg_efficiency":    float(np.mean(efficiencies)),
        "avg_slot_reuse":    float(np.mean(slot_reuses)),
        "avg_departures":    float(np.mean(departures)),
        "avg_fast_fwd":      float(np.mean(fast_fwds)),
        "avg_occ_util":      float(np.mean(occ_utils)),
        "avg_fallbacks":     float(np.mean(n_fallbacks)),
        "slot_distribution": dict(slot_counts),
        # raw lists for stability analysis
        "_rewards":    rewards,
        "_conflicts":  conflicts,
        "_waits":      waits,
        "_fallbacks":  n_fallbacks,
        "_departures": departures,
        "_throughputs": throughputs,
    }


def evaluate_all(
    model_path: str  = DEFAULT_MODEL_PATH,
    n_episodes: int  = N_EVAL_EPISODES,
    env_kwargs: dict | None = None,
) -> dict[str, dict]:
    """Compare Random, Heuristic V1-V4, and PPO over n_episodes each."""
    ppo_model: Any | None = None
    active = ["random", "heuristic", "heuristic_v2", "heuristic_v3", "heuristic_v4"]

    try:
        from sb3_contrib import MaskablePPO
        zp = model_path if model_path.endswith(".zip") else model_path + ".zip"
        lp = (model_path if os.path.exists(model_path) else
              zp          if os.path.exists(zp)         else None)
        if lp:
            ppo_model = MaskablePPO.load(lp)
            active.append("ppo")
            print(f"[eval] Loaded PPO model ← {lp}")
        else:
            print(f"[eval] No model at '{model_path}' — skipping PPO")
    except ImportError:
        print("[eval] sb3-contrib not available — skipping PPO")

    print(f"\nEvaluating {n_episodes} episodes × {len(active)} policies …\n")
    results: dict[str, dict] = {}
    for ptype in active:
        print(f"  [{ptype:>9}] …", end=" ", flush=True)
        results[ptype] = evaluate_policy(
            ptype,
            model=ppo_model if ptype == "ppo" else None,
            n_episodes=n_episodes,
            env_kwargs=env_kwargs,
        )
        print("done")

    _print_results(results)
    _check_stability(results)
    _print_slot_histogram(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_COL = 13


def _fmt(val: Any, width: int = _COL) -> str:
    if isinstance(val, float):
        return f"{val:>{width}.3f}"
    return f"{str(val):>{width}}"


def _print_results(results: dict[str, dict]) -> None:
    policies = list(results.keys())
    sep = "─" * (32 + _COL * len(policies))

    print("\n" + "═" * len(sep))
    print("EVALUATION RESULTS  (departure-augmented env, MAX_STEPS=64)")
    print("═" * len(sep))

    header = f"  {'Metric':<30}" + "".join(f"{p.upper():>{_COL}}" for p in policies)
    print(header)
    print("  " + sep)

    rows = [
        ("avg_reward",       "Avg Reward"),
        ("std_reward",       "Reward Std"),
        ("avg_conflicts",    "Avg Conflicts / ep"),
        ("avg_waits",        "Avg WAITs / ep"),
        ("avg_wait_rate",    "WAIT rate (waits/steps)"),
        ("avg_max_consec",   "Avg Max Consec WAITs"),
        ("avg_throughput",   "Avg Throughput (slots)"),
        ("avg_slot_reuse",   "Avg Slot Reuse (>8 assigns)"),
        ("avg_departures",   "Avg Departures / ep"),
        ("avg_fast_fwd",     "Avg Fast-Forwards / ep"),
        ("avg_occ_util",     "Occupancy Utilisation (0–1)"),
        ("avg_efficiency",   "Efficiency (slots/step)"),
        ("avg_travel_time",  "Avg Travel Time (s)"),
        ("avg_fallbacks",    "Avg Fallback Masks / ep"),
    ]

    for key, label in rows:
        row = f"  {label:<30}"
        for p in policies:
            row += _fmt(results[p].get(key, float("nan")))
        print(row)

    print()
    for ptype in policies:
        r = results[ptype]
        print(
            f"  {ptype.capitalize():<10}"
            f" reward: {r['avg_reward']:+.3f}±{r['std_reward']:.2f}"
            f"  conflicts: {r['avg_conflicts']:.2f}"
            f"  waits: {r['avg_waits']:.1f} ({r['avg_wait_rate']:.1%})"
            f"  throughput: {r['avg_throughput']:.2f}"
            f"  departures: {r['avg_departures']:.1f}"
            f"  occ: {r['avg_occ_util']:.2%}"
        )


def _check_stability(results: dict[str, dict]) -> None:
    print("\n" + "═" * 60)
    print("STABILITY CHECKS")
    print("═" * 60)

    for ptype, r in results.items():
        rewards    = r["_rewards"]
        conflicts  = r["_conflicts"]
        waits      = r["_waits"]
        fallbacks  = r["_fallbacks"]
        departures = r["_departures"]

        nan_eps    = sum(1 for x in rewards if math.isnan(x) or math.isinf(x))
        reward_std = float(np.std(rewards))

        dist  = r["slot_distribution"]
        total = sum(dist.values()) or 1
        probs = [v / total for v in dist.values()]
        entropy     = -sum(p * math.log(p + 1e-9) for p in probs)
        max_entropy = math.log(8)
        entropy_pct = entropy / max_entropy * 100

        avg_fb    = float(np.mean(fallbacks))
        avg_wait  = float(np.mean(waits))
        n_eps     = len(rewards)
        fb_eps    = sum(1 for f in fallbacks if f > 0)
        avg_conf  = float(np.mean(conflicts))
        avg_dep   = float(np.mean(departures))
        avg_mc    = float(r.get("avg_max_consec", 0))

        issues = []
        if nan_eps > 0:
            issues.append(f"⚠  {nan_eps} episodes with NaN/Inf reward")
        if reward_std < 1e-3:
            issues.append("⚠  reward std ≈ 0 — possible policy collapse")
        if entropy_pct < 30:
            issues.append(f"⚠  low action entropy ({entropy_pct:.0f}%) — overspecialised")
        if avg_fb > 4:
            issues.append(f"⚠  high fallback rate ({avg_fb:.1f}/ep) — conflicts dominate")
        if avg_dep < 1:
            issues.append("⚠  avg departures < 1/ep — departure dynamics not triggering")
        # WAIT-collapse check: if waits > 30% of MAX_STEPS policy is over-waiting
        from .parking_env import MAX_STEPS, MAX_CONSECUTIVE_WAITS
        wait_collapse_threshold = MAX_STEPS * 0.30
        if avg_wait > wait_collapse_threshold:
            issues.append(
                f"⚠  WAIT collapse risk: avg {avg_wait:.1f} waits/ep "
                f"({avg_wait/MAX_STEPS*100:.0f}% of MAX_STEPS)"
            )
        if avg_mc >= MAX_CONSECUTIVE_WAITS:
            issues.append(
                f"⚠  hit MAX_CONSECUTIVE_WAITS={MAX_CONSECUTIVE_WAITS} "
                f"— forced fallback triggered"
            )

        print(f"\n  {ptype.upper()}")
        print(f"    NaN / Inf rewards    : {nan_eps} / {n_eps} episodes")
        print(f"    Reward std           : {reward_std:.4f}")
        print(f"    Action entropy       : {entropy:.3f} / {max_entropy:.3f}  ({entropy_pct:.1f}%)")
        print(f"    Fallback triggers    : {avg_fb:.2f} avg / ep  ({fb_eps}/{n_eps} affected)")
        print(f"    Avg conflicts / ep   : {avg_conf:.2f}")
        print(f"    Avg WAITs / ep       : {avg_wait:.2f}  (max consec: {avg_mc:.1f})")
        print(f"    Avg departures / ep  : {avg_dep:.2f}")
        if issues:
            for msg in issues:
                print(f"    {msg}")
        else:
            print("    ✓ No stability issues detected")


def _print_slot_histogram(results: dict[str, dict]) -> None:
    from .parking_env import SLOT_NAMES

    print("\n" + "═" * 60)
    print("SLOT USAGE HISTOGRAM")
    print("═" * 60)

    all_biased: set[str] = set()

    for ptype, r in results.items():
        dist  = r["slot_distribution"]
        total = sum(dist.values()) or 1

        print(f"\n  {ptype.upper()}")
        print(f"  {'Slot':<6} {'Count':>6}  {'%':>6}  {'Bar':<30}")
        print(f"  {'─' * 55}")

        for name in SLOT_NAMES:
            cnt = dist.get(name, 0)
            pct = cnt / total * 100
            bar = "█" * int(pct / 2.5)
            flag = "  ← bias?" if pct > 20 else ""
            print(f"  {name:<6} {cnt:>6}  {pct:>5.1f}%  {bar:<30}{flag}")
            if pct > 20:
                all_biased.add(name)

    if all_biased:
        print(f"\n  ⚠  Heavily-used slots (>20% in ≥1 policy): {sorted(all_biased)}")
    else:
        print("\n  ✓ No slot shows >20% usage bias across any policy")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/evaluate MaskablePPO — departure-augmented parking env"
    )
    p.add_argument("--eval-only",  action="store_true",
                   help="Skip training; evaluate saved model only")
    p.add_argument("--timesteps",  type=int, default=DEFAULT_TIMESTEPS, metavar="N",
                   help=f"Training timesteps (default: {DEFAULT_TIMESTEPS:,})")
    p.add_argument("--episodes",   type=int, default=N_EVAL_EPISODES,   metavar="N",
                   help=f"Eval episodes per policy (default: {N_EVAL_EPISODES})")
    p.add_argument("--n-envs",     type=int, default=DEFAULT_N_ENVS,    metavar="N",
                   help=f"Parallel envs for training (default: {DEFAULT_N_ENVS})")
    p.add_argument("--model",      default=DEFAULT_MODEL_PATH,          metavar="PATH",
                   help=f"Model path (default: {DEFAULT_MODEL_PATH})")
    p.add_argument("--tb-log",     default=DEFAULT_TB_LOG,              metavar="DIR",
                   help=f"TensorBoard log dir (default: {DEFAULT_TB_LOG})")
    p.add_argument("--device",     default=None,                         metavar="DEV",
                   help="torch device: cuda / mps / cpu (auto if omitted)")
    p.add_argument("--bench",      action="store_true",
                   help="Benchmark DummyVecEnv vs SubprocVecEnv before training")
    # ── WAIT-policy sweep knobs ────────────────────────────────────────────────
    p.add_argument("--wait-penalty",      type=float, default=None, metavar="X",
                   help="WAIT_PENALTY_BASE override (e.g. -0.2 / -1.0 / -2.0)")
    p.add_argument("--wait-increment",    type=float, default=None, metavar="X",
                   help="Penalty added per extra consecutive WAIT (default: 0.1)")
    p.add_argument("--wait-time",         type=float, default=None, metavar="S",
                   help="Seconds advanced per WAIT action (default: 1.0)")
    p.add_argument("--max-consec-waits",  type=int,   default=None, metavar="N",
                   help="Cap on consecutive WAITs before masking (default: 5)")
    p.add_argument("--assignment-bonus",  type=float, default=0.0,  metavar="X",
                   help="Bonus reward per successful assignment (default: 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Collect env kwargs (only set what user provided so defaults remain)
    env_kwargs: dict[str, Any] = {}
    if args.wait_penalty     is not None: env_kwargs["wait_penalty_base"]      = args.wait_penalty
    if args.wait_increment   is not None: env_kwargs["wait_penalty_increment"] = args.wait_increment
    if args.wait_time        is not None: env_kwargs["wait_time"]              = args.wait_time
    if args.max_consec_waits is not None: env_kwargs["max_consecutive_waits"]  = args.max_consec_waits
    if args.assignment_bonus != 0.0:      env_kwargs["assignment_bonus"]       = args.assignment_bonus

    if env_kwargs:
        print(f"[env] overrides: {env_kwargs}")

    if not args.eval_only:
        train(
            total_timesteps=args.timesteps,
            output_path=args.model,
            tensorboard_log=args.tb_log,
            n_envs=args.n_envs,
            run_bench=args.bench,
            device=args.device,
            env_kwargs=env_kwargs or None,
        )

    evaluate_all(
        model_path=args.model,
        n_episodes=args.episodes,
        env_kwargs=env_kwargs or None,
    )
