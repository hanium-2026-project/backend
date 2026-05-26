"""Visualization utilities for ParkingRoutingEnv evaluation.

Outputs
-------
outputs/random_timeline.png      — reservation Gantt chart (random policy)
outputs/heuristic_timeline.png   — reservation Gantt chart (heuristic policy)
outputs/ppo_timeline.png         — reservation Gantt chart (PPO policy)
outputs/slot_usage.png           — side-by-side slot usage comparison
outputs/traffic_animation.mp4    — N-panel animated parking traffic
   (falls back to .gif if ffmpeg is unavailable)

Animation design
----------------
Each panel shows vehicles moving **simultaneously** through waypoints.
Position is derived from route_intervals (same physics as advance_time).
Each vehicle has a colored dot + route highlight line + slot label.
Parked vehicles stay visible as faint dots at their slot.

Usage (CLI)
-----------
    python -m rl.visualize                                 # use saved model
    python -m rl.visualize --model models/...zip --seed 7

Usage (Python)
--------------
    from rl.visualize import run_visualization
    run_visualization(model_path="models/sb3_parking_policy.zip", seed=42)
"""
from __future__ import annotations

import argparse
import math
import os
import random as _random
from typing import Any

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Slot display colors (consistent across all plots)
SLOT_COLORS: dict[str, str] = {
    "A1": "#1f77b4", "A2": "#6baed6", "A3": "#2171b5", "A4": "#08519c",
    "B1": "#31a354", "B2": "#74c476", "B3": "#238b45", "B4": "#006d2c",
}
CONFLICT_COLOR = "#e74c3c"
CONFLICT_ALPHA = 0.50


def _node_coords() -> dict[str, tuple[float, float]]:
    """Return NODE_COORDINATES from the env module (single source of truth)."""
    from .parking_env import NODE_COORDINATES
    return NODE_COORDINATES


# ═══════════════════════════════════════════════════════════════════════════════
# Physics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _vehicle_pos_at(
    vehicle: dict[str, Any],
    t: float,
    node_coords: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """Return (x_mm, y_mm) for *vehicle* at simulation time *t*.

    Uses route_intervals (same logic as advance_time) so position is always
    consistent with the traffic simulator layer.

    Returns
    -------
    (x, y)   — interpolated position while moving
    (x, y)   — final node position after parking
    None     — vehicle hasn't arrived yet
    """
    intervals = vehicle["route_intervals"]
    route     = vehicle["route"]

    if t < intervals[0][0]:
        return None          # not yet arrived

    for i, (t_enter, t_exit) in enumerate(intervals):
        if t_enter <= t < t_exit:
            frac = (t - t_enter) / max(t_exit - t_enter, 1e-9)
            p1   = node_coords.get(route[i],     (0.0, 0.0))
            p2   = node_coords.get(route[i + 1] if i + 1 < len(route) else route[i], p1)
            return (p1[0] + frac * (p2[0] - p1[0]),
                    p1[1] + frac * (p2[1] - p1[1]))

    # t ≥ last exit → parked at final node
    return node_coords.get(route[-1])


def _t_max(trajectory: dict) -> float:
    """Latest simulation time in a recorded trajectory."""
    times = [
        seg["t_exit"]
        for v in trajectory["vehicles"] + trajectory["conflict_attempts"]
        for seg in v["route_timed"]
    ]
    return max(times) + 1.5 if times else 12.0


# ═══════════════════════════════════════════════════════════════════════════════
# Episode recording
# ═══════════════════════════════════════════════════════════════════════════════

def record_episode(
    policy_type: str,
    model: Any | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one full episode and collect every data point for visualization.

    Each vehicle entry now contains ``route_intervals`` (the exact time windows
    used by the traffic simulator) so the animation can reconstruct the true
    simultaneous position of every vehicle at any simulation time t.

    Returns
    -------
    dict with keys:
        policy            : str
        events            : list of per-step dicts
        vehicles          : list of successfully assigned vehicle dicts
        conflict_attempts : list of failed attempt dicts
        reservations      : { node → [(t_start, t_end, slot, is_conflict)] }
    """
    from .parking_env import (
        ParkingRoutingEnv, SLOT_ROUTES, NODE_TRAVEL_TIME, SLOT_NAMES,
    )
    from .inference import heuristic_policy

    env = ParkingRoutingEnv()
    obs, _ = env.reset(seed=seed)

    events: list[dict]            = []
    vehicles: list[dict]          = []   # successful assignments
    conflict_attempts: list[dict] = []   # failed attempts
    reservations: dict[str, list] = {}   # node → [(ts, te, slot, is_conflict)]

    while True:
        masks = env.action_masks()
        if not masks.any():
            break

        # ── select action ────────────────────────────────────────────────────
        if policy_type == "random":
            valid  = [i for i, ok in enumerate(masks) if ok]
            action = _random.choice(valid)
        elif policy_type == "heuristic":
            action = heuristic_policy(masks)
        else:  # ppo
            if model is None:
                raise ValueError("model must be provided for ppo policy")
            obs_b = obs.reshape(1, -1)
            a, _  = model.predict(obs_b, action_masks=masks, deterministic=True)
            action = int(np.asarray(a).flat[0])

        t_arrive = env._current_time
        obs, reward, terminated, truncated, info = env.step(action)

        slot        = info["slot"]
        is_conflict = info.get("conflict", False)
        is_taken    = info.get("reason") == "slot_already_taken"

        events.append({
            "step":     info["step"],
            "slot":     slot,
            "arrive":   t_arrive,
            "iat":      env._current_time - t_arrive,
            "reward":   reward,
            "conflict": is_conflict,
            "taken":    is_taken,
        })

        # ── build timed route (for timeline chart) ───────────────────────────
        route        = SLOT_ROUTES[slot]
        t            = t_arrive
        route_timed: list[dict] = []
        route_intervals: list[tuple[float, float]] = []

        for node in route:
            t_exit = t + NODE_TRAVEL_TIME
            route_timed.append({"node": node, "t_enter": t, "t_exit": t_exit})
            route_intervals.append((t, t_exit))
            reservations.setdefault(node, []).append(
                (t, t_exit, slot, is_conflict)
            )
            t += NODE_TRAVEL_TIME

        record = {
            "id":             env._vehicle_id_counter,
            "slot":           slot,
            "route":          route,
            "route_intervals": route_intervals,   # ← physics time windows
            "route_timed":    route_timed,         # ← for timeline chart
            "enter_time":     t_arrive,
            "reward":         reward,
            "arrive":         t_arrive,            # alias for animation code
            "color":          SLOT_COLORS.get(slot, "#888"),
        }
        if is_conflict:
            conflict_attempts.append(record)
        elif not is_taken:
            vehicles.append(record)

        if terminated or truncated:
            break

    return {
        "policy":            policy_type,
        "events":            events,
        "vehicles":          vehicles,
        "conflict_attempts": conflict_attempts,
        "reservations":      reservations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Reservation Timeline
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reservation_timeline(
    trajectory: dict,
    out_path: str,
) -> None:
    """Gantt-style horizontal bar chart of node reservations.

    - Successful reservations : slot color, solid
    - Conflict attempts       : red, semi-transparent, hatched border
    - Bottleneck nodes        : shaded row background
    """
    from .parking_env import BOTTLENECK_NODES, _ALL_NODES

    reservations = trajectory["reservations"]
    policy       = trajectory["policy"]
    n_vehicles   = len(trajectory["vehicles"])
    n_conflicts  = len(trajectory["conflict_attempts"])

    # Order: bottlenecks first, then the rest alphabetically
    ordered = BOTTLENECK_NODES + [
        n for n in sorted(_ALL_NODES) if n not in BOTTLENECK_NODES
    ]
    ordered = [n for n in ordered if n in reservations]

    fig, ax = plt.subplots(figsize=(15, max(5, len(ordered) * 0.6 + 2)))
    legend_seen: dict[str, mpatches.Patch] = {}

    for yi, node in enumerate(ordered):
        if node in BOTTLENECK_NODES:
            ax.axhspan(yi - 0.45, yi + 0.45, color="#fef9e7", zorder=0)

        for ts, te, slot, is_conflict in reservations[node]:
            color = CONFLICT_COLOR if is_conflict else SLOT_COLORS.get(slot, "#888")
            alpha = CONFLICT_ALPHA if is_conflict else 0.85
            hatch = "///" if is_conflict else None
            ax.barh(
                yi, te - ts, left=ts, height=0.65,
                color=color, alpha=alpha,
                edgecolor="darkred" if is_conflict else "white",
                linewidth=1.5 if is_conflict else 0.4,
                hatch=hatch,
            )
            label = f"{slot} ✗ conflict" if is_conflict else slot
            if label not in legend_seen:
                legend_seen[label] = mpatches.Patch(
                    facecolor=color, alpha=alpha,
                    edgecolor="darkred" if is_conflict else "none",
                    hatch=hatch, label=label,
                )

    for ev in trajectory["events"]:
        if not ev["taken"]:
            lc = CONFLICT_COLOR if ev["conflict"] else "#555"
            ls = "--" if ev["conflict"] else ":"
            ax.axvline(ev["arrive"], color=lc, linewidth=0.8, linestyle=ls, alpha=0.4)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=9)
    ax.set_xlabel("Simulation Time (s)", fontsize=11)
    ax.set_title(
        f"Reservation Timeline  —  {policy.upper()} policy\n"
        f"{n_vehicles} successful assignments   |   "
        f"{n_conflicts} conflict attempts",
        fontsize=12, pad=10,
    )
    ax.legend(
        handles=list(legend_seen.values()),
        loc="upper right", fontsize=8, ncol=3, framealpha=0.9,
    )
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.invert_yaxis()

    for yi, node in enumerate(ordered):
        if node in BOTTLENECK_NODES:
            ax.text(
                -0.01, yi, "⚡", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, color="#e67e22",
            )

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Slot Usage Heatmap / Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_slot_usage(
    results_dict: dict[str, dict],
    out_path: str,
) -> None:
    """Side-by-side horizontal bar charts (one per policy) + 2×4 heatmap."""
    from .parking_env import SLOT_NAMES

    policies = list(results_dict.keys())
    n_pol    = len(policies)
    uniform  = 100.0 / len(SLOT_NAMES)

    fig = plt.figure(figsize=(5 * n_pol + 1, 9))
    gs  = fig.add_gridspec(2, n_pol, height_ratios=[3, 1.4], hspace=0.45, wspace=0.35)

    axes_bar = [fig.add_subplot(gs[0, j]) for j in range(n_pol)]
    max_pct  = 0.0

    for j, (ax, ptype) in enumerate(zip(axes_bar, policies)):
        dist   = results_dict[ptype].get("slot_distribution", {})
        total  = sum(dist.values()) or 1
        pcts   = [dist.get(n, 0) / total * 100 for n in SLOT_NAMES]
        colors = [SLOT_COLORS.get(n, "#888") for n in SLOT_NAMES]
        max_pct = max(max_pct, max(pcts))

        bars = ax.barh(SLOT_NAMES, pcts, color=colors,
                       edgecolor="white", linewidth=0.8)
        for bar, pct in zip(bars, pcts):
            ax.text(
                bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left", fontsize=8,
            )
        ax.axvline(uniform, color="#e74c3c", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"Uniform {uniform:.1f}%")
        ax.set_title(f"{ptype.upper()}", fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("Usage (%)", fontsize=9)
        ax.tick_params(labelsize=9)
        for i, n in enumerate(SLOT_NAMES):
            ax.axhspan(i - 0.4, i + 0.4,
                       color="#eaf4fb" if n.startswith("A") else "#eafbea",
                       zorder=0)
        if j == 0:
            ax.legend(fontsize=7, loc="lower right")

    for ax in axes_bar:
        ax.set_xlim(0, max_pct * 1.25 + 2)

    ax_hm = fig.add_subplot(gs[1, :])
    mat   = np.zeros((len(SLOT_NAMES), n_pol))
    for j, ptype in enumerate(policies):
        dist  = results_dict[ptype].get("slot_distribution", {})
        total = sum(dist.values()) or 1
        for i, n in enumerate(SLOT_NAMES):
            mat[i, j] = dist.get(n, 0) / total * 100

    im = ax_hm.imshow(mat.T, aspect="auto", cmap="YlOrRd",
                      vmin=0, vmax=max(mat.max(), uniform * 2))
    ax_hm.set_xticks(range(len(SLOT_NAMES)))
    ax_hm.set_xticklabels(SLOT_NAMES, fontsize=9)
    ax_hm.set_yticks(range(n_pol))
    ax_hm.set_yticklabels([p.upper() for p in policies], fontsize=9)
    ax_hm.set_title("Heatmap: Usage % per slot (row=policy)", fontsize=10)
    for i in range(len(SLOT_NAMES)):
        for j in range(n_pol):
            ax_hm.text(i, j, f"{mat[i, j]:.0f}", ha="center", va="center",
                       fontsize=8, color="black" if mat[i, j] < 20 else "white")

    cbar = fig.colorbar(im, ax=ax_hm, orientation="vertical",
                        fraction=0.02, pad=0.01)
    cbar.set_label("Usage (%)", fontsize=8)

    fig.suptitle("Slot Usage Distribution per Policy  (100 episodes each)",
                 fontsize=13, fontweight="bold", y=0.98)
    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Parking Traffic Animation
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_lot(
    ax: plt.Axes,
    node_coords: dict[str, tuple[float, float]],
    title: str = "",
) -> None:
    """Draw the static parking lot background.

    Includes: lot outline, aisles, slot boxes, route edges, waypoint nodes,
    entrance arrow.  Uses *node_coords* (imported NODE_COORDINATES) so the
    background exactly matches the traffic simulator geometry.
    """
    from .parking_env import SLOT_COORDINATES, SLOT_NAMES, SLOT_ROUTES, BOTTLENECK_NODES

    ax.set_xlim(-80, 1320)
    ax.set_ylim(-120, 1300)
    ax.set_aspect("equal")
    ax.set_facecolor("#c8d6e5")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)

    # ── lot boundary ─────────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1200, 1200, boxstyle="square,pad=0",
        facecolor="#f0f0f0", edgecolor="#2c3e50", linewidth=2, zorder=1,
    ))

    # ── aisles ───────────────────────────────────────────────────────────────
    for rect in [
        (0,    0, 270, 1200),    # main vertical aisle
        (270, 900, 930,  180),   # A cross-aisle
        (270,   0, 930,  200),   # B cross-aisle
    ]:
        ax.add_patch(mpatches.Rectangle(
            (rect[0], rect[1]), rect[2], rect[3],
            facecolor="#d5d8dc", edgecolor="none", zorder=1,
        ))

    # Lane dividers
    for y in [950, 980]:
        ax.plot([270, 1200], [y, y], "-", color="#bdc3c7", linewidth=0.7, zorder=2)
    for y in [150, 180]:
        ax.plot([270, 1200], [y, y], "-", color="#bdc3c7", linewidth=0.7, zorder=2)

    # ── slot boxes ───────────────────────────────────────────────────────────
    for name in SLOT_NAMES:
        sx, sy = SLOT_COORDINATES[name]
        ax.add_patch(mpatches.FancyBboxPatch(
            (sx - 50, sy - 30), 100, 60, boxstyle="round,pad=3",
            facecolor=SLOT_COLORS[name], edgecolor="white",
            linewidth=1.5, alpha=0.45, zorder=3,
        ))
        ax.text(sx, sy, name, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=4)

    # ── route graph edges ────────────────────────────────────────────────────
    drawn: set[tuple[str, str]] = set()
    for route in SLOT_ROUTES.values():
        for n1, n2 in zip(route[:-1], route[1:]):
            key = tuple(sorted([n1, n2]))
            if key in drawn:
                continue
            drawn.add(key)
            p1 = node_coords.get(n1)
            p2 = node_coords.get(n2)
            if p1 and p2:
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    "-", color="#85929e", linewidth=1.5, alpha=0.55, zorder=2,
                )

    # ── waypoint nodes ───────────────────────────────────────────────────────
    for node, (nx, ny) in node_coords.items():
        is_front  = "_front" in node
        is_bottle = node in BOTTLENECK_NODES
        color  = "#e74c3c" if is_bottle else ("#95a5a6" if is_front else "#2c3e50")
        size   = 8 if is_bottle else (4 if is_front else 5)
        ax.plot(nx, ny, "o", color=color, markersize=size,
                zorder=5, markeredgecolor="white", markeredgewidth=0.8)
        if not is_front:
            ax.text(nx + 14, ny + 14, node, fontsize=5,
                    color="#34495e", zorder=6)

    # ── entrance arrow ───────────────────────────────────────────────────────
    ax.annotate("", xy=(150, 28), xytext=(150, -90),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2.2))
    ax.text(150, -95, "ENTER", ha="center", va="top",
            fontsize=7, color="#e74c3c", fontweight="bold")


def create_traffic_animation(
    trajectories: dict[str, dict],
    out_path: str,
    fps: int = 15,
    sim_speed: float = 3.0,
) -> None:
    """Create a multi-panel animated parking traffic visualization.

    Design
    ------
    - Each panel = one policy.
    - At every frame (simulation time t), ALL assigned vehicles are drawn
      simultaneously using their route_intervals for position interpolation —
      identical to what advance_time() computes inside the env.
    - Moving vehicles: large colored circle + slot label + route highlight line.
    - Parked vehicles: small faded circle at final slot position.
    - Conflict flashes: red ✗ at entrance, fades over 0.8 s.
    - Three-line stats overlay: t, parked/moving counts, conflict count.

    Parameters
    ----------
    trajectories : {policy → trajectory from record_episode()}
    out_path     : destination .mp4 (or .gif fallback)
    fps          : frames per second
    sim_speed    : simulation-seconds per real-second of video
    """
    from .parking_env import SLOT_COORDINATES

    node_coords = _node_coords()
    policies    = list(trajectories.keys())
    n_panels    = len(policies)
    t_end       = max(_t_max(traj) for traj in trajectories.values())
    dt_frame    = sim_speed / fps
    n_frames    = int(t_end / dt_frame) + 12

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(8.5 * n_panels, 9.0),
        gridspec_kw={"wspace": 0.04},
    )
    if n_panels == 1:
        axes = [axes]

    fig.patch.set_facecolor("#1a252f")

    # ── static lot backgrounds ───────────────────────────────────────────────
    for ax, ptype in zip(axes, policies):
        _draw_lot(ax, node_coords, title=f"{ptype.upper()} policy")

    # ── per-panel animated artists ───────────────────────────────────────────
    panel_states: list[dict] = []

    for ax, ptype in zip(axes, policies):
        traj    = trajectories[ptype]
        vehicles = traj["vehicles"]
        confs    = traj["conflict_attempts"]

        # Route highlight lines (one per vehicle, drawn once, alpha toggled)
        route_lines = []
        for v in vehicles:
            xs = [node_coords[n][0] for n in v["route"] if n in node_coords]
            ys = [node_coords[n][1] for n in v["route"] if n in node_coords]
            ln, = ax.plot(xs, ys, "-",
                          color=v["color"], linewidth=2.8,
                          alpha=0.0, zorder=3)
            route_lines.append(ln)

        # Vehicle dots (moving: large; parked: small faded)
        v_dots = [
            ax.plot([], [], "o",
                    color=v["color"],
                    markersize=14,
                    markeredgecolor="white",
                    markeredgewidth=1.2,
                    zorder=8, alpha=0.0)[0]
            for v in vehicles
        ]
        # Vehicle slot labels
        v_labels = [
            ax.text(0, 0, v["slot"], ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white",
                    zorder=9, alpha=0.0)
            for v in vehicles
        ]
        # Conflict flash markers
        conf_dots = [
            ax.plot([], [], "X",
                    markersize=13, color=CONFLICT_COLOR,
                    markeredgewidth=2.0, zorder=10, alpha=0.0)[0]
            for _ in confs
        ]
        # Conflict slot labels
        conf_labels = [
            ax.text(0, 0, c["slot"], ha="center", va="bottom",
                    fontsize=6, color=CONFLICT_COLOR, zorder=11, alpha=0.0)
            for c in confs
        ]

        # Stats overlay
        time_txt = ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            fontsize=9, va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a252f", alpha=0.80),
        )
        stat_txt = ax.text(
            0.02, 0.91, "", transform=ax.transAxes,
            fontsize=7.5, va="top", color="#cccccc",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a252f", alpha=0.70),
        )

        panel_states.append({
            "traj":         traj,
            "vehicles":     vehicles,
            "confs":        confs,
            "route_lines":  route_lines,
            "v_dots":       v_dots,
            "v_labels":     v_labels,
            "conf_dots":    conf_dots,
            "conf_labels":  conf_labels,
            "time_txt":     time_txt,
            "stat_txt":     stat_txt,
        })

    # Collect all animated artists for blit
    all_artists: list = []
    for ps in panel_states:
        all_artists += (
            ps["route_lines"] +
            ps["v_dots"] +
            ps["v_labels"] +
            ps["conf_dots"] +
            ps["conf_labels"] +
            [ps["time_txt"], ps["stat_txt"]]
        )

    # ── animation callbacks ───────────────────────────────────────────────────

    def init():
        for ps in panel_states:
            for ln in ps["route_lines"]:
                ln.set_alpha(0.0)
            for dot in ps["v_dots"] + ps["conf_dots"]:
                dot.set_data([], [])
                dot.set_alpha(0.0)
            for lbl in ps["v_labels"] + ps["conf_labels"]:
                lbl.set_alpha(0.0)
            ps["time_txt"].set_text("")
            ps["stat_txt"].set_text("")
        return all_artists

    def update(frame: int):
        t = frame * dt_frame

        for ps in panel_states:
            vehicles = ps["vehicles"]
            confs    = ps["confs"]

            # ── stats ────────────────────────────────────────────────────────
            n_moving  = 0
            n_parked  = 0
            for v in vehicles:
                t_end_v = v["route_intervals"][-1][1]
                if v["enter_time"] <= t < t_end_v:
                    n_moving += 1
                elif t >= t_end_v:
                    n_parked += 1
            n_cf = sum(1 for c in confs if abs(t - c["arrive"]) <= 0.85)

            ps["time_txt"].set_text(f"t = {t:.2f} s")
            ps["stat_txt"].set_text(
                f"moving: {n_moving}  parked: {n_parked}  conflicts: {n_cf}"
            )

            # ── vehicles ─────────────────────────────────────────────────────
            for v, dot, lbl, ln in zip(
                vehicles, ps["v_dots"], ps["v_labels"], ps["route_lines"]
            ):
                t_start = v["enter_time"]
                t_end_v = v["route_intervals"][-1][1]

                if t < t_start:
                    # Not arrived yet
                    dot.set_data([], [])
                    dot.set_alpha(0.0)
                    lbl.set_alpha(0.0)
                    ln.set_alpha(0.0)

                elif t < t_end_v:
                    # Moving: interpolate position using route_intervals
                    pos = _vehicle_pos_at(v, t, node_coords)
                    if pos:
                        dot.set_data([pos[0]], [pos[1]])
                        dot.set_color(v["color"])
                        dot.set_markersize(14)
                        dot.set_alpha(0.92)
                        lbl.set_position((pos[0], pos[1]))
                        lbl.set_alpha(0.95)
                        ln.set_alpha(0.40)

                else:
                    # Parked: faint dot at final slot position
                    sx, sy = SLOT_COORDINATES[v["slot"]]
                    dot.set_data([sx], [sy])
                    dot.set_color(v["color"])
                    dot.set_markersize(6)
                    dot.set_alpha(0.30)
                    lbl.set_position((sx, sy))
                    lbl.set_alpha(0.0)    # label off once parked
                    ln.set_alpha(0.08)

            # ── conflict flashes ─────────────────────────────────────────────
            ent_x, ent_y = node_coords.get("entrance", (150.0, 20.0))
            for c, cdot, clbl in zip(confs, ps["conf_dots"], ps["conf_labels"]):
                dt_from = t - c["arrive"]
                if -0.05 <= dt_from <= 0.85:
                    alpha = float(np.clip(1.0 - dt_from / 0.85, 0.0, 1.0))
                    cdot.set_data([ent_x], [ent_y])
                    cdot.set_alpha(alpha)
                    clbl.set_position((ent_x + 15, ent_y + 18))
                    clbl.set_alpha(alpha)
                else:
                    cdot.set_data([], [])
                    cdot.set_alpha(0.0)
                    clbl.set_alpha(0.0)

        return all_artists

    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        init_func=init,
        blit=True,
        interval=1000 // fps,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    saved_path = out_path
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(
            fps=fps,
            metadata={"title": "Parking Traffic Simulator"},
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        anim.save(out_path, writer=writer, dpi=100)
    except Exception as mp4_err:
        saved_path = out_path.replace(".mp4", ".gif")
        print(f"  [viz] ffmpeg unavailable ({type(mp4_err).__name__}), "
              f"saving GIF instead …")
        try:
            from matplotlib.animation import PillowWriter
            anim.save(saved_path, writer=PillowWriter(fps=fps), dpi=80)
        except Exception as gif_err:
            print(f"  [viz] GIF also failed: {gif_err}")
            saved_path = None

    plt.close(fig)
    if saved_path:
        print(f"  [viz] Animation saved → {saved_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualization(
    model_path: str = "models/sb3_parking_policy.zip",
    out_dir: str = "outputs",
    seed: int = 42,
    n_eval_episodes: int = 100,
) -> None:
    """Record episodes + aggregate stats, then produce all visualizations."""
    from .train_sb3 import evaluate_policy

    os.makedirs(out_dir, exist_ok=True)

    # ── load PPO model ────────────────────────────────────────────────────────
    ppo_model = None
    active    = ["random", "heuristic"]
    try:
        from sb3_contrib import MaskablePPO
        zip_p = model_path if model_path.endswith(".zip") else model_path + ".zip"
        lp    = (model_path if os.path.exists(model_path)
                 else zip_p  if os.path.exists(zip_p) else None)
        if lp:
            ppo_model = MaskablePPO.load(lp)
            active.append("ppo")
            print(f"[viz] Loaded PPO model ← {lp}")
        else:
            print(f"[viz] No model at '{model_path}' — skipping PPO")
    except ImportError:
        print("[viz] sb3-contrib not installed — skipping PPO")

    # ── aggregate evaluation for slot usage chart ─────────────────────────────
    print(f"\n[viz] Evaluating {n_eval_episodes} episodes × {len(active)} policies …")
    agg_results: dict[str, dict] = {}
    for ptype in active:
        print(f"  {ptype:<10}", end=" ", flush=True)
        agg_results[ptype] = evaluate_policy(
            ptype,
            model=ppo_model if ptype == "ppo" else None,
            n_episodes=n_eval_episodes,
        )
        print("done")

    print("\n[viz] Generating slot usage chart …")
    plot_slot_usage(agg_results, os.path.join(out_dir, "slot_usage.png"))

    # ── per-policy timeline + animation data ──────────────────────────────────
    print("\n[viz] Recording single episodes for timeline + animation …")
    trajectories: dict[str, dict] = {}
    for ptype in active:
        print(f"  {ptype:<10}", end=" ", flush=True)
        traj = record_episode(
            ptype,
            model=ppo_model if ptype == "ppo" else None,
            seed=seed,
        )
        trajectories[ptype] = traj
        print(f"  {len(traj['vehicles'])} assigned, "
              f"{len(traj['conflict_attempts'])} conflicts")

        plot_reservation_timeline(
            traj,
            os.path.join(out_dir, f"{ptype}_timeline.png"),
        )

    # ── combined traffic animation ─────────────────────────────────────────────
    print("\n[viz] Rendering traffic animation (simultaneous vehicle movement) …")
    create_traffic_animation(
        trajectories,
        os.path.join(out_dir, "traffic_animation.mp4"),
    )

    print(f"\n[viz] ✓ All outputs → {os.path.abspath(out_dir)}/")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {f:<40} {size/1024:>7.1f} KB")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parking RL visualizer")
    p.add_argument("--model",    default="models/sb3_parking_policy.zip")
    p.add_argument("--out-dir",  default="outputs")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--episodes", type=int, default=100)
    args = p.parse_args()

    run_visualization(
        model_path=args.model,
        out_dir=args.out_dir,
        seed=args.seed,
        n_eval_episodes=args.episodes,
    )
