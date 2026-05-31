"""2D Safety vs Throughput Pareto scatter plot.

Visualises the safety-throughput trade-off across all evaluated policies.

  - X axis: Avg Conflicts per Episode   (lower = safer)
  - Y axis: Avg Throughput per Episode  (higher = more efficient)
  - Each policy = distinct marker/color
  - PPO is highlighted with a large star
  - Pareto frontier connects non-dominated policies
  - "Ideal region" arrow points to top-left (low conflict, high throughput)

Reuses the existing evaluation pipeline — no new metrics computed; just
reads avg_conflicts / avg_throughput from `evaluate_all()` results.

Usage
-----
    python -m rl.plot_pareto                       # 100 episodes
    python -m rl.plot_pareto --episodes 200        # custom
    python -m rl.plot_pareto --out outputs/x.png   # custom path
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt

from .train_sb3 import evaluate_all

DEFAULT_OUTPUT = "outputs/pareto_safety_throughput.png"

# Distinct marker / color per policy.
# `offset` is the (dx, dy) in display points for the annotation text.
# Cluster policies (V1/V2/V4 sit very close in conflict/throughput space)
# are pushed to different directions so labels never collide.
POLICY_STYLE: dict[str, dict] = {
    "random":       {"color": "#7f7f7f", "marker": "o", "label": "Random",
                     "size": 180, "offset": (10, -22)},
    "heuristic":    {"color": "#1f77b4", "marker": "s", "label": "V1 Nearest",
                     "size": 180, "offset": (14, -10)},
    "heuristic_v2": {"color": "#2ca02c", "marker": "D", "label": "V2 Congestion",
                     "size": 180, "offset": (10,  22)},
    "heuristic_v3": {"color": "#9467bd", "marker": "^", "label": "V3 ProactiveWAIT",
                     "size": 180, "offset": (12,  10)},
    "heuristic_v4": {"color": "#ff7f0e", "marker": "v", "label": "V4 RouteLen",
                     "size": 180, "offset": (-105, -10)},
    "ppo":          {"color": "#d62728", "marker": "*", "label": "PPO",
                     "size": 520, "offset": (14,  14)},
}


# ─── Pareto frontier ─────────────────────────────────────────────────────────

def compute_pareto_frontier(
    points: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Non-dominated subset under "minimize X, maximize Y" preference.

    A point (x, y) is dominated when some other point (x', y') satisfies
    x' ≤ x AND y' ≥ y with at least one strict inequality.  Returns the
    surviving points sorted by ascending x.
    """
    frontier: list[tuple[float, float, str]] = []
    for i, (x, y, name) in enumerate(points):
        dominated = False
        for j, (x2, y2, _) in enumerate(points):
            if i == j:
                continue
            if x2 <= x and y2 >= y and (x2 < x or y2 > y):
                dominated = True
                break
        if not dominated:
            frontier.append((x, y, name))
    frontier.sort(key=lambda p: p[0])
    return frontier


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_pareto(results: dict, out_path: str = DEFAULT_OUTPUT) -> str:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("#f8f9fa")

    # Scatter every policy
    points: list[tuple[float, float, str]] = []
    for ptype, r in results.items():
        if "avg_conflicts" not in r or "avg_throughput" not in r:
            continue
        x = float(r["avg_conflicts"])
        y = float(r["avg_throughput"])
        n_waits  = float(r.get("avg_waits", 0.0))
        wait_pct = float(r.get("avg_wait_rate", 0.0)) * 100.0
        st = POLICY_STYLE.get(ptype, {"color": "#000", "marker": "o",
                                     "label": ptype, "size": 180})
        # Distinct edge for PPO so its star pops
        edge_w = 2.2 if ptype == "ppo" else 1.4
        ax.scatter(x, y,
                   c=st["color"], marker=st["marker"], s=st["size"],
                   edgecolor="black", linewidth=edge_w, zorder=5,
                   label=st["label"])
        # Label + WAIT count on a single line: e.g. "PPO (45.6 waits)"
        annotation = f"{st['label']} ({n_waits:.1f} waits)"
        offset = st.get("offset", (10, 10))
        ax.annotate(annotation, xy=(x, y), xytext=offset,
                    textcoords="offset points",
                    fontsize=9.5, fontweight="bold",
                    color=st["color"],
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor=st["color"],
                              linewidth=0.8, alpha=0.85))
        points.append((x, y, ptype))

    # Pareto frontier
    frontier = compute_pareto_frontier(points)
    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, "--", color="#16a085", linewidth=2.4,
                alpha=0.75, zorder=3,
                label=f"Pareto frontier ({len(frontier)} policies)")

    # Axis bounds with a little headroom
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_pad = max(0.4, (max(xs) - min(xs)) * 0.15)
    y_pad = max(0.6, (max(ys) - min(ys)) * 0.12)
    x_min, x_max = min(xs) - x_pad, max(xs) + x_pad
    y_min, y_max = min(ys) - y_pad, max(ys) + y_pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Shade ideal region (top-left quadrant relative to a midpoint)
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    ax.axvspan(x_min, mid_x, ymin=0.5, ymax=1.0,
               facecolor="#27ae60", alpha=0.06, zorder=1)
    ax.text(x_min + 0.04 * (x_max - x_min),
            y_max - 0.06 * (y_max - y_min),
            "IDEAL REGION\n(low conflict, high throughput)",
            fontsize=10, color="#1e8449", fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#eafaf1", edgecolor="#16a085",
                      alpha=0.85))

    # Direction arrows on axes
    ax.annotate("safer →", xy=(0.02, 1.02), xycoords="axes fraction",
                fontsize=9, color="#16a085", fontweight="bold")
    ax.annotate("← more efficient",
                xy=(1.02, 0.02), xycoords="axes fraction",
                rotation=90, fontsize=9, color="#16a085", fontweight="bold")

    # Axes / cosmetics
    ax.set_xlabel("Avg Conflicts per Episode   (lower is safer ←)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Throughput per Episode  (higher is better ↑)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Safety vs Throughput Trade-off",
                 fontsize=15, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.invert_xaxis()  # so "safer" is on the right? — keep left=low for clarity
    ax.invert_xaxis()  # restore (no-op pair, explicit comment for readers)

    # Legend outside plot to the right so it never overlaps a data point.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.95,
              title="Policy", title_fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Safety vs Throughput Pareto scatter plot")
    p.add_argument("--episodes", type=int, default=100,
                   help="Eval episodes per policy (default 100)")
    p.add_argument("--model",    default="models/sb3_parking_policy.zip")
    p.add_argument("--out",      default=DEFAULT_OUTPUT)
    args = p.parse_args()

    print(f"[pareto] Running evaluation (episodes={args.episodes}) …")
    results = evaluate_all(model_path=args.model, n_episodes=args.episodes)

    print("\n[pareto] Generating Pareto plot …")
    out_path = plot_pareto(results, args.out)
    print(f"[pareto] Saved → {out_path}")

    # Console summary of frontier
    points = [(r["avg_conflicts"], r["avg_throughput"], k)
              for k, r in results.items()]
    frontier = compute_pareto_frontier(points)
    print("\n[pareto] Pareto-optimal policies:")
    for x, y, name in frontier:
        print(f"  {name:<14}  conflicts={x:>5.2f}   throughput={y:>5.2f}")
    dominated = sorted(set(p[2] for p in points) - set(f[2] for f in frontier))
    if dominated:
        print(f"\n[pareto] Dominated: {', '.join(dominated)}")


if __name__ == "__main__":
    main()
