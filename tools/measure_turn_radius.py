"""Run 기록에서 실제 선회 반경·속도를 뽑는다.

수동으로 최대 조향을 걸고 돈 구간을 찾아 원을 맞춰 반경을 잰다. 화면으로
눈대중하는 것보다 정확하고, 경로 생성기의 계획 반경
(`parking.waypoints.MIN_TURN_RADIUS_MM`)을 이 숫자로 정한다.

두 가지 방법으로 각각 계산해 서로 검증한다.

1. **원 피팅** — 궤적 점들에 최소제곱으로 원을 맞춘다 (Kåsa 법).
   조향을 일정하게 유지한 구간에서 가장 정확하다.
2. **v/ω** — 속도를 각속도로 나눈다. 순간값이라 구간이 짧아도 되지만
   heading 잡음에 민감하다.

사용::

    python tools/measure_turn_radius.py runs/run_20260812_161555
    python tools/measure_turn_radius.py runs/run_*/ --min-steer 0.8

`--min-steer` 이상으로 조향이 유지된 구간만 본다. 최소 선회 반경을 재려면
조향을 끝까지 걸고 돌아야 하므로 기본값이 높다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(path: Path, name: str) -> list[dict]:
    f = path / name
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def fit_circle(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """최소제곱 원 피팅 (Kåsa). 반환 (cx, cy, r). 직선에 가까우면 None.

    (x-cx)^2 + (y-cy)^2 = r^2 을 x^2+y^2 = 2cx·x + 2cy·y + (r^2-cx^2-cy^2)
    로 펴서 선형 최소제곱으로 푼다.
    """
    n = len(points)
    if n < 3:
        return None
    sx = sy = sxx = syy = sxy = sxz = syz = sz = 0.0
    for x, y in points:
        z = x * x + y * y
        sx += x; sy += y; sz += z
        sxx += x * x; syy += y * y; sxy += x * y
        sxz += x * z; syz += y * z
    # 정규방정식 3x3
    a = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
    b = [sxz, syz, sz]
    # 가우스 소거
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(a[r][i]))
        if abs(a[p][i]) < 1e-12:
            return None                       # 특이 — 점들이 일직선
        a[i], a[p] = a[p], a[i]
        b[i], b[p] = b[p], b[i]
        for r in range(i + 1, 3):
            f = a[r][i] / a[i][i]
            for c in range(i, 3):
                a[r][c] -= f * a[i][c]
            b[r] -= f * b[i]
    sol = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        s = b[i] - sum(a[i][c] * sol[c] for c in range(i + 1, 3))
        sol[i] = s / a[i][i]
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def steer_spans(controls: list[dict], min_steer: float, min_ms: float
                ) -> list[tuple[float, float, float]]:
    """조향이 한 방향으로 min_steer 이상 유지된 구간 [(t0, t1, 평균조향)]."""
    spans, start, sign, acc, cnt = [], None, 0, 0.0, 0
    for row in controls:
        st = row.get("steering_cmd")
        t = row.get("t_s")
        if st is None or t is None:
            continue
        s = 1 if st > 0 else -1
        if abs(st) >= min_steer and (start is None or s == sign):
            if start is None:
                start, sign, acc, cnt = t, s, 0.0, 0
            acc += st; cnt += 1
            last = t
        else:
            if start is not None and (last - start) * 1000.0 >= min_ms:
                spans.append((start, last, acc / max(cnt, 1)))
            start = None
    if start is not None and (last - start) * 1000.0 >= min_ms:
        spans.append((start, last, acc / max(cnt, 1)))
    return spans


def analyse(run: Path, min_steer: float, min_ms: float) -> None:
    poses = _load(run, "pose.jsonl")
    controls = _load(run, "control.jsonl")
    if not poses:
        print(f"{run.name}: pose.jsonl 이 비었습니다 (차량이 car_id 에 매핑됐는지 확인)")
        return

    print(f"\n=== {run.name} ===")
    print(f"pose {len(poses)}행, control {len(controls)}행")

    # ─ 전체 궤적 요약 ─
    xs = [p["x_mm"] for p in poses]
    ys = [p["y_mm"] for p in poses]
    print(f"이동 범위  x {min(xs)/10:.1f}~{max(xs)/10:.1f}cm   "
          f"y {min(ys)/10:.1f}~{max(ys)/10:.1f}cm")
    srcs = {p.get("heading_source") for p in poses}
    print(f"heading 출처: {', '.join(sorted(s for s in srcs if s))}")

    spans = steer_spans(controls, min_steer, min_ms)
    if not spans:
        print(f"조향 |{min_steer}| 이상을 {min_ms:.0f}ms 넘게 유지한 구간이 없습니다.")
        print("  → 최대 조향으로 한 바퀴 돌아주세요 (A 또는 D 를 끝까지 유지)")
        return

    print(f"\n최대 조향 구간 {len(spans)}개:")
    print(f"{'구간(s)':>14s} {'조향':>6s} {'점':>4s} {'원피팅 R':>10s} "
          f"{'v/ω R':>9s} {'속도':>9s} {'회전':>7s}")
    print("-" * 66)
    best = []
    for t0, t1, avg in spans:
        pts = [(p["x_mm"], p["y_mm"]) for p in poses
               if t0 <= p.get("t_s", -1) <= t1]
        seq = [p for p in poses if t0 <= p.get("t_s", -1) <= t1]
        if len(pts) < 5:
            continue
        fit = fit_circle(pts)
        r_fit = fit[2] if fit else None

        # v/ω — 구간 양끝의 heading 변화와 이동 거리
        dist = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts, pts[1:]))
        h0 = seq[0].get("heading_deg")
        h1 = seq[-1].get("heading_deg")
        dt = t1 - t0
        r_vw = turn = None
        if h0 is not None and h1 is not None:
            turn = abs(_wrap180(h1 - h0))
            if turn > 5.0:
                r_vw = dist / math.radians(turn)
        speed = dist / dt / 10.0 if dt > 0 else 0.0     # cm/s

        print(f"{t0:6.2f}~{t1:6.2f} {avg:+6.2f} {len(pts):4d} "
              f"{(f'{r_fit/10:.1f}cm' if r_fit else '—'):>10s} "
              f"{(f'{r_vw/10:.1f}cm' if r_vw else '—'):>9s} "
              f"{speed:6.1f}cm/s {(f'{turn:.0f}°' if turn else '—'):>7s}")
        if r_fit and turn and turn > 30.0:
            best.append(r_fit)

    print()
    if best:
        lo, hi = min(best), max(best)
        print(f"▶ 30° 이상 돈 구간의 원피팅 반경: {lo/10:.1f} ~ {hi/10:.1f}cm "
              f"(중앙 {sorted(best)[len(best)//2]/10:.1f}cm)")
        print(f"  parking/waypoints.py 의 MIN_TURN_RADIUS_MM 을 "
              f"{hi:.0f} 이상으로 두세요 (여유 포함).")
    else:
        print("30° 이상 선회한 구간이 없어 최소 반경을 확정할 수 없습니다.")
        print("  → 최대 조향으로 최소 1/4 바퀴(90°) 이상 돌아주세요.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run 디렉터리 (runs/run_YYYYmmdd_HHMMSS)")
    ap.add_argument("--min-steer", type=float, default=0.8,
                    help="이 이상 조향이 걸린 구간만 본다 (기본 0.8)")
    ap.add_argument("--min-ms", type=float, default=600.0,
                    help="구간 최소 지속 시간 ms (기본 600)")
    args = ap.parse_args()
    for r in args.runs:
        analyse(Path(r), args.min_steer, args.min_ms)


if __name__ == "__main__":
    main()
