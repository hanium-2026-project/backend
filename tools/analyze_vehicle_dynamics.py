"""실차 Run 기록에서 차량 동역학 관측값을 뽑아낸다 (offline 전용).

이 도구는 **원시 기록만 읽고** 파생 물리량을 계산한다. recorder 는 원시값만
남기고(steering/throttle 명령, pose, encoder, timestamp), 반경/속도/정지거리
같은 유도값은 전부 여기서 만든다. 실시간 경로에는 아무 영향이 없다.

왜 전용 도구인가
----------------
같은 로그라도 **표본 추출 방법에 따라 결론이 정반대로 나온다.** 실제로
2026-09-03 audit 에서 연속 pose 3점으로 원을 맞추는 방식으로 계산했더니
"계획 반경 대비 실측 오차 중앙값 64%, 단조성 없음" 이라는 결과가 나왔는데,
그건 **조향이 계속 바뀌는 구간을 한 원호로 묶은 분석 artifact** 였다.

조향이 일정한 구간만 골라 호 길이/heading 변화로 다시 재면:

    FORWARD  |steer| 1.0 -> 623~681mm
    FORWARD  |steer| 0.8 -> 823mm
    REVERSE  |steer| 1.0 -> 700~726mm

로 물리적으로 타당하고 단조성도 맞다 (실측 최소 선회반경 610mm 와도 정합).

그래서 추출 규칙을 코드로 고정한다 — 매번 즉석 스크립트로 재분석하면 같은
함정에 다시 빠진다.

사용법
------
    python tools/analyze_vehicle_dynamics.py runs/run_XXXX [runs/run_YYYY ...]
    python tools/analyze_vehicle_dynamics.py --json out.json runs/run_*
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

# 곡률/속도 추정에 쓸 수 있는 heading 출처. LAST_VALID 은 이전 값을 복사한
# 것이라 heading 변화가 0 으로 나와 반경이 무한대로 튄다 — 반드시 제외한다.
TRUSTED_HEADING = frozenset({"FRONT_CUSHION", "TRAJECTORY"})

# ─── 표본 채택 문턱 ──────────────────────────────────────────────────────────
# 카메라가 약 4fps 라 한두 프레임 차분으로는 잡음이 신호를 덮는다.
STEER_WINDOW_EPS = 0.05        # 이 이상 조향이 변하면 다른 구간으로 끊는다
MIN_WINDOW_SAMPLES = 4         # 창 하나에 필요한 서로 다른 관측 수
MIN_ARC_MM = 60.0              # 창의 최소 이동거리
MIN_TURN_DEG = 8.0             # 창의 최소 heading 변화 (반경 추정용)
STRAIGHT_STEER_MAX = 0.2       # 속도 추정은 거의 직진인 구간만
MIN_SPEED_ARC_MM = 40.0
MIN_SPEED_DT_S = 0.3
STOP_SETTLE_S = 0.8            # 이만큼 안 움직이면 정지로 본다
STOP_MOVE_EPS_MM = 2.0


@dataclass
class ArcSample:
    """조향이 일정한 구간 하나에서 얻은 선회 반경 관측."""
    run: str
    phase: str | None
    direction: str
    steering: float
    samples: int
    arc_mm: float
    heading_change_deg: float
    radius_mm: float
    planned_radius_mm: float | None


@dataclass
class SpeedSample:
    run: str
    phase: str | None
    direction: str
    throttle: float
    samples: int
    arc_mm: float
    dt_s: float
    speed_mm_s: float


@dataclass
class StopSample:
    run: str
    throttle_before: float
    coast_mm: float
    coast_s: float


@dataclass
class DeadbandSample:
    """구동 명령을 유지했는데 거의 움직이지 않은 구간.

    이건 버릴 데이터가 아니라 **물리 데이터**다: "이 throttle 에서는 차가
    신뢰할 만한 움직임을 시작하지 못한다". 실측 run_20260903_125917 에서
    throttle 0.10 을 2.4초 유지했는데 약 10mm 만 움직였고(PWM 16 =
    PWM_FORWARD_MIN), 속도 표본 기준으로는 n=0 이라 그냥 사라졌다.
    """
    run: str
    direction: str
    throttle: float
    duration_s: float
    displacement_mm: float
    motion_detected: bool


def _rows(run_dir: str) -> list[dict[str, Any]]:
    path = os.path.join(run_dir, "control.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _usable_pose(row: dict[str, Any]) -> bool:
    return (row.get("pose_x_mm") is not None
            and row.get("pose_heading_deg") is not None
            and row.get("pose_heading_source") in TRUSTED_HEADING)


def _dedup(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 관측이 여러 tick 에 반복 기록되므로 좌표가 바뀔 때만 남긴다."""
    out: list[dict[str, Any]] = []
    prev = None
    for r in rows:
        key = (r.get("pose_x_mm"), r.get("pose_y_mm"))
        if key == prev:
            continue
        prev = key
        out.append(r)
    return out


def _arc_and_turn(win: Sequence[dict[str, Any]]) -> tuple[float, float]:
    arc = sum(math.hypot(b["pose_x_mm"] - a["pose_x_mm"],
                         b["pose_y_mm"] - a["pose_y_mm"])
              for a, b in zip(win, win[1:]))
    turn = 0.0
    for a, b in zip(win, win[1:]):
        turn += (b["pose_heading_deg"] - a["pose_heading_deg"] + 180.0) % 360.0 - 180.0
    return arc, turn


def _direction(row: dict[str, Any]) -> str | None:
    """주행 방향. waypoint 가 없으면 throttle 부호로 정한다.

    계측 run 에는 route/waypoint 가 없어 motion_direction 이 None 이다
    (실측 run_20260903_125917: motion_direction=None, phase=None). 이걸
    필수로 요구하면 **모든 계측 구간이 통째로 버려진다** — 계측 도구를 만든
    목적이 사라진다. 계측에서는 throttle 부호가 곧 방향이라 모호하지 않다.
    """
    explicit = row.get("motion_direction")
    if explicit in ("FORWARD", "REVERSE"):
        return explicit
    throttle = row.get("throttle_cmd")
    if throttle:
        return "FORWARD" if throttle > 0 else "REVERSE"
    return None


def _split_constant(rows: Sequence[dict[str, Any]], key: str, eps: float):
    """key 명령과 주행 방향이 일정한 구간으로 자른다."""
    window: list[dict[str, Any]] = []
    for r in rows:
        direction = _direction(r)
        if not _usable_pose(r) or r.get(key) is None or direction is None:
            if window:
                yield window
            window = []
            continue
        if window and (abs(r[key] - window[0][key]) > eps
                       or direction != _direction(window[0])):
            yield window
            window = []
        window.append(r)
    if window:
        yield window


def arc_samples(run_dir: str, rows: Sequence[dict[str, Any]]) -> list[ArcSample]:
    """조향 명령 -> 실제 선회 반경.

    반경은 3점 원 맞춤이 아니라 **호 길이 / heading 변화** 로 낸다. 4fps 잡음에
    훨씬 강하고, 창 전체를 쓰므로 한두 점의 튐이 결과를 지배하지 않는다.
    """
    out: list[ArcSample] = []
    for win in _split_constant(_dedup(rows), "wire_steering", STEER_WINDOW_EPS):
        if len(win) < MIN_WINDOW_SAMPLES:
            continue
        arc, turn = _arc_and_turn(win)
        if arc < MIN_ARC_MM or abs(turn) < MIN_TURN_DEG:
            continue
        curvature = win[0].get("curvature")
        out.append(ArcSample(
            run=os.path.basename(run_dir), phase=win[0].get("phase"),
            direction=_direction(win[0]) or "FORWARD",
            steering=float(win[0]["wire_steering"]), samples=len(win),
            arc_mm=arc, heading_change_deg=turn,
            radius_mm=arc / abs(math.radians(turn)),
            planned_radius_mm=(1.0 / abs(curvature)) if curvature else None))
    return out


def speed_samples(run_dir: str, rows: Sequence[dict[str, Any]]) -> list[SpeedSample]:
    """throttle 명령 -> 실제 병진 속도 (거의 직진인 구간만)."""
    # steering 0.0 은 **직진 명령**이지 결측이 아니다. `or` 로 기본값을 주면
    # 0.0 이 falsy 라 정확히 우리가 원하는 직진 구간이 전부 걸러진다.
    straight = [r for r in _dedup(rows)
                if r.get("throttle_cmd")
                and r.get("wire_steering") is not None
                and abs(r["wire_steering"]) < STRAIGHT_STEER_MAX]
    out: list[SpeedSample] = []
    for win in _split_constant(straight, "throttle_cmd", 0.02):
        if len(win) < MIN_WINDOW_SAMPLES:
            continue
        arc, _turn = _arc_and_turn(win)
        dt = win[-1]["t_s"] - win[0]["t_s"]
        if dt < MIN_SPEED_DT_S or arc < MIN_SPEED_ARC_MM:
            continue
        out.append(SpeedSample(
            run=os.path.basename(run_dir), phase=win[0].get("phase"),
            direction=_direction(win[0]) or "FORWARD",
            throttle=abs(float(win[0]["throttle_cmd"])), samples=len(win),
            arc_mm=arc, dt_s=dt, speed_mm_s=arc / dt))
    return out


def stop_samples(run_dir: str, rows: Sequence[dict[str, Any]]) -> list[StopSample]:
    """throttle 이 0 이 된 뒤 물리적으로 멈출 때까지의 타행 거리."""
    out: list[StopSample] = []
    previous = None
    run_state: dict[str, Any] | None = None
    for r in rows:
        if r.get("pose_x_mm") is None:
            previous = r
            continue
        throttle = r.get("throttle_cmd")
        if previous is not None and (previous.get("throttle_cmd") or 0.0) != 0.0 \
                and throttle == 0.0:
            run_state = {"t0": r["t_s"], "x0": r["pose_x_mm"], "y0": r["pose_y_mm"],
                         "before": abs(previous.get("throttle_cmd") or 0.0),
                         "last": (r["pose_x_mm"], r["pose_y_mm"]), "t_last": r["t_s"]}
        elif run_state is not None:
            if throttle not in (0.0, None):
                run_state = None            # 다시 구동 — 타행 관측 아님
            else:
                moved = math.hypot(r["pose_x_mm"] - run_state["last"][0],
                                   r["pose_y_mm"] - run_state["last"][1])
                if moved > STOP_MOVE_EPS_MM:
                    run_state["last"] = (r["pose_x_mm"], r["pose_y_mm"])
                    run_state["t_last"] = r["t_s"]
                elif r["t_s"] - run_state["t_last"] > STOP_SETTLE_S:
                    out.append(StopSample(
                        run=os.path.basename(run_dir),
                        throttle_before=run_state["before"],
                        coast_mm=math.hypot(
                            run_state["last"][0] - run_state["x0"],
                            run_state["last"][1] - run_state["y0"]),
                        coast_s=run_state["t_last"] - run_state["t0"]))
                    run_state = None
        previous = r
    return out


# 이만큼 명령을 유지했는데
DEADBAND_MIN_S = 1.0
# 이만큼도 못 움직이면 "움직이지 않았다" 로 본다 (정지 pose 잡음 ~5mm 의 3배)
DEADBAND_MOVE_MM = 15.0


def deadband_samples(run_dir: str,
                     rows: Sequence[dict[str, Any]]) -> list[DeadbandSample]:
    """throttle 별 "실제로 움직였는가" 판정.

    speed_samples 는 속도를 재기 위해 최소 이동거리를 요구하므로, 안 움직인
    조건은 통째로 사라진다. 여기서는 그 조건을 **명시적으로** 남긴다.
    """
    out: list[DeadbandSample] = []
    for win in _split_constant(_dedup(rows), "throttle_cmd", 0.02):
        if len(win) < 2 or not win[0].get("throttle_cmd"):
            continue
        dt = win[-1]["t_s"] - win[0]["t_s"]
        if dt < DEADBAND_MIN_S:
            continue
        # **직선 변위**를 쓴다. 누적 경로길이(_arc_and_turn)는 4fps pose 잡음이
        # 지그재그로 쌓여 부풀기 때문에, 제자리에서 떨리는 차가 "움직였다" 로
        # 보인다 — 실측 run_20260903_125917 은 순변위 약 10mm 인데 경로길이는
        # 15.8mm 였다. deadband 판정에서 묻는 것은 "실제로 어딘가 갔는가" 다.
        net = math.hypot(win[-1]["pose_x_mm"] - win[0]["pose_x_mm"],
                         win[-1]["pose_y_mm"] - win[0]["pose_y_mm"])
        out.append(DeadbandSample(
            run=os.path.basename(run_dir),
            direction=_direction(win[0]) or "FORWARD",
            throttle=abs(float(win[0]["throttle_cmd"])), duration_s=dt,
            displacement_mm=net, motion_detected=net >= DEADBAND_MOVE_MM))
    return out


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    v = sorted(values)
    return (v[max(0, int(len(v) * 0.1))], v[len(v) // 2],
            v[min(len(v) - 1, int(len(v) * 0.9))])


def confidence(n: int) -> str:
    """표본 수만으로 매기는 보수적 신뢰도. 적으면 적다고 말한다."""
    if n >= 20:
        return "MEDIUM"      # 통제 실험이 아니므로 HIGH 는 주지 않는다
    if n >= 8:
        return "LOW"
    return "UNUSABLE"


def analyze(run_dirs: Sequence[str]) -> dict[str, Any]:
    arcs: list[ArcSample] = []
    speeds: list[SpeedSample] = []
    stops: list[StopSample] = []
    deadbands: list[DeadbandSample] = []
    for d in run_dirs:
        rows = _rows(d)
        if not rows:
            continue
        arcs.extend(arc_samples(d, rows))
        speeds.extend(speed_samples(d, rows))
        stops.extend(stop_samples(d, rows))
        deadbands.extend(deadband_samples(d, rows))
    return {"arcs": [asdict(a) for a in arcs],
            "speeds": [asdict(s) for s in speeds],
            "stops": [asdict(s) for s in stops],
            "deadband": [asdict(s) for s in deadbands]}


def _report(result: dict[str, Any]) -> None:
    bands = result.get("deadband", [])
    print(f"=== throttle -> motion / deadband  (n={len(bands)} windows)")
    if bands:
        buckets = defaultdict(list)
        for b in bands:
            buckets[(b["direction"], round(b["throttle"], 2))].append(b)
        print(f"{'dir':<8}{'|thr|':>7}{'n':>5}{'moved':>7}"
              f"{'median mm':>11}{'  verdict':>12}")
        for key in sorted(buckets, key=lambda k: (k[0], k[1])):
            v = buckets[key]
            moved = sum(1 for x in v if x["motion_detected"])
            med = sorted(x["displacement_mm"] for x in v)[len(v) // 2]
            verdict = "MOVES" if moved else "DEADBAND"
            print(f"{key[0]:<8}{key[1]:>7.2f}{len(v):>5}{moved:>4}/{len(v):<2}"
                  f"{med:>11.1f}{verdict:>12}")
    print()
    arcs = result["arcs"]
    print(f"=== steering -> turn radius  (n={len(arcs)} constant-steering windows)")
    if arcs:
        buckets = defaultdict(list)
        for a in arcs:
            buckets[(a["direction"], round(a["steering"], 1))].append(a["radius_mm"])
        print(f"{'dir':<8}{'steer':>7}{'n':>5}{'p10':>8}{'median':>8}{'p90':>8}  conf")
        for key in sorted(buckets, key=lambda k: (k[0], k[1])):
            v = buckets[key]
            p10, med, p90 = _quantiles(v)
            print(f"{key[0]:<8}{key[1]:>7.1f}{len(v):>5}{p10:>8.0f}{med:>8.0f}"
                  f"{p90:>8.0f}  {confidence(len(v))}")
    speeds = result["speeds"]
    print(f"\n=== throttle -> speed  (n={len(speeds)} straight windows)")
    if speeds:
        buckets = defaultdict(list)
        for s in speeds:
            buckets[(s["direction"], round(s["throttle"], 2))].append(s["speed_mm_s"])
        print(f"{'dir':<8}{'|thr|':>7}{'n':>5}{'p10':>8}{'median':>8}{'p90':>8}  conf")
        for key in sorted(buckets, key=lambda k: (k[0], k[1])):
            v = buckets[key]
            p10, med, p90 = _quantiles(v)
            print(f"{key[0]:<8}{key[1]:>7.2f}{len(v):>5}{p10:>8.0f}{med:>8.0f}"
                  f"{p90:>8.0f}  {confidence(len(v))}")
    stops = result["stops"]
    print(f"\n=== zero command -> coasting  (n={len(stops)})")
    if stops:
        buckets = defaultdict(list)
        for s in stops:
            buckets[round(s["throttle_before"], 2)].append((s["coast_mm"], s["coast_s"]))
        print(f"{'|thr| before':>13}{'n':>5}{'coast p10':>11}{'median':>8}{'p90':>8}"
              f"{'  t_med':>8}  conf")
        for key in sorted(buckets):
            v = buckets[key]
            p10, med, p90 = _quantiles([x[0] for x in v])
            _, tmed, _ = _quantiles([x[1] for x in v])
            print(f"{key:>13.2f}{len(v):>5}{p10:>11.0f}{med:>8.0f}{p90:>8.0f}"
                  f"{tmed:>8.2f}  {confidence(len(v))}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run 디렉터리들")
    ap.add_argument("--json", help="원시 표본을 이 경로에 JSON 으로 저장")
    args = ap.parse_args(argv)
    result = analyze(args.runs)
    _report(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nraw samples -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
