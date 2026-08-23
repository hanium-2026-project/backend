#!/usr/bin/env python3
"""calib_pose_logger CSV → 인수인계 §14 비교표.

§13 의 지적대로 mAP 가 아니라 production 지표로 본다. 특히 평균 검출률보다
"연속 몇 프레임을 놓치는가" 가 제어에 직결된다.

검출 층을 분리해서 본다 — 셋은 서로 다른 이유로 실패한다:

    쿠션 검출률   YOLO 가 박스를 냈는가              (모델/임계값/영상)
    association   그 박스가 차량과 매칭됐는가         (매칭 로직)
    heading 채택  그 결과가 heading 으로 쓰였는가     (점프 게이트 등)

주의: 이 CSV 는 **rc_car 가 잡힌 프레임만** 행을 쓴다. 그래서 행 수만 세면
차량 미검출 구간이 통째로 사라진다. frames_since_reset 으로 실제 프레임 수를
복원해서 분모로 쓴다.

사용법::

    python tools/analyze_calib_csv.py runs/calib_pose_XXXX.csv
    python tools/analyze_calib_csv.py old.csv new.csv     # 구/신 모델 비교
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import OrderedDict
from pathlib import Path


def fnum(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def analyze(path: Path) -> "OrderedDict[str, dict]":
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r.get("analysis_valid") == "1"]
    by_cond: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        by_cond.setdefault(r.get("condition") or "-", []).append(r)

    out: "OrderedDict[str, dict]" = OrderedDict()
    for cond, rs in by_cond.items():
        # 분모: 측정 창 안의 실제 프레임 수 (행이 없는 프레임 = 차량 미검출)
        idx = [int(r["frames_since_reset"]) for r in rs if r["frames_since_reset"]]
        total = max(idx) - min(idx) + 1 if idx else len(rs)

        seen = {int(r["frames_since_reset"]) for r in rs if r["frames_since_reset"]}
        cushion_frames = {int(r["frames_since_reset"]) for r in rs
                          if fnum(r.get("det_front_cushion", "")) or 0}
        assoc_frames = sum(1 for r in rs if (fnum(r.get("assoc_pairs", "")) or 0) > 0)
        adopted = sum(1 for r in rs if r.get("heading_source") == "FRONT_CUSHION")

        confs = [c for c in (fnum(r.get("front_cushion_conf", "")) for r in rs)
                 if c is not None]
        car_confs = [c for c in (fnum(r.get("rc_car_conf", "")) for r in rs)
                     if c is not None]

        # 연속 미검출: 행이 없는 프레임도 미검출로 센다.
        lo, hi = (min(idx), max(idx)) if idx else (0, -1)
        streak = worst = 0
        for i in range(lo, hi + 1):
            if i in cushion_frames:
                streak = 0
            else:
                streak += 1
                worst = max(worst, streak)

        tids = [r["rc_car_track_id"] for r in rs if r.get("rc_car_track_id")]
        switches = sum(1 for a, b in zip(tids, tids[1:]) if a != b)

        xs = [c for c in (fnum(r.get("x_mm", "")) for r in rs) if c is not None]
        ys = [c for c in (fnum(r.get("y_mm", "")) for r in rs) if c is not None]
        hs = [c for c in (fnum(r.get("heading_deg", "")) for r in rs) if c is not None]

        def pct(n: int) -> float:
            return 100.0 * n / total if total else 0.0

        out[cond] = {
            "frames": total,
            "rc_car_%": pct(len(seen)),
            "cushion_%": pct(len(cushion_frames)),
            "assoc_%": pct(assoc_frames),
            "adopt_%": pct(adopted),
            "conf_mean": statistics.fmean(confs) if confs else None,
            "conf_min": min(confs) if confs else None,
            "conf_p5": (statistics.quantiles(confs, n=20)[0]
                        if len(confs) >= 20 else (min(confs) if confs else None)),
            "car_conf_mean": statistics.fmean(car_confs) if car_confs else None,
            "max_miss": worst,
            "tid_switch": switches,
            "pose": (statistics.fmean(xs) if xs else 0,
                     statistics.fmean(ys) if ys else 0,
                     statistics.fmean(hs) if hs else 0),
        }
    return out


def fmt(v, spec="6.1f", dash="   -  "):
    return dash if v is None else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description="calib_pose CSV 분석")
    ap.add_argument("csv", nargs="+")
    a = ap.parse_args()

    results = []
    for p in a.csv:
        path = Path(p)
        if not path.is_file():
            raise SystemExit(f"[ERROR] 파일이 없습니다: {path}")
        results.append((path.name, analyze(path)))

    for name, res in results:
        print(f"\n{'='*95}\n{name}\n{'='*95}")
        # 한글은 터미널에서 2칸을 차지하므로 폭 계산에서 글자수만큼 빼준다.
        def h(label: str, width: int) -> str:
            wide = sum(1 for ch in label if ord(ch) > 0x2E80)
            return f"{label:>{max(width - wide, 0)}}"
        print(h("조건", 6) + h("프레임", 8) + h("차량%", 8) + h("쿠션%", 8)
              + h("assoc%", 8) + h("채택%", 8) + h("conf평균", 10)
              + h("conf최소", 10) + h("conf p5", 10) + h("최대연속", 10)
              + h("tid변경", 9))
        print("-" * 95)
        for cond, m in res.items():
            print(f"{cond:<6}{m['frames']:>8}{m['rc_car_%']:>8.1f}{m['cushion_%']:>8.1f}"
                  f"{m['assoc_%']:>8.1f}{m['adopt_%']:>8.1f}"
                  f"{fmt(m['conf_mean'],'10.3f','         -')}"
                  f"{fmt(m['conf_min'],'10.3f','         -')}"
                  f"{fmt(m['conf_p5'],'10.3f','         -')}"
                  f"{m['max_miss']:>10}{m['tid_switch']:>9}")
        print("-" * 95)
        for cond, m in res.items():
            x, y, h = m["pose"]
            print(f"  {cond} 실측 pose ≈ ({x:.0f}, {y:.0f}, {h:.1f}°)")

    if len(results) == 2:
        (n0, r0), (n1, r1) = results
        print(f"\n{'='*70}\n비교: {n0} → {n1}\n{'='*70}")
        print(f"{'조건':<7}{'쿠션% 변화':>16}{'채택% 변화':>16}{'최대연속 변화':>18}")
        print("-" * 70)
        for cond in r0:
            if cond not in r1:
                continue
            a0, b0 = r0[cond], r1[cond]
            print(f"{cond:<7}"
                  f"{a0['cushion_%']:>7.1f}→{b0['cushion_%']:<8.1f}"
                  f"{a0['adopt_%']:>7.1f}→{b0['adopt_%']:<8.1f}"
                  f"{a0['max_miss']:>8}→{b0['max_miss']:<9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
