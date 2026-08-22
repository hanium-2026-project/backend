#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# production perception 경로를 그대로 쓴다. 이 로거만의 임시 association/heading
# 을 쓰면 pose.jsonl 과 숫자를 비교할 수 없다 (그래서 정적 진단이 무의미해진다).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cv.association import associate                 # noqa: E402
from cv.heading import HeadingEstimator              # noqa: E402
from cv.vehicle_detector import (LABEL_CAR, LABEL_CUSHION,   # noqa: E402
                                 YoloVehicleDetector)


SIDECAR_DEFAULT = str(Path("runs") / "_current_command.json")
# sidecar 가 이 시간보다 오래되면 "지금 명령이 아니다"로 본다. bridge 가
# 주행 중 0.1s 마다 갱신하므로 넉넉히 잡아도 이전 run 과 섞이지 않는다.
STALE_MS = 1500.0


class CommandSidecar:
    """bridge 가 쓰는 현재 명령 파일을 읽어 pose row 에 붙인다.

    설계 원칙 — 이게 없어도 pose 기록은 계속돼야 한다:
      - 파일이 없거나 JSON 이 깨져도 예외를 올리지 않고 빈 값을 준다
      - phase 가 FINISHED 면 명령을 active 로 기록하지 않는다
        (주행이 끝난 뒤 프레임에 이전 run 의 조향이 붙는 것을 막는다)
      - updated_timestamp 가 STALE_MS 를 넘으면 stale 로 보고 비운다
      - run_id 는 run 마다 새로 생기므로 프로세스가 죽어도 섞이지 않는다
    """

    FIELDS = ["run_id", "commanded_steering", "commanded_throttle",
              "command_duration_s", "command_phase", "command_timestamp",
              "command_age_ms", "applied_steering", "applied_throttle",
              "encoder_count", "control_seq", "status_seq", "status_timestamp"]

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._warned = False

    def blank(self) -> dict:
        return {k: "" for k in self.FIELDS}

    def read(self) -> dict:
        row = self.blank()
        try:
            raw = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return row
        try:
            d = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            if not self._warned:
                print("[WARN] sidecar JSON parse 실패 — 명령 컬럼을 비우고 계속")
                self._warned = True
            return row
        if not isinstance(d, dict):
            return row

        now = time.time()
        updated = d.get("updated_timestamp")
        age_ms = (now - float(updated)) * 1000.0 if isinstance(updated, (int, float)) else None
        phase = str(d.get("phase", "") or "")

        row["run_id"] = d.get("run_id", "") or ""
        row["command_phase"] = phase
        if age_ms is not None:
            row["command_age_ms"] = f"{age_ms:.0f}"

        # FINISHED / stale 이면 명령값을 싣지 않는다 (run 경계 오염 방지).
        active = phase in ("STEERING_SETTLE", "MOVING") and \
            (age_ms is not None and age_ms <= STALE_MS)
        if active:
            for key, col in (("commanded_steering", "commanded_steering"),
                             ("commanded_throttle", "commanded_throttle"),
                             ("command_duration_s", "command_duration_s"),
                             ("command_timestamp", "command_timestamp")):
                v = d.get(key)
                if isinstance(v, (int, float)):
                    row[col] = f"{v:.4f}" if col != "command_timestamp" else f"{v:.3f}"
            st = d.get("status")
            if isinstance(st, dict):
                for src_key, col in (("applied_steering", "applied_steering"),
                                     ("applied_throttle", "applied_throttle"),
                                     ("encoder_count", "encoder_count"),
                                     ("control_seq", "control_seq"),
                                     ("seq", "status_seq")):
                    v = st.get(src_key)
                    if v is not None:
                        row[col] = v
            sts = d.get("status_timestamp")
            if isinstance(sts, (int, float)):
                row["status_timestamp"] = f"{sts:.3f}"
        return row


def norm_label(name: str) -> str:
    k = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "rc_car": "rc_car", "rccar": "rc_car", "car": "rc_car", "vehicle": "rc_car",
        "front_cushion": "front_cushion", "frontcushion": "front_cushion",
        "cushion": "front_cushion", "front": "front_cushion",
    }
    return aliases.get(k, k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--calibration", default="calibration_new.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--measure-s", type=float, default=10.0,
                    help="조건당 측정 시간(초). 이 창 안의 행만 analysis_valid=1")
    ap.add_argument("--conditions",
                    default="G1,P1,P2,P3,C1,C2",
                    help="정적 진단 조건 라벨 순서 (n 키로 다음 조건 + 상태 reset)")
    ap.add_argument("--sidecar", default=SIDECAR_DEFAULT,
                    help="bridge 가 쓰는 현재 명령 JSON (없어도 동작)")
    a = ap.parse_args()

    calib = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    src = np.asarray(calib["homography_src"], dtype=np.float32)
    W = float(calib.get("lot_width_mm", 1200.0))
    Hm = float(calib.get("lot_height_mm", 1200.0))
    dst = np.asarray([[0,Hm],[W,Hm],[W,0],[0,0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    if not a.out:
        a.out = str(Path("runs") / ("calib_pose_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] camera-only logger: NO vehicle control")
    print("[INFO] CSV:", out)
    print("[INFO] keys:  n = 다음 조건 10s 측정 시작   r = 현재 조건 재측정   q = 종료")
    print("[INFO] 차를 놓고 손을 치운 뒤 n 을 누르세요. 측정 창 밖은 analysis_valid=0.")

    detector = YoloVehicleDetector(weights_path=a.weights,
                                   confidence_threshold=a.conf,
                                   custom_model=True, imgsz=a.imgsz,
                                   device=a.device)
    heading_est = HeadingEstimator()
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.camera}")

    sidecar = CommandSidecar(a.sidecar)
    print("[INFO] sidecar:", sidecar.path, "(없으면 명령 컬럼만 빈다)")
    cols = [
        "wall_time","t_s","frame_idx","x_mm","y_mm","heading_deg","heading_source",
        "rc_car_conf","front_cushion_conf","rc_car_track_id","front_cushion_track_id",
        # perception provenance — production pose.jsonl 과 같은 이름/의미.
        # 이 로거는 원래 rc_car+cushion 이 **둘 다** 잡힌 프레임만 기록해서
        # "미검출"을 관측할 수 없었다. 이제 매 프레임 기록한다.
        "det_total","det_rc_car","det_front_cushion","assoc_pairs",
        "assoc_unpaired_cars",
        # 조건 간 estimator 상태 carry-over 를 막기 위한 구간 표식.
        "condition","frames_since_reset","analysis_valid",
    ] + CommandSidecar.FIELDS
    t0 = time.monotonic()
    frame_idx = rows = 0
    last_heading: dict[int, float | None] = {}

    # ── 조건 간 독립성 ────────────────────────────────────────────────────
    # HeadingEstimator 는 track_id 별로 _history/_last_valid/_last_source/
    # _pending_front_jump 를 들고 있고, associate() 는 previous_heading 으로
    # 쿠션 후보를 가산한다. 손으로 큰 거리/각도를 옮기면 이전 조건의 heading 이
    # 다음 조건의 jump 게이트와 association 을 오염시킨다 — 그런데 그 두 값이
    # 바로 우리가 측정하려는 대상이다. 그래서 조건마다 명시적으로 비운다.
    # 카메라 세션은 유지한다 (auto-exposure 를 리셋하면 조명 비교가 깨진다).
    # 측정 창(measurement window) 밖의 프레임 — 차를 손으로 옮기는 동안,
    # auto-exposure 안정화 대기 — 은 analysis_valid=0 으로 남긴다. 기록은 하되
    # 분석에서 제외한다. reset 은 "측정 시작" 순간에만 수행한다.
    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    cond_idx = -1
    frames_since_reset = 0
    measure_until: float | None = None
    measured_rows = 0

    def start_measurement(idx: int) -> None:
        nonlocal heading_est, frames_since_reset, measure_until, measured_rows
        heading_est = HeadingEstimator()
        last_heading.clear()
        frames_since_reset = 0
        measured_rows = 0
        measure_until = time.monotonic() + a.measure_s
        print(f"[COND] {conditions[idx]} 측정 시작 "
              f"({a.measure_s:.0f}s) — estimator/association 초기화")


    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_idx += 1
                frames_since_reset += 1
                measuring = (measure_until is not None
                             and time.monotonic() < measure_until)
                if measure_until is not None and not measuring:
                    print(f"[COND] {conditions[cond_idx]} 측정 완료 "
                          f"— 유효 {measured_rows} frames")
                    measure_until = None
                dets = detector.detect_and_track(frame)

                def _img_heading(car_px, front_px):
                    q = cv2.perspectiveTransform(
                        np.asarray([[car_px, front_px]], dtype=np.float32), H)[0]
                    cx, cy = map(float, q[0]); fx, fy = map(float, q[1])
                    if math.hypot(fx - cx, fy - cy) < 1e-6:
                        return None
                    return math.degrees(math.atan2(fy - cy, fx - cx)) % 360.0

                prev = {t: h for t, h in last_heading.items() if h is not None}
                pairs, unpaired = associate(dets, prev,
                                            image_heading_of=_img_heading)
                perception = {
                    "det_total": len(dets),
                    "det_rc_car": sum(1 for d in dets if d.label == LABEL_CAR),
                    "det_front_cushion": sum(1 for d in dets
                                             if d.label == LABEL_CUSHION),
                    "assoc_pairs": len(pairs),
                    "assoc_unpaired_cars": len(unpaired),
                }

                # 차체가 잡힌 프레임은 쿠션 유무와 무관하게 기록한다.
                car_det = front_px = None
                cushion_conf = cushion_tid = None
                if pairs:
                    car_det = pairs[0].car
                    front_px = pairs[0].cushion_center_px
                    cu = getattr(pairs[0], "cushion", None)
                    cushion_conf = getattr(cu, "confidence", None)
                    cushion_tid = getattr(cu, "track_id", None)
                elif unpaired:
                    car_det = unpaired[0]

                txt = "waiting rc_car"
                if car_det is not None:
                    x1, y1, x2, y2 = car_det.bbox
                    car_px = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    q = cv2.perspectiveTransform(
                        np.asarray([[car_px]], dtype=np.float32), H)[0]
                    x, y = map(float, q[0])
                    fmap = None
                    if front_px is not None:
                        qf = cv2.perspectiveTransform(
                            np.asarray([[front_px]], dtype=np.float32), H)[0]
                        fmap = (float(qf[0][0]), float(qf[0][1]))
                    tid = car_det.track_id if car_det.track_id is not None else -1
                    hr = heading_est.update(tid, (x, y), front_point=fmap)
                    last_heading[tid] = hr.heading_deg
                    ts = time.monotonic() - t0
                    cmdrow = sidecar.read()
                    w.writerow({
                        **cmdrow, **perception,
                        "condition": (conditions[cond_idx]
                                      if 0 <= cond_idx < len(conditions) else ""),
                        "frames_since_reset": frames_since_reset,
                        "analysis_valid": 1 if measuring else 0,
                        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                        "t_s": f"{ts:.4f}", "frame_idx": frame_idx,
                        "x_mm": f"{x:.2f}", "y_mm": f"{y:.2f}",
                        "heading_deg": ("" if hr.heading_deg is None
                                        else f"{hr.heading_deg:.3f}"),
                        "heading_source": hr.source or "",
                        "rc_car_conf": f"{car_det.confidence:.4f}",
                        "front_cushion_conf": ("" if cushion_conf is None
                                               else f"{cushion_conf:.4f}"),
                        "rc_car_track_id": "" if car_det.track_id is None else car_det.track_id,
                        "front_cushion_track_id": "" if cushion_tid is None else cushion_tid,
                    })
                    f.flush()
                    rows += 1
                    hd = "-" if hr.heading_deg is None else f"{hr.heading_deg:.1f}"
                    if measuring:
                        measured_rows += 1
                    cond = (conditions[cond_idx]
                            if 0 <= cond_idx < len(conditions) else "-")
                    mark = f"{cond} MEASURING {measured_rows}" if measuring else f"{cond} idle"
                    txt = (f"[{mark}] x={x:.0f} y={y:.0f} "
                           f"hdg={hd} [{hr.source}] "
                           f"cushion={perception['det_front_cushion']} "
                           f"pairs={perception['assoc_pairs']}")
                    if a.show:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if pairs:
                            cu = getattr(pairs[0], "cushion", None)
                            if cu is not None:
                                bx1, by1, bx2, by2 = cu.bbox
                                cv2.rectangle(frame, (bx1, by1), (bx2, by2),
                                              (0, 165, 255), 2)

                if a.show:
                    cv2.putText(frame, txt, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                    cv2.imshow("CALIB POSE LOGGER", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("n"):
                        if cond_idx + 1 < len(conditions):
                            cond_idx += 1
                            start_measurement(cond_idx)
                        else:
                            print("[COND] 모든 조건 완료 (q 로 종료)")
                    elif key == ord("r"):
                        if cond_idx >= 0:
                            start_measurement(cond_idx)
                        else:
                            print("[COND] 아직 시작 전입니다 — n 으로 G1 시작")
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print(f"[DONE] rows={rows}")
    print("[DONE] saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
