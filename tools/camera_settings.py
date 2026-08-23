#!/usr/bin/env python3
"""카메라 노출/포커스 제어 가능 여부를 실측하고, 되는 것만 고정한다.

배경: 주행 중 노출·포커스가 흔들리면 FRONT_CUSHION confidence 가 프레임마다
출렁인다. 데이터셋을 찍기 **전에** 고정해야 학습 분포와 production 분포가
맞는다 (나중에 고정하면 재촬영이다).

문제는 OpenCV 의 카메라 속성 지원이 백엔드마다 다르다는 것이다. Windows
(DirectShow)는 대부분 먹지만 macOS(AVFoundation)는 상당수를 조용히 무시한다
— set() 이 True 를 돌려줘도 실제로는 안 바뀌는 경우가 있다. 그래서 이 도구는
**set 후 read-back 으로 실제 반영 여부를 확인**한다.

사용법::

    # 1) 무엇이 제어 가능한지 조사 (원래 값으로 되돌린다)
    python tools/camera_settings.py --camera 0 --probe

    # 2) 되는 것만 고정하고 camera_settings.json 으로 저장
    python tools/camera_settings.py --camera 0 --lock

    # 3) 저장해둔 설정을 다시 적용 (촬영/측정 세션마다)
    python tools/camera_settings.py --camera 0 --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# (이름, CAP_PROP, 고정에 쓸 값). 값이 None 이면 "현재 값을 그대로 고정"이다.
# AUTO_* 를 먼저 끈 뒤 수동값을 넣어야 반영되는 카메라가 많아 순서가 중요하다.
AUTO_PROPS = [
    ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),  # 0.25=수동, 0.75=자동
    ("autofocus", cv2.CAP_PROP_AUTOFOCUS, 0.0),
    ("auto_wb", cv2.CAP_PROP_AUTO_WB, 0.0),
]
MANUAL_PROPS = [
    ("exposure", cv2.CAP_PROP_EXPOSURE, None),
    ("focus", cv2.CAP_PROP_FOCUS, None),
    ("wb_temperature", cv2.CAP_PROP_WB_TEMPERATURE, None),
    ("brightness", cv2.CAP_PROP_BRIGHTNESS, None),
    ("contrast", cv2.CAP_PROP_CONTRAST, None),
    ("gain", cv2.CAP_PROP_GAIN, None),
]
INFO_PROPS = [
    ("frame_width", cv2.CAP_PROP_FRAME_WIDTH),
    ("frame_height", cv2.CAP_PROP_FRAME_HEIGHT),
    ("fps", cv2.CAP_PROP_FPS),
]


def backend() -> int:
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def settable(cap, prop: int, value: float) -> tuple[bool, float, float]:
    """set 후 read-back 으로 실제 반영 여부를 판정한다.

    set() 의 반환값은 믿을 수 없다 — 반영하지 않고도 True 를 주는 백엔드가 있다.
    """
    before = cap.get(prop)
    cap.set(prop, value)
    # 일부 카메라는 다음 프레임을 읽어야 값이 갱신된다.
    cap.read()
    after = cap.get(prop)
    return (abs(after - before) > 1e-6 or abs(after - value) < 1e-6), before, after


def main() -> int:
    ap = argparse.ArgumentParser(description="카메라 노출/포커스 제어 실측·고정")
    ap.add_argument("--camera", type=int, default=0)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true", help="제어 가능 여부만 조사")
    mode.add_argument("--lock", action="store_true", help="되는 것을 고정하고 저장")
    mode.add_argument("--apply", action="store_true", help="저장된 설정 재적용")
    ap.add_argument("--out", default="camera_settings.json")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera, backend())
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] 카메라 {a.camera} 를 열 수 없습니다. "
                         "인덱스와 카메라 권한을 확인하세요.")
    for _ in range(5):   # auto 알고리즘이 자리잡을 시간
        cap.read()

    print(f"[INFO] 백엔드 {'DSHOW' if backend()==cv2.CAP_DSHOW else 'AVFOUNDATION' if backend()==cv2.CAP_AVFOUNDATION else 'ANY'}")
    for name, prop in INFO_PROPS:
        print(f"       {name:14s} = {cap.get(prop)}")

    out = Path(a.out)

    if a.apply:
        if not out.is_file():
            raise SystemExit(f"[ERROR] 설정 파일이 없습니다: {out} (먼저 --lock)")
        saved = json.loads(out.read_text(encoding="utf-8"))
        applied, failed = [], []
        for name, prop, _ in AUTO_PROPS + MANUAL_PROPS:
            if name not in saved:
                continue
            cap.set(prop, saved[name])
            cap.read()
            got = cap.get(prop)
            (applied if abs(got - saved[name]) < 1e-3 else failed).append(
                f"{name}={saved[name]}(실제 {got})")
        print("\n[적용됨]", ", ".join(applied) or "없음")
        if failed:
            print("[반영 안 됨]", ", ".join(failed))
        cap.release()
        return 0

    print("\n=== 제어 가능 여부 (set → read-back) ===")
    original: dict[str, float] = {}
    result: dict[str, float] = {}
    for name, prop, target in AUTO_PROPS + MANUAL_PROPS:
        cur = cap.get(prop)
        original[name] = cur
        # 수동 속성은 "현재 값 그대로 고정"이 목표이므로, 반영 여부만 보려고
        # 살짝 다른 값을 시도한 뒤 원래 값으로 되돌린다.
        probe_value = target if target is not None else (cur + 1.0)
        ok, before, after = settable(cap, prop, probe_value)
        mark = "가능" if ok else "무시됨"
        print(f"  {name:14s} {mark:6s}  {before:>10.3f} → {after:>10.3f}")
        if ok:
            result[name] = probe_value if target is not None else cur
        # probe 모드거나 반영이 안 된 경우 원상 복구
        if a.probe or not ok:
            cap.set(prop, cur)
            cap.read()

    if a.probe:
        print("\n[PROBE] 원래 값으로 되돌렸습니다. 고정하려면 --lock 으로 다시 실행하세요.")
        cap.release()
        return 0

    # --lock: 수동 속성은 '지금 auto 가 고른 값'을 그대로 굳힌다.
    for name, prop, target in MANUAL_PROPS:
        if name in result:
            cap.set(prop, result[name])
            cap.read()
            result[name] = cap.get(prop)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[LOCK] 고정 성공 {len(result)}개 → {out}")
    print("       " + (", ".join(f"{k}={v:g}" for k, v in result.items()) or "없음"))
    if not result:
        print("\n[경고] OpenCV 로는 아무것도 고정하지 못했습니다.")
        print("       macOS 라면 외부 유틸(uvcc, Webcam Settings 앱)이나")
        print("       Windows(DirectShow) 쪽에서 고정해야 합니다.")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
