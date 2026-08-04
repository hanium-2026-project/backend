"""실행 중인 파이프라인에 가짜 ESP32 를 붙인다 (하드웨어 없이 전체 흐름 확인).

run_pipeline 은 차량이 TCP 로 접속해야만 슬롯 배정·waypoint 전송 단계로 넘어간다.
실제 ESP32 가 준비되기 전에 그 흐름을 확인하려면 이 도구로 접속시킨다.

펌웨어 계약 검증에 쓰는 comm/tests/mock_firmware.py 를 그대로 재사용하므로,
필수 필드가 빠지거나 범위를 벗어난 명령은 실제 펌웨어처럼 거절되고 로그에 남는다.

사용법 (터미널 2개)::

    # 터미널 1 — 파이프라인
    python manage.py run_pipeline --camera 0 --weights best.pt \
        --calibration calibration.json --port 5050 --show

    # 터미널 2 — 가짜 차량
    python tools/fake_esp32.py --port 5050
    python tools/fake_esp32.py --port 5050 --car-id 2   # 두 번째 차량
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comm.tests.mock_firmware import MockFirmware


def main() -> int:
    ap = argparse.ArgumentParser(description="가짜 ESP32 (파이프라인 연결 확인용)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--car-id", type=int, default=1, help="차량 번호 (1, 2, ...)")
    args = ap.parse_args()

    car = f"CAR_{args.car_id:02d}"
    print(f"{car} → {args.host}:{args.port} 접속 시도")
    try:
        esp = MockFirmware(args.port, host=args.host,
                           car_id=car, boot_id=f"B{args.car_id:07d}")
    except OSError as exc:
        print(f"접속 실패: {exc}\n  파이프라인이 먼저 떠 있어야 합니다 (--port 확인).")
        return 1

    # HELLO_ACK 대기
    for _ in range(50):
        if esp.hello_result is not None:
            break
        time.sleep(0.1)

    if esp.hello_result != "READY_ALLOWED":
        print(f"핸드셰이크 거절됨: {esp.hello_result}")
        for reason, line in esp.rejects[:3]:
            print(f"  거절 사유: {reason}")
        esp.close()
        return 1

    print(f"접속 성공 — session={esp.session_id}, state={esp.state}")
    print("차량을 카메라 진입 위치(entrance)에 두면 슬롯이 배정됩니다.")
    print("Ctrl+C 로 종료\n")

    last = None
    try:
        while True:
            now = (esp.state, esp.wait_reason,
                   (esp.target or {}).get("route_id"),
                   (esp.target or {}).get("waypoint_id"),
                   (esp.target or {}).get("phase"))
            if now != last:
                last = now
                state, reason, route, wp, phase = now
                target = f"route={route} wp={wp} {phase}" if route else "target 없음"
                extra = f" ({reason})" if reason not in ("NONE", "") else ""
                print(f"[{time.strftime('%H:%M:%S')}] state={state}{extra}  {target}")
            if esp.rejects:
                for reason, line in esp.rejects:
                    print(f"  ⚠ 계약 위반: {reason}\n     {line[:140]}")
                esp.rejects.clear()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        wp_count = len([m for m in esp.received if m.get("type") == "WAYPOINT"])
        print(f"\n수신 요약 — WAYPOINT {wp_count}건, HEARTBEAT {esp.heartbeats}건, "
              f"최종 state={esp.state}")
        esp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
