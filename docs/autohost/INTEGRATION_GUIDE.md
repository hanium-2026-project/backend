# INTEGRATION_GUIDE — backend 결선

## AUTO_HOST 결선 (control/auto_host_runner.py 사용)
```python
from control.auto_host_runner import AutoHostRunner   # production_patch/auto_host_runner.py 템플릿
runner = AutoHostRunner(server, car_id=1, backend_waypoints=route)  # attach()로 callback fan-out
runner.start()          # send_set_mode->seq → ACCEPTED 확인 → arm + direct_control_enabled=True → 100ms loop
# CV 새 프레임마다(관측 시각 보존):
runner.on_camera_pose(x_mm, y_mm, heading_deg, obs_time=frame_time)
# stale/comm fault 후 재출발(SET_MODE 재협상):
runner.re_arm()
```

## 책임 분리
- host: pose→waypoint 계산, authority, mission 진행, stale/fault 판정, zero 강제.
- VehicleServer: session_id / control_seq / DIRECT_CONTROL wire / 100ms 스트림 소유.
- standalone controller 는 production wire seq 를 만들지 않는다.

## callback (실제 arity)
attach() 가 아래를 fan-out 으로 연결(기존 pipeline/orchestrator callback 보존):
```
on_comm_fail(car_id, info)        → FAULTED + stop_control(car_id)
on_comm_recovered(car_id)         → FAULTED 유지(자동 복귀 금지)   ← 인자 1개
on_resync(car_id, hello)          → 이전 handshake 무효 + zero
on_command_rejected(car_id, result, msg)   → FAULTED + zero (보존)
on_command_result(car_id, seq, result, msg) → ACCEPTED 확인(server 패치로 추가)
```

## SET_MODE handshake (비동기)
```
READY → send_set_mode(car_id, "REMOTE_DIRECT") → seq
      → 해당 seq 의 terminal ACCEPTED(on_command_result) 확인
      → direct_control_enabled=True → arm → non-zero DIRECT_CONTROL 허용
ACCEPTED 전: throttle=0, steering=0. 실패/거절/timeout: FAULTED + 0,0.
```

## direct_control_enabled (다중 차량 주의)
- AUTO_HOST arm 시 True 보장. 이 값이 아니면 push_control 해도 실제 TCP 스트림이 안 나감.
- **server-global boolean** 이다. 한 차량 fault 로 끄지 않는다(다른 AUTO_HOST 차량 stream 유지).
  fault 차량은 stop_control(car_id) 로만 zero. 전체 종료 시에만 global False.

## camera stale
- control loop(100ms)는 camera 와 독립. 새 관측 없으면 pose.timestamp 갱신 안 함.
- age > threshold → FAULTED + push_control(car_id,0,0) → latest_control zero(마지막 non-zero
  재전송 방지).

## steering 부호
음수=LEFT / 0=CENTER / 양수=RIGHT. push_control 직전 최종값 기준. (참고 servo 50°/86°/122°)

## CLI
`--control-mode auto-host` 선택 시 이 host controller 만 DIRECT_CONTROL owner. legacy direct/
WAYPOINT_AUTO 모드는 기존 SW 경로 보존.

## 실제 backend 적용
production_patch/README.md + production_patch/backend.patch 참조. 최소 변경: comm/server.py
(on_command_result), pipeline/config.py, pipeline/runner.py, run_pipeline.py.
