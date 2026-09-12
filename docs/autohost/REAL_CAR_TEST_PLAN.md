# REAL_CAR_TEST_PLAN — actuator OFF 첫 통합시험

ESP32 는 반드시 `ENABLE_ACTUATOR_OUTPUT = 0`(모터 OFF)로 시작한다. 아래는 모터를 돌리지 않고
production path 전체를 검증하는 순서다.

## 첫 명령 흐름
```
backend 시작 → ESP32 HELLO → READY
→ AUTO_HOST 선택(--control-mode auto-host --car 1)
→ send_set_mode(1, REMOTE_DIRECT) → seq 로그
→ 해당 seq ACCEPTED 확인(on_command_result)
→ direct_control_enabled = True
→ ControlScheduler 100ms
→ Camera Pose + Host waypoint → controller
→ DIRECT_CONTROL 로그(mock actuator log)
→ Camera stale → DIRECT_CONTROL 0/0
```

## 확인 항목
1. SET_MODE REMOTE_DIRECT 가 ACCEPTED 되는가(그 전 non-zero 없음).
2. DIRECT_CONTROL 이 100ms 로 스트리밍되는가.
3. camera 를 가리면(stale) latest_control 이 0/0 으로 떨어지고 FAULTED latch 되는가.
4. steering 좌/우 명령 시 firmware wire 부호(음수=LEFT / 양수=RIGHT)가 일치하는가.
5. explicit re-arm 전에는 자동 재출발하지 않는가.
6. AUTO_HOST 동안 WAYPOINT/GO wire 전송이 0회인가.

여기까지 모두 확인된 뒤에야 단일 waypoint 저속 실차 주행(모터 ON) 단계로 넘어간다.

## 아직 실측 필요(모두 provisional)
throttle↔cm/s, 정지거리, servo 명령각↔실제 바퀴각, 축거/조향각, encoder count↔거리/속도.
이 값들이 보정되기 전에는 throttle 을 실제 속도로 취급하지 않는다.
