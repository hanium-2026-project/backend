# SOURCE_COMPATIBILITY — 실제 backend 사용 여부 / E2E 판정

| 항목 | 값 |
|---|---|
| repo / branch / 최신 HEAD | hanium-2026-project/backend · develop · 96901f2268d227d2bda70c4e34b42bf03015e056 |
| 최신 VehicleServer full source 사용 | 아니오(제공 안 됨) |
| ACTUAL_BACKEND_E2E (최신 comm.server.VehicleServer full path) | **SKIPPED** |
| REAL_MOCKFW_WIRE_E2E (실제 protocol+reliability+MockFirmware over TCP) | **PASS** |

## SKIPPED 이유
제공된 backend 소스는 `claude_v4_real_backend_handoff` 안의 **pre-B-merge 스냅샷**뿐이다. 이
스냅샷의 `comm/server.py` 에는 `push_control`/`stop_control`/`direct_control_enabled`/
`on_comm_recovered`/`on_command_result` 가 아직 없다(PR#41 B-merge 에서 추가). 따라서 최신
VehicleServer 객체로 DIRECT_CONTROL full path 를 실행할 수 없다.

## REAL_MOCKFW_WIRE_E2E 경계
- 사용한 실제 backend 코드: `comm.protocol`(make_hello_ack/make_set_mode/make_direct_control/
  make_heartbeat/wire_car_id), `comm.reliability.ReliableSender`,
  `comm.tests.mock_firmware.MockFirmware`.
- 재현(실제 객체 아님): 최신 VehicleServer 의 push_control/direct_control_enabled/latest-only
  100ms 재전송 의미만 얇은 harness 로. control_seq 는 harness(서버 역할)가 소유.
- 즉 "실제 protocol/reliability/mock over 실제 TCP" + "서버 제어 의미 재현". 최신 VehicleServer
  객체 자체의 E2E 는 아니다.

## 실행
```bash
HANIUM_BACKEND_SRC=/path/to/backend python -m unittest tests.real_backend.test_real_mockfw_wire_e2e -v
# 미지정 시 해당 3개 SKIP, 나머지는 계약 재현(spec)으로 통과.
```
comm/__init__.py 가 rl/gymnasium 을 eager import 하므로, wire E2E 는 bare comm 패키지로 실제
protocol/reliability/mock_firmware 파일만 로드한다(tests/real_backend/real_comm_loader.py).

## 최신 full source 확보 시
`VehicleServer(port=0, known_car_ids={1})` + backend.patch 적용(on_command_result) +
`MockFirmware(server.bound_port, car_id="CAR_01")` 로 동일 소켓 패턴의 실제 객체 E2E 를 추가해
ACTUAL_BACKEND_E2E=PASS 로 승격할 수 있다(push_control 반환 None, send_set_mode 반환 seq 검증).
