# TEST_REPORT (FINAL)

Python 3 (stdlib only). 실행:
```bash
python -m unittest discover -s . -p 'test_*.py' -t .
```

## 결과
- HANIUM_BACKEND_SRC 미설정(배포 기본): `Ran 118 tests — OK (skipped=3)` → 115 OK + 3 skip.
- HANIUM_BACKEND_SRC=<backend> 설정: `Ran 118 tests — OK` → 118 OK (real wire E2E 3개 실행).

## E2E 정직성 라벨 (SOURCE_COMPATIBILITY.md)
- `ACTUAL_BACKEND_E2E = SKIPPED` — 최신 comm.server.VehicleServer full source 부재
  (제공된 backend 는 pre-B-merge 스냅샷뿐이라 push_control/direct_control_enabled 미포함).
- `REAL_MOCKFW_WIRE_E2E = PASS` — 실제 comm.protocol + comm.reliability +
  comm.tests.mock_firmware 를 실제 TCP 로 결선. 서버측 push_control/direct_control_enabled
  의미만 계약대로 재현한 harness.

spec double 통과를 actual backend E2E 로 표현하지 않는다.

## 스위트별 개수
| 스위트 | tests |
|---|---|
| controller.tests.test_geometry | 7 |
| controller.tests.test_pose_controller | 16 |
| controller.tests.test_convergence | 6 |
| controller.tests.test_independence | 3 |
| host_control.tests.test_authority | 12 |
| host_control.tests.test_producers | 9 |
| host_control.tests.test_mission | 8 |
| host_control.tests.test_pose_source | 6 |
| host_control.tests.test_transport | 7 |
| host_control.tests.test_host_controller | 12 |
| host_control.tests.test_mission_simulation | 5 |
| integration.tests.test_integration_independence | 4 |
| integration.tests.test_production_integration | 20 |
| tests.real_backend.test_real_mockfw_wire_e2e | 3 (미설정 시 skip) |
| **합계** | **118** |

## 프롬프트 17장 필수 최소 검증 매핑
- controller 핵심: controller/*, host_control/* 스위트.
- LEFT→wire<0 / RIGHT→wire>0: test_wire_steering_sign, real wire E2E.
- stale→0,0: test_F_stale_zeros_latest_control.
- fault 후 자동 non-zero resume 없음: test_G_no_auto_resume, test_H_comm_fail_recovered.
- push_control(int car_id,thr,steer)->None: test_A_*.
- SET_MODE ACCEPTED 전 non-zero 없음 / 후 arm: test_C_*.
- WAYPOINT=0 / GO=0: test_I_progression_no_waypoint_go, real wire E2E.
- final arrival→0,0: mission_simulation, test_I.
- legacy 와 동시 owner 아님: production_patch(runner 분기) + owner 단일 검증.
- callback arity(on_comm_recovered 1-arg): TestV5CallbackContract.

## real wire E2E(3)
- test_ready_handshake: 실제 HELLO/HELLO_ACK → mock READY.
- test_set_mode_accepted_then_direct_stream: 실제 reliable SET_MODE→ACCEPTED, DIRECT count>0,
  control_seq monotonic, LEFT<0/RIGHT>0, stale→0/0, WAYPOINT/GO=0, rejects=[].
- test_real_session_handshake_and_stream: 실제 RemoteDirectSession 을 real mock 에 결선 —
  실제 ACCEPTED → session arm → 실제 DIRECT 스트림(LEFT<0) 확인.
