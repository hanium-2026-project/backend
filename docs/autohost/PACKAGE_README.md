# host_autonomous_control_FINAL — B안 AUTO_HOST 제어기 (실차 준비본)

2026 한이음 RC카 자율주차. 노트북(host)이 camera pose + host 내부 waypoint 로 throttle/steering
을 계산해 DIRECT_CONTROL 로 ESP32(REMOTE_DIRECT)에 스트리밍한다. 이 버전은 실차 테스트를
시작할 수 있는 **최소한의 안정된 코드**다(새 architecture 추가 없음).

## 제어 흐름
```
Camera → Pose(관측 timestamp 보존) → HostWaypointMission → WaypointController
      → throttle/steering(음수=LEFT) → VehicleServer.push_control(car_id:int,thr,steer)
      → ESP32 REMOTE_DIRECT
```
AUTO_HOST 주행 중 ESP32 로 WAYPOINT/GO 를 보내지 않는다(host 내부 target).

## 실행
```bash
bash run_tests.sh                                   # 로직/계약 테스트
HANIUM_BACKEND_SRC=/path/to/backend bash run_tests.sh   # 실제 protocol/mock wire E2E 포함
python -m examples.run_auto_host --control-mode auto-host --car 1   # actuator OFF 흐름 데모
```
- **118 tests**. HANIUM_BACKEND_SRC 미설정: 115 OK + 3 skip(real wire). 설정 시: 118 OK.

## 실제 backend 계약(요지)
- `send_set_mode(car_id:int, mode)->seq:int` (비동기; ACCEPTED 는 나중에 확인)
- `push_control(car_id:int, throttle, steering)->None` (control_seq/session/wire 는 서버 소유)
- `stop_control(car_id:int)->None`
- 속성 callback: on_comm_fail(car_id,info) / **on_comm_recovered(car_id)** / on_resync(car_id,hello)
  / on_command_rejected(car_id,result,msg). ACCEPTED 관찰은 server 에 추가하는
  on_command_result(car_id,seq,result,msg) 로(production_patch/backend.patch).
- `direct_control_enabled=True` 여야 실제 TCP 스트림이 나감(AUTO_HOST arm 시 보장, server-global).
- car_id 는 int(1,2). "CAR_01" 은 wire 경계에서만.

## 최종 답변(프롬프트 24)
1. **control flow**: 위 그림 그대로.
2. **AUTO_HOST control owner**: 이 패키지의 HostController/ControlScheduler 단 하나.
   legacy `_update_control()`/`_on_vehicle_ready()` direct 경로는 auto-host 에서 비활성.
3. **WAYPOINT/GO 전송?**: 아니오(0회).
4. **SET_MODE ACCEPTED 확인**: send_set_mode→seq 저장 → 해당 seq 의 terminal ACCEPTED 를
   on_command_result 로 확인 후 arm. 다른 seq 는 무시. ACCEPTED 전 non-zero 금지.
5. **stale pose 정지**: FAULTED latch + push_control(car_id,0,0) → latest_control zero.
6. **통신 복구 시 자동 출발?**: 아니오. FAULTED 유지, 명시적 re-arm(SET_MODE 재협상) 후에만.
7. **wire steering 부호**: 음수=LEFT, 0=CENTER, 양수=RIGHT.
8. **실제 backend full-source E2E 실행?**: 최신 VehicleServer full source 부재로 그 객체
   E2E 는 SKIPPED. 대신 실제 protocol/reliability/MockFirmware 로 TCP wire E2E 는 PASS.
9. **PASS/FAIL**: REAL_MOCKFW_WIRE_E2E=PASS.
10. **SKIPPED 이유**: 제공 backend 는 pre-B-merge 스냅샷뿐이라 최신 push_control/
    direct_control_enabled 가진 VehicleServer 가 없음(SOURCE_COMPATIBILITY.md).
11. **backend 수정 파일**: comm/server.py(on_command_result 최소 추가), pipeline/config.py
    (control_mode), pipeline/runner.py(AUTO_HOST owner 분기 + legacy 비활성),
    run_pipeline.py(--control-mode). + 신규 control/auto_host_runner.py. protocol/reliability/
    orchestrator/CV/RL/waypoint generator 무수정.
12. **actuator OFF 시작**: REAL_CAR_TEST_PLAN.md 참조.
13. **아직 실측할 값**: throttle↔cm/s, 정지거리, servo각↔실제 바퀴각, 축거/조향각, encoder 물리량.

## 구조
```
controller/      제어 수학(pose→waypoint→throttle/steering)
host_control/    authority / HostWaypointMission / pose_source / scheduler 연결
integration/     backend_adapter(push_control 위임) / control_scheduler(100ms) /
                 remote_direct_session(비동기 handshake+callback) / backend_contract(계약 재현)
tests/           로직 테스트 + tests/real_backend(실제 protocol/mock wire E2E)
examples/run_auto_host.py   AUTO_HOST 데모
production_patch/ backend.patch + auto_host_runner.py + README(적용 지침)
```

## 주의
throttle 은 normalized(실제 cm/s 아님). encoder 는 미보정이라 PID 미사용. 로직/wire 검증이며
실차 성능이 아니다. ENABLE_ACTUATOR_OUTPUT=0 로 첫 통합시험을 시작한다.
