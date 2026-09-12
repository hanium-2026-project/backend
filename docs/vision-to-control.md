# Vision-to-Control Integration

CV가 차량을 잘 검출하는 것과, 그 결과가 실제로 안전하게 제어 명령이 되는 것은 다른 문제였다. 이 문서는 `Detection → Association → Heading → Homography → Pose → HostController → ESP32 → Feedback` 경로에서 실제로 발생했던 통합 문제와 해결 방식을 정리한다.

## 1. Detection → Association: 잘못 묶이면 방향이 반대가 된다

`cv/association.py`는 검출된 차량(`rc_car`)과 전면 마커(`front_cushion`) 박스를 짝짓는다. 마커가 잘못된 차량에 묶이면 heading이 반대 방향으로 산출된다. 이를 막기 위해 컨테인먼트/인접도, 박스 대각선 대비 거리, 궤적 연속성, heading 급변 페널티, confidence를 함께 고려한 그리디 1:1 매칭을 수행하고, 애매한 경우에는 매칭을 포기하고 궤적 기반 heading으로 자연스럽게 넘어가도록 되어 있다.

## 2. Heading 추정: 마커 우선 3단 폴백

`cv/heading.py`의 `HeadingEstimator.update()`는 다음 우선순위로 heading을 결정한다.

1. `FRONT_CUSHION` (마커 기반, 가장 신뢰) — 차량이 정지 중이거나 후진 중이면 궤적으로는 방향을 알 수 없어 마커가 유일한 단서다.
2. `TRAJECTORY` (이동 궤적)
3. `LAST_VALID` (직전 유효값)

45도를 초과하는 급격한 heading 변화는 2프레임 연속으로 같은 값이 나올 때만 채택한다 — 한 프레임의 오매칭이 트랙 전체를 오염시키는 것을 막는 장치다.

## 3. Homography: 좌표계를 하나로 통일

`cv/homography.py`가 카메라 pixel 좌표를 실제 주차장 mm 좌표로 변환한다. 이 좌표계는 주차면 배정(RL), 경로 생성(waypoints), 제어(controller) 전체가 공유한다 — 카메라 위치가 바뀌면 `tools/calibrate_camera.py`로 재보정해야 한다.

## 4. Pose Freshness: "연결이 복구됐다"가 "안전하다"는 뜻이 아니다

TCP 연결이 복구되었다는 사실만으로 차량의 물리적 상태가 안전하다고 보장할 수 없다. 이전 session의 명령이나 오래된 카메라 pose가 그대로 재사용되면 비정상 재출발이 일어날 수 있다. 이를 막기 위해:

- `integration/camera_adapter.py`: pose timestamp는 **새로운 관측이 있을 때만** 갱신한다. tick 시각으로 덮어쓰면 카메라가 죽어도 pose가 "살아있는 것처럼" 보이게 된다 — 이 구분이 없으면 정지된 카메라를 정지된 카메라로 인식할 수 없다.
- `control/waypoint_controller.py` / `controller/pose_controller.py`: 드라이브 명령을 내리기 전에 `POSE_INVALID → NO_HEADING → POSE_STALE`(`max_pose_age_s=0.5`) 순으로 안전 게이트를 통과해야 한다.
- `host_control/final_pose_guard.py`의 `FinalPoseGuard`는 최종 waypoint 도착을 **신선한 카메라 관측 3회 연속**으로 확인해야 확정한다 — "인식이 한 번 도착했다고 말했다"를 곧바로 믿지 않는다.
- `host_control/approach_guard.py`의 `ApproachProgressGuard`는 COARSE/FINE 캡처 반경 기준으로 접근 진행을 추적하고, 목표 지점을 지나치면(`APPROACH_MISSED`) `REPLAN_REQUIRED`를 발생시킨다.

## 5. Control Scheduler: 카메라가 멈춰도 명령이 계속 나가던 문제

가장 실제적인 통합 버그는 이것이었다: 제어 루프가 **카메라 콜백 안에서만** 실행되면, 카메라 피드가 끊겨도 새 계산이 없으니 `VehicleServer`는 마지막으로 계산된 non-zero 명령을 계속 재전송한다. ESP32의 500ms fail-safe watchdog는 "패킷이 안 온다"만 감지하므로, 패킷이 (오래된 값이라도) 계속 들어오면 **watchdog가 아예 발동하지 않는다.**

해결: `integration/control_scheduler.py`가 카메라 콜백과 독립된 100ms 주기 `host.tick()` 루프를 둔다. 이 루프가 pose 신선도를 직접 검사해서, 카메라가 멈추면 스스로 `FAULTED` + zero 출력으로 전환한다 — ESP32의 watchdog에 의존하지 않는다.

## 6. Control Authority: 한 번에 하나의 입력만

`host_control/authority.py`의 `ControlAuthority`는 `DISARMED / MANUAL / AUTO_HOST / FAULTED` 단일 상태로 관리되어, 수동 조작과 자율 주행이 구조적으로 동시에 활성화될 수 없다. 카메라 관측이 끊기면 `FAULTED`로 래치되고, 여기서 벗어나려면 명시적 재무장(`arm_manual` / `arm_auto`)이 필요하다 — 자동으로 조용히 복구되지 않는다.

## 7. 신뢰성 있는 명령 전달

`comm/reliability.py`의 `ReliableSender`는 반드시 도착해야 하는 명령을 seq 번호로 추적하며, 차량당 미확인 명령을 1개로 제한해 순서 역전을 막는다. 응답이 없으면 같은 seq·같은 내용으로 최대 5회 재전송한다. `comm/protocol.py`는 STOP > WAIT > 일반 순으로 우선순위 선점을 정의해, 위험 명령이 대기 중인 일반 명령에 막히지 않도록 한다.

## System Impact

이 레이어의 안전장치들(pose freshness, control scheduler, authority state machine, final pose guard)은 개별로는 작은 방어 로직이지만, 실제 실차 테스트에서 발견된 구체적 실패(카메라 정지 후 계속 주행, 재출발 오류)를 막기 위해 하나씩 추가된 것이다. 이 문제들이 실제로 실차에서 어떻게 드러났는지는 [`sim-to-real.md`](sim-to-real.md)에서 다룬다.
