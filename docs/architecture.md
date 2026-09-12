# System Architecture

## 개요

이 시스템은 3개의 독립된 git 저장소로 구성된 하나의 closed-loop 시스템입니다.

| 저장소 | 역할 | 비고 |
|---|---|---|
| `backend` (이 저장소) | 인식(CV), 배정(RL), 제어, 통신, 웹 서버 | 이 문서가 설명하는 대상 |
| `frontend` | React 대시보드 | 별도 GitHub 저장소 |
| 하드웨어 저장소 | ESP32 펌웨어(FreeRTOS), HIL 브리지 | 별도 GitHub 저장소, `integrated/esp32_main/` |

세 저장소가 물리적으로 하나의 루프를 이룹니다: 카메라 영상은 이 `backend`에서 처리되고, 계산된 제어값은 Wi-Fi로 하드웨어 저장소의 ESP32 펌웨어에 전달되며, 그 결과는 다시 카메라로 관측되어 `backend`로 돌아옵니다. 사람이 보는 화면은 `frontend`가 Django Channels(WebSocket)로 받아 그립니다.

## 데이터 흐름

```
천장 고정 카메라
  │
  ▼
YOLO 2-class 검출 (rc_car, front_cushion)     cv/vehicle_detector.py
  │
  ▼
차량 ↔ 마커 Association                        cv/association.py
  │
  ▼
Heading 추정 (marker > trajectory > last_valid) cv/heading.py
  │
  ▼
Homography (pixel → mm)                        cv/homography.py
  │
  ▼
Vehicle Pose (x, y, heading, timestamp)
  │
  ├─▶ PPO 주차면 배정 (MaskablePPO)              rl/parking_env.py, rl/bridge.py
  │        │
  │        ▼
  │     Route / Waypoint 생성 + 안전 검증        parking/waypoints.py, parking/trajectory_safety.py
  │        │
  ▼        ▼
HostController 폐루프 제어                      controller/pose_controller.py, control/waypoint_controller.py
  │
  ▼
TCP/NDJSON 통신 (재전송, 우선순위 선점)          comm/protocol.py, comm/reliability.py
  │
  ▼
ESP32 REMOTE_DIRECT (FreeRTOS Task 분리)        [하드웨어 저장소] integrated/esp32_main/
  │
  ▼
DC 모터 / 서보 / 엔코더
  │
  └─▶ 카메라가 이동 결과를 재관측 → 루프 반복 (closed loop)

병렬로: Django Channels + Redis → WebSocket → React 대시보드
```

각 단계의 문제 해결 과정은 다음 문서에서 다룹니다. 이 문서는 "무엇이 어디 있는지"만 설명하고, "왜 그렇게 됐는지"는 중복하지 않습니다.

- 인식 단계의 실패/재학습 → [`cv-perception.md`](cv-perception.md)
- Pose → Control 연결부의 통합 문제 → [`vision-to-control.md`](vision-to-control.md)
- 실차에서 드러난 시뮬레이션과의 격차 → [`sim-to-real.md`](sim-to-real.md)
- PPO 배정 알고리즘 자체 → [`rl-parking-assignment.md`](rl-parking-assignment.md)
- Django/Redis/통신/배포 구조 → [`system-integration.md`](system-integration.md)

## 설계 상 핵심 결정

- **ESP32는 waypoint 판단을 하지 않는다.** 초기에는 ESP32가 전역 waypoint를 스스로 추종하는 구조(`WAYPOINT_AUTO`)였으나, 판단을 노트북(HostController)으로 옮기고 ESP32는 `REMOTE_DIRECT` 명령을 저수준으로 실행 + 즉시 안전정지만 담당하도록 재설계했다. 이전 프로토콜은 호환성을 위해 코드에는 남아 있지만 production 경로가 아니다.
- **waypoint는 목표 자체가 아니라 목표 pose(최종 slot pose)로 가는 중간 참조점이다.** 추종 오차는 피드백으로 흡수하고, 오차가 커지면 정지 → fresh pose 확보 → 재계획한다.
- **RL(PPO) 배정과 Safety Shield(경로 충돌 검사)는 독립적으로 동작한다.** RL이 잘못된 배정을 하더라도 Shield가 충돌 경로를 차단한다 — 자세한 내용은 [`rl-parking-assignment.md`](rl-parking-assignment.md).
