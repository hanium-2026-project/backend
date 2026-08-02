# SW팀 → HW팀 전달: 통신 계약 정합 완료 및 요청사항

기준일: 2026-08-02
SW 저장소: `hanium-2026-project/backend`, 브랜치 `feat/comm-firmware-contract` (PR #30)
대조 대상: `rjsdl5389/2026-hanium-smart-parking` @ `bcebbb6` (`integrated/esp32_main/main/protocol.c`)

---

## 1. 요약

펌웨어 `protocol.c` 의 파싱 코드를 직접 대조해, SW 측 통신 모듈을 **펌웨어 계약에 맞춰 수정 완료**했습니다.
펌웨어 변경 없이 현재 상태 그대로 연결 가능합니다.

이전에 논의된 **`car_id` 를 정수로 바꾸자는 SW 측 제안은 철회**합니다. 펌웨어의 `"CAR_01"` 문자열을 그대로 유지하고, SW 내부에서만 정수로 변환해 씁니다 (아래 5절).

---

## 2. SW 측에서 수정한 항목 (펌웨어 변경 불필요)

| # | 항목 | 수정 전 증상 |
|---|---|---|
| 1 | `car_id` 를 `"CAR_01"` 문자열로 송신 | 정수 송신 시 `strcmp` 불일치로 전 메시지 거절 |
| 2 | `HELLO_ACK` 에 `boot_id` 에코 + `command_seq_start` 추가 | 필수 필드 누락 → 핸드셰이크 실패 |
| 3 | `HEARTBEAT` 250ms 주기 송신 | 미송신 → 연결 1초 뒤 차량 `COMM_TIMEOUT` |
| 4 | `waypoint_id` 를 1부터 발급 | 하한 1 위반 → 첫 waypoint 거절 |
| 5 | `target_heading_deg` 를 항상 실수로 송신 | `heading_required=false` 일 때 `null` → 파싱 실패 |
| 6 | `WAIT` 에 `route_id`/`waypoint_id` 포함 | 필수 필드 누락 → **안전정지 명령 거절** |
| 7 | `WAIT` `reason` 을 펌웨어 enum 으로 제한 | `WP_SWITCH` 등 임의값 → 거절 |
| 8 | ACK 판정을 `command_result` terminal 기준으로 교체 | 주기 STATUS 의 `command_result=NONE` 을 승인으로 오인 |

8번과 함께 **거절(negative result) 처리 경로**를 신설했습니다. 이전에는 성공만 가정해서, 예를 들어 `GO` 가 `INVALID_STATE` 로 거절되면 SW 상태기계가 영구 대기에 빠졌습니다. 이제 `STALE_ROUTE`/`TARGET_MISMATCH` 계열은 경로 재생성, `POSE_REQUIRED`/`INVALID_STATE` 는 대기 후 재개, `LOCKED_STATE`/`SESSION_MISMATCH` 계열은 자동 재개 금지로 분기합니다.

기타: 서버 기본 포트를 **5000** 으로 맞췄습니다 (`app_config.h` 의 `SERVER_PORT` 와 일치).

---

## 3. HW팀 요청사항

### 3.1 `POSE_UPDATE` 수신 구현 (우선순위 최상)

`implementation_status.md` 기준 현재 미구현이며, `protocol.c` 의 수신 타입 분기에도 없습니다.
**이것이 없으면 waypoint 자동주행이 원리적으로 불가능합니다.** ESP32 가 자기 위치를 모르기 때문입니다.

SW 측은 이미 송신 코드를 갖추고 있고, 펌웨어 미지원이라 현재 플래그로 꺼둔 상태입니다
(`comm/protocol.py` 의 `POSE_UPDATE_ENABLED = False`). 펌웨어에 추가되면 플래그만 켜면 됩니다.

송신 예정 형식 (통합 문서 §19 준수):

```json
{"version":1,"type":"POSE_UPDATE","car_id":"CAR_01","session_id":"S82F19C4",
 "pose_seq":5021,"x_cm":42.5,"y_cm":76.0,"heading_deg":91.2,
 "position_confidence":0.96,"heading_confidence":0.90,
 "heading_source":"TRAJECTORY","measurement_age_ms":42,"valid":true}
```

- 주기: 100ms (조정 가능)
- `pose_seq` 로 최신값 판별, 재전송·ACK 없음
- `valid=false` 인 값은 제어에 사용하지 말 것
- `heading_source` 는 현재 `TRAJECTORY` 또는 `LAST_VALID` (쿠션 학습 전)

### 3.2 전방 쿠션 제작 사양 (차량 제작 시 반영 요청)

통합 문서 §6.4 의 `FRONT_CUSHION` 입니다. **차량 제작이 끝난 뒤에는 반영이 어려우므로 제작 단계에서 함께 부탁드립니다.**

용도: 정지 상태와 후진에서의 heading 측정. 현재 1차 구현은 이동 궤적으로 heading 을 추정하는데, 원리상 **정지 중에는 방향을 알 수 없고 후진 시 180° 반대로 나옵니다.** 정밀 정렬이 필요한 ALIGN·ENTRY·FINAL 구간이 하필 저속·정지 구간이라, 쿠션이 있어야 최종 주차 정확도가 확보됩니다.

| 항목 | 요구사항 | 이유 |
|---|---|---|
| 색상 | 빨강 또는 파랑 계열 **원색** | 바닥(회색)·차체(검정)과 대비 확보 |
| 금지 색 | **노랑 계열** | 주차선과 혼동 |
| 크기 | 위에서 봤을 때 최소 **차폭의 1/3 이상** | 천장 720p 기준 화면에서 15~20px 이상이어야 안정 탐지 |
| 위치 | 차량 **앞쪽 끝** | 중심→쿠션 벡터가 곧 heading 이므로 앞뒤 구분이 명확해야 함 |
| 개수 | **앞에만 1개** | 뒤에도 비슷한 것이 있으면 앞뒤 구분 불가 |
| 형태 | 위에서 볼 때 좌우 대칭, 납작해도 무방 | 탑뷰 카메라만 사용 |
| 고정 | 주행 진동에 흔들리지 않게 | 흔들리면 heading 이 떨린다 |

색상·크기가 확정되면 알려주세요. 확정 후 SW 측에서 2클래스(`RC_CAR`, `FRONT_CUSHION`) 재학습을 진행합니다.

**촬영 일정 관련**: 재학습용 데이터셋은 **최종 차량 형태(ESP32·배터리·배선·쿠션 장착 완료) + 천장 카메라 설치 후**에 촬영해야 의미가 있습니다. 중간 형태로 찍으면 다시 찍어야 하므로, 차량 2대 제작과 카메라 설치가 끝나는 시점을 알려주시면 그때 맞춰 촬영하겠습니다.

### 3.3 확인 요청

1. **`ARRIVED` 제거 확인**
   도착 판정을 노트북이 수행하기로 확정했습니다. 펌웨어에서 `ARRIVED`/`EVENT_ACK`/`event_id` 계층을 빼도 되는지 확인 부탁드립니다. (SW 측은 하위호환으로, `ARRIVED` 를 받으면 `EVENT_ACK` 만 응답하고 무시합니다.)
   → 전환 방식: 노트북이 도착을 판정하면 `WAIT(reason=WAYPOINT_REACHED)` → 다음 `WAYPOINT` → `GO` 순으로 보냅니다. 펌웨어 §18.4 의 "WAITING 상태에서 target 교체" 규칙을 그대로 사용하므로 상태기계 변경은 없습니다.

2. **`command_result` / `rejected_seq` 필드 실제 송신 여부**
   SW 의 ACK 판정이 이 두 필드에 의존합니다. `session.py` 기준으로는 STATUS 에 포함되는 것으로 이해했는데, 실펌웨어 STATUS 에도 항상 포함되는지 확인 부탁드립니다.

3. **`MAX_ROUTE_ID` / `MAX_WAYPOINT_ID` / `MAX_COORDINATE_CM` 실제 값**
   `protocol.c` 에서 상수로만 참조되어 헤더에서 값을 찾지 못했습니다. SW 측은 임시로 route/waypoint 9999, 좌표 500cm 로 가정해 검증했습니다. 맵이 120×120cm 이므로 좌표 상한만 확인되면 충분합니다.

4. **`WAIT` 의 `route_id` 하한**
   주행 중이 아닐 때(경로 없음) `route_id=0, waypoint_id=0` 으로 보내고 있습니다. 파싱 하한이 0 이라 통과할 것으로 보이는데, 상태기계에서 문제 없는지 확인 부탁드립니다.

---

## 4. 통합 테스트 절차 제안

1. **1차 — 링크 확인**: SW 서버 기동 → ESP32 접속 → `HELLO_ACK` 승인 → `STATUS` 수신 → HEARTBEAT 유지 확인 (COMM_TIMEOUT 미발생)
2. **2차 — 단일 명령**: `WAIT` / `STOP` / `RESET` 각각 `ACCEPTED` 확인, 거절 시 사유 확인
3. **3차 — 단일 waypoint**: `POSE_UPDATE` 구현 후, waypoint 1개 전송 → 실제 주행 → 노트북 도착 판정 → 정지
4. **4차 — 전 구간**: A4 슬롯 CRUISE~FINAL 7개 waypoint 연속 주행
5. **5차 — 안전**: 주행 중 `WAIT` → 정지 확인 → 경로 재생성 → `GO` 재개

SW 측은 1·2차를 위한 서버가 준비된 상태입니다. 실차 없이 확인하려면 SW 저장소의
`comm/tests/mock_firmware.py` (펌웨어 파싱 규칙 이식본) 를 사용할 수 있습니다.

---

## 5. `car_id` 표기 관련 (제안 철회 근거)

앞서 SW 측이 `car_id` 정수 통일을 제안했으나, 검토 결과 **펌웨어의 `"CAR_01"` 유지가 낫다**고 판단했습니다.

- 펌웨어·브리지·문서·51개 테스트가 모두 문자열 전제로 작성되어 있어 변경 범위가 큽니다.
- SW 내부는 DB 기본키(`vehicle_id`, 정수)와 맞물려 정수를 써야 하는데, **직렬화 경계에서만 변환**하면 양쪽 모두 자연스럽습니다.
- 실제로 SW 측은 변환 함수 2개(`wire_car_id`/`parse_car_id`)만 두고 나머지 코드는 손대지 않았습니다.

참고로 향후 펌웨어를 정수로 바꾸더라도 SW 측 영향은 **함수 1개 수정**입니다 (검증 완료). 따라서 이 건은 HW팀 편의대로 결정하시면 됩니다.

---

## 6. SW 측 현재 상태

- 검증: `python -m unittest comm.tests.test_firmware_contract` — 15개 통과
  (핸드셰이크 / HEARTBEAT / 전 phase waypoint / WAIT enum / 멱등 재전송 / 거절 통지 / 주기 STATUS 오인 방지 / A4 전 구간 주행 후 PARKED)
- 준비 완료: 슬롯 배정(PPO), 경로·waypoint 생성, 충돌 감지, 도착 판정, 경로 재생성
- 통합 파이프라인(`pipeline/`) 결선 완료 — 카메라 프레임 → 탐지·추적 → 좌표·heading →
  슬롯 배정 → waypoint 전송 → 도착 판정 → 충돌 시 정지·재개 까지 한 루프로 동작.
  실카메라·실차 없이 검증 완료(9종). 실장비가 준비되면 카메라 소스와 캘리브레이션
  네 점만 교체하면 그대로 돌아갑니다.
- 대기 중: 카메라 실물 설치 후 Homography 캘리브레이션, RC카 2호기
