# SW팀 → HW팀 전달 (2026-08-04): 실카메라 검증 완료, ESP32 연동 준비됨

이전 문서: `docs/hw_team_handoff_2026-08-02.md` (통신 계약 정합 내용은 그대로 유효)
SW 저장소: `hanium-2026-project/backend` @ `develop`

---

## 1. 한 줄 요약

**노트북 쪽은 실제 카메라로 전 구간 검증이 끝났습니다.** 차량이 접속만 하면 슬롯 배정부터 주차 완료 판정까지 동작합니다. 지금 남은 건 ESP32 쪽 `POSE_UPDATE` 수신 구현 하나이고, 그 전에 할 수 있는 링크 시험이 있으니 아래 3절부터 시도해 주세요.

---

## 2. 8/2 이후 SW 측 진행 상황

### 2.1 완료된 것

| 항목 | 결과 |
|---|---|
| 천장 카메라 설치·캘리브레이션 | 바닥판 1200×1200mm 실측, 슬롯 8칸 좌표 일치 확인 |
| 전방 쿠션 학습 | 2클래스 모델(`rc_car` + `front_cushion`) mAP50 0.98 |
| **정지 상태 heading** | **확보됨** — 쿠션 기반이라 차가 멈춰 있어도 방향을 압니다 |
| 실카메라 전 구간 | 접속 → 슬롯 배정 → waypoint 7단계 → 주차 완료 판정, **최종 오차 14mm** |

검증 로그 (가짜 ESP32를 붙여 실제 카메라로 측정):

```
state=READY
state=MOVING  route=1 wp=2 CRUISE
state=MOVING  route=1 wp=3 CRUISE
state=MOVING  route=1 wp=4 APPROACH
state=MOVING  route=1 wp=5 ALIGN
state=MOVING  route=1 wp=6 ENTRY
state=MOVING  route=1 wp=7 FINAL
state=WAITING (FINAL_WAYPOINT_REACHED)   ← 정지 확인 단계
→ PARKED 확정
```

계약 위반(필드 누락·범위 초과) **0건**으로 통과했습니다.

### 2.2 슬롯 배치가 바뀌었습니다 (확인 요망)

실물 바닥판을 캘리브레이션하다 발견했습니다. **입구에서 가까운 줄이 A행**입니다.

| 행 | 위치 | 좌표 (mm) |
|---|---|---|
| A1~A4 | 입구 쪽 (아래) | y = 150, x = 425 / 650 / 875 / 1100 |
| B1~B4 | 출구 쪽 (위) | y = 1050, x = 425 / 650 / 875 / 1100 |

번호는 양쪽 다 **입구에서 가까운 순으로 1→4**입니다. 코드·DB 모두 이 기준으로 통일했습니다. 펌웨어에서 슬롯 이름을 별도로 갖고 있다면 맞춰 주세요 (없다면 무관 — 좌표만 받으므로).

### 2.3 쿠션 색상 변경

검정 브래킷은 차체·바닥과 대비가 부족해 탐지가 불안정했습니다. **흰색 폼을 씌워** 해결했고, 그 상태로 학습·검증을 마쳤습니다. 2호기 제작 시 동일하게 **흰색(또는 원색, 노랑 제외)** 으로 부탁드립니다.

---

## 3. 지금 바로 시도해 주실 것 (POSE_UPDATE 없이 가능)

### 3.1 1차 — 링크 시험 (오늘이라도 가능)

노트북 서버는 준비돼 있습니다. ESP32를 접속시켜 아래가 되는지만 봐 주세요.

**노트북 쪽 실행** (저희가 실행합니다):
```bash
python manage.py run_pipeline --camera 0 --weights best.pt \
    --calibration calibration.json --port 5050 --show
```

**확인 항목**

| # | 확인할 것 | 정상 동작 |
|---|---|---|
| 1 | ESP32 → 노트북 TCP 접속 | 노트북에 `car 1 ready` 로그 |
| 2 | `HELLO` → `HELLO_ACK` | ESP32가 `READY` 상태로 전환 |
| 3 | `STATUS` 주기 송신 | 노트북이 상태를 계속 받음 |
| 4 | `HEARTBEAT` 수신 | 노트북이 250ms 주기로 보냅니다. **끊기면 1초 뒤 COMM_TIMEOUT** 이 정상 |
| 5 | 통신 끊기 테스트 | Wi-Fi 끄면 차량이 안전정지하는지 |

이 단계만 통과해도 **프로토콜 계약이 실물끼리 맞는다**는 게 확인됩니다.

### 3.2 2차 — 명령 수신 시험 (POSE_UPDATE 없이 가능)

노트북에서 `WAIT` / `STOP` / `RESET` 을 수동으로 보내겠습니다. 각각에 대해:

- `command_result` 가 `ACCEPTED` 로 오는지
- 거절 시 사유(`INVALID_STATE` 등)가 실려 오는지
- `STOP` 후 `RESET` 으로만 복구되는지 (GO로는 안 풀려야 정상)

**주의**: 실제 모터를 돌리지 않는 상태(`ENABLE_ACTUATOR_OUTPUT 0`)로 먼저 하시는 걸 권합니다.

### 3.3 3차 — `WAYPOINT` 수신 시험 (POSE_UPDATE 없이 가능)

노트북이 실제 waypoint를 보냅니다. **추종하지 않아도 됩니다.** 파싱만 되는지 보면 됩니다.

실제로 전송되는 형태:
```json
{"version":1,"type":"WAYPOINT","car_id":"CAR_01","session_id":"S9C74D034","seq":3,
 "route_id":1,"waypoint_id":4,"phase":"APPROACH",
 "x_cm":42.5,"y_cm":78.0,"target_heading_deg":270.0,
 "motion_direction":"FORWARD","arrival_mode":"STOP",
 "speed_cm_s":8.0,"position_tolerance_cm":6.0,"heading_tolerance_deg":20.0,
 "heading_required":false,"is_final":false}
```

- 파싱 성공 → `target` 저장 → `WAITING` 유지되는지
- 필드 거절이 나면 어떤 필드인지 알려 주세요 (저희가 맞추겠습니다)

---

## 4. 반드시 필요한 것: `POSE_UPDATE` 수신 구현

**이게 자동주행의 유일한 선행조건입니다.** ESP32는 자기 위치를 모르기 때문에, 노트북이 카메라로 측정한 위치를 계속 내려줘야 waypoint를 추종할 수 있습니다.

현재 `protocol.c` 의 수신 타입 분기에 `POSE_UPDATE` 가 없어서, 저희 쪽은 송신을 꺼둔 상태입니다 (`comm/protocol.py` 의 `POSE_UPDATE_ENABLED = False`). 펌웨어에 추가되면 **플래그만 켜면 됩니다.**

송신할 형태 (통합 문서 §19 그대로):

```json
{"version":1,"type":"POSE_UPDATE","car_id":"CAR_01","session_id":"S9C74D034",
 "pose_seq":5021,"x_cm":42.5,"y_cm":76.0,"heading_deg":91.2,
 "position_confidence":0.96,"heading_confidence":0.90,
 "heading_source":"FRONT_CUSHION","measurement_age_ms":42,"valid":true}
```

| 필드 | 처리 방법 |
|---|---|
| `pose_seq` | 이 값이 큰 것만 사용 (오래된 건 버림). ACK·재전송 없음 |
| `x_cm`, `y_cm` | 바닥판 좌하단 원점, 우측 +x, 위쪽 +y |
| `heading_deg` | 0~360°, 오른쪽 0° / 위쪽 90° |
| `heading_source` | `FRONT_CUSHION`(정확) / `TRAJECTORY` / `LAST_VALID`(추정) |
| `valid` | **`false` 면 제어에 쓰지 마세요** (카메라가 놓친 프레임) |

- 송신 주기: 100ms (조정 가능)
- 최신값만 유지하면 됩니다 (길이 1 큐 또는 구조체)

---

## 5. 확인·조율 필요 (짧은 것들)

### 5.1 포트 번호 ⚠️

펌웨어 `SERVER_PORT` 가 **5000**인데, **macOS 는 5000번을 AirPlay Receiver 가 점유**합니다. 저희 검증은 **5050**으로 했습니다.

- 시연 노트북이 macOS 라면 → 펌웨어를 5050(또는 다른 번호)으로 바꾸는 게 안전합니다
- 또는 노트북에서 AirPlay Receiver 를 끄고 5000 유지

어느 쪽으로 할지 정해 주세요. 저희는 `--port` 로 아무 값이나 맞출 수 있습니다.

### 5.2 이전 문서에서 아직 답 못 받은 것

`hw_team_handoff_2026-08-02.md` §3.3 의 4가지입니다.

1. **`ARRIVED` 제거해도 되는지** — 도착 판정은 노트북이 합니다. 펌웨어에서 `ARRIVED`/`EVENT_ACK` 계층을 빼면 구현이 꽤 줄어듭니다
2. **`command_result` / `rejected_seq` 가 STATUS 에 항상 실려 오는지** — 저희 ACK 판정이 이 두 필드에 의존합니다
3. **`MAX_ROUTE_ID` / `MAX_WAYPOINT_ID` / `MAX_COORDINATE_CM` 실제 값** — 맵이 120×120cm 이니 좌표 상한만 알면 됩니다
4. **`WAIT` 의 `route_id=0` 허용 여부** — 주행 중이 아닐 때 0으로 보냅니다

### 5.3 실차 주행 특성 (엔코더 관련)

waypoint 추종을 시작하면 필요합니다.

- **정지 거리**: 목표 속도(4~12cm/s)에서 STOP 후 실제로 몇 cm 더 가는지
- **엔코더 환산**: 1회전당 count, count당 이동거리(cm)
- **최소 구동 속도**: 이보다 느리면 안 움직이는 PWM 하한

정지 거리는 특히 중요합니다. 저희 도착 판정 허용오차가 phase별로 4~8cm 인데, 실제 정지 거리가 이보다 크면 계속 지나치게 됩니다.

---

## 6. 하드웨어 없이 그쪽에서 미리 해볼 수 있는 것

저희 저장소에 **가짜 노트북 서버 역할을 할 수 있는 것**이 없어서, 대신 이렇게 하실 수 있습니다.

**`comm/tests/mock_firmware.py`** — 저희가 펌웨어 계약 검증에 쓰는 목입니다. `protocol.c` 의 필수 필드 검증을 그대로 이식해서, 필드가 빠지면 실제 펌웨어처럼 거절합니다. 펌웨어 수정 후 이 목과 비교해 보시면 계약 불일치를 미리 잡을 수 있습니다.

노트북 서버를 띄운 상태로 시험하고 싶으시면 언제든 말씀해 주세요. 저희가 서버만 올려두겠습니다 (카메라 없어도 실행됩니다).

---

## 7. 현재 병목과 일정 감각

```
[완료] 카메라·인식·좌표·슬롯배정·waypoint·도착판정   ← SW, 실물 검증 끝
[대기] POSE_UPDATE 구현                              ← HW, 자동주행의 유일한 선행조건
[대기] 실차 주행 튜닝 (정지거리·tolerance)            ← 위가 끝나야 시작
[대기] 2호기 → 다중 차량 시나리오
```

3절의 1~3차 시험은 `POSE_UPDATE` 와 **병렬로 진행 가능**합니다. 링크·명령 시험을 먼저 통과시켜 두면, `POSE_UPDATE` 가 붙는 즉시 자동주행 시험으로 넘어갈 수 있습니다.
