# SW팀 → HW팀 전달 (2026-08-07): 피드백 4건 반영 + B안 제어기 구현

이전 문서: `docs/hw_team_handoff_2026-08-04.md`
SW 저장소: `hanium-2026-project/backend`

---

## 1. 한 줄 요약

보내주신 **4건 모두 고쳤고**, **B안 제어기(노트북이 throttle/steering 계산)** 를 구현했습니다.
이제 `--direct-control` 플래그만 켜면 노트북이 100ms 주기로 `DIRECT_CONTROL` 을 내려보냅니다.
**먼저 `ENABLE_ACTUATOR_OUTPUT=0` 인 상태로 값만 확인**해 주시고, 아래 3절의 질문 3개에 답을 주시면 실차 튜닝으로 넘어갑니다.

---

## 2. 피드백 4건 반영 결과

| # | 지적사항 | 수정 내용 |
|---|---|---|
| 1 | `WAYPOINT` 수신 시 실물은 `READY → WAITING` | 목(`comm/tests/mock_firmware.py`)의 상태 전이를 실물에 맞췄습니다. 덤으로 목이 **200ms 주기 STATUS** 와 **HELLO 재전송** 도 하도록 해서 실물 송신 패턴과 같아졌습니다 |
| 2 | — | `HELLO_ACK=HOLD` 인데 노트북이 소켓을 닫아 복구 경로가 없었습니다. 이제 **HOLD 는 연결을 유지**하고 다음 `HELLO` 로 재판정합니다. `REJECTED` 만 닫습니다 |
| 3 | COMM_FAIL callback 이 같은 장애 동안 반복 발생 | **장애 시작 시 1회만** 통지하도록 바꾸고, 미션 로직에 결선했습니다 (아래 2.1) |
| 4 | — | `WAIT` 이 응답 대기 중인 `STOP` 을 취소할 수 있었습니다. **`STOP > WAIT > 일반`** 우선순위를 넣어 막았습니다 |

### 2.1 통신 장애 시 노트북 동작 (확인 부탁드립니다)

```
COMM_TIMEOUT 감지 → 미션 정지 (명령은 보내지 않음)
                  → 차량은 펌웨어 자체 safe-stop 에 맡김
STATUS 수신 재개  → 현재 카메라 pose 로 경로 재계획 (새 route_id)
                  → 기존 경로 재개(GO)는 하지 않음
```

**기존 경로를 재개하지 않는 이유**: 단절 동안 차가 얼마나 굴러갔는지 알 수 없어서 옛 waypoint 를 그대로 쓰면 위험합니다. 재접속(`HELLO`) 이든 링크 복구든 항상 재계획합니다.

또한 장애·재계획 시 **미응답 명령을 폐기**합니다. 안 그러면 복구 후 모든 명령이 "이전 명령 응답 대기 중" 으로 막힙니다.

---

## 3. B안 제어기 — 구현 완료, 확인 필요한 3가지

`control/waypoint_controller.py` 에 구현했습니다. 순수 계산 모듈이라 하드웨어 없이도 검증했습니다 (단위 테스트 22종 + 자전거 모델 수렴 시뮬레이션).

```
카메라 pose ─┐
             ├→ 거리·heading 오차 → throttle / steering → DIRECT_CONTROL (100ms)
목표 waypoint ┘
```

### 3.1 ❓ 부호 규약 — 가장 중요합니다

노트북은 이렇게 보냅니다. **반대면 알려주세요** (`--steering-sign -1` 로 바로 뒤집습니다).

| 값 | 규약 |
|---|---|
| `steering` **+1.0** | **좌회전** (heading 증가 방향 = 반시계) |
| `steering` **-1.0** | 우회전 |
| `steering` **0.0** | 직진 |
| `throttle` **0.0~1.0** | 전진. 현재 경로 설계는 후진을 쓰지 않습니다 |

`steering = ±1.0` 이 **서보 최대 조향각**에 대응한다고 가정했습니다. 실제 최대 조향각(°)을 알려주시면 제어기 정규화에 반영합니다 (현재 30° 가정).

### 3.2 ❓ `DIRECT_CONTROL` 이 어느 모드에서 동작하나요

노트북은 `HELLO_ACK` 직후 `SET_MODE REMOTE_DIRECT` 를 보내도록 해뒀습니다.

- **`WAYPOINT_AUTO` 모드에서도 `DIRECT_CONTROL` 이 먹는다면** → `SET_MODE` 를 생략하겠습니다
- **`REMOTE_DIRECT` 모드에서 `WAYPOINT`/`GO` 가 거절된다면** → 알려주세요. 지금은 `WAYPOINT`/`GO` 로 target·MOVING 상태를 잡고 그 위에 제어값을 흘리는 구조라, 거절되면 구조를 바꿔야 합니다

### 3.3 ❓ 스트림이 끊기면 어떻게 되나요

`DIRECT_CONTROL` 은 ack 없는 스트림(최신값만)이라 마지막 값이 남습니다.

- 펌웨어가 **일정 시간 `DIRECT_CONTROL` 이 없으면 자동으로 0 으로 떨어뜨리는지** 확인 부탁드립니다
- 없다면 넣어주시는 게 안전합니다 (권장 300~500ms). 노트북 쪽도 정지 구간에서는 `throttle=0` 을 명시적으로 계속 보냅니다

### 3.4 실측값 3개를 주시면 바로 반영합니다

기본값은 보수적 추정치입니다. 실측치로 바꾸면 정확도가 크게 올라갑니다.

| 파라미터 | 현재 가정 | 필요한 실측 |
|---|---|---|
| `min_throttle` | 0.22 | **최소 구동값** — 이보다 낮으면 안 움직이는 값 |
| `stop_distance_cm` | 3.0 | **정지 거리** — 목표 속도(4~12cm/s)에서 정지 명령 후 실제로 더 가는 거리 |
| `max_steer_deg` | 30.0 | **최대 조향각** |

---

## 4. 검증 순서 (제안하신 순서 그대로)

### 4.1 1단계 — 모터 OFF, 제어값만 확인

```bash
python manage.py run_pipeline --camera 0 --weights best.pt \
    --calibration calibration.json --port 5050 --show --direct-control
```

`ENABLE_ACTUATOR_OUTPUT=0` 상태로 붙여주세요. 노트북 화면에 차량마다 이렇게 뜹니다:

```
car1 DRIVING B1 wp3/7  남은거리 412mm
   DRIVE thr +0.31 str +0.42 err +18deg
```

조향 방향은 **화살표로도 표시**되니, 차를 손으로 옮겨가며 화살표가 목표 쪽을 향하는지 봐주시면 3.1의 부호 규약을 바로 확인할 수 있습니다.

ESP32 쪽에서는 `DIRECT_CONTROL` 이 100ms 주기로 들어오는지, `control_seq` 가 단조 증가하는지만 봐주세요.

### 4.2 2단계 — actuator ON, 단일 waypoint

`--max-throttle 0.25` 정도로 낮춰 시작하시길 권합니다.

```bash
python manage.py run_pipeline ... --direct-control --max-throttle 0.25
```

여기서 정지 거리를 실측하시면 3.4에 반영합니다.

### 4.3 3단계 — 복수 waypoint → 4단계 주차 한 사이클

노트북 쪽은 이미 전 구간(진입 → 슬롯 배정 → waypoint 7단계 → PARKED) 검증이 끝나 있어서, 제어값이 실차에서 동작하기 시작하면 바로 갈 수 있습니다.

---

## 5. 안전장치 (노트북이 자동으로 0 을 보내는 조건)

| 조건 | 사유 코드 |
|---|---|
| 카메라가 차량을 놓침 | `POSE_STALE` (0.5초 이상) |
| heading 을 모름 | `NO_HEADING` — 어디로 틀지 모르는 상태로는 안 움직입니다 |
| 충돌 위험 감지 | `DRIVE_NOT_ALLOWED` |
| 통신 장애 | `DRIVE_NOT_ALLOWED` (+ 펌웨어 자체 safe-stop) |
| waypoint 전환 중 | `DRIVE_NOT_ALLOWED` |
| 목표 도달 | `ARRIVED` |
| 위치는 맞는데 방향이 안 맞음 | `ALIGN` — 제자리 회전이 안 되므로 멈추고 재접근 경로를 다시 만듭니다 |

`throttle` 상한은 기본 **0.45** 로 묶어뒀습니다.

---

## 6. 네트워크 (실측 반영)

iPhone 핫스팟에서 STATUS 지연이 크다는 것 확인했습니다. **시연은 Windows 모바일 핫스팟 또는 전용 AP** 로 가는 것에 동의합니다.

포트는 여전히 조율이 필요합니다 — macOS 는 5000번을 AirPlay Receiver 가 점유해서 저희는 **5050** 으로 검증 중입니다 (`--port` 로 아무 값이나 맞출 수 있습니다).

---

## 7. 아직 답을 못 받은 것 (8/2 문서 §3.3)

1. `ARRIVED` 제거해도 되는지 (도착 판정은 노트북이 합니다)
2. `command_result` / `rejected_seq` 가 STATUS 에 항상 실려 오는지
3. `MAX_ROUTE_ID` / `MAX_WAYPOINT_ID` / `MAX_COORDINATE_CM` 실제 값
4. `WAIT` 의 `route_id=0` 허용 여부

B안으로 가면서 **`POSE_UPDATE` 수신 구현은 급하지 않게 됐습니다.** 1차 데모 이후 고도화 대상으로 미뤄도 됩니다.

---

# 부록: 실물 ESP32 검증 결과 (2026-08-10 새벽, SW팀 단독 확인)

USB 연결 상태에서 `ENABLE_ACTUATOR_OUTPUT` 0 → 1 로 올려가며 확인했습니다.
**3절의 질문 3개와 5절 실측 항목 일부가 여기서 닫혔습니다.**

## 확정된 것

| 항목 | 결과 |
|---|---|
| `SET_MODE REMOTE_DIRECT` | **ACCEPTED** (READY 상태에서만 수락) |
| `DIRECT_CONTROL` → 액추에이터 | **동작함** (`vehicle_control.c` → `actuator_apply_direct`) |
| **조향 부호** | **음수 = 좌회전 — 실물 앞바퀴로 확인 완료** |
| `REMOTE_DIRECT` 에서 `WAYPOINT` | **INVALID_STATE 로 거절** |
| 제어 스트림 단절 | **500ms 후 safeStop, MOVING → WAITING** |
| `MAX_COORDINATE_CM` 등 | `app_config.example.h` 에서 확인 (500.0 / route·waypoint id 각 100만) |

### 조향 스윕 실측 (서보 각도)

| wire steering | servo | 실제 방향 |
|---|---|---|
| -1.0 | 50.0° | 좌 최대 ✓ |
| -0.5 | 68.0° | 좌 |
| 0.0 | 86.0° | 중립 |
| +0.5 | 104.0° | 우 |
| +1.0 | 122.0° | 우 최대 |

### 참고 실측
- `throttle 0.3` → PWM **19/255** (직진 프로파일)

## 이것이 SW 설계에 미친 영향

**`WAYPOINT_AUTO` 와 `REMOTE_DIRECT` 는 상호 배타적입니다.** WAYPOINT 는
`CONTROL_MODE_WAYPOINT_AUTO` 에서만, DIRECT_CONTROL 은 `CONTROL_MODE_REMOTE_DIRECT`
에서만 처리됩니다. 따라서 "WAYPOINT/GO 로 상태를 몰면서 그 위에 제어값을 얹는"
구조는 실물에서 불가능하며, **`--control-mode auto-host` (host 내부 waypoint +
DIRECT_CONTROL 만 사용)가 유일한 경로**입니다.

## 아직 실측이 필요한 것 (현장에서만 가능)

1. **최소 구동값** — 실제로 바퀴가 도는 최소 throttle
2. **정지 거리** — 목표 속도에서 정지 명령 후 실제로 더 가는 거리
3. **최대 조향 바퀴각** — 서보 50~122° 가 실제 바퀴 몇 도인지

## 네트워크 재확인

아이폰 핫스팟으로 연동 시험 중 약 10초 만에 `COMM_TIMEOUT` 1회 발생 후 자동
재접속했습니다. 지적하신 아이폰 핫스팟 문제가 재현됩니다. **시연은 윈도우
모바일 핫스팟 또는 전용 AP로 갑니다.**

추가로, 아이폰 핫스팟은 macOS 클라이언트에 IPv6 전용 주소를 주는 반면 ESP32 는
IPv4(172.20.10.x)를 받아 서로 통신이 안 됩니다. 현장에서 이 조합은 피해 주세요.

---

# 부록 2: 자율주행 실차 측정 (2026-08-11 새벽)

카메라 → 제어기 → DIRECT_CONTROL → 실차 폐루프로 **단일 waypoint 자동 주행이
반복 성공**했습니다. 그 과정에서 나온 실측값입니다.

## 🔴 최소 선회 반경 57cm — 이게 지금 제일 큰 벽입니다

| 조건 | 최소 선회 반경 |
|---|---|
| 서보 ±36° (기존 50~122) | ~50~60cm |
| 서보 **±40°** (46~126 으로 확대) | **~57cm** |

**서보 운용각을 11% 넓혔는데 반경이 줄지 않았습니다.** 서보는 더 도는데
앞바퀴가 따라오지 않는다는 뜻입니다.

반경에서 실제 바퀴각을 역산하면 (자전거 모델 `R = L / tan δ`):

| 축거 | 실제 바퀴각 |
|---|---|
| 140mm | 13.8° |
| 160mm | 15.7° |
| 180mm | 17.5° |

**서보를 40° 돌려도 바퀴는 14~17° 밖에 안 꺾입니다.** 링키지가 이미 한계이거나
조향 암 기하가 서보각을 옮기지 못하는 것으로 보입니다.

### 왜 문제인가

바닥판이 **120×120cm** 인데 선회 지름이 **114cm** 입니다. 맵 한 변과 같아서
슬롯 진입 경로를 만들 수가 없습니다. 실제로 목표가 옆쪽에 있으면 차가 잡지
못하고 주위를 돕니다 (거리 11.6cm 까지 접근했다가 다시 벌어짐).

### 요청

1. **축거 실측값**을 알려주세요 (앞바퀴축 ↔ 뒷바퀴축)
2. **최대 조향 시 앞바퀴 각도**를 각도기로 재주세요 — 위 역산값과 맞는지
3. 링키지·서보 혼으로 바퀴각을 더 확보할 수 있는지 검토 부탁드립니다

소프트웨어로는 더 줄일 방법이 없습니다. 반경이 이대로면 **후진을 쓰는 주차
기동**으로 경로를 다시 설계해야 합니다 (현재 제어기는 전진 전용).

## 그 밖의 실측

| 항목 | 값 |
|---|---|
| 주행 속도 | 13~19 cm/s (강회전 duty 기준) |
| 도착 정확도 | 9.7~10.9cm (허용 8cm + 제동여유 3cm) |
| `throttle 0.3` → PWM | 19 (직진) — 계산식과 일치 |

## 🟡 회전 중 속도를 노트북이 제어할 수 없습니다

`throttle_to_duty()` 는 `|steering| > 0.5` 구간에서 duty 를 강회전 기본값으로
올리고 **throttle 을 무시**합니다. `|steering| = 1.0` 에서는 min = default 라
throttle 값과 무관하게 고정입니다.

실측에서도 `--max-throttle` 을 0.25 → 0.6 으로 올려도 전 구간 `thr 0.22` 로
동일했고, 결과 속도도 같았습니다.

**정할 것**: 회전 시 속도 책임을 어디가 지는지.
- (A) 펌웨어가 throttle 을 선형으로 받고, 회전 감속은 노트북이 계산 ← 권장
- (B) 지금 구조 유지 → 노트북 제어기의 회전 감속 로직을 제거해야 함
  (이중 보정이라 예측 불가)

## ⚠️ 캘리브레이션 값이 저장소에 없습니다

`app_config.h` 가 `.gitignore` 대상이라, 새로 세팅하는 사람은 전부
`app_config.example.h` 의 값을 씁니다. 저희가 실차 시험에 쓴 값은 아래인데
**저장소에는 반영돼 있지 않습니다.**

```c
#define SERVO_LEFT_STRONG_DEG  46.0     // example: 50.0
#define SERVO_LEFT_WEAK_DEG    66.0     // example: 68.0
#define SERVO_RIGHT_WEAK_DEG  106.0     // example: 104.0
#define SERVO_RIGHT_STRONG_DEG 126.0    // example: 122.0
#define PWM_FORWARD_DEFAULT     22      // example: 27
#define PWM_TURN_DEFAULT        40      // example: 45
#define PWM_STRONG_TURN_DEFAULT 50      // example: 55
```

`app_config.example.h` 를 실측값으로 갱신하거나, 캘리브레이션만 별도 헤더로
빼서 커밋해 주시면 좋겠습니다. `HW_CALIBRATION_2026-07-29.md` 도 7/30 실측과
값이 달라(30/60/86/112/122) 혼동을 줍니다.

## 로그

`tools/drive_logger.py` 로 10Hz CSV 를 남깁니다. `motor_pwm` 과
`servo_angle_deg` 는 STATUS 에 없어서(512바이트 상한) 펌웨어 매핑을 재현해
계산했고 컬럼명에 `_calc` 를 붙였습니다. **STATUS 에 이 두 값을 넣어주시면**
계산값이 아닌 실제값으로 검증할 수 있습니다.
