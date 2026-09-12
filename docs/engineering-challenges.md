# Engineering Challenges & Problem Solving

이 문서는 Hanium 프로젝트를 "YOLO + PPO 스마트 주차"라는 기술 나열이 아니라, **Perception → Planning(RL) → Control → Communication 으로 이어지는 하나의 closed-loop 시스템에서 실제로 무엇이 깨졌고, 어떻게 원인을 좁혀 고쳤는가**의 관점에서 정리한다.

아래 5개 사례는 backend(`hanium-2026-project/backend`)와 hardware(`hanium-2026-project/hardware`) 두 저장소의 `main` 브랜치 코드·커밋·테스트·문서에서 직접 확인된 것만 다룬다. 수치는 전부 코드 주석, 커밋 메시지, 테스트/문서에 실제로 적힌 것을 인용했고, 별도로 재현하지 않은 값은 "development measurement" 또는 "커밋 기록상 주장"으로 명시해 구분했다.

각 사례는 `Problem → Root Cause Analysis → Engineering Decision → Implementation → Validation → Lesson` 순서로 정리했다.

---

## 1. Production E2E Integration — Lifecycle 전반의 Failure Mode 분석

**시스템**: hardware repo, `docs/troubleshooting.md` / `docs/development_log.md` / `docs/test_log_summary.md`

### Problem
2026-09-07 production 통합에서 개별 모듈(인식/계획/제어/통신)은 각자 단위 테스트를 통과했지만, 실제 E2E 실행에서는 여러 계층이 동시에 얽히며 예상하지 못한 실패가 반복됐다. 8개의 독립적인 실패 모드가 문서화되어 있으며, 아래는 그중 시스템 lifecycle 전 구간(인식 신선도 → 계획 상태 → 제어 루프 → 통신 세션)을 대표하는 4가지다.

| 실패 모드 | 증상 |
|---|---|
| Route 후보 실패 전파 | 첫 슬롯 route 생성이 불가능하면 개별 후보 거부가 아니라 카메라/TCP 파이프라인 전체가 종료됨 |
| Stale heading으로 미션 시작 | 일반 제어용 heading fallback(`LAST_VALID`)이 계획(route 생성) 단계에서도 그대로 허용되어, 신뢰할 수 없는 방향으로 미션이 시작됨 |
| Waypoint overshoot 재계획 루프 | 핸드오프 지점을 20–25mm 지나친 뒤 동일 waypoint를 향해 반복 재계획하며 진행이 멈춤 |
| 통신 재접속 후 stale 세션 재사용 | `COMM_TIMEOUT` 이후 재연결됐다는 사실만으로 이전 세션의 명령/상태를 그대로 재사용할 위험 |

### Root Cause Analysis
공통 원인은 "각 계층이 자기 계약만 지키면 된다"는 암묵적 가정이었다.
- Route 실패: 예외(`InfeasibleRouteError`)가 후보 단위 경계를 넘어 파이프라인 레벨까지 전파됨.
- Stale heading: 제어 루프의 관대한 fallback 정책이 계획 단계의 엄격한 신뢰 요구와 구분되지 않음.
- Overshoot 루프: "정확히 그 지점 재획득"을 요구하면, 관성으로 지나친 차량이 같은 지점을 향해 영원히 재시도함 — 종료 조건이 위치 일치가 아니라 진행 여부여야 했다.
- Stale 세션: TCP 연결 복구는 물리적 안전 상태를 보장하지 않는다 — 소켓 재연결과 미션 컨텍스트 복구가 결합되어 있지 않았다.

### Engineering Decision
"복구 가능한 실패는 국소적으로, 불확실한 상태는 항상 정지 후 재확인"을 계층 공통 원칙으로 세웠다.

### Implementation
- Route 실패: 실패한 후보만 거부하고 다음 슬롯을 시도, 전부 실패하면 zero + WAIT/FAULT로 안전 정지.
- Stale heading: 초기 배정/핸드오프/후면주차/recovery 경로 생성 시 fresh/trusted heading을 별도로 요구.
- Overshoot: terminal capture → STOP → fresh Pose 확보 → 다음 단계 전환 순서로 반복 재계획을 원천 차단.
- Stale 세션: zero latch → 이전 세션 폐기 → HELLO 재협상 → RESET → SET_MODE → fresh Pose/heading 확인 → 현재 위치 기준 재계획.

### Validation
`docs/troubleshooting.md`와 `docs/development_log.md` 양쪽에 동일 문제-해결 쌍이 교차 문서화되어 있다. `docs/test_log_summary.md`에는 5개의 실제 차량 run(`run_20260831_002703`, `run_20260903_230921`, `run_20260904_000722`, `run_20260904_183055`, `run_20260904_183503`)이 PARKED 도달·recovery 시나리오와 함께 기록되어 있으나, 위 8개 failure mode 각각이 특정 run ID로 1:1 추적되지는 않는다 — "실제 E2E로 확인됨"이라는 문서 서술까지가 현재 근거의 범위다.

### Lesson
개별 모듈 테스트 통과는 시스템이 동작한다는 증거가 아니다. 실패는 항상 "어느 계층의 가정이 다른 계층의 가정과 충돌했는가"의 형태로 나타났고, 고립된 버그 수정이 아니라 계층 간 계약(fresh 여부, 실패 전파 범위, 재연결 시 상태 소유권)을 다시 정의해야 했다.

**Evidence**: `docs/troubleshooting.md`(2026-09-07 production 통합 문제 요약), `docs/development_log.md`(주요 실차 blocker와 해결 방향), `docs/test_log_summary.md`

---

## 2. FINAL Reverse Blind-Travel Overshoot

**시스템**: backend repo, `controller/pose_controller.py`

### Problem
실측 run `run_20260901_154551`에서, 목표까지 65mm 남은 상태로 후진 중이던 차량이 **547ms 동안 새 카메라 관측 없이 78.7mm를 더 이동**해 맵 경계 밖 56.8mm로 이탈했다. 이때 `BOUNDARY_HARD`가 먼저 발동해 `FINAL_POSE_EVAL` 자체가 실행되지 못했다.

### Root Cause Analysis
기존 안전 게이트 `max_pose_age_s`는 **시간만** 제한하고 그 시간 동안 실제로 얼마나 이동할 수 있는지는 고려하지 않았다. 코드 주석에 남은 실측 근거:
- FINAL 후진 실측 속도 약 **134mm/s** (요구 속도 40mm/s의 3.3배)
- 슬롯 깊이 여유 **25mm**
- 카메라가 정상(4.3fps, 프레임 간 233ms) 동작해도 한 프레임 사이 약 **31mm** 이동 — 이미 여유(25mm)를 초과
- `max_pose_age_s`(500ms)를 다 쓰면 최대 67mm 이동 — 여유의 2.7배

즉 정상적인 카메라 프레임레이트에서도 "시간 기준" 게이트만으로는 종단 근처의 여유 공간을 지킬 수 없는 구조적 문제였다.

### Engineering Decision
"관측 없이 목표를 지나칠 수 있으면 움직이지 않는다"는 **거리 기반** 계약을 세우고, 이를 정지가 필요한 FINAL 단계에만 한정 적용한다 (통과해야 하는 중간 waypoint에 걸면 진행 자체가 끊긴다).

### Implementation
`controller/pose_controller.py`에 blind-travel guard를 추가: 허용 가능한 "관측 없는 이동 거리"를 목표까지 남은 거리와 비교해, 멀면 여유 있게 허용하고 목표에 가까워질수록 0으로 수렴시켜 종점 근처에서는 신선한 관측을 강제한다(`POSE_BLIND_TRAVEL` 정지 사유). `max_pose_age_s` 자체는 낮추지 않아 다른 구간의 동작은 그대로 유지했다.

### Validation
`controller/tests/test_pose_controller.py`(16 tests, `docs/autohost/TEST_REPORT.md`에 스위트 목록으로 편입)로 회귀 커버됨. 단, `run_20260901_154551`은 코드 주석에 남은 실측 사례이지 이 fix 이후의 "재현 안 됨"을 보여주는 별도의 before/after 비교 로그는 저장소에서 확인되지 않았다 — unit test 통과와 실차 재현 여부는 구분해서 이해해야 한다.

### Lesson
"시간 기준" 안전장치는 그 시간 동안의 실제 물리적 이동량을 반드시 함께 검토해야 한다. 카메라 fps, 실측 속도, 물리적 여유 공간 세 가지를 하나의 부등식으로 엮어야 실제로 안전한 임계값이 나온다는 것을, 실패 사례를 역산해서 확인했다.

**Evidence**: `controller/pose_controller.py`(blind travel 공간 계약 블록, `_final_terminal_geometry`/`_is_arrival_candidate`), `controller/tests/test_pose_controller.py`, `docs/autohost/TEST_REPORT.md`

---

## 3. Geometry-Derived Final Parking Tolerance

**시스템**: backend repo, `parking/final_alignment.py`

### Problem
Waypoint 도착이 곧 주차 완료(PARKED)가 아니다. 위치에 도착해도 차체가 비스듬하면 물리적으로 슬롯을 벗어날 수 있는데, "허용 오차"를 얼마로 잡아야 하는지가 임의의 hyperparameter가 되기 쉬운 지점이었다.

### Root Cause Analysis
실측 바닥판 기준 슬롯은 200mm(폭)×300mm(깊이), 차량은 250mm(길이)×150mm(폭) — 완벽히 정렬해도 좌우 여유는 각 25mm뿐이다. heading 오차 θ일 때 차체가 슬롯 폭 방향으로 차지하는 반폭은 `125·sin|θ| + 75·cos|θ|`이며, 코드에 남은 표는 다음과 같다.

| θ | 반폭 | 남는 여유 |
|---|---|---|
| 0° | 75.0mm | 25.0mm |
| 5° | 85.6mm | 14.4mm |
| 10° | 95.6mm | 4.4mm |
| 12° | 99.4mm | 0.6mm |
| 13° | 101.2mm | 초과(슬롯 이탈) |

기존에 쓰이던 12° 허용오차는 이 표에서 보듯 횡방향 여유를 **거의 전부 소진**하는 값이었다 — 사실상 안전 여유가 없는 임계값을 관용적으로 쓰고 있었다는 뜻이다.

### Engineering Decision
허용오차를 임의로 정하지 않고, "횡오차를 약 15mm까지는 허용한다"는 물리적 목표를 먼저 세운 뒤 위 공식을 역산해 임계각을 도출한다. 판정 자체도 각도 하나가 아니라 실제 차체 footprint 다각형과 슬롯 다각형의 포함 관계로 수행한다.

### Implementation
`125·sinθ + 75·cosθ ≤ 85`의 해가 약 4.7°이며, 여기서 `ALIGNED_HEADING_TOLERANCE_DEG = 4.5`로 정했다(코드 주석에 유도 과정 포함). 이 각도보다 heading이 틀어져 있으면 직선 후진으로 회복 불가능하다고 보고 `FINAL_ALIGNMENT`로 보낸다.

추가로 "직선 후진이 흡수할 수 있는 횡오차"도 임의로 정하지 않고, closed-loop 실측(staging 275mm 구간)으로 확인했다:

| 시작 횡오차 | 끝 횡오차 / 끝 heading 오차 | 판정 |
|---|---|---|
| 40mm | 13.0mm / 10.9° | 불합격 (4.5° 기준 초과) |
| 20mm | 3.9mm / 4.1° | 합격 |
| 10mm | 3.3mm / 2.7° | 합격 |
| 5mm | 1.6mm / 1.4° | 합격 |

이 실측으로 "20mm 이 실측상 한계"임을 확인하고, 정렬 목표(`FINAL_LATERAL_BAND_MM`)와 직선후진 수용 한계(`STRAIGHT_REVERSE_LATERAL_LIMIT_MM`)를 **같은 값(20.0mm)**으로 묶었다 — 두 값이 어긋나면 "정렬 → 아직 크다고 판단 → 다시 정렬"을 반복하는 진동이 생기기 때문이다.

### Validation
위 두 표 모두 `parking/final_alignment.py` 모듈 docstring에 유도 과정·실측치와 함께 그대로 남아 있다. 이 문서화 자체가 검증 근거다 — 다만 이 표를 산출한 실측 run의 별도 로그 파일이나 자동화된 재현 테스트명은 이번 조사에서 확인하지 못했다.

### Lesson
Control threshold는 "대략 이 정도면 되겠지"가 아니라 실제 기구/슬롯 치수에서 역산해야 하고, 두 개의 연관된 임계값(정렬 목표·후진 수용 한계)은 독립적으로 정하면 진동을 만든다는 것을 기하학적으로, 그리고 실측으로 확인했다.

**Evidence**: `parking/final_alignment.py`(모듈 docstring 전체, `ALIGNED_HEADING_TOLERANCE_DEG`, `FINAL_LATERAL_BAND_MM`, `STRAIGHT_REVERSE_LATERAL_LIMIT_MM`)

---

## 4. PPO Reward Design Failure — Over-Conflict-Avoidance와 WAIT Deadlock

**시스템**: backend repo, `rl/reward.py`, `rl/sweep_wait_penalty.py`

### Problem
두 개의 독립된 실패가 같은 근본 이슈(보상 신호의 불균형)에서 비롯되어 하나의 사례로 묶인다.

**A. 과도한 충돌 회피** — v1 보상에서 `CONFLICT_PENALTY = -10`이 다른 모든 신호를 압도해, PPO가 "충돌 회피 전문가"가 되어 지나치게 WAIT를 선택하고 처리량(throughput)을 희생시켰다. 정작 핵심 목표인 처리량 자체에는 보상 신호가 전혀 없었다.

**B. WAIT 남용 deadlock** — `WAIT_PENALTY_BASE = -0.2`로 설정했을 때, WAIT가 항상 합리적인 선택이 되어 에이전트가 사실상 아무 행동도 하지 않는 상태로 수렴했다.

### Root Cause Analysis
A: `rl/reward.py` 커밋 메시지(`601095d`)에 원인이 직접 명시되어 있다 — "CONFLICT_PENALTY=-10 dominated all signals... The core goal (flow efficiency / vehicles processed) had zero reward signal."

B: `rl/sweep_wait_penalty.py` docstring에 산술적 근거가 그대로 남아 있다. `WAIT_PENALTY_BASE=-0.2`, 충돌 페널티 -10 기준:
```
5회 연속 WAIT = -0.2 -0.3 -0.4 -0.5 -0.6 = -2.0
충돌 1회 회피 = +10.0
```
→ WAIT의 누적 비용(-2.0)이 충돌 회피 이득(+10.0)보다 항상 작아, 에이전트가 구조적으로 WAIT를 선호하게 된다.

### Engineering Decision
"충돌은 목표가 아니라 처리량을 해치는 수단(means)이다"라는 관점으로 보상 구조를 재설계한다 — 충돌 페널티를 낮춰 다른 신호와 경쟁하게 하고, 처리량 자체에 대한 명시적 보상을 추가한다. WAIT penalty는 breakeven 지점을 찾기 위해 여러 값을 스윕한다.

### Implementation
`rl/reward.py` v2:
- `CONFLICT_PENALTY: -10 → -5`
- `R_flow` 신규 추가(δ=0.8) — 슬롯 재사용 보너스(`SLOT_REUSE_BONUS=0.5`, 해당 슬롯에서 이미 한 번 이상 배정이 있었다면 = 회전이 발생했다는 뜻)와 빈 슬롯 분산 보너스(`FREE_SLOT_BONUS_MAX=0.3`, 남은 빈 슬롯 비율에 비례)로 구성.
- `R_efficiency`(α=1.0, 거리 기반), `R_congestion`(β=0.5, 병목 부하)은 변경 없음.

`rl/sweep_wait_penalty.py`: `WAIT_PENALTY_BASE ∈ {-0.2, -1.0, -2.0, -3.0}`를 학습+평가 전체 파이프라인으로 스윕해 conflict-vs-throughput trade-off의 breakeven 지점을 탐색.

### Validation
커밋 `601095d` 메시지에 "PPO doesn't sacrifice 9% throughput just to shave 5% off conflict rate"라는 development measurement가 기록되어 있다. **이 9%/5% 수치는 커밋 히스토리에 남은 개발 당시 측정 기록이며, 이 조사에서 별도로 재현하거나 독립된 벤치마크 로그로 확인한 것이 아니다** — 확정된 성능 지표로 인용하지 않는다. WAIT-penalty 스윕이 최종적으로 어떤 값을 채택했는지를 보여주는 결과 파일은 main에서 확인되지 않았다.

**중요**: 이 PPO 정책이 실제 RC카 주행에서 검증됐다고 서술하지 않는다. `docs/test_log_summary.md`(hardware repo)에 따르면 최종 실차 검증 시점에는 `sb3-contrib`가 배포 환경에 없어 **PPO가 아니라 deterministic heuristic fallback이 실제로 실행**되었다.

### Lesson
보상 함수의 한 항이 다른 항을 수치적으로 압도하면, 정책은 "설계자가 원한 목표"가 아니라 "보상이 실제로 가리키는 목표"를 정확히 학습한다. 실패를 사후에 설명하는 대신 산술적으로(deadlock의 경우 페널티 누적값과 이득값을 직접 비교) 원인을 특정한 뒤에 보상을 재설계했다.

**Evidence**: `rl/reward.py`(모듈 docstring, `CONFLICT_PENALTY`/`SLOT_REUSE_BONUS`/`FREE_SLOT_BONUS_MAX`), `rl/sweep_wait_penalty.py`(docstring), commit `601095d`, `docs/test_log_summary.md`(hardware repo, PPO 미실행·heuristic fallback 명시)

---

## 5. Vision Occupancy → Planning Failure

**시스템**: backend repo, `pipeline/runner.py`, `pipeline/tests/test_vision_static_obstacle.py`, `pipeline/tests/test_vision_slot_occupancy.py`

### Problem
전원이 꺼진 채 슬롯에 세워둔 차량은 `rc_car`는 매 프레임 검출되지만 `front_cushion`(전면 마커)이 거의 검출되지 않는다. 실측 run `run_20260906_190856`/`_191008`에서 해당 차량의 `heading_source`가 각각 프레임의 74%/95%에서 `LAST_VALID`로 고착됐다(반대로 `front_cushion`이 실제 검출된 비율은 각 25.9%/4.7% — 두 수치는 서로 보완적 관계로, 코드의 두 지점에 각각 남아 있다). 그 결과 "자세를 신뢰할 수 없는 차량"으로 분류되어, **planner가 생성하는 모든 route/recovery가 `OTHER_VEHICLE_POSE_UNCERTAIN`으로 거절**됐다 — 두 run 모두 정확히 이 사유로 끝났다(`190856 t=18.358 RECOVERY_REJECTED OTHER_VEHICLE_POSE_UNCERTAIN`, `191008 t=29.788` 동일).

### Root Cause Analysis
정적 장애물(움직이지 않는 차)과 이동 중인 차량에 **동일한 heading 신뢰 기준**을 요구하고 있었다. 하지만 이미 확정 주차된(PARKED) 차량에는 "마지막으로 검증된 정적 자세를 쓰고 heading 신선도를 요구하지 않는다"는 별도의 처리(`_parked_obstacles`)가 존재했다 — 전원 꺼진 정차 차량도 본질적으로 같은 범주인데 그 처리 경로를 타지 못하고 있었을 뿐이다.

### Engineering Decision
"확정된 슬롯의 차는 자세를 모르는 차가 아니라 그 슬롯에 세워둔 차"라는 관점으로 재정의한다. 장애물 목록에서 제외하는 게 아니라, 믿을 수 없는 heading 축 하나만 슬롯의 주차 방향으로 대체한다.

### Implementation
`pipeline/runner.py`의 `_update_vision_occupancy()`가 카메라 pose와 슬롯 기하만으로(CAR_ID 바인딩이나 ESP 연결 불필요) 정차 여부를 판정해 `RealtimeAllocator.set_vision_occupied()`로 슬롯 상태에 vision overlay를 얹는다(`effective_slot_statuses = max(slot_statuses, vision_occupied)`, base 예약/PARKED 상태는 건드리지 않음). 계획 단계에서는 이 vision-확정 차량을 `_parked_obstacles`와 같은 정적 의미로 취급하되, heading만 슬롯 방향으로 대체한다. 판정 조건(`in_final_region`, `is_stationary`, 연속 관측 횟수)은 전부 기존 계약을 재사용했다.

판정 신뢰도는 `vision_occupancy_enabled=False`로 끄면 배정·계획 모두 기존 동작으로 완전히 되돌아가도록 플래그로 게이트했다.

### Validation
전용 테스트 스위트 `pipeline/tests/test_vision_static_obstacle.py`, `pipeline/tests/test_vision_slot_occupancy.py`가 실측 run의 로그 발췌(위 타임스탬프)를 테스트 docstring에 직접 인용하며 동일 시나리오를 회귀 테스트로 고정하고 있다.

**명시적으로 남겨진 한계**: 정적 차량의 점유 확정에는 실측상 약 4.5초가 걸리며, 그 전에 자율주행 차량 배정이 일어나면 아직 확정되지 않은 슬롯이 배정될 수 있다(실측 0.4~0.8초 차이로 경합 발생 가능). **이 경합을 막는 allocation gate는 이번 변경에 포함되지 않았다** — 코드 주석과 문서에 제약으로 명시되어 있으며, 이 문서에서도 그대로 밝힌다.

### Lesson
CV 단계의 불확실성(front_cushion 미검출)은 CV 문제로 끝나지 않고, 다운스트림 planning 로직의 암묵적 가정(모든 차는 신뢰 가능한 heading이 있어야 한다)과 만나 시스템 전체를 멈추는 실패로 전파됐다. 해결책은 새 CV 모델이 아니라, 이미 존재하던 "정적 차량" 개념을 정차 차량까지 일관되게 확장하는 것이었다.

**Evidence**: commit `15043f3`, `pipeline/runner.py`(`_update_vision_occupancy`, line ~502–561), `pipeline/config.py`(`vision_occupancy_enabled`), `pipeline/tests/test_vision_static_obstacle.py`, `pipeline/tests/test_vision_slot_occupancy.py`

---

## Additional Engineering Fixes

아래는 위 5개만큼 길게 다루지는 않지만, 실제 repository evidence가 있는 추가 문제 해결 사례다.

- **Host/Firmware Configuration Drift** — host 쪽 `FirmwareConstants`(PWM 상수)가 firmware(`app_config.example.h`)의 실제 calibration 값과 어긋난 채 방치됐다. 두 파일이 서로 다른 관심사(호스트 설정 vs 펌웨어 소스)라 git merge conflict로 드러나지 않는, "문법적으로는 조용하지만 의미적으로 틀린" 종류의 버그였다. 커밋 `cb15e2d`로 host 값을 firmware 기준으로 재동기화했고, hardware repo `docs/troubleshooting.md`에 "firmware 상수를 바꾸면 FirmwareConstants도 반드시 같이 확인한다"는 재발 방지 경고를 영구히 남겼다.
- **Communication Reconnect Infinite Loop** — 통신이 한 번 끊기면 `EMERGENCY_STOP` 이력 때문에 재연결이 다시 실패하는 복구 불가 고리가 실차에서 발견되어 커밋 `84faa33`으로 수정했다.
- **Bidirectional Safety Shield** — RL 충돌 검사가 입차 경로에만 적용되고 출차(exit) 경로는 빠져 있던 버그를 `_check_conflict_excluding()`으로 수정(커밋 `33b2e8e`), `_process_departures()`에서 출차 차량도 동일하게 충돌 검사를 받도록 했다.
- **Slot Topology Sim-to-Real Correction** — 시뮬레이션 그래프가 실제 L자형 단일차선 테스트베드 배치와 어긋나 있던 문제를 4개 커밋(`3012d00`, `5b3fe14`, `e358c8f`, `189acdf`)에 걸쳐 순차적으로 수정, A/B 행 배치를 실제 입구 기준으로 재정렬했다.
- **ML Dependency Fallback** — `sb3-contrib`(MaskablePPO 의존성)가 배포 환경에 없어도 카메라·제어 루프 전체가 죽지 않도록, `rl/inference.py`의 `load_policy()`가 예외 대신 `None`을 반환하고 `select_action()`이 규칙 기반 `heuristic_policy()`(v2~v4)로 투명하게 전환되도록 설계했다. 이 설계가 실제로 의미가 있었다는 것은, 위 4번 사례에서 언급했듯 최종 실차 검증에서 정확히 이 경로(fallback)가 실행됐다는 사실로 확인된다.

---

## 참고

이 문서는 hardware repo의 `docs/troubleshooting.md`, `docs/development_log.md`, `docs/test_log_summary.md`를 대체하지 않는다 — 그 문서들은 원본 상세 기록으로 그대로 유지되며, 이 문서는 포트폴리오 관점에서 대표 사례를 선별·요약·연결하는 상위 문서다.
