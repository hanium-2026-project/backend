# Hanium Smart Parking Backend

Django REST Framework backend for the intelligent parking scheduling and control MVP.
**Strictly Docker-based development & deployment** — no local Python/conda/venv setup.

## Prerequisites

- Docker / [OrbStack](https://orbstack.dev) (Apple Silicon 권장) — `docker compose v2` 사용
- `gh` CLI (선택, 이슈/PR 자동화용)

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

- Backend (Daphne ASGI): http://localhost:8000
- Redis: localhost:6379

`compose`가 `hanium` 네트워크를 만들고 Redis 컨테이너를 별칭 `redis`로 붙여서, `.env`의 `REDIS_URL=redis://redis:6379/0`가 곧바로 동작한다.

## Common commands

모든 명령은 컨테이너 안에서 실행한다.

```bash
# 단위 테스트
docker compose exec backend python manage.py test

# 데모 데이터 재시드 (idempotent)
docker compose exec backend python manage.py seed_demo

# 마이그레이션 (Dockerfile CMD가 부팅 시 자동 호출하므로 보통 불필요)
docker compose exec backend python manage.py migrate

# Django shell
docker compose exec backend python manage.py shell

# 로그 실시간
docker compose logs backend -f

# 전체 종료 + 정리
docker compose down

# 코드 변경 후 재빌드
docker compose up --build -d
```

## Main APIs

- `GET/POST /api/vehicles/`
- `GET /api/parking-lots/` — `lot_width`, `lot_height` 포함 (캔버스 스케일링)
- `GET/PATCH /api/parking-spots/` — `coord_x`, `coord_y`
- `POST /api/entry/` — 차량 입차 + 경로(`waypoints`) 자동 생성
- `POST /api/exit/`
- `GET /api/recommendations/spots/`
- `POST /api/cameras/{camera_id}/heartbeat/`
- `GET /api/dashboard/`
- `WS /ws/dashboard/`

## 자율주차 제어 경로 (실차)

REST/WS API 와 별개로, RC카 자동주차는 아래 경로로 동작한다.

```
천장 고정 카메라
  → YOLO / OpenCV / Homography          (cv/)
  → Vehicle Pose (x, y, heading)        (cv/heading.py, cv/association.py)
  → 슬롯 배정                            (rl/, parking/)
  → GLOBAL route                        (parking/waypoints.py)
  → parking setup / recovery
  → 후면주차 route (ALIGN/ENTRY/FINAL)
  → 실행 전 경로 안전 검증               (parking/trajectory_safety.py)
  → 노트북 AUTO_HOST                    (control/, controller/, host_control/)
  → TCP / NDJSON                        (comm/)
  → ESP32 REMOTE_DIRECT
  → DC 모터 / 서보 / 엔코더
  → 카메라 재관측 (closed loop)
```

핵심 계약 두 가지:

- **production wire 는 `AUTO_HOST → REMOTE_DIRECT` 다.** throttle/steering 은
  노트북이 계산해서 내려주고 ESP32 는 `DIRECT_CONTROL` 을 실행만 한다.
  ESP32 의 `WAYPOINT_AUTO` 는 이 경로와 배타적이며 production 이 아니다.
- **waypoint 는 목표 자체가 아니다.** 최종 slot pose 가 목표다. 작은 추종
  오차는 피드백으로 흡수하고, 오차가 크거나 경로가 불가능해지면 정지 →
  fresh pose → 재계획한다 (과거 waypoint 로 무조건 되돌아가지 않는다).

파이프라인 실행 (production 경로는 `--control-mode auto-host` 와 후면주차 모드를
명시한다. 실제 실차에서는 검증된 calibration/weight 경로를 사용한다):

```bash
python manage.py run_pipeline --control-mode auto-host --parking-mode rear --calibration <validated-calibration.json> --weights weights/best.pt --record runs --record-video --show
```

제어/주차 관련 테스트는 Django 없이도 돌아간다:

```bash
python -m unittest discover -s . -p "test_*.py" -t .
python tools/run_auto_parking_tests.py
```

### 현재 개발 상태

단일 자율주행 차량의 카메라 Pose 기반 후면주차 E2E는 실제 차량에서 완료 사례를
확보했다. production 제어는 `AUTO_HOST → DIRECT_CONTROL → REMOTE_DIRECT`이며,
통신 단절·stale Pose·전후진 전환·경계 위험에서는 먼저 zero command를 보낸 뒤
fresh observation과 현재 Pose 기준 재계획을 요구한다.

현재 구현 범위와 한계는 다음과 같다.

- allocator의 슬롯 상태와 정상 `PARKED` 결과는 예약/장애물 정보로 반영된다.
- 카메라에 보이는 정적 차량으로 슬롯을 `VISION_OCCUPIED` 처리한다. CAR_ID
  binding 도 차량 전원도 필요 없고, pose 와 슬롯 기하만 쓴다. 확정된 칸은
  allocator 배정에서 제외되고, 그 차량은 기존 `PARKED` 차량과 같은 정적
  장애물(`STATIC_PARKED`)로 경로계획에 남는다.
  `vision_occupancy_enabled=False` 로 두면 이 기능이 없던 동작과 같아진다.
- **운용 순서 제약**: 정적 차량의 점유가 확정된 뒤에 자율주행 차량을
  활성화해야 한다. 확정에는 정지 이력과 연속 관측이 필요해 실측 약 4.5초가
  걸리는데, 그 전에 배정이 일어나면 아직 확정되지 않은 칸이 배정될 수 있다
  (실측에서 배정이 확정보다 0.4~0.8초 빨랐다). 이 경합을 막는 allocation
  gate 는 구현되어 있지 않다. 검증된 절차는
  `정적 차량 배치 → 점유 확정(VISION_SLOT_OCCUPIED 로그) → 자율주행 차량 활성화`
  뿐이다.
- 여러 차량이 연결될 수 있지만 `AUTO_HOST` 이동 권한은 한 번에 한 차량만 갖는다.
  두 차량 동시 자율주행 완료로 해석하면 안 된다.
- PPO 환경·정책 로딩 경로는 구현되어 있으나, 모델 또는 `sb3-contrib`가 없으면
  deterministic nearest-slot heuristic을 사용한다. 개별 HIL run의 policy provenance는
  현재 recorder만으로 확정할 수 없다.
- Dashboard REST/WebSocket은 구현되어 있으며 Redis 설정이 없으면 in-memory channel
  layer를 사용한다. Dashboard 전송 실패는 차량 제어와 안전정지를 막지 않는다.

## Live Viz Demo

차량 라우팅을 시각화하는 standalone HTML 데모는 별도 프론트 레포에 있다:
[`hanium-2026-project/frontend → viz/`](https://github.com/hanium-2026-project/frontend/tree/main/viz)

로컬 백엔드를 띄운 상태에서 viz의 README에 따라 `python3 -m http.server 5173`으로 서빙하면 즉시 동작한다.

## Troubleshooting

**Port 8000 in use** — 보통 이전 `docker compose down`이 빠진 경우.
```bash
docker compose down
docker ps -a | grep 8000   # 외부 컨테이너가 잡고 있는지 확인
```

**`redis` 호스트네임 풀이 실패** — 컨테이너가 `hanium` 네트워크 밖에 있는 경우.
```bash
docker network inspect hanium     # backend + redis 둘 다 있어야 함
docker compose up -d              # compose가 알아서 attach
```

**호스트 `db.sqlite3`가 컨테이너에 섞임** — `.dockerignore`가 호스트 DB를 빌드 컨텍스트에서 제외하므로, 새 빌드는 항상 깨끗한 컨테이너 DB로 시작한다.
