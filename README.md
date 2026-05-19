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
