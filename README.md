# Hanium Smart Parking Backend

Django REST Framework backend for the intelligent parking scheduling and control MVP.

## Quick start with Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

Backend serves on http://localhost:8000, Redis on 6379. The compose file
attaches both containers to the `hanium` network and aliases Redis as
`redis` so `REDIS_URL=redis://redis:6379/0` in `.env` works out of the box.

## Local setup (conda / venv)

```bash
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo
python manage.py runserver 127.0.0.1:8000 --noreload
```

## Test

```bash
python manage.py test
```

## Main APIs

- `GET/POST /api/vehicles/`
- `GET /api/parking-lots/`
- `GET/PATCH /api/parking-spots/`
- `POST /api/entry/`
- `POST /api/exit/`
- `GET /api/recommendations/spots/`
- `POST /api/cameras/{camera_id}/heartbeat/`
- `GET /api/dashboard/`
- `WS /ws/dashboard/`
