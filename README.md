# Hanium Smart Parking Backend

Django REST Framework backend for the intelligent parking scheduling and control MVP.

## Setup

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
