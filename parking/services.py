"""Domain services for parking recommendation, assignment, and dashboard state."""

from __future__ import annotations


from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db import transaction
from django.db.models import Count, Q
from django.utils import timezone
from rest_framework.exceptions import ValidationError

from .models import (
    Camera,
    EntryExit,
    ParkingAssignment,
    ParkingLot,
    ParkingSpot,
    RLPolicyLog,
    RoutePlan,
    Vehicle,
)


ENTRY_POINT: tuple[float, float] = (150.0, 0.0)      # 주차장 입구 — 좌측 도로(300mm) 중심, 하단
EXIT_POINT: tuple[float, float] = (150.0, 1200.0)    # 주차장 출구 — 좌측 도로(300mm) 중심, 상단

SPOT_PREFERENCE_BY_VEHICLE = {
    "ev": ["ev", "standard", "compact"],
    "compact": ["compact", "standard"],
    "disabled": ["disabled"],
    "suv": ["standard"],
    "sedan": ["standard", "compact"],
}


def seed_demo_data() -> None:
    """Create a deterministic lot, spots, and cameras for first-run demos.

    The frontend should have useful data immediately after `manage.py migrate`
    plus this function, while tests can call it repeatedly because lookups are
    idempotent.
    """
    lot, _ = ParkingLot.objects.get_or_create(
        lot_id=1,
        defaults={
            "name": "Hanium Smart Parking",
            "address": "Seoul Demo Campus",
            "total_capacity": 8,
            "lot_width": 1200.0,
            "lot_height": 1200.0,
        },
    )
    # Layout dimensions (mm): spot 200x300, line 25, aisle 550
    # Origin (0, 0) = 주차장 입구 (bottom-left)
    _PITCH = 225.0    # spot_width(200) + line_width(25)
    _START_X = 425.0  
    _A_Y = 1050.0     # B_depth(300) + line(25) + aisle(550) + line(25) + half_depth(150)
    _B_Y = 150.0      # half_depth(150) from bottom

    specs: list[tuple[str, str, float, float]] = [
        # A열 — 출구 방향 (위쪽)
        ("A1", "standard", _START_X,               _A_Y),
        ("A2", "standard", _START_X + _PITCH,       _A_Y),
        ("A3", "standard", _START_X + _PITCH * 2,   _A_Y),
        ("A4", "standard", _START_X + _PITCH * 3,   _A_Y),
        # B열 — 입구 방향 (아래쪽)
        ("B1", "standard", _START_X,               _B_Y),
        ("B2", "standard", _START_X + _PITCH,       _B_Y),
        ("B3", "standard", _START_X + _PITCH * 2,   _B_Y),
        ("B4", "standard", _START_X + _PITCH * 3,   _B_Y),
    ]
    for section, spot_type, x, y in specs:
        ParkingSpot.objects.update_or_create(
            lot=lot,
            section=section,
            defaults={"spot_type": spot_type, "coord_x": x, "coord_y": y},
        )
    lot.total_capacity = lot.spots.count()
    lot.save(update_fields=["total_capacity"])
    first_spot = lot.spots.order_by("spot_id").first()
    Camera.objects.get_or_create(
        camera_id=1,
        defaults={"lot": lot, "spot": first_spot, "location_desc": "Main entrance camera", "status": "online"},
    )
    Camera.objects.get_or_create(
        camera_id=2,
        defaults={"lot": lot, "spot": None, "location_desc": "Central aisle overview", "status": "online"},
    )


def _distance_from_entry(spot: ParkingSpot) -> float:
    """Score a spot by distance from the current MVP entry coordinate."""
    return ((spot.coord_x - ENTRY_POINT[0]) ** 2 + (spot.coord_y - ENTRY_POINT[1]) ** 2) ** 0.5


def recommend_spot(lot_id: int | None = None, vehicle_type: str = "sedan") -> ParkingSpot:
    """Return the best available spot for a vehicle using a replaceable heuristic.

    The preference order separates policy logic from API code. A future RL
    inference call can replace this function while preserving the response shape.
    """
    queryset = ParkingSpot.objects.select_related("lot").filter(status="vacant")
    if lot_id:
        queryset = queryset.filter(lot_id=lot_id)
    candidates = list(queryset)
    if not candidates:
        raise ValidationError({"detail": "No vacant parking spot is available."})

    preferences = SPOT_PREFERENCE_BY_VEHICLE.get(vehicle_type, ["standard", "compact"])
    preference_rank = {spot_type: index for index, spot_type in enumerate(preferences)}

    def score(spot: ParkingSpot) -> tuple[int, float, int]:
        type_score = preference_rank.get(spot.spot_type, len(preferences) + 1)
        return (type_score, _distance_from_entry(spot), spot.spot_id)

    recommended = sorted(candidates, key=score)[0]
    RLPolicyLog.objects.create(
        policy_name="heuristic-v1",
        state={"lot_id": lot_id, "vehicle_type": vehicle_type, "vacant_count": len(candidates)},
        action={"spot_id": recommended.spot_id},
        reward=1.0,
        metadata={"reason": "lowest type preference rank then nearest coordinate"},
    )
    return recommended


def build_route_plan(vehicle: Vehicle, target_spot: ParkingSpot, start: tuple[float, float] = ENTRY_POINT) -> RoutePlan:
    """Create a simple waypoint route that can later be replaced by RL output."""
    start_x, start_y = start
    waypoints = [
        {"x": start_x, "y": start_y, "label": "entry"},
        {"x": target_spot.coord_x, "y": start_y, "label": "aisle"},
        {"x": target_spot.coord_x, "y": target_spot.coord_y, "label": target_spot.section},
    ]
    return RoutePlan.objects.create(
        vehicle=vehicle,
        target_spot=target_spot,
        start_x=start_x,
        start_y=start_y,
        waypoints=waypoints,
        policy_name="heuristic-route-v1",
    )


def _broadcast_state(event: str, payload: dict) -> None:
    """Publish a state update to dashboard WebSocket clients when available."""
    channel_layer = get_channel_layer()
    if not channel_layer:
        return
    async_to_sync(channel_layer.group_send)(
        "parking_dashboard",
        {"type": "parking.state", "payload": {"event": event, **payload}},
    )


@transaction.atomic
def process_entry(license_plate: str, vehicle_type: str = "sedan", lot_id: int | None = None) -> tuple[EntryExit, RoutePlan]:
    """Create an entry transaction, occupy a spot, and persist a route plan."""
    vehicle, created = Vehicle.objects.get_or_create(
        license_plate=license_plate,
        defaults={"vehicle_type": vehicle_type, "is_registered": False, "discount_type": "none"},
    )
    if not created and vehicle.vehicle_type != vehicle_type:
        vehicle.vehicle_type = vehicle_type
        vehicle.save(update_fields=["vehicle_type"])
    if EntryExit.objects.filter(vehicle=vehicle, exit_time__isnull=True).exists():
        raise ValidationError({"detail": "Vehicle already has an active parking transaction."})

    spot = recommend_spot(lot_id=lot_id, vehicle_type=vehicle.vehicle_type)
    spot.status = "occupied"
    spot.save(update_fields=["status"])
    transaction_record = EntryExit.objects.create(vehicle=vehicle, spot=spot)
    ParkingAssignment.objects.create(vehicle=vehicle, spot=spot, status="occupied")
    route_plan = build_route_plan(vehicle=vehicle, target_spot=spot)
    _broadcast_state(
        "entry",
        {"license_plate": license_plate, "spot_id": spot.spot_id, "transaction_id": transaction_record.transaction_id},
    )
    return transaction_record, route_plan


@transaction.atomic
def process_exit(license_plate: str) -> EntryExit:
    """Complete an active transaction and release the occupied spot."""
    try:
        vehicle = Vehicle.objects.get(license_plate=license_plate)
    except Vehicle.DoesNotExist as exc:
        raise ValidationError({"detail": "Vehicle is not registered in the system."}) from exc

    try:
        transaction_record = EntryExit.objects.select_for_update().get(vehicle=vehicle, exit_time__isnull=True)
    except EntryExit.DoesNotExist as exc:
        raise ValidationError({"detail": "No active parking transaction for this vehicle."}) from exc

    transaction_record.exit_time = timezone.now()
    transaction_record.save(update_fields=["exit_time"])
    spot = transaction_record.spot
    spot.status = "vacant"
    spot.save(update_fields=["status"])
    ParkingAssignment.objects.filter(vehicle=vehicle, spot=spot, status="occupied").update(
        status="completed",
        released_at=timezone.now(),
    )
    _broadcast_state(
        "exit",
        {"license_plate": license_plate, "spot_id": spot.spot_id, "transaction_id": transaction_record.transaction_id},
    )
    return transaction_record


def update_camera_heartbeat(camera: Camera, status: str = "online") -> Camera:
    """Record that a camera is alive and publish the health change."""
    camera.status = status
    camera.last_heartbeat = timezone.now()
    camera.save(update_fields=["status", "last_heartbeat"])
    _broadcast_state("camera_heartbeat", {"camera_id": camera.camera_id, "status": camera.status})
    return camera


def parking_lot_queryset_with_counts():
    """Return lots annotated with status counts for dashboard and list APIs."""
    return ParkingLot.objects.annotate(
        vacant_count=Count("spots", filter=Q(spots__status="vacant")),
        occupied_count=Count("spots", filter=Q(spots__status="occupied")),
        reserved_count=Count("spots", filter=Q(spots__status="reserved")),
    )


def dashboard_state() -> dict:
    """Build an aggregated dashboard payload in one stable API shape."""
    lots = parking_lot_queryset_with_counts()
    spots = ParkingSpot.objects.select_related("lot").all()
    cameras = Camera.objects.select_related("lot", "spot").all()
    recent_transactions = EntryExit.objects.select_related("vehicle", "spot").order_by("-entry_time")[:10]
    return {
        "lots": [
            {
                "lot_id": lot.lot_id,
                "name": lot.name,
                "address": lot.address,
                "total_capacity": lot.total_capacity,
                "vacant_count": lot.vacant_count,
                "occupied_count": lot.occupied_count,
                "reserved_count": lot.reserved_count,
            }
            for lot in lots
        ],
        "summary": {
            "total_spots": spots.count(),
            "vacant": spots.filter(status="vacant").count(),
            "occupied": spots.filter(status="occupied").count(),
            "reserved": spots.filter(status="reserved").count(),
            "disabled": spots.filter(status="disabled").count(),
            "cameras_online": cameras.filter(status="online").count(),
            "cameras_offline": cameras.filter(status="offline").count(),
        },
        "cameras": [
            {
                "camera_id": camera.camera_id,
                "lot_id": camera.lot_id,
                "spot_id": camera.spot_id,
                "location_desc": camera.location_desc,
                "status": camera.status,
                "last_heartbeat": camera.last_heartbeat.isoformat() if camera.last_heartbeat else None,
            }
            for camera in cameras
        ],
        "recent_transactions": [
            {
                "transaction_id": item.transaction_id,
                "license_plate": item.vehicle.license_plate,
                "spot_id": item.spot_id,
                "entry_time": item.entry_time.isoformat(),
                "exit_time": item.exit_time.isoformat() if item.exit_time else None,
            }
            for item in recent_transactions
        ],
    }
