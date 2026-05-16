"""Database models for the parking scheduling and control MVP."""

from django.db import models
from django.utils import timezone


class Vehicle(models.Model):
    """Vehicle master data used by registration, entry, and routing flows."""

    VEHICLE_TYPES = [
        ("sedan", "Sedan"),
        ("suv", "SUV"),
        ("compact", "Compact"),
        ("ev", "EV"),
        ("disabled", "Disabled"),
    ]
    DISCOUNT_TYPES = [
        ("none", "None"),
        ("compact", "Compact"),
        ("ev", "EV"),
        ("disabled", "Disabled"),
        ("resident", "Resident"),
    ]

    vehicle_id = models.BigAutoField(primary_key=True)
    license_plate = models.CharField(max_length=32, unique=True)
    vehicle_type = models.CharField(max_length=20, choices=VEHICLE_TYPES, default="sedan")
    is_registered = models.BooleanField(default=False)
    discount_type = models.CharField(max_length=20, choices=DISCOUNT_TYPES, default="none")

    class Meta:
        ordering = ["license_plate"]

    def __str__(self) -> str:
        return self.license_plate


class ParkingLot(models.Model):
    """Physical parking lot metadata and capacity baseline."""

    lot_id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=120)
    address = models.CharField(max_length=255)
    total_capacity = models.PositiveIntegerField(default=0)
    lot_width = models.FloatField(default=0.0)   # mm, x축 전체 폭
    lot_height = models.FloatField(default=0.0)  # mm, y축 전체 높이

    class Meta:
        ordering = ["lot_id"]

    def __str__(self) -> str:
        return self.name


class ParkingSpot(models.Model):
    """Individual parking spot with map coordinates and operational state."""

    SPOT_TYPES = [
        ("standard", "Standard"),
        ("compact", "Compact"),
        ("ev", "EV"),
        ("disabled", "Disabled"),
    ]
    STATUSES = [
        ("vacant", "Vacant"),
        ("occupied", "Occupied"),
        ("reserved", "Reserved"),
        ("disabled", "Disabled"),
    ]

    spot_id = models.BigAutoField(primary_key=True)
    lot = models.ForeignKey(ParkingLot, related_name="spots", on_delete=models.CASCADE)
    section = models.CharField(max_length=32)
    spot_type = models.CharField(max_length=20, choices=SPOT_TYPES, default="standard")
    status = models.CharField(max_length=20, choices=STATUSES, default="vacant")
    coord_x = models.FloatField(default=0.0)
    coord_y = models.FloatField(default=0.0)

    class Meta:
        ordering = ["lot_id", "section", "spot_id"]

    def __str__(self) -> str:
        return f"{self.lot.name}-{self.section}-{self.spot_id}"


class Camera(models.Model):
    """Camera device state; a camera may watch a full lot or one spot."""

    STATUSES = [
        ("online", "Online"),
        ("offline", "Offline"),
        ("maintenance", "Maintenance"),
    ]

    camera_id = models.BigAutoField(primary_key=True)
    lot = models.ForeignKey(ParkingLot, related_name="cameras", on_delete=models.CASCADE)
    spot = models.ForeignKey(
        ParkingSpot,
        related_name="cameras",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    location_desc = models.CharField(max_length=160)
    status = models.CharField(max_length=20, choices=STATUSES, default="offline")
    last_heartbeat = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["camera_id"]

    def __str__(self) -> str:
        return f"Camera {self.camera_id} ({self.status})"


class EntryExit(models.Model):
    """Entry/exit transaction for a vehicle occupying a parking spot."""

    transaction_id = models.BigAutoField(primary_key=True)
    vehicle = models.ForeignKey(Vehicle, related_name="transactions", on_delete=models.CASCADE)
    spot = models.ForeignKey(ParkingSpot, related_name="transactions", on_delete=models.PROTECT)
    entry_time = models.DateTimeField(default=timezone.now)
    exit_time = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-entry_time"]

    @property
    def is_active(self) -> bool:
        """Return True while the vehicle has not completed exit processing."""
        return self.exit_time is None

    def __str__(self) -> str:
        return f"{self.vehicle.license_plate} -> {self.spot_id}"


class RoutePlan(models.Model):
    """Persisted route recommendation so the frontend can retrieve waypoints."""

    route_id = models.BigAutoField(primary_key=True)
    vehicle = models.ForeignKey(Vehicle, related_name="route_plans", on_delete=models.CASCADE)
    target_spot = models.ForeignKey(ParkingSpot, related_name="route_plans", on_delete=models.CASCADE)
    start_x = models.FloatField(default=0.0)
    start_y = models.FloatField(default=0.0)
    waypoints = models.JSONField(default=list)
    policy_name = models.CharField(max_length=80, default="heuristic-v1")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]


class SensorEvent(models.Model):
    """Normalized event from cameras, spot sensors, or simulated devices."""

    event_id = models.BigAutoField(primary_key=True)
    camera = models.ForeignKey(Camera, related_name="sensor_events", on_delete=models.SET_NULL, null=True, blank=True)
    spot = models.ForeignKey(ParkingSpot, related_name="sensor_events", on_delete=models.SET_NULL, null=True, blank=True)
    event_type = models.CharField(max_length=80)
    payload = models.JSONField(default=dict)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]


class ParkingAssignment(models.Model):
    """Assignment lifecycle connecting a vehicle to a recommended or occupied spot."""

    STATUSES = [
        ("assigned", "Assigned"),
        ("occupied", "Occupied"),
        ("completed", "Completed"),
        ("cancelled", "Cancelled"),
    ]

    assignment_id = models.BigAutoField(primary_key=True)
    vehicle = models.ForeignKey(Vehicle, related_name="assignments", on_delete=models.CASCADE)
    spot = models.ForeignKey(ParkingSpot, related_name="assignments", on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUSES, default="assigned")
    reason = models.CharField(max_length=160, default="heuristic nearest vacant spot")
    assigned_at = models.DateTimeField(default=timezone.now)
    released_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-assigned_at"]


class RLPolicyLog(models.Model):
    """Trace table for future RL inference decisions and reward analysis."""

    log_id = models.BigAutoField(primary_key=True)
    policy_name = models.CharField(max_length=80)
    state = models.JSONField(default=dict)
    action = models.JSONField(default=dict)
    reward = models.FloatField(default=0.0)
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]


class CameraFrameLog(models.Model):
    """Frame-level CV audit log for detector outputs and future model debugging."""

    frame_id = models.BigAutoField(primary_key=True)
    camera = models.ForeignKey(Camera, related_name="frame_logs", on_delete=models.CASCADE)
    frame_uri = models.CharField(max_length=255)
    detected_objects = models.JSONField(default=list)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
