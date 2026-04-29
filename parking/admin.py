"""Admin registrations for quick inspection during MVP development."""

from django.contrib import admin

from .models import (
    Camera,
    CameraFrameLog,
    EntryExit,
    ParkingAssignment,
    ParkingLot,
    ParkingSpot,
    RLPolicyLog,
    RoutePlan,
    SensorEvent,
    Vehicle,
)


admin.site.register(Vehicle)
admin.site.register(ParkingLot)
admin.site.register(ParkingSpot)
admin.site.register(Camera)
admin.site.register(EntryExit)
admin.site.register(RoutePlan)
admin.site.register(SensorEvent)
admin.site.register(ParkingAssignment)
admin.site.register(RLPolicyLog)
admin.site.register(CameraFrameLog)
