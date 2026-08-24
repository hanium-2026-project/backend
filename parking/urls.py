"""URL routing for the parking REST API."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    CameraViewSet,
    EntryAPIView,
    EntryExitViewSet,
    ExitAPIView,
    ParkingLotViewSet,
    ParkingSpotViewSet,
    RoutePlanViewSet,
    VehicleViewSet,
    camera_stream_view,
    dashboard_view,
    recommend_spot_view,
)


router = DefaultRouter()
router.register("vehicles", VehicleViewSet, basename="vehicle")
router.register("parking-lots", ParkingLotViewSet, basename="parking-lot")
router.register("parking-spots", ParkingSpotViewSet, basename="parking-spot")
router.register("cameras", CameraViewSet, basename="camera")
router.register("transactions", EntryExitViewSet, basename="transaction")
router.register("routes", RoutePlanViewSet, basename="route")

urlpatterns = [
    path("", include(router.urls)),
    path("entry/", EntryAPIView.as_view(), name="entry"),
    path("exit/", ExitAPIView.as_view(), name="exit"),
    path("recommendations/spots/", recommend_spot_view, name="recommend-spot"),
    path("dashboard/", dashboard_view, name="dashboard"),
    path("cameras/<int:camera_id>/stream/", camera_stream_view, name="camera-stream"),
]
