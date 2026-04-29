"""REST API views for the parking scheduling and control backend."""

from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Camera, EntryExit, ParkingLot, ParkingSpot, RoutePlan, Vehicle
from .serializers import (
    CameraSerializer,
    EntryExitSerializer,
    EntryRequestSerializer,
    ExitRequestSerializer,
    HeartbeatSerializer,
    ParkingLotSerializer,
    ParkingSpotSerializer,
    RoutePlanSerializer,
    RouteRequestSerializer,
    VehicleSerializer,
)
from .services import (
    build_route_plan,
    dashboard_state,
    parking_lot_queryset_with_counts,
    process_entry,
    process_exit,
    recommend_spot,
    update_camera_heartbeat,
)


class VehicleViewSet(viewsets.ModelViewSet):
    """CRUD API for vehicles managed by operators or entry simulation."""

    queryset = Vehicle.objects.all()
    serializer_class = VehicleSerializer
    lookup_field = "vehicle_id"

    @action(detail=True, methods=["get", "post"], url_path="route")
    def route(self, request, vehicle_id: int | None = None) -> Response:
        """Return or create a route recommendation for a specific vehicle."""
        vehicle = self.get_object()
        serializer = RouteRequestSerializer(data=request.data if request.method == "POST" else request.query_params)
        serializer.is_valid(raise_exception=True)
        start_values = serializer.validated_data.get("start") or [0.0, 0.0]
        target_spot_id = serializer.validated_data.get("target_spot_id")

        if target_spot_id:
            target_spot = get_object_or_404(ParkingSpot, spot_id=target_spot_id)
            route_plan = build_route_plan(vehicle, target_spot, start=(start_values[0], start_values[1]))
        else:
            route_plan = vehicle.route_plans.select_related("target_spot").first()
            if route_plan is None:
                active = EntryExit.objects.filter(vehicle=vehicle, exit_time__isnull=True).select_related("spot").first()
                target_spot = active.spot if active else recommend_spot(vehicle_type=vehicle.vehicle_type)
                route_plan = build_route_plan(vehicle, target_spot, start=(start_values[0], start_values[1]))

        return Response(RoutePlanSerializer(route_plan).data)


class ParkingLotViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only parking lot API with status counts for dashboard use."""

    serializer_class = ParkingLotSerializer
    lookup_field = "lot_id"

    def get_queryset(self):
        return parking_lot_queryset_with_counts()


class ParkingSpotViewSet(viewsets.ModelViewSet):
    """Parking spot API used by map UI and manual status operations."""

    queryset = ParkingSpot.objects.select_related("lot").all()
    serializer_class = ParkingSpotSerializer
    lookup_field = "spot_id"

    @action(detail=True, methods=["patch"], url_path="set-status")
    def set_status(self, request, spot_id: int | None = None) -> Response:
        """Change a single spot status without touching other spot metadata."""
        spot = self.get_object()
        status_value = request.data.get("status")
        valid_statuses = {choice[0] for choice in ParkingSpot.STATUSES}
        if status_value not in valid_statuses:
            return Response({"detail": "Invalid parking spot status."}, status=status.HTTP_400_BAD_REQUEST)
        spot.status = status_value
        spot.save(update_fields=["status"])
        return Response(self.get_serializer(spot).data)


class CameraViewSet(viewsets.ModelViewSet):
    """Camera API including heartbeat state refresh for device simulators."""

    queryset = Camera.objects.select_related("lot", "spot").all()
    serializer_class = CameraSerializer
    lookup_field = "camera_id"

    @action(detail=True, methods=["post"], url_path="heartbeat")
    def heartbeat(self, request, camera_id: int | None = None) -> Response:
        """Mark a camera as alive and update its operational status."""
        camera = self.get_object()
        serializer = HeartbeatSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        updated = update_camera_heartbeat(camera, serializer.validated_data["status"])
        return Response(self.get_serializer(updated).data)


class EntryExitViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only transaction history API for dashboards and audits."""

    queryset = EntryExit.objects.select_related("vehicle", "spot").all()
    serializer_class = EntryExitSerializer
    lookup_field = "transaction_id"


class RoutePlanViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only route plan API for route visualization."""

    queryset = RoutePlan.objects.select_related("vehicle", "target_spot").all()
    serializer_class = RoutePlanSerializer
    lookup_field = "route_id"


class EntryAPIView(APIView):
    """Process vehicle entry and return assigned spot plus route plan."""

    def post(self, request) -> Response:
        serializer = EntryRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        record, route_plan = process_entry(**serializer.validated_data)
        return Response(
            {
                "transaction": EntryExitSerializer(record).data,
                "route": RoutePlanSerializer(route_plan).data,
                "recommended_spot": ParkingSpotSerializer(record.spot).data,
            },
            status=status.HTTP_201_CREATED,
        )


class ExitAPIView(APIView):
    """Process vehicle exit and release the occupied parking spot."""

    def post(self, request) -> Response:
        serializer = ExitRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        record = process_exit(serializer.validated_data["license_plate"])
        return Response({"transaction": EntryExitSerializer(record).data})


@api_view(["GET"])
def recommend_spot_view(request) -> Response:
    """Return a vacant spot recommendation without changing spot state."""
    lot_id = request.query_params.get("lot_id")
    vehicle_type = request.query_params.get("vehicle_type", "sedan")
    vehicle_id = request.query_params.get("vehicle_id")
    if vehicle_id:
        vehicle = get_object_or_404(Vehicle, vehicle_id=vehicle_id)
        vehicle_type = vehicle.vehicle_type
    spot = recommend_spot(lot_id=int(lot_id) if lot_id else None, vehicle_type=vehicle_type)
    return Response({"recommended_spot": ParkingSpotSerializer(spot).data})


@api_view(["GET"])
def dashboard_view(request) -> Response:
    """Return full parking lot, camera, and recent transaction state."""
    return Response(dashboard_state())
