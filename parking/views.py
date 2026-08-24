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
    ENTRY_POINT,
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
        raw_start = serializer.validated_data.get("start")
        start = (raw_start[0], raw_start[1]) if raw_start else ENTRY_POINT
        target_spot_id = serializer.validated_data.get("target_spot_id")

        if target_spot_id:
            target_spot = get_object_or_404(ParkingSpot, spot_id=target_spot_id)
            route_plan = build_route_plan(vehicle, target_spot, start=start)
        else:
            route_plan = vehicle.route_plans.select_related("target_spot").first()
            if route_plan is None:
                active = EntryExit.objects.filter(vehicle=vehicle, exit_time__isnull=True).select_related("spot").first()
                target_spot = active.spot if active else recommend_spot(vehicle_type=vehicle.vehicle_type)
                route_plan = build_route_plan(vehicle, target_spot, start=start)

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


import time  # noqa: E402  (CCTV 스트림 유휴 판정용)


# ─── CCTV MJPEG 중계 ──────────────────────────────────────────────────────
# 카메라는 한 프로세스만 열 수 있어서 웹서버가 장치를 직접 못 읽는다.
# run_pipeline 이 주석 그린 프레임을 Redis 에 올리고, 여기서 그걸 꺼내
# multipart/x-mixed-replace 로 흘린다. 프론트는 <img src> 하나면 된다.

_MJPEG_BOUNDARY = "frame"
# 이만큼 새 프레임이 없으면 파이프라인이 멈춘 것으로 보고 스트림을 끝낸다.
# 계속 붙들고 있으면 <img> 가 옛 화면을 띄운 채 굳는다.
_STREAM_IDLE_TIMEOUT_S = 5.0
# 폴링 간격. 퍼블리셔가 기본 8fps 이므로 그보다 촘촘하게 본다.
_STREAM_POLL_S = 0.04


async def _mjpeg_frames(camera_id: int):
    """Redis 의 최신 프레임을 multipart 청크로 흘린다.

    **비동기 제너레이터여야 한다.** Django 의 ASGI 핸들러는 동기 이터레이터를
    받으면 끝까지 소비한 뒤에야 응답을 내보내는데, 이 스트림은 끝이 없으므로
    응답이 영원히 나가지 않는다 (실제로 그렇게 타임아웃났다).
    """
    import asyncio

    from django.conf import settings
    from redis import asyncio as aioredis

    from pipeline.camera_stream import frame_key

    url = getattr(settings, "REDIS_URL", "") or ""
    if not url:
        return
    try:
        client = aioredis.from_url(url)
    except Exception:
        return

    key = frame_key(camera_id)
    last = None
    idle_since = time.monotonic()
    try:
        while True:
            try:
                data = await client.get(key)
            except Exception:
                return                              # Redis 끊김 → 스트림 종료
            if data and data != last:
                last = data
                idle_since = time.monotonic()
                yield (b"--" + _MJPEG_BOUNDARY.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                       + data + b"\r\n")
            elif time.monotonic() - idle_since > _STREAM_IDLE_TIMEOUT_S:
                return                              # 파이프라인 정지로 판단
            await asyncio.sleep(_STREAM_POLL_S)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


def camera_stream_view(request, camera_id: int):
    """실시간 카메라 프레임을 MJPEG 로 중계한다 (탐지 박스가 그려진 화면)."""
    from django.http import StreamingHttpResponse

    response = StreamingHttpResponse(
        _mjpeg_frames(camera_id),
        content_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
    )
    # 프록시·브라우저가 스트림을 버퍼링하거나 캐시하면 화면이 밀린다.
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["X-Accel-Buffering"] = "no"
    return response
