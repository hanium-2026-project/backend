"""Backend, CV, and RL tests for the MVP parking system."""

import numpy as np
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from cv.camera_capture import create_synthetic_frame
from cv.homography import compute_homography, warp_point
from cv.plate_detector import MockPlateDetector
from cv.vehicle_detector import MockVehicleDetector
from parking.models import Camera, EntryExit, ParkingLot, ParkingSpot, Vehicle
from parking.protocol import VehicleTelemetryMessage
from parking.services import process_entry, process_exit, recommend_spot, seed_demo_data
from rl.inference import heuristic_policy, load_policy
from rl.parking_env import ParkingRoutingEnv


class ParkingModelTests(APITestCase):
    """Validate core ERD entities and relationships."""

    def test_model_relationships(self) -> None:
        lot = ParkingLot.objects.create(name="Test Lot", address="Seoul", total_capacity=1)
        spot = ParkingSpot.objects.create(lot=lot, section="A1", spot_type="standard", coord_x=1, coord_y=2)
        camera = Camera.objects.create(lot=lot, spot=spot, location_desc="Gate")
        vehicle = Vehicle.objects.create(license_plate="11가1111", vehicle_type="sedan", is_registered=True)
        tx = EntryExit.objects.create(vehicle=vehicle, spot=spot)

        self.assertEqual(camera.spot, spot)
        self.assertTrue(tx.is_active)
        self.assertEqual(str(vehicle), "11가1111")


class ParkingApiTests(APITestCase):
    """Validate public REST API behavior for operators and the frontend."""

    def setUp(self) -> None:
        seed_demo_data()

    def test_vehicle_crud(self) -> None:
        create_response = self.client.post(
            reverse("vehicle-list"),
            {"license_plate": "22나2222", "vehicle_type": "ev", "is_registered": True, "discount_type": "ev"},
            format="json",
        )
        self.assertEqual(create_response.status_code, status.HTTP_201_CREATED)
        vehicle_id = create_response.data["vehicle_id"]

        list_response = self.client.get(reverse("vehicle-list"))
        self.assertEqual(list_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(list_response.data), 1)

        patch_response = self.client.patch(
            reverse("vehicle-detail", kwargs={"vehicle_id": vehicle_id}),
            {"vehicle_type": "sedan"},
            format="json",
        )
        self.assertEqual(patch_response.status_code, status.HTTP_200_OK)
        self.assertEqual(patch_response.data["vehicle_type"], "sedan")

        delete_response = self.client.delete(reverse("vehicle-detail", kwargs={"vehicle_id": vehicle_id}))
        self.assertEqual(delete_response.status_code, status.HTTP_204_NO_CONTENT)

    def test_recommendation_prefers_vehicle_type(self) -> None:
        response = self.client.get(reverse("recommend-spot"), {"vehicle_type": "ev"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["recommended_spot"]["spot_type"], "ev")

    def test_entry_and_exit_flow_updates_spot_status(self) -> None:
        entry_response = self.client.post(
            reverse("entry"),
            {"license_plate": "33다3333", "vehicle_type": "compact", "lot_id": 1},
            format="json",
        )
        self.assertEqual(entry_response.status_code, status.HTTP_201_CREATED)
        spot_id = entry_response.data["recommended_spot"]["spot_id"]
        self.assertEqual(ParkingSpot.objects.get(spot_id=spot_id).status, "occupied")

        exit_response = self.client.post(reverse("exit"), {"license_plate": "33다3333"}, format="json")
        self.assertEqual(exit_response.status_code, status.HTTP_200_OK)
        self.assertEqual(ParkingSpot.objects.get(spot_id=spot_id).status, "vacant")
        self.assertIsNotNone(exit_response.data["transaction"]["exit_time"])

    def test_spot_status_and_camera_heartbeat(self) -> None:
        spot = ParkingSpot.objects.first()
        response = self.client.patch(
            reverse("parking-spot-set-status", kwargs={"spot_id": spot.spot_id}),
            {"status": "reserved"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "reserved")

        camera = Camera.objects.first()
        heartbeat_response = self.client.post(
            reverse("camera-heartbeat", kwargs={"camera_id": camera.camera_id}),
            {"status": "online"},
            format="json",
        )
        self.assertEqual(heartbeat_response.status_code, status.HTTP_200_OK)
        self.assertIsNotNone(heartbeat_response.data["last_heartbeat"])

    def test_dashboard_contains_summary(self) -> None:
        process_entry("44라4444", vehicle_type="sedan", lot_id=1)
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(response.data["summary"]["occupied"], 1)
        self.assertGreaterEqual(len(response.data["recent_transactions"]), 1)

    def test_vehicle_route_endpoint(self) -> None:
        record, _ = process_entry("55마5555", vehicle_type="sedan", lot_id=1)
        response = self.client.get(reverse("vehicle-route", kwargs={"vehicle_id": record.vehicle_id}))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data["waypoints"]), 2)


class ParkingServiceTests(APITestCase):
    """Validate domain services without HTTP serialization noise."""

    def setUp(self) -> None:
        seed_demo_data()

    def test_duplicate_entry_is_rejected(self) -> None:
        process_entry("66바6666", vehicle_type="sedan", lot_id=1)
        with self.assertRaises(Exception):
            process_entry("66바6666", vehicle_type="sedan", lot_id=1)

    def test_exit_requires_active_transaction(self) -> None:
        Vehicle.objects.create(license_plate="77사7777", vehicle_type="sedan")
        with self.assertRaises(Exception):
            process_exit("77사7777")

    def test_recommend_spot_returns_vacant_spot(self) -> None:
        spot = recommend_spot(lot_id=1, vehicle_type="sedan")
        self.assertEqual(spot.status, "vacant")


class ComputerVisionTests(APITestCase):
    """Validate mock CV components and homography math."""

    def test_mock_detectors_return_deterministic_results(self) -> None:
        frame = create_synthetic_frame()
        vehicle_detections = MockVehicleDetector().detect(frame.image)
        plate_detections = MockPlateDetector().detect(frame.image)
        self.assertEqual(vehicle_detections[0].label, "vehicle")
        self.assertEqual(plate_detections[0].text, "12가3456")

    def test_homography_projects_point(self) -> None:
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        dst = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
        matrix = compute_homography(src, dst)
        x, y = warp_point((5, 5), matrix)
        self.assertAlmostEqual(x, 10.0, places=4)
        self.assertAlmostEqual(y, 10.0, places=4)


class ReinforcementLearningTests(APITestCase):
    """Validate the Gymnasium-style environment and mock policy."""

    def test_environment_step_and_heuristic_policy(self) -> None:
        env = ParkingRoutingEnv(spot_types=[0, 1], coordinates=[(3, 0), (1, 0)])
        observation, _ = env.reset(options={"vehicle_type": 1})
        action = heuristic_policy(observation)
        self.assertEqual(action, 1)
        _, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["assigned_index"], 1)
        self.assertGreater(reward, 0)

    def test_load_policy_returns_heuristic_without_model(self) -> None:
        policy = load_policy()
        observation = {
            "vehicle_type": 0,
            "spot_statuses": np.array([1, 0]),
            "spot_types": np.array([0, 0]),
            "spot_coordinates": np.array([[0, 0], [2, 0]]),
        }
        self.assertEqual(policy(observation), 1)


class ProtocolTests(APITestCase):
    """Validate WebSocket/MQTT-compatible telemetry schema."""

    def test_vehicle_telemetry_round_trip(self) -> None:
        payload = {
            "car_id": 1,
            "license_plate": "12가3456",
            "pos": [10.5, 3.2],
            "status": "moving",
            "target_spot_id": 3,
        }
        message = VehicleTelemetryMessage.from_dict(payload)
        wire = message.to_dict()
        # 기존 필드는 그대로 보존되어야 한다
        self.assertEqual({k: wire[k] for k in payload}, payload)
        # CV 파이프라인이 채우는 확장 필드는 값이 없으면 None 으로 나간다
        self.assertIsNone(wire["heading_deg"])
        self.assertIsNone(wire["parking_phase"])
        # 왕복 후에도 동일한 메시지로 복원된다
        self.assertEqual(VehicleTelemetryMessage.from_dict(wire), message)
