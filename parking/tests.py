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
from rl.inference import heuristic_policy, load_policy, select_action
from rl.parking_env import (NUM_SLOTS, SLOT_NAMES, STATE_DIM, WAIT_ACTION,
                            ParkingRoutingEnv)


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
        # seed_demo_data 는 실제 주차장 8칸(전부 standard)만 만든다. 선호 타입
        # 우선순위(SPOT_PREFERENCE_BY_VEHICLE)를 확인하려면 그 타입의 칸이
        # 하나는 있어야 하므로 여기서 EV 칸을 하나 추가한다.
        lot = ParkingLot.objects.first()
        ParkingSpot.objects.create(lot=lot, section="EV1", spot_type="ev",
                                   coord_x=9, coord_y=9, status="vacant")
        response = self.client.get(reverse("recommend-spot"), {"vehicle_type": "ev"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["recommended_spot"]["spot_type"], "ev")

    def test_recommendation_falls_back_when_preferred_type_is_absent(self) -> None:
        """EV 칸이 없으면 선호 목록의 다음 타입(standard)으로 내려간다."""
        response = self.client.get(reverse("recommend-spot"), {"vehicle_type": "ev"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["recommended_spot"]["spot_type"], "standard")

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
        # 2클래스 실차 모델로 바뀌면서 라벨이 rc_car 로 확정됐다
        # (cv/vehicle_detector.py::LABEL_CAR, 짝이 되는 front_cushion 과 함께).
        self.assertEqual(vehicle_detections[0].label, "rc_car")
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
        # 환경은 이제 실제 주차장(8칸 고정)을 그대로 쓴다 — spot_types/coordinates
        # 를 주입하지 않는다. 정책도 관측이 아니라 action_masks 로 고른다.
        env = ParkingRoutingEnv()
        env.reset(seed=0)
        masks = env.action_masks()
        self.assertEqual(masks.shape, (NUM_SLOTS + 1,))

        action = heuristic_policy(masks)
        self.assertIn(action, range(NUM_SLOTS))
        self.assertTrue(bool(masks[action]), "마스크가 막은 칸을 골랐다")

        _obs, _reward, terminated, truncated, _info = env.step(action)
        self.assertFalse(truncated)
        self.assertIsInstance(terminated, bool)

    def test_heuristic_policy_never_picks_a_masked_slot(self) -> None:
        masks = np.zeros(NUM_SLOTS + 1, dtype=bool)
        masks[SLOT_NAMES.index("A2")] = True
        masks[WAIT_ACTION] = True
        self.assertEqual(heuristic_policy(masks), SLOT_NAMES.index("A2"))

    def test_heuristic_policy_waits_when_every_slot_is_taken(self) -> None:
        masks = np.zeros(NUM_SLOTS + 1, dtype=bool)
        masks[WAIT_ACTION] = True
        self.assertEqual(heuristic_policy(masks), WAIT_ACTION)

    def test_select_action_falls_back_to_the_heuristic_without_a_policy(self):
        """학습 정책을 못 불러오면 결정론적 heuristic 으로 떨어진다.

        load_policy 는 예외를 던지지 않고 None 을 돌려주고(파일이 없거나
        sb3-contrib 미설치), select_action 이 그 자리에서 heuristic 으로
        대체한다. 실차 실행이 실제로 타는 경로가 이쪽이다.
        """
        self.assertIsNone(load_policy("models/__no_such_policy__.zip"))

        observation = np.zeros(STATE_DIM, dtype=np.float32)
        masks = np.zeros(NUM_SLOTS + 1, dtype=bool)
        masks[SLOT_NAMES.index("A2")] = True
        masks[WAIT_ACTION] = True
        action = select_action(observation, masks,
                               model_path="models/__no_such_policy__.zip")
        self.assertEqual(action, SLOT_NAMES.index("A2"))
        self.assertEqual(action, heuristic_policy(masks))


class ProtocolTests(APITestCase):
    """Validate WebSocket/MQTT-compatible telemetry schema."""

    def test_vehicle_telemetry_round_trip(self) -> None:
        payload = {
            "car_id": 1,
            "license_plate": "12가3456",
            "pos": [10.5, 3.2],
            "status": "moving",
            "target_spot_id": 3,
            "track_id": 7,
            "assigned_slot": "B1",
            "parking_stage": "PARKING",
            "connection_state": "CONNECTED",
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
