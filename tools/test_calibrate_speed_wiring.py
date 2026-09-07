"""`--calibrate-speed` 의 throttle 배선이 실제로 살아 있는지 고정한다.

실차 run_20260903_123402 에서 차가 전혀 움직이지 않았다. 원인은 actuator 가
아니라 계측 루프였다: pose 조회가 ``track_of_car`` 에 의존했는데, ``_bind_car``
는 차가 entry_nodes(junction/entrance) 에 있을 때만 호출된다. 통로 한가운데
(node=lane_pt_3)에 놓인 차는 171 프레임 내내 car_id=None 이었고, 계측 루프는
첫 primitive 를 시작하지도 못한 채 대기만 했다 (throttle_cmd 는 0.0/None 뿐).

여기서 두 가지를 고정한다.

1. 사람이 WASD 로 못 내는 크기(0.10)가 production 경로를 그대로 타고
   wire 까지 나간다 — 새 actuator 경로를 만들지 않았다는 증거.
2. 계측 루프가 의존해도 되는 것과 안 되는 것 (binding 없이도 pose 를 찾는다).
"""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from host_control import HostController, HostWaypointMission
from host_control.producers import ManualInput


class _RecordingSender:
    """VehicleServerDirectSender 자리에 끼워 wire 값을 그대로 받는다."""

    def __init__(self) -> None:
        self.sent: list[tuple[float, float]] = []

    def send_command(self, cmd) -> dict:
        self.sent.append((cmd.throttle, cmd.steering))
        return {}

    def send_zero(self) -> dict:
        self.sent.append((0.0, 0.0))
        return {}


def _manual_host() -> tuple[HostController, _RecordingSender]:
    """HybridControlMux.switch_to_manual 이 만드는 것과 같은 수동 구성."""
    sender = _RecordingSender()
    # 수동 모드는 자체 한계를 쓴다 (mux 가 max_throttle=1.0 으로 교체한다).
    host = HostController(config=ControllerConfig(max_throttle=1.0,
                                                  allow_reverse=True),
                          mission=HostWaypointMission([]), sender=sender)
    host.arm_manual()
    return host, sender


class CalibrationThrottleReachesTheWire(unittest.TestCase):
    """요청문 10절: 0.10 요청 -> wire 0.10 -> primitive 종료 시 0."""

    def test_forward_010_reaches_the_wire(self):
        host, sender = _manual_host()
        result = host.tick(100.0, manual_input=ManualInput(0.10, 0.0))
        self.assertAlmostEqual(result.command.throttle, 0.10, places=6)
        self.assertAlmostEqual(sender.sent[-1][0], 0.10, places=6)

    def test_steering_stays_centred(self):
        host, sender = _manual_host()
        host.tick(100.0, manual_input=ManualInput(0.10, 0.0))
        self.assertAlmostEqual(sender.sent[-1][1], 0.0, places=6)

    def test_primitive_end_sends_zero(self):
        host, sender = _manual_host()
        host.tick(100.0, manual_input=ManualInput(0.10, 0.0))
        host.tick(100.1, manual_input=ManualInput(0.0, 0.0))
        self.assertAlmostEqual(sender.sent[-1][0], 0.0, places=6)

    def test_reverse_010_reaches_the_wire(self):
        host, sender = _manual_host()
        result = host.tick(100.0, manual_input=ManualInput(-0.10, 0.0))
        self.assertAlmostEqual(result.command.throttle, -0.10, places=6)
        self.assertAlmostEqual(sender.sent[-1][0], -0.10, places=6)

    def test_each_calibration_level_survives_the_wire(self):
        """0.10/0.15/0.25 셋 다 그대로 나가야 계측에 의미가 있다."""
        for level in (0.10, 0.15, 0.25):
            with self.subTest(level=level):
                host, sender = _manual_host()
                host.tick(100.0, manual_input=ManualInput(level, 0.0))
                self.assertAlmostEqual(sender.sent[-1][0], level, places=6)


class WasdCannotProduceCalibrationLevels(unittest.TestCase):
    """계측 runner 가 왜 필요한지 — WASD 로는 이 크기를 못 낸다."""

    def test_wasd_throttle_is_binary(self):
        from control.wasd_logic import compute_throttle
        self.assertEqual(compute_throttle({"w"}), 1.0)
        self.assertEqual(compute_throttle({"s"}), -1.0)
        self.assertEqual(compute_throttle(set()), 0.0)

    def test_wasd_cannot_reach_any_calibration_level(self):
        from control.wasd_logic import compute_throttle
        reachable = {compute_throttle(k) for k in ({"w"}, {"s"}, set(),
                                                   {"w", "s"})}
        for level in (0.10, 0.15, 0.25):
            self.assertNotIn(level, reachable)


class CalibrationPoseLookup(unittest.TestCase):
    """계측 루프는 car binding 없이도 pose 를 찾아야 한다 (123402 회귀)."""

    @staticmethod
    def _pose_lookup(views: dict, track_of_car: dict, car_id: int = 1):
        """run_pipeline._run_speed_calibration 의 pose() 와 같은 규칙."""
        bound = track_of_car.get(car_id)
        if bound is not None and bound in views:
            view = views[bound]
        else:
            seen = [v for v in views.values() if v.heading_deg is not None]
            if len(seen) != 1:
                return None
            view = seen[0]
        if view.heading_deg is None:
            return None
        return (view.position_mm[0], view.position_mm[1], view.heading_deg)

    @staticmethod
    def _view(x, y, h):
        from pipeline.runner import VehicleView
        return VehicleView(track_id=1, car_id=None, node="lane_pt_3",
                           position_mm=(x, y), heading_deg=h,
                           heading_source="FRONT_CUSHION", last_obs_time=1.0)

    def test_unbound_single_track_is_usable(self):
        """실차 123402 의 상황: node=lane_pt_3, car_id=None, track 하나."""
        views = {1: self._view(606.0, 669.0, 355.4)}
        self.assertEqual(self._pose_lookup(views, {}),
                         (606.0, 669.0, 355.4))

    def test_binding_is_preferred_when_present(self):
        views = {1: self._view(606.0, 669.0, 355.4),
                 2: self._view(200.0, 200.0, 90.0)}
        self.assertEqual(self._pose_lookup(views, {1: 2}),
                         (200.0, 200.0, 90.0))

    def test_ambiguous_tracks_refuse_to_drive(self):
        """어느 것이 차인지 모르면 움직이지 않는다."""
        views = {1: self._view(606.0, 669.0, 355.4),
                 2: self._view(200.0, 200.0, 90.0)}
        self.assertIsNone(self._pose_lookup(views, {}))

    def test_no_tracks_refuse_to_drive(self):
        self.assertIsNone(self._pose_lookup({}, {}))


class _FakePipeline:
    """계측 루프가 쓰는 표면만 흉내낸다 (실차 123402 상황 그대로)."""

    def __init__(self) -> None:
        from pipeline.runner import VehicleView
        self.views = {1: VehicleView(
            track_id=1, car_id=None, node="lane_pt_3",
            position_mm=(606.0, 669.0), heading_deg=355.4,
            heading_source="FRONT_CUSHION", last_obs_time=1.0)}
        self.track_of_car: dict = {}          # ← 묶이지 않은 상태
        self.hybrid_controls = {1: object()}
        self.drive: list[tuple[float, float]] = []
        self.manual_stopped = False

    def run_camera(self, **_kw) -> None:
        import time
        time.sleep(30)

    def set_manual_drive(self, _car_id, throttle, steering) -> None:
        self.drive.append((round(throttle, 3), round(steering, 3)))

    def switch_to_manual(self, _car_id) -> None:
        pass

    def manual_stop(self, _car_id) -> None:
        self.manual_stopped = True

    def stop(self) -> None:
        pass


class SinglePrimitiveDryRun(unittest.TestCase):
    """실차 123402 자세에서 --calibrate-once 가 실제로 구동을 낸다.

    그 run 은 node=lane_pt_3 / car_id=None 이라 계측 루프가 pose 를 못 찾고
    첫 primitive 를 시작조차 못했다 (throttle_cmd 는 0.0/None 뿐). 여기서는
    **실제 _run_speed_calibration 을** 같은 조건으로 돌린다.
    """

    def _run(self, **overrides):
        from parking.management.commands.run_pipeline import Command
        cmd = Command()
        logs: list[str] = []
        cmd.stdout = type("O", (), {
            "write": lambda _s, m, *a, **k: logs.append(str(m))})()
        cmd.style = type("S", (), {
            "WARNING": staticmethod(lambda x: x),
            "SUCCESS": staticmethod(lambda x: x)})()
        pipeline = _FakePipeline()
        options = {"calibrate_throttles": "0.10", "calibrate_seconds": 0.4,
                   "calibrate_once": True, "parking_throttle": 0.25,
                   "max_frames": None, "show": False}
        options.update(overrides)
        cmd._run_speed_calibration(pipeline, options)
        return pipeline, logs

    def test_first_primitive_commands_010_forward(self):
        pipeline, _logs = self._run()
        commanded = {t for t, _s in pipeline.drive}
        self.assertIn(0.10, commanded, "0.10 이 한 번도 명령되지 않았다")

    def test_steering_stays_zero_throughout(self):
        pipeline, _logs = self._run()
        self.assertEqual({s for _t, s in pipeline.drive}, {0.0})

    def test_primitive_ends_at_zero(self):
        pipeline, _logs = self._run()
        self.assertEqual(pipeline.drive[-1], (0.0, 0.0))
        self.assertTrue(pipeline.manual_stopped)

    def test_once_mode_runs_forward_only(self):
        """단일 모드는 후진 primitive 를 만들지 않는다."""
        pipeline, _logs = self._run()
        self.assertEqual([t for t, _s in pipeline.drive if t < 0.0], [])

    def test_throttle_is_clamped_to_parking_limit(self):
        """계측이라도 주차 상한을 넘지 않는다."""
        pipeline, _logs = self._run(calibrate_throttles="0.90")
        self.assertEqual(max(t for t, _s in pipeline.drive), 0.25)


if __name__ == "__main__":
    unittest.main()
