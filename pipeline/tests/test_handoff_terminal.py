"""GLOBAL handoff capture and repeated-replan regressions from real E2E."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from control.auto_host_runner import MissionStatus
from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline, VehicleView


def target(*, final: bool = True):
    return SimpleNamespace(
        x_mm=425.0, y_mm=600.0, target_heading_deg=0.0,
        position_tolerance_cm=5.0, heading_tolerance_deg=12.0,
        phase="FINAL", is_final=final, route_id=1, waypoint_id=1,
        motion_direction=SimpleNamespace(value="FORWARD"),
    )


class Mission:
    def __init__(self) -> None:
        self.is_recovering = False
        self.requests: list[str] = []

    def request_replan(self, reason: str) -> None:
        self.requests.append(reason)


class Runner:
    def __init__(self, current) -> None:
        self.status = MissionStatus.RUNNING
        self.current_target = current
        self.failed_target = current
        self.replan_reason = "PATH_DEVIATION"
        self.mission = Mission()
        self.prepared = 0
        self.terminal = bool(current.is_final)

    @property
    def current_is_terminal(self) -> bool:
        return self.terminal

    def prepare_route_switch(self) -> None:
        self.prepared += 1


class TestHandoffTerminal(unittest.TestCase):
    def setUp(self) -> None:
        p = ParkingPipeline.__new__(ParkingPipeline)
        p.config = PipelineConfig(parking_mode="rear", deviation_frames=3)
        p._parking_stage = {}
        p._parking_plan_wait = {}
        p._parking_setup_wait = {}
        p._parking_recovery_attempts = {}
        p._deviation_streak = {}
        p._reverse_closest = {}
        p._forward_closest = {}
        p._heading_wait_state = {}
        p._last_replan_signature = {}
        p._replan_attempts = {}
        p._auto_host_slot = {1: "B1"}
        p.track_of_car = {1: 7}
        p.dashboard = SimpleNamespace(push_event=lambda *a, **kw: None)
        p.zeroed = []
        p.server = SimpleNamespace(stop_control=lambda car: p.zeroed.append(car))
        p.events = []
        p.on_event_record = lambda name, **fields: p.events.append((name, fields))
        self.p = p
        self.t = target()
        self.runner = Runner(self.t)
        self.p.auto_hosts = {1: self.runner}
        self.view = VehicleView(
            track_id=7, car_id=1, position_mm=(441.0, 598.0),
            heading_deg=349.0, heading_source="FRONT_CUSHION",
            last_obs_time=1.0)
        self.p.views = {7: self.view}

    def _observe_stopped(self) -> None:
        for i in range(self.p.config.stationary_window):
            self.view.recent.append(self.view.position_mm)
            self.view.last_obs_time += 0.1 + i * 0.001

    def test_15_to_25mm_overshoot_never_requests_global_replan(self) -> None:
        for i, pose in enumerate(((441.0, 598.0), (446.0, 600.0),
                                  (450.0, 602.0), (447.0, 600.0)), 1):
            self.view.position_mm = pose
            self.view.last_obs_time = float(i)
            self.p._check_path_deviation(self.view, self.runner)
        self.assertEqual(self.runner.mission.requests, [])

        # HostController's existing FINAL confirmation reports DONE; pipeline
        # then waits for a distinct observation before invoking parking.
        self.p._on_auto_host_status(1, MissionStatus.RUNNING,
                                    MissionStatus.DONE)
        self.assertEqual(self.p._parking_stage[1],
                         "PARKING_HANDOFF_PENDING")
        started = []
        self.p._start_rear_parking_stage = lambda car: started.append(car) or True
        self.p._maybe_start_rear_after_stop(self.view)
        self.assertEqual(started, [])
        self._observe_stopped()
        self.p._maybe_start_rear_after_stop(self.view)
        self.assertEqual(started, [1])

    def test_outside_capture_in_handoff_region_transitions_without_replan(self) -> None:
        self.view.position_mm = (500.0, 600.0)  # 75 mm: outside FINAL 50 mm
        for _ in range(3):
            self.p._check_path_deviation(self.view, self.runner)
        self.assertEqual(self.runner.mission.requests, [])
        self.assertEqual(self.p._parking_stage[1],
                         "PARKING_HANDOFF_PENDING")
        self.assertGreaterEqual(self.runner.prepared, 1)
        self.assertIn(1, self.p.zeroed)
        started = []
        self.p._start_rear_parking_stage = lambda car: started.append(car) or True
        self._observe_stopped()
        self.p._maybe_start_rear_after_stop(self.view)
        self.assertEqual(started, [1])

    def test_phase_boundary_waits_for_physical_stop_not_just_new_frame(self) -> None:
        self.p._on_auto_host_status(1, MissionStatus.RUNNING,
                                    MissionStatus.DONE)
        started = []
        self.p._start_rear_parking_stage = lambda car: started.append(car) or True
        for i, x in enumerate((455.0, 470.0, 485.0), 1):
            self.view.position_mm = (x, 600.0)
            self.view.recent.append(self.view.position_mm)
            self.view.last_obs_time += 0.1
            self.p._maybe_start_rear_after_stop(self.view)
        self.assertEqual(started, [])
        self.assertIn(1, self.p.zeroed)
        self.assertTrue(any(name == "WAIT_FOR_PHYSICAL_STOP"
                            for name, _ in self.p.events))
        self.view.position_mm = (485.0, 600.0)
        self.view.recent.clear()
        self._observe_stopped()
        self.p._maybe_start_rear_after_stop(self.view)
        self.assertEqual(started, [1])

    def test_175219_175349_setup_done_motion_cannot_activate_rear(self) -> None:
        fixtures = (
            ((242.5, 627.4), (212.0, 631.9)),
            ((232.0, 621.0), (202.0, 625.0)),
        )
        for setup_done, first_rear_pose in fixtures:
            with self.subTest(setup_done=setup_done):
                self.p._parking_stage[1] = "SETUP"
                self.view.position_mm = setup_done
                self.view.recent.clear()
                self.view.recent.append(setup_done)
                self.view.last_obs_time += 1.0
                self.p._on_auto_host_status(
                    1, MissionStatus.RUNNING, MissionStatus.DONE)
                started = []
                self.p._start_rear_parking_stage = (
                    lambda car, out=started: out.append(car) or True)

                self.view.position_mm = first_rear_pose
                self.view.recent.append(first_rear_pose)
                self.view.last_obs_time += 0.1
                self.p._maybe_start_rear_after_stop(self.view)
                self.assertEqual(started, [],
                                 "rear route activated while car was still moving")

                self.view.recent.clear()
                self._observe_stopped()
                self.p._maybe_start_rear_after_stop(self.view)
                self.assertEqual(started, [1])

    def test_intermediate_waypoint_keeps_path_deviation_behavior(self) -> None:
        self.runner.current_target = target(final=False)
        self.runner.terminal = False
        self.view.position_mm = (515.0, 600.0)  # 90 mm, outside 50 mm capture
        for _ in range(3):
            self.p._check_path_deviation(self.view, self.runner)
        self.assertEqual(self.runner.mission.requests, ["PATH_DEVIATION"])

    def test_intermediate_capture_wins_before_deviation_monitor(self) -> None:
        self.runner.current_target = target(final=False)
        self.runner.terminal = False
        self.view.position_mm = (450.0, 600.0)  # 25 mm, inside 50 mm capture
        for _ in range(3):
            self.p._check_path_deviation(self.view, self.runner)
        self.assertEqual(self.runner.mission.requests, [])

    def test_setup_terminal_uses_recovery_capture_then_replans_outside(self) -> None:
        setup_end = target(final=False)
        setup_end.phase = "RECOVERY"
        setup_end.position_tolerance_cm = 8.0
        self.runner.current_target = setup_end
        self.runner.terminal = True
        self.p._parking_stage[1] = "SETUP"

        self.view.position_mm = (490.0, 600.0)  # 65 mm: captured by existing 80 mm
        self.p._check_path_deviation(self.view, self.runner)
        self.assertEqual(self.runner.mission.requests, [])

        self.view.position_mm = (515.0, 600.0)  # 90 mm: outside setup completion
        for _ in range(3):
            self.p._check_path_deviation(self.view, self.runner)
        self.assertEqual(self.runner.mission.requests, ["PATH_DEVIATION"])

    def test_repeated_same_target_reason_and_pose_is_faulted_before_reload(self) -> None:
        self.view.position_mm = (450.0, 600.0)
        self.runner.failed_target = target(final=False)
        self.p._last_replan_signature[1] = (
            "PATH_DEVIATION", 425.0, 600.0, 447.0, 600.0)

        self.p._replan_auto_host(1)

        self.assertEqual(self.p._parking_stage[1], "WAIT_REPEATED_REPLAN")
        self.assertEqual(self.p.zeroed, [1])
        self.assertIn("REPEATED_IDENTICAL_REPLAN",
                      [fields["reason"] for name, fields in self.p.events
                       if name == "FAULT"])

    def test_replan_attempt_limit_remains_three(self) -> None:
        self.assertEqual(self.p.config.max_replan_attempts, 3)


if __name__ == "__main__":
    unittest.main()
