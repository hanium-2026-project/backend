from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from controller.config import FirmwareConstants
from tools.run_recorder import (RunRecorder, _execution_gate_reason,
                                motor_duty_for)


class TestFirmwareDutyContract(unittest.TestCase):
    def test_zero_and_deadband_always_remain_zero(self):
        self.assertEqual(motor_duty_for(0.0, 0.0), 0)
        self.assertEqual(motor_duty_for(0.02, 1.0), 0)
        self.assertEqual(motor_duty_for(-0.02, -1.0), 0)

    def test_forward_reverse_share_duty_magnitude_and_keep_sign_external(self):
        for steering in (0.0, 0.5, 1.0):
            for throttle in (0.1, 0.25, 0.5, 1.0):
                self.assertEqual(motor_duty_for(throttle, steering),
                                 motor_duty_for(-throttle, steering))

    def test_duty_is_monotonic_and_saturated(self):
        fw = FirmwareConstants()
        for steering in (0.0, 0.5, 1.0):
            duties = [motor_duty_for(t, steering)
                      for t in (0.0, 0.03, 0.1, 0.25, 0.5, 1.0, 2.0)]
            self.assertEqual(duties, sorted(duties))
            self.assertLessEqual(max(duties), fw.motor_pwm_max_duty)

    def test_turning_compensation_semantics_are_preserved(self):
        for throttle in (0.25, 0.5, 1.0):
            straight = motor_duty_for(throttle, 0.0)
            weak = motor_duty_for(throttle, 0.5)
            strong = motor_duty_for(throttle, 1.0)
            self.assertLess(straight, weak)
            self.assertLess(weak, strong)


class TestExecutionGateReason(unittest.TestCase):
    def session(self, **overrides):
        values = {"alive": True, "comm_failed": False, "control_held": False}
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_running_zero_behind_comm_latch_is_never_silent(self):
        reason = _execution_gate_reason(
            self.session(control_held=True), "ARMED", None, "RUNNING",
            0.0, 0.0, None)
        self.assertEqual(reason, "COMM_ZERO_LATCH")

    def test_controller_and_firmware_zero_are_distinguished(self):
        self.assertEqual(_execution_gate_reason(
            self.session(), "ARMED", None, "RUNNING", 0.0, 0.0,
            "DIRECTION_CHANGE_INTERLOCK"), "DIRECTION_CHANGE_INTERLOCK")
        self.assertEqual(_execution_gate_reason(
            self.session(), "ARMED", None, "RUNNING", 0.2, 0.0, None),
            "ESP_APPLIED_ZERO")

    def test_route_done_is_a_route_gate_not_parked(self):
        self.assertEqual(_execution_gate_reason(
            self.session(), "ARMED", None, "DONE", 0.0, 0.0, None),
            "MISSION_DONE")


class TestCommFaultSummary(unittest.TestCase):
    def test_explicit_backend_comm_fail_is_counted_without_esp_state_change(self):
        server = SimpleNamespace(sessions={}, last_status=lambda _car_id: {})
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RunRecorder(tmp, server)
            recorder.event("COMM_FAIL", reason="COMM_TIMEOUT")
            summary = recorder.stop(outcome="ABORTED")
        self.assertEqual(summary["comm_fault_events"], 1)

    def test_stationary_zero_is_reported_as_zero_not_null(self):
        server = SimpleNamespace(sessions={}, last_status=lambda _car_id: {})
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RunRecorder(tmp, server)
            recorder._collect(
                {}, 0.0, 0.0, None, None, None, None, None, None, {})
            summary = recorder.stop(outcome="ABORTED")
        self.assertEqual(summary["throttle"], {"max": 0.0, "mean": 0.0})


class TestRunVideoRecorder(unittest.TestCase):
    def test_record_video_off_creates_no_video_artifacts(self):
        server = SimpleNamespace(sessions={}, last_status=lambda _car_id: {})
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RunRecorder(tmp, server)
            run_dir = recorder.dir
            summary = recorder.stop(outcome="ABORTED")
            self.assertNotIn("video", summary)
            self.assertFalse((run_dir / "e2e.mp4").exists())
            self.assertFalse((run_dir / "video_frames.jsonl").exists())

    def test_mp4_and_timing_sidecar_share_run_directory_and_are_playable(self):
        server = SimpleNamespace(sessions={}, last_status=lambda _car_id: {})
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RunRecorder(tmp, server)
            path = recorder.start_video(fps=4.0)
            for index in range(1, 7):
                image = np.full((480, 640, 3), index * 20, dtype=np.uint8)
                state = SimpleNamespace(
                    frame_index=index,
                    timestamp=recorder.t0 + index * 0.25,
                )
                recorder.log_video_frame(image, state)
            summary = recorder.stop(outcome="ABORTED")

            video = summary["video"]
            self.assertEqual(Path(video["path"]), path)
            self.assertEqual(path.parent, recorder.dir)
            self.assertGreater(path.stat().st_size, 0)
            timing_rows = [json.loads(line) for line in
                           Path(video["frames_path"]).read_text(
                               encoding="utf-8").splitlines()]
            self.assertEqual(len(timing_rows), 6)
            self.assertEqual(timing_rows[0]["tracker_frame_index"], 1)
            self.assertAlmostEqual(timing_rows[-1]["elapsed_s"], 1.5)

            capture = cv2.VideoCapture(str(path))
            try:
                ok, frame = capture.read()
                self.assertTrue(capture.isOpened())
                self.assertTrue(ok)
                self.assertEqual(frame.shape[:2], (480, 640))
            finally:
                capture.release()


class TestControllerTelemetryRecording(unittest.TestCase):
    def test_control_jsonl_preserves_controller_diagnostics_and_overrides(self):
        command = SimpleNamespace(
            throttle=0.2, steering=-0.3, logical_steering=0.3,
            heading_error_deg=12.0, target_bearing_deg=25.0, reason="",
            telemetry={
                "steering_raw": 0.42,
                "steering_proportional_term": 0.40,
                "steering_derivative_term": 0.02,
                "throttle_requested_raw": 0.24,
                "cross_track_error_mm": 7.0,
                "cross_track_definition": "SIGNED_ARC_RADIAL_ERROR",
                "pose_fresh": True,
            },
        )
        target = SimpleNamespace(
            x_mm=300.0, y_mm=200.0, target_heading_deg=20.0,
            route_id=3, waypoint_id=2, phase="ENTRY",
            motion_direction="REVERSE", speed_cm_s=8.0, curvature=0.001,
            path_capture_tolerance_cm=10.0, capture_tolerance_cm=None,
            position_tolerance_cm=5.0, heading_tolerance_deg=5.0,
            heading_required=True,
        )
        mission = SimpleNamespace(
            status=SimpleNamespace(value="RUNNING"), current_phase="ENTRY",
            index=1, total=4, replan_reason=None, recovery_attempts=0,
        )
        runner = SimpleNamespace(
            current_target=target, mission=mission,
            last_tick_result=SimpleNamespace(command=command),
            config=SimpleNamespace(
                feedforward_steering=lambda *_args, **_kwargs: 0.1),
            host=SimpleNamespace(
                approach_guard=SimpleNamespace(stage=SimpleNamespace(value="FINE"),
                                               best_distance_cm=4.0),
                final_pose_guard=SimpleNamespace(count=0),
                authority=SimpleNamespace(state=SimpleNamespace(value="ARMED"),
                                          fault_reason=None),
                reverse_observation_state="REVERSE_TRACK_PRIMARY"),
        )
        session = SimpleNamespace(
            alive=True, comm_failed=False, control_held=False,
            latest_control={"throttle": 0.2, "steering": -0.3},
            firmware_version="test", last_rx_gap_ms=10.0,
            max_rx_gap_ms=20.0,
        )
        status = {
            "applied_throttle": 0.2, "applied_steering": -0.3,
            "latest_control_seq": 8, "encoder_count": 12,
        }
        server = SimpleNamespace(
            sessions={1: session}, last_status=lambda _car_id: status)
        pose = {
            "x_mm": 100.0, "y_mm": 100.0, "heading_deg": 10.0,
            "heading_source": "FRONT_CUSHION", "obs_time": 1.0,
        }
        lifecycle = {
            "slot_id": "A1", "slot_center_x_mm": 425.0,
            "slot_center_y_mm": 150.0, "parked_heading_deg": 270.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            recorder = RunRecorder(
                tmp, server, pose_provider=lambda: pose,
                runner_provider=lambda: runner,
                lifecycle_provider=lambda: lifecycle)
            control_path = recorder.dir / "control.jsonl"
            recorder._tick()
            recorder.stop(outcome="ABORTED")
            row = json.loads(control_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(row["steering_raw"], 0.42)
        self.assertEqual(row["cross_track_error_mm"], 7.0)
        self.assertEqual(row["throttle_requested_raw"], 0.24)
        self.assertEqual(row["throttle_command_final"], 0.2)
        self.assertEqual(row["steering_command_final"], -0.3)
        self.assertEqual(row["waypoint_index"], 1)
        self.assertEqual(row["waypoint_total_count"], 4)
        self.assertEqual(row["slot_id"], "A1")
        self.assertEqual(row["control_override_reason"], "EXECUTING")


if __name__ == "__main__":
    unittest.main()
