"""HostController 통합 테스트: authority 중재 / stale→fault / 자동재출발 차단 / transport."""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import (ControlCommand, ControlMode, MotionDirection,
                               Pose, Waypoint)
from host_control import (
    Authority,
    HostController,
    HostWaypointMission,
    ManualInput,
    MissionStatus,
)


def fresh_pose(x=0.0, y=0.0, h=0.0, t=100.0):
    return Pose(x, y, h, timestamp=t)


def two_wp_mission():
    return HostWaypointMission([
        Waypoint(400, 50, position_tolerance_cm=8),
        Waypoint(900, 200, position_tolerance_cm=8, is_final=True),
    ])


class TestAuthorityArbitration(unittest.TestCase):
    def test_disarmed_zero(self) -> None:
        hc = HostController(mission=two_wp_mission())
        r = hc.tick(100.0, observation=fresh_pose())
        self.assertEqual(r.authority, Authority.DISARMED)
        self.assertEqual(r.command.throttle, 0.0)
        self.assertEqual(r.command.steering, 0.0)
        self.assertEqual(r.payload["type"], "DIRECT_CONTROL")

    def test_manual_only_reflects_manual(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_manual()
        r = hc.tick(100.0, observation=fresh_pose(),
                    manual_input=ManualInput(throttle=0.3, steering=1.0))
        self.assertEqual(r.authority, Authority.MANUAL)
        self.assertGreater(r.command.throttle, 0.0)
        self.assertLess(r.command.steering, 0.0)  # 논리 LEFT → wire 음수

    def test_manual_ignores_pose_waypoint(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_manual()
        # manual_input 없음 → 사람 입력 없음 → zero (waypoint 로 자율주행하지 않음)
        r = hc.tick(100.0, observation=fresh_pose())
        self.assertEqual(r.command.throttle, 0.0)

    def test_auto_only_reflects_autonomous(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_auto()
        # AUTO 인데 manual_input 을 줘도 무시되어야 함
        r = hc.tick(100.0, observation=fresh_pose(),
                    manual_input=ManualInput(throttle=1.0, steering=-1.0))
        self.assertEqual(r.authority, Authority.AUTO_HOST)
        # 자율 출력(전방 목표) 이 반영, manual 의 강한 우회전(-1 논리)이 아님
        self.assertGreater(r.command.throttle, 0.0)

    def test_faulted_zero(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_auto()
        hc.fault("TEST")
        r = hc.tick(100.0, observation=fresh_pose())
        self.assertEqual(r.authority, Authority.FAULTED)
        self.assertEqual(r.command.throttle, 0.0)


class TestStaleAndFault(unittest.TestCase):
    def test_stale_observation_faults_and_zero(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_auto()
        # 관측 시각 100.0 로 한 번 관측
        hc.tick(100.0, observation=fresh_pose(t=100.0))
        # 이후 새 관측 없이 0.7s 경과 → stale → FAULTED latch
        r = hc.tick(100.7)
        self.assertEqual(r.authority, Authority.FAULTED)
        self.assertEqual(r.command.throttle, 0.0)
        self.assertIn(r.command.reason, ("POSE_STALE",))

    def test_no_auto_resume_after_stale(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_auto()
        hc.tick(100.0, observation=fresh_pose(t=100.0))
        hc.tick(100.7)  # → FAULTED
        # 신선한 관측이 다시 들어와도 자동으로 non-zero 로 복귀하면 안 됨
        r = hc.tick(101.0, observation=fresh_pose(x=10, t=101.0))
        self.assertEqual(r.authority, Authority.FAULTED)
        self.assertEqual(r.command.throttle, 0.0)
        # 명시적 re-arm 후에만 주행 가능
        hc.re_arm_auto()
        r2 = hc.tick(101.1, observation=fresh_pose(x=10, t=101.1))
        self.assertEqual(r2.authority, Authority.AUTO_HOST)

    def test_invalid_pose_faults(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_auto()
        r = hc.tick(100.0, observation=Pose(0, 0, 0.0, timestamp=100.0, valid=False))
        self.assertEqual(r.authority, Authority.FAULTED)

    def test_no_observation_yet_holds_without_fault(self) -> None:
        # 관측 전(WARMUP): fault 아님, zero 유지
        hc = HostController(mission=two_wp_mission())
        hc.arm_auto()
        r = hc.tick(100.0)  # 관측 없음
        self.assertEqual(r.authority, Authority.AUTO_HOST)
        self.assertEqual(r.command.throttle, 0.0)

    def test_stop_latches(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_auto()
        hc.stop()
        r = hc.tick(100.0, observation=fresh_pose())
        self.assertEqual(r.authority, Authority.FAULTED)
        self.assertEqual(r.command.throttle, 0.0)


class TestRecoveryHold(unittest.TestCase):
    @staticmethod
    def _reverse_host(*, timeout: float = 0.5) -> HostController:
        mission = HostWaypointMission([
            Waypoint(
                -500.0, 200.0, target_heading_deg=10.0,
                position_tolerance_cm=4.0, phase="ENTRY",
                motion_direction=MotionDirection.REVERSE,
                curvature=1.0 / 800.0,
            )
        ])
        host = HostController(
            mission=mission,
            config=ControllerConfig(
                allow_reverse=True,
                reverse_heading_wait_timeout_s=timeout,
            ),
        )
        host.arm_auto()
        return host

    def test_fresh_unsafe_reverse_heading_has_bounded_replan_timeout(self) -> None:
        mission = HostWaypointMission([
            Waypoint(
                500.0, 0.0, target_heading_deg=0.0,
                position_tolerance_cm=4.0, phase="ENTRY",
                motion_direction=MotionDirection.REVERSE,
                curvature=1.0 / 800.0,
            )
        ])
        hc = HostController(
            mission=mission,
            config=ControllerConfig(
                allow_reverse=True, reverse_heading_wait_timeout_s=0.5))
        hc.arm_auto()

        def trajectory_pose(t: float) -> Pose:
            return Pose(0.0, 0.0, 0.0, timestamp=t,
                        heading_source="TRAJECTORY")

        first = hc.tick(100.0, observation=trajectory_pose(100.0))
        waiting = hc.tick(100.4, observation=trajectory_pose(100.4))
        timed_out = hc.tick(100.5, observation=trajectory_pose(100.5))
        self.assertEqual(first.command.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(waiting.command.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(timed_out.command.reason, "REVERSE_HEADING_TIMEOUT")
        self.assertIs(timed_out.mission_status, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(timed_out.command.throttle, 0.0)

    def test_reverse_last_valid_holds_zero_then_fresh_heading_resumes(self) -> None:
        hc = self._reverse_host()

        moving = hc.tick(
            100.0,
            observation=Pose(
                0.0, 0.0, 0.0, timestamp=100.0,
                heading_source="FRONT_CUSHION"),
        )
        held = hc.tick(
            100.1,
            observation=Pose(
                -10.0, 3.0, 0.0, timestamp=100.1,
                heading_source="LAST_VALID"),
        )
        resumed = hc.tick(
            100.2,
            observation=Pose(
                -20.0, 6.0, 1.0, timestamp=100.2,
                heading_source="FRONT_CUSHION"),
        )

        self.assertLess(moving.command.throttle, 0.0)
        self.assertEqual(held.command.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(held.command.throttle, 0.0)
        self.assertLess(resumed.command.throttle, 0.0)
        self.assertIs(resumed.authority, Authority.AUTO_HOST)
        self.assertIs(resumed.mission_status, MissionStatus.RUNNING)

    def test_reverse_pose_gap_holds_zero_without_latching_then_resumes(self) -> None:
        hc = self._reverse_host(timeout=0.5)
        hc.tick(
            100.0,
            observation=Pose(
                0.0, 0.0, 0.0, timestamp=100.0,
                heading_source="FRONT_CUSHION"),
        )

        stale = hc.tick(100.6)
        resumed = hc.tick(
            100.7,
            observation=Pose(
                -20.0, 6.0, 1.0, timestamp=100.7,
                heading_source="FRONT_CUSHION"),
        )

        self.assertEqual(stale.command.reason, "POSE_STALE")
        self.assertEqual(stale.command.throttle, 0.0)
        self.assertIs(stale.authority, Authority.AUTO_HOST)
        self.assertIs(stale.mission_status, MissionStatus.RUNNING)
        self.assertLess(resumed.command.throttle, 0.0)

    def test_reverse_pose_gap_times_out_to_replan_not_silent_fault(self) -> None:
        hc = self._reverse_host(timeout=0.3)
        hc.tick(
            100.0,
            observation=Pose(
                0.0, 0.0, 0.0, timestamp=100.0,
                heading_source="FRONT_CUSHION"),
        )
        held = hc.tick(100.6)
        timed_out = hc.tick(100.9)

        self.assertEqual(held.command.reason, "POSE_STALE")
        self.assertEqual(timed_out.command.reason, "REVERSE_HEADING_TIMEOUT")
        self.assertEqual(timed_out.command.throttle, 0.0)
        self.assertIs(timed_out.authority, Authority.AUTO_HOST)
        self.assertIs(timed_out.mission_status, MissionStatus.REPLAN_REQUIRED)

    def test_reverse_no_pose_after_direction_interlock_is_bounded(self) -> None:
        align = Waypoint(
            60.0, 0.0, target_heading_deg=0.0,
            position_tolerance_cm=2.0, heading_required=True,
            heading_tolerance_deg=5.0, route_id=8, waypoint_id=1,
            phase="ALIGN", motion_direction=MotionDirection.FORWARD,
            curvature=1.0 / 1000.0, path_capture_tolerance_cm=10.0,
        )
        entry = Waypoint(
            -500.0, 0.0, route_id=8, waypoint_id=2, phase="ENTRY",
            motion_direction=MotionDirection.REVERSE,
            curvature=1.0 / 800.0,
        )
        hc = HostController(
            mission=HostWaypointMission([align, entry]),
            config=ControllerConfig(
                allow_reverse=True, reverse_heading_wait_timeout_s=0.3),
        )
        hc.arm_auto()
        for i, x in enumerate((0.0, 15.0, 30.0, 60.0)):
            t = 10.0 + i * 0.1
            hc.tick(
                t, observation=Pose(
                    x, 0.0, 0.0, timestamp=t,
                    heading_source="TRAJECTORY"),
            )
        interlock = hc.tick(10.4)
        no_pose = hc.tick(10.5)
        timed_out = hc.tick(10.81)

        self.assertEqual(interlock.command.reason, "DIRECTION_CHANGE_STOP")
        self.assertEqual(no_pose.command.reason, "NO_POSE")
        self.assertEqual(timed_out.command.reason, "REVERSE_HEADING_TIMEOUT")
        self.assertIs(timed_out.mission_status, MissionStatus.REPLAN_REQUIRED)

    def test_reverse_continue_uses_direction_corrected_quality_trajectory(self) -> None:
        """161237: fresh centres remain usable after a short cushion dropout."""
        hc = self._reverse_host(timeout=2.5)
        primary = hc.tick(
            15.09,
            observation=Pose(
                476.1, 586.0, 349.9, timestamp=15.09,
                heading_source="FRONT_CUSHION"),
        )
        acquiring = hc.tick(
            15.31,
            observation=Pose(
                457.9, 589.0, 349.9, timestamp=15.31,
                heading_source="LAST_VALID"),
        )
        fallback = hc.tick(
            15.53,
            observation=Pose(
                432.9, 592.0, 349.9, timestamp=15.53,
                heading_source="LAST_VALID"),
        )

        self.assertLess(primary.command.throttle, 0.0)
        self.assertEqual(acquiring.command.throttle, 0.0)
        self.assertLess(fallback.command.throttle, 0.0)
        self.assertEqual(
            hc.reverse_observation_state,
            "REVERSE_TRACK_TRAJECTORY_FALLBACK",
        )

    def test_reverse_continue_rejects_jittering_last_valid(self) -> None:
        hc = self._reverse_host(timeout=2.5)
        hc.tick(
            10.0,
            observation=Pose(
                0.0, 0.0, 0.0, timestamp=10.0,
                heading_source="FRONT_CUSHION"),
        )
        held = None
        for i, (x, y) in enumerate(((2.0, 1.0), (-1.0, 2.0), (1.0, -1.0)), 1):
            held = hc.tick(
                10.0 + i * 0.2,
                observation=Pose(
                    x, y, 0.0, timestamp=10.0 + i * 0.2,
                    heading_source="LAST_VALID"),
            )
        self.assertIsNotNone(held)
        self.assertEqual(held.command.throttle, 0.0)
        self.assertEqual(held.command.reason, "REVERSE_HEADING_UNSAFE")
        self.assertEqual(
            hc.reverse_observation_state, "REVERSE_OBSERVATION_LOST")

    def test_replan_required_holds_zero_until_recovery_and_fresh_pose(self) -> None:
        mission = HostWaypointMission([
            Waypoint(
                30.0, 0.0,
                target_heading_deg=90.0,
                position_tolerance_cm=8.0,
                heading_tolerance_deg=12.0,
                heading_required=True,
                phase="APPROACH",
                is_final=True,
            )
        ])
        hc = HostController(mission=mission)
        hc.arm_auto()

        r1 = hc.tick(100.0, observation=fresh_pose(h=0.0, t=100.0))
        self.assertIs(r1.mission_status, MissionStatus.REPLAN_REQUIRED)
        self.assertEqual(r1.command.throttle, 0.0)

        r2 = hc.tick(100.1, observation=fresh_pose(h=0.0, t=100.1))
        self.assertEqual(r2.command.throttle, 0.0)
        self.assertEqual(r2.command.reason, "MISSION_REPLAN_REQUIRED")

        hc.prepare_route_switch()
        mission.load_recovery([Waypoint(500.0, 0.0, phase="RECOVERY")])

        r3 = hc.tick(100.2)
        self.assertIs(r3.mission_status, MissionStatus.RUNNING)
        self.assertEqual(r3.command.throttle, 0.0)
        self.assertEqual(r3.command.reason, "NO_POSE")

        r4 = hc.tick(100.3, observation=fresh_pose(h=0.0, t=100.3))
        self.assertGreater(r4.command.throttle, 0.0)

    def test_recovery_failed_is_latched_zero(self) -> None:
        mission = HostWaypointMission([
            Waypoint(500.0, 0.0, phase="APPROACH", is_final=True)
        ], max_recovery_attempts=1)
        align = ControlCommand(
            0.0, 0.0, ControlMode.ALIGN, False, 2.0, 30.0, 0.0,
            "HEADING_OUT_OF_TOLERANCE",
        )
        mission.notify_result(align)
        mission.load_recovery([Waypoint(0.0, -100.0, phase="RECOVERY")])
        mission.notify_result(align)
        status = mission.load_recovery([Waypoint(-50.0, -100.0, phase="RECOVERY")])
        self.assertIs(status, MissionStatus.RECOVERY_FAILED)

        hc = HostController(mission=mission)
        hc.arm_auto()
        r = hc.tick(100.0, observation=fresh_pose(t=100.0))
        self.assertEqual(r.command.throttle, 0.0)
        self.assertEqual(r.command.steering, 0.0)
        self.assertEqual(r.command.reason, "MISSION_RECOVERY_FAILED")


class TestTransportContract(unittest.TestCase):
    def test_every_tick_sends_direct_control_with_monotonic_seq(self) -> None:
        hc = HostController(mission=two_wp_mission())
        hc.arm_auto()
        seqs = []
        for i in range(5):
            r = hc.tick(100.0 + i * 0.1, observation=fresh_pose(x=10 * i, t=100.0 + i * 0.1))
            self.assertEqual(r.payload["type"], "DIRECT_CONTROL")
            seqs.append(r.payload["control_seq"])
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))  # 유일/증가

    def test_zero_state_still_sends_direct_control(self) -> None:
        hc = HostController(mission=two_wp_mission())  # DISARMED
        r = hc.tick(100.0)
        self.assertEqual(r.payload["type"], "DIRECT_CONTROL")
        self.assertEqual(r.payload["throttle"], 0.0)
        self.assertEqual(r.payload["steering"], 0.0)


if __name__ == "__main__":
    unittest.main()
