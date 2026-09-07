"""HostController 통합 테스트: authority 중재 / stale→fault / 자동재출발 차단 / transport."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

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


class TestReverseStartAnchorCapture(unittest.TestCase):
    """run_20260824_192746: ALIGN 도착 tick 의 heading 출처가 anchor 를 좌우한다.

    실측 실패: 도착 tick 이 FRONT_CUSHION 이라 anchor 가 만들어지지 않았고, 다음
    프레임부터 쿠션이 사라져 ENTRY 24 tick 이 전부 throttle 0 →
    REVERSE_HEADING_TIMEOUT. 후진이 단 한 번도 시작되지 않았다.
    """

    # ALIGN 종점 직전 실제 관측(값은 run_20260824_192746 pose.jsonl).
    FORWARD_TRACK = (
        (677.7, 330.1, 318.7, 44.74),
        (687.6, 322.8, 320.1, 44.95),
        (698.8, 312.6, 318.5, 45.17),
        (710.1, 305.2, 322.1, 45.39),
    )
    ARRIVAL = (720.0, 293.6, 315.9, 45.58)      # ALIGN wp4 도착 tick
    AFTER = (729.9, 283.3, 317.0, 45.80)        # 쿠션 소실 직후 첫 프레임

    def _align_then_entry_host(self) -> HostController:
        mission = HostWaypointMission([
            Waypoint(747.2, 272.2, target_heading_deg=315.0,
                     position_tolerance_cm=4.0, heading_tolerance_deg=5.0,
                     motion_direction=MotionDirection.FORWARD,
                     phase="ALIGN", route_id=18, waypoint_id=4),
            Waypoint(635.1, 403.4, target_heading_deg=306.0,
                     position_tolerance_cm=4.0, heading_tolerance_deg=12.0,
                     motion_direction=MotionDirection.REVERSE,
                     phase="ENTRY", route_id=18, waypoint_id=5),
        ])
        hc = HostController(
            mission=mission,
            config=ControllerConfig(allow_reverse=True,
                                    reverse_heading_wait_timeout_s=2.5),
        )
        hc.arm_auto()
        return hc

    def _drive_to_align_terminal(self, hc: HostController,
                                 arrival_source: str) -> None:
        for x, y, h, t in self.FORWARD_TRACK:
            hc.tick(t, observation=Pose(x, y, h, timestamp=t,
                                        heading_source="FRONT_CUSHION"))
        x, y, h, t = self.ARRIVAL
        hc.tick(t, observation=Pose(x, y, h, timestamp=t,
                                    heading_source=arrival_source))

    def test_front_cushion_align_terminal_still_yields_reverse_start(self) -> None:
        hc = self._align_then_entry_host()
        self._drive_to_align_terminal(hc, "FRONT_CUSHION")

        # 방향 전환 interlock: 첫 tick 은 반드시 zero 이고 pose 를 버린다.
        x, y, h, t = self.AFTER
        flip = hc.tick(t, observation=Pose(x, y, h, timestamp=t,
                                           heading_source="TRAJECTORY"))
        self.assertEqual(flip.command.throttle, 0.0)

        started = hc.tick(t + 0.11,
                          observation=Pose(x, y, h, timestamp=t + 0.11,
                                           heading_source="TRAJECTORY"))
        self.assertLess(started.command.throttle, 0.0)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_START_TRAJECTORY_ANCHOR")

    def test_trajectory_align_terminal_keeps_existing_behaviour(self) -> None:
        hc = self._align_then_entry_host()
        self._drive_to_align_terminal(hc, "TRAJECTORY")

        x, y, h, t = self.AFTER
        hc.tick(t, observation=Pose(x, y, h, timestamp=t,
                                    heading_source="TRAJECTORY"))
        started = hc.tick(t + 0.11,
                          observation=Pose(x, y, h, timestamp=t + 0.11,
                                           heading_source="TRAJECTORY"))
        self.assertLess(started.command.throttle, 0.0)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_START_TRAJECTORY_ANCHOR")

    def test_last_valid_align_terminal_never_becomes_anchor(self) -> None:
        hc = self._align_then_entry_host()
        self._drive_to_align_terminal(hc, "LAST_VALID")

        x, y, h, t = self.AFTER
        held = hc.tick(t, observation=Pose(x, y, h, timestamp=t,
                                           heading_source="TRAJECTORY"))
        held = hc.tick(t + 0.11,
                       observation=Pose(x, y, h, timestamp=t + 0.11,
                                        heading_source="TRAJECTORY"))
        self.assertEqual(held.command.throttle, 0.0)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_WAIT_PRIMARY_HEADING")

    def test_anchor_still_bounded_by_distance(self) -> None:
        hc = self._align_then_entry_host()
        self._drive_to_align_terminal(hc, "FRONT_CUSHION")

        # anchor 로부터 35mm 를 넘어선 첫 프레임은 bootstrap 대상이 아니다.
        ax, ay, _, _ = self.ARRIVAL
        far = Pose(ax + 40.0, ay - 20.0, 317.0, timestamp=45.80,
                   heading_source="TRAJECTORY")
        hc.tick(45.80, observation=far)
        held = hc.tick(45.91, observation=replace(far, timestamp=45.91))
        self.assertEqual(held.command.throttle, 0.0)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_WAIT_PRIMARY_HEADING")

    def test_anchor_still_bounded_by_age(self) -> None:
        hc = self._align_then_entry_host()
        self._drive_to_align_terminal(hc, "FRONT_CUSHION")

        # 0.75s 를 넘겨 도착한 첫 후진 프레임은 anchor 를 쓰지 못한다.
        x, y, h, _ = self.AFTER
        late = 45.58 + 0.9
        hc.tick(late, observation=Pose(x, y, h, timestamp=late,
                                       heading_source="TRAJECTORY"))
        held = hc.tick(late + 0.11,
                       observation=Pose(x, y, h, timestamp=late + 0.11,
                                        heading_source="TRAJECTORY"))
        self.assertEqual(held.command.throttle, 0.0)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_WAIT_PRIMARY_HEADING")

    def test_front_cushion_inconsistent_with_motion_is_not_anchored(self) -> None:
        """쿠션 heading 이 최근 전진 궤적과 20° 넘게 어긋나면 승격하지 않는다."""
        hc = self._align_then_entry_host()
        for x, y, h, t in self.FORWARD_TRACK:
            hc.tick(t, observation=Pose(x, y, h, timestamp=t,
                                        heading_source="FRONT_CUSHION"))
        x, y, _, t = self.ARRIVAL
        hc.tick(t, observation=Pose(x, y, 180.0, timestamp=t,
                                    heading_source="FRONT_CUSHION"))

        ax, ay, ah, _ = self.AFTER
        hc.tick(t + 0.22,
                observation=Pose(ax, ay, ah, timestamp=t + 0.22,
                                 heading_source="TRAJECTORY"))
        held = hc.tick(t + 0.33,
                       observation=Pose(ax, ay, ah, timestamp=t + 0.33,
                                        heading_source="TRAJECTORY"))
        self.assertEqual(held.command.throttle, 0.0)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_WAIT_PRIMARY_HEADING")


class TestReverseMotionConfirmation(unittest.TestCase):
    """run_20260824_204134: 후진 명령 != 실제 후진.

    실측: ALIGN 종료 직후 차는 관성으로 앞으로 밀리는 중이었는데, 첫 후진 명령이
    나간 순간부터 그 전진 pose 들이 reverse trajectory 표본이 됐다. 표본에서
    유도한 진행방향에 reverse 보정 180° 가 붙어 차체 방향과 177° 어긋났고,
    궤적 fallback 이 거부돼 OBSERVATION_LOST → REVERSE_HEADING_TIMEOUT 이 됐다.
    """

    BODY_DEG = 333.0                     # ALIGN 종료 시 차체 방향(실측 근사)
    ANCHOR = (809.1, 360.6, 45.30)       # ENTRY 진입 직전 pose

    def _entry_host(self) -> HostController:
        mission = HostWaypointMission([
            Waypoint(700.0, 480.0, target_heading_deg=self.BODY_DEG,
                     position_tolerance_cm=4.0, heading_tolerance_deg=12.0,
                     motion_direction=MotionDirection.REVERSE,
                     phase="ENTRY", route_id=4, waypoint_id=3),
        ])
        hc = HostController(
            mission=mission,
            config=ControllerConfig(allow_reverse=True,
                                    reverse_heading_wait_timeout_s=2.5),
        )
        hc.arm_auto()
        # ALIGN 이 검증한 body heading anchor 를 직접 세운다 (H-1 이 만드는 것).
        anchor_pose = Pose(self.ANCHOR[0], self.ANCHOR[1], self.BODY_DEG,
                           timestamp=self.ANCHOR[2],
                           heading_source="REVERSE_START_TRAJECTORY_ANCHOR")
        hc._reverse_start_anchor = anchor_pose
        hc._reverse_start_anchor_route_id = 4
        hc._last_trusted_reverse_heading = self.BODY_DEG
        return hc

    @staticmethod
    def _along(x: float, y: float, ds: float, deg: float):
        rad = math.radians(deg)
        return (x + ds * math.cos(rad), y + ds * math.sin(rad))

    def _feed(self, hc, start, offsets, t0, *, source="TRAJECTORY", step=0.22):
        """차체축 기준 부호 있는 변위열을 pose 로 먹인다. 마지막 결과 반환."""
        result = None
        x, y = start
        travelled = 0.0
        for i, ds in enumerate(offsets, start=1):
            travelled += ds
            px, py = self._along(x, y, travelled, self.BODY_DEG)
            t = t0 + i * step
            result = hc.tick(t, observation=Pose(px, py, self.BODY_DEG,
                                                 timestamp=t,
                                                 heading_source=source))
        return result

    def test_forward_inertia_never_enters_reverse_window(self) -> None:
        hc = self._entry_host()
        # 후진 명령 + 관성 전진 3프레임 (실측 t=32.6~33.2 구간과 같은 형태)
        self._feed(hc, self.ANCHOR[:2], [+22.0, +25.0, +7.0], self.ANCHOR[2])
        self.assertTrue(hc._reverse_motion_started)
        self.assertFalse(hc._reverse_motion_confirmed)
        self.assertEqual(len(hc._reverse_observations), 0)

    def test_forward_inertia_alone_never_reaches_trajectory_fallback(self) -> None:
        hc = self._entry_host()
        last = self._feed(hc, self.ANCHOR[:2],
                          [+22.0, +25.0, +7.0, +5.0, +3.0], self.ANCHOR[2])
        self.assertNotEqual(hc.reverse_observation_state,
                            "REVERSE_TRACK_TRAJECTORY_FALLBACK")
        self.assertEqual(last.command.throttle, 0.0)

    def test_confirmed_reverse_motion_reaches_trajectory_fallback(self) -> None:
        """관성 구간 뒤 실제 후진이 나타나면 궤적 fallback 이 서야 한다."""
        hc = self._entry_host()
        self._feed(hc, self.ANCHOR[:2], [+22.0, +25.0, +7.0], self.ANCHOR[2])
        self.assertFalse(hc._reverse_motion_confirmed)

        # 실제 후진 — 기존 품질 계약(3관측 / 30mm / 선형성 0.90)을 만족시킨다.
        apex = self._along(self.ANCHOR[0], self.ANCHOR[1], 54.0, self.BODY_DEG)
        self._feed(hc, apex, [-16.0, -16.0, -16.0], self.ANCHOR[2] + 0.66)

        self.assertTrue(hc._reverse_motion_confirmed)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_TRACK_TRAJECTORY_FALLBACK")
        # 전진 관성 표본이 창에 남아 있으면 안 된다.
        self.assertTrue(all(
            hc._is_rearward(a, b)
            for a, b in zip(hc._reverse_observations,
                            list(hc._reverse_observations)[1:])))

    def test_stationary_vehicle_never_confirms_reverse(self) -> None:
        """차가 전혀 안 움직이면 확인도 fallback 도 없고 bounded 하게 끝난다.

        단 **출발 자체는 막지 않는다** — 정지 상태에서 bounded START_ANCHOR 로
        첫 후진 명령을 내보내는 것이 H-1 의 목적이다. 여기서 막으면 다시
        "움직여야 움직일 수 있다"는 교착이 된다.
        """
        hc = self._entry_host()
        self._feed(hc, self.ANCHOR[:2], [0.0, 0.0, 0.0, 0.0], self.ANCHOR[2])
        self.assertFalse(hc._reverse_motion_confirmed)
        self.assertEqual(len(hc._reverse_observations), 0)
        self.assertNotEqual(hc.reverse_observation_state,
                            "REVERSE_TRACK_TRAJECTORY_FALLBACK")

        # bootstrap 한계(1.5s)를 넘기면 더 이상 anchor 를 쓰지 않는다.
        late = self.ANCHOR[2] + 2.0
        stalled = hc.tick(late, observation=Pose(
            self.ANCHOR[0], self.ANCHOR[1], self.BODY_DEG, timestamp=late,
            heading_source="TRAJECTORY"))
        self.assertEqual(stalled.command.throttle, 0.0)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_OBSERVATION_LOST")

    def test_last_valid_alone_never_confirms_or_falls_back(self) -> None:
        hc = self._entry_host()
        last = self._feed(hc, self.ANCHOR[:2], [-16.0, -16.0, -16.0],
                          self.ANCHOR[2], source="LAST_VALID")
        self.assertNotEqual(hc.reverse_observation_state,
                            "REVERSE_TRACK_TRAJECTORY_FALLBACK")
        self.assertNotEqual(hc.reverse_observation_state,
                            "REVERSE_START_TRAJECTORY_ANCHOR")
        self.assertEqual(last.command.throttle, 0.0)

    def test_front_cushion_return_keeps_body_authority_with_motion_guidance(self) -> None:
        hc = self._entry_host()
        self._feed(hc, self.ANCHOR[:2], [+22.0, +25.0], self.ANCHOR[2])
        apex = self._along(self.ANCHOR[0], self.ANCHOR[1], 47.0, self.BODY_DEG)
        self._feed(hc, apex, [-16.0, -16.0, -16.0], self.ANCHOR[2] + 0.44)
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_TRACK_TRAJECTORY_FALLBACK")

        back = self._along(self.ANCHOR[0], self.ANCHOR[1], -1.0, self.BODY_DEG)
        t = self.ANCHOR[2] + 1.5
        hc.tick(t, observation=Pose(back[0], back[1], self.BODY_DEG,
                                    timestamp=t,
                                    heading_source="FRONT_CUSHION"))
        self.assertEqual(hc.reverse_observation_state,
                         "REVERSE_TRACK_PRIMARY_MOTION_GUIDANCE")
        self.assertEqual(hc._last_trusted_reverse_heading, self.BODY_DEG)

    def test_sign_reference_never_taken_from_untrusted_heading(self) -> None:
        """anchor/trusted heading 이 없으면 어떤 변위도 후진으로 인정하지 않는다."""
        hc = self._entry_host()
        hc._reverse_start_anchor = None
        hc._last_trusted_reverse_heading = None
        self.assertIsNone(hc._trusted_body_heading())
        a = Pose(800.0, 360.0, 333.0, timestamp=1.0, heading_source="LAST_VALID")
        b = Pose(780.0, 370.0, 333.0, timestamp=1.2, heading_source="LAST_VALID")
        self.assertFalse(hc._is_rearward(a, b))


if __name__ == "__main__":
    unittest.main()
