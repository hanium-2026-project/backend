"""HostController 통합 테스트: authority 중재 / stale→fault / 자동재출발 차단 / transport."""

from __future__ import annotations

import unittest

from controller.models import ControlCommand, ControlMode, Pose, Waypoint
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
