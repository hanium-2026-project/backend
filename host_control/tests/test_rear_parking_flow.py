"""후면주차 B1 경로를 AUTO_HOST 미션에 그대로 태워 phase 진행을 확인한다 (목 검증).

실차 없이 "생성기가 만든 route 를 host 가 끝까지 몰 수 있는가"만 본다.
카메라 대신 계획 자세를 그대로 관측으로 넣으므로 **추종 오차는 보지 않는다** —
이건 계약 검증이지 제어 성능 검증이 아니다.

실행: python -m unittest host_control.tests.test_rear_parking_flow -v
"""

from __future__ import annotations

import unittest

from controller.config import ControllerConfig
from controller.models import MotionDirection, Pose
from host_control import HostController, HostWaypointMission, MissionStatus
from integration.backend_adapter import waypoints_from_backend
from parking.waypoints import (AISLE_Y, build_rear_parking_waypoints,
                               choose_rear_parking_plan, default_slot_specs)

SPECS = default_slot_specs()
START = (150.0, AISLE_Y)
TICK = 0.1


def planned_poses() -> list[tuple[float, float, float]]:
    """route 의 각 waypoint 에서 차량이 취해야 할 자세 (x, y, heading)."""
    plan, _ = choose_rear_parking_plan(SPECS["B1"], from_pose=START)
    poses: list[tuple[float, float, float]] = []
    wps = build_rear_parking_waypoints(SPECS["B1"], route_id=1, from_pose=START,
                                       from_heading_deg=0.0)
    align = list(plan.align_poses)
    entry = list(plan.entry_poses)
    for w in wps:
        if w.phase in ("CRUISE", "APPROACH"):
            poses.append((w.x, w.y, plan.aisle_heading_deg))
        elif w.phase == "ALIGN":
            poses.append(align.pop(0))
        elif w.phase == "ENTRY":
            poses.append(entry.pop(0))
        else:
            poses.append(plan.final_pose)
    return poses


def make_host():
    wps = waypoints_from_backend(
        build_rear_parking_waypoints(SPECS["B1"], route_id=1, from_pose=START,
                                     from_heading_deg=0.0))
    mission = HostWaypointMission(wps, max_recovery_attempts=3)
    host = HostController(
        config=ControllerConfig(allow_reverse=True, steer_kd=0.0,
                                final_confirm_observations=3,
                                approach_capture_tolerance_cm=10.0,
                                approach_pass_margin_cm=1.0),
        mission=mission)
    host.arm_auto()
    return host, mission, wps


class RearParkingFlow(unittest.TestCase):
    """route 를 끝까지 몰아보고 각 구간에서 관측된 것을 기록한다."""

    def drive(self):
        """waypoint 마다 '직전 자세에서 달린다 → 목표 자세에 도착한다' 로 민다.

        목표 자세를 처음부터 넣으면 매 waypoint 가 즉시 도착 처리돼 구동 명령이
        한 번도 안 나온다 (인터록도, 조향도 못 본다).
        """
        host, mission, wps = make_host()
        plan, _ = choose_rear_parking_plan(SPECS["B1"], from_pose=START)
        poses = planned_poses()
        t = 100.0
        log: list[dict] = []

        def tick(idx, pose):
            nonlocal t
            t += TICK
            r = host.tick(t, observation=Pose(pose[0], pose[1], pose[2], timestamp=t))
            log.append({
                "wp": idx + 1, "phase": mission.current_phase,
                "throttle": r.command.throttle, "steering": r.command.steering,
                "reason": r.command.reason, "status": mission.status,
                "arrived": r.command.arrived,
            })
            return r

        prev = (START[0], START[1], plan.aisle_heading_deg)
        for idx in range(len(wps)):
            target = poses[idx]
            # ① 직전 자세에서 목표를 향해 달린다 (방향 전환이면 zero 가 한 번 낀다)
            for _ in range(4):
                if tick(idx, prev).command.throttle != 0.0:
                    break
            # ② 목표 자세 도착
            for _ in range(12):
                tick(idx, target)
                if mission.status is MissionStatus.DONE or mission.index > idx:
                    break
            if mission.status is MissionStatus.DONE:
                break
            prev = target
        return host, mission, log


class TestRearParkingMockRun(RearParkingFlow):
    def test_mission_reaches_done(self) -> None:
        _, mission, log = self.drive()
        self.assertIs(mission.status, MissionStatus.DONE,
                      f"마지막 상태={mission.status} 마지막로그={log[-3:]}")

    def test_phase_progression_is_complete(self) -> None:
        _, _, log = self.drive()
        seen = []
        for r in log:
            if not seen or seen[-1] != r["phase"]:
                seen.append(r["phase"])
        for phase in ("APPROACH", "ALIGN", "ENTRY", "FINAL"):
            self.assertIn(phase, seen, f"{phase} 를 거치지 않았다: {seen}")
        self.assertEqual(seen.index("ALIGN") < seen.index("ENTRY"), True)
        self.assertEqual(seen.index("ENTRY") < seen.index("FINAL"), True)

    def test_no_reverse_rejection_anywhere(self) -> None:
        """⑥ ENTRY/FINAL 후진이 게이트에서 막히면 안 된다."""
        _, _, log = self.drive()
        bad = [r for r in log if r["reason"] in
               ("REVERSE_NOT_ALLOWED", "REVERSE_PHASE_NOT_ALLOWED")]
        self.assertEqual(bad, [], f"후진이 거절됐다: {bad[:3]}")

    def test_direction_change_interlock_between_align_and_entry(self) -> None:
        """⑦ 전진 ALIGN → 후진 ENTRY 사이에 1 tick zero 가 들어간다."""
        _, _, log = self.drive()
        stops = [r for r in log if r["reason"] == "DIRECTION_CHANGE_STOP"]
        self.assertTrue(stops, "방향 전환 인터록이 없다")
        for r in stops:
            self.assertEqual(r["throttle"], 0.0)

    def test_entry_and_final_command_negative_throttle(self) -> None:
        _, _, log = self.drive()
        rev = [r for r in log if r["phase"] in ("ENTRY", "FINAL")
               and r["throttle"] != 0.0]
        self.assertTrue(rev, "후진 구간에서 구동값이 전혀 안 나왔다")
        for r in rev:
            self.assertLess(r["throttle"], 0.0,
                            f"후진 구간인데 throttle {r['throttle']}")

    def test_entry_actually_steers(self) -> None:
        """phase 분리의 핵심 — ENTRY 후진에서 조향이 살아 있어야 원호를 탄다."""
        _, _, log = self.drive()
        steers = [r["steering"] for r in log
                  if r["phase"] == "ENTRY" and r["throttle"] != 0.0]
        self.assertTrue(steers, "ENTRY 구동 tick 이 없다")
        self.assertTrue(any(abs(s) > 1e-6 for s in steers),
                        "ENTRY 후진 조향이 전부 0 이다 — 11자로 묶여 있다")

    def test_final_requires_three_fresh_observations(self) -> None:
        """⑧ FINAL 은 서로 다른 fresh 관측 3회를 세고 나서 DONE."""
        _, _, log = self.drive()
        confirms = [r["reason"] for r in log
                    if r["reason"].startswith("FINAL_CONFIRMING")]
        self.assertIn("FINAL_CONFIRMING_1_OF_3", confirms)
        self.assertIn("FINAL_CONFIRMING_2_OF_3", confirms)
        self.assertTrue(any(r["arrived"] for r in log))

    def test_stale_repeat_is_not_counted_as_new_observation(self) -> None:
        """같은 pose.timestamp 를 다시 조회한 것은 새 관측으로 세지 않는다."""
        host, mission, wps = make_host()
        poses = planned_poses()
        t = 100.0
        for idx, (wp, (px, py, ph)) in enumerate(zip(wps, poses)):
            for _ in range(12):
                t += TICK
                host.tick(t, observation=Pose(px, py, ph, timestamp=t))
                if mission.index > idx or mission.status is MissionStatus.DONE:
                    break
            if mission.current_phase == "FINAL":
                break
        fx, fy, fh = poses[-1]
        t += TICK
        first = host.tick(t, observation=Pose(fx, fy, fh, timestamp=t))
        self.assertTrue(first.command.reason.startswith("FINAL_CONFIRMING"))
        again = host.tick(t + 0.03)               # 관측 없음 = 같은 pose 재조회
        self.assertEqual(again.command.reason, first.command.reason)


if __name__ == "__main__":
    unittest.main()
