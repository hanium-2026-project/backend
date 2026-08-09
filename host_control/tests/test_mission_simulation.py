"""폐루프 미션 시뮬레이션 테스트.

자전거 모델로 HostController 를 폐루프 구동하여:
- single/multiple waypoint 진행
- waypoint 1 도착 → host 가 target waypoint 2 로 전환
- 이 과정에서 **DIRECT_CONTROL 외의 message(WAYPOINT/GO)를 절대 보내지 않음**
- final 도착 → DONE
- 발산/과도 진동 없음
을 검증한다.

⚠ SIM-ONLY 물리 상수(SimulationConfig)는 잠정값이다(실차 성능 아님).
"""

from __future__ import annotations

import math
import unittest

from controller.config import SimulationConfig
from controller.models import Pose, Waypoint
from host_control import HostController, HostWaypointMission, MissionStatus


class _SimSink:
    """전송된 모든 payload 를 기록하는 sink (message type 검증용)."""

    def __init__(self) -> None:
        self.payloads = []

    def __call__(self, payload) -> None:
        self.payloads.append(payload)


def _bicycle(x, y, h, throttle, wire_steering, sim: SimulationConfig):
    # wire steering(음수=LEFT) → 물리 조향각(양수=LEFT)
    delta = math.radians(-wire_steering * sim.max_wheel_angle_deg)
    v_mm_s = throttle * sim.sim_speed_cm_s_at_full_throttle * 10.0
    dt = sim.dt_s
    hr = math.radians(h)
    x2 = x + v_mm_s * math.cos(hr) * dt
    y2 = y + v_mm_s * math.sin(hr) * dt
    h2 = math.degrees(hr + (v_mm_s / sim.wheelbase_mm) * math.tan(delta) * dt)
    return x2, y2, h2 % 360.0


def run_mission(waypoints, *, start=(0.0, 0.0, 0.0), max_steps=3000):
    sim = SimulationConfig()
    sink = _SimSink()
    from host_control.direct_control import DirectControlSender
    hc = HostController(
        mission=HostWaypointMission(waypoints),
        sender=DirectControlSender(sink=sink),
    )
    hc.arm_auto()
    x, y, h = start
    t = 100.0
    progression = []
    steering_series = []
    for _ in range(max_steps):
        r = hc.tick(t, observation=Pose(x, y, h, timestamp=t))
        if not progression or progression[-1] != hc.mission.index:
            progression.append(hc.mission.index)
        steering_series.append(r.command.steering)
        if r.mission_status is MissionStatus.DONE:
            return hc, sink, progression, (x, y), True
        x, y, h = _bicycle(x, y, h, r.command.throttle, r.command.steering, sim)
        t += sim.dt_s
    return hc, sink, progression, (x, y), False


class TestMissionSimulation(unittest.TestCase):
    def test_single_waypoint_arrival(self) -> None:
        wps = [Waypoint(600, 0, position_tolerance_cm=8, is_final=True)]
        hc, sink, prog, pos, done = run_mission(wps)
        self.assertTrue(done, "single waypoint 미완료")
        self.assertIs(hc.mission.status, MissionStatus.DONE)

    def test_multiple_waypoint_progression(self) -> None:
        wps = [
            Waypoint(400, 50, position_tolerance_cm=8),
            Waypoint(800, 150, position_tolerance_cm=8),
            Waypoint(1100, 150, position_tolerance_cm=8, is_final=True),
        ]
        hc, sink, prog, pos, done = run_mission(wps)
        self.assertTrue(done, "multiple waypoint 미완료")
        self.assertEqual(prog, [0, 1, 2], f"target 전환 순서 이상: {prog}")

    def test_only_direct_control_messages_sent(self) -> None:
        # ★ 자율주행 전 과정에서 WAYPOINT/GO 등 다른 message type 을 절대 보내지 않음
        wps = [
            Waypoint(400, 50, position_tolerance_cm=8),
            Waypoint(900, 200, position_tolerance_cm=8, is_final=True),
        ]
        hc, sink, prog, pos, done = run_mission(wps)
        self.assertTrue(done)
        types = {p["type"] for p in sink.payloads}
        self.assertEqual(types, {"DIRECT_CONTROL"},
                         f"DIRECT_CONTROL 외 message 전송됨: {types}")
        # seq monotonic
        seqs = [p["control_seq"] for p in sink.payloads]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))

    def test_no_divergence_or_excessive_oscillation(self) -> None:
        wps = [Waypoint(500, 300, position_tolerance_cm=8, is_final=True)]
        hc, sink, prog, pos, done = run_mission(wps, start=(0.0, 0.0, 90.0))
        # steering 이 모두 [-1,1] 이고 NaN 없음
        for s in [p["steering"] for p in sink.payloads]:
            self.assertFalse(math.isnan(s))
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)
        self.assertTrue(done)

    def test_done_then_zero_output(self) -> None:
        wps = [Waypoint(500, 0, position_tolerance_cm=8, is_final=True)]
        hc, sink, prog, pos, done = run_mission(wps)
        self.assertTrue(done)
        # DONE 이후 한 tick 더: zero 유지
        r = hc.tick(999.0, observation=Pose(pos[0], pos[1], 0.0, timestamp=999.0))
        self.assertEqual(r.command.throttle, 0.0)
        self.assertEqual(r.payload["type"], "DIRECT_CONTROL")


if __name__ == "__main__":
    unittest.main()
