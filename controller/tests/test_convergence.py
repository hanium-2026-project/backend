"""수렴 시뮬레이션 테스트 (자전거 모델).

⚠ 이 시뮬레이션의 물리 상수는 SIMULATION-ONLY / PROVISIONAL 이다.
목적은 실차 성능 검증이 아니라 controller 의 **부호 정합성과 수렴 거동**을
motor-OFF 로 확인하는 것이다.

wire steering → 물리 조향각 변환 (실제 ESP32 부호 반영)
------------------------------------------------------
wire steering 음수 = LEFT = CCW 회전(heading 증가).
자전거 모델에서 조향각 δ 는 δ>0 을 좌회전(CCW)으로 정의하므로:
    δ = -wire_steering * max_wheel_angle
(wire=-1 → δ=+max → 좌회전, wire=+1 → δ=-max → 우회전)
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from controller.config import ControllerConfig, SimulationConfig
from controller.models import ControlMode, Pose, Waypoint
from controller.pose_controller import PoseWaypointController


@dataclass
class SimState:
    x_mm: float
    y_mm: float
    heading_deg: float


class BicycleSim:
    """SIM-ONLY 자전거 운동학 시뮬레이터."""

    def __init__(self, sim: SimulationConfig) -> None:
        self.sim = sim

    def step(self, st: SimState, throttle: float, wire_steering: float) -> SimState:
        sim = self.sim
        # wire steering(음수=LEFT) → 물리 조향각 δ(양수=LEFT)
        delta = math.radians(-wire_steering * sim.max_wheel_angle_deg)
        # throttle → 속도 (SIM-ONLY 잠정)
        v_cm_s = throttle * sim.sim_speed_cm_s_at_full_throttle
        v_mm_s = v_cm_s * 10.0
        dt = sim.dt_s
        h = math.radians(st.heading_deg)
        x = st.x_mm + v_mm_s * math.cos(h) * dt
        y = st.y_mm + v_mm_s * math.sin(h) * dt
        # yaw_rate = v/L * tan(delta)
        yaw = (v_mm_s / sim.wheelbase_mm) * math.tan(delta) * dt
        heading = math.degrees(h + yaw)
        return SimState(x, y, (heading % 360.0))


def run_sim(start: SimState, wp: Waypoint, *, max_steps=1200,
            cfg=None, sim=None):
    cfg = cfg or ControllerConfig()
    sim = sim or SimulationConfig()
    ctl = PoseWaypointController(cfg)
    bike = BicycleSim(sim)
    st = SimState(start.x_mm, start.y_mm, start.heading_deg)
    t = 100.0
    trajectory = []
    last_cmd = None
    for _ in range(max_steps):
        pose = Pose(st.x_mm, st.y_mm, st.heading_deg, timestamp=t, valid=True)
        cmd = ctl.compute(pose, wp, now=t)
        last_cmd = cmd
        trajectory.append((st.x_mm, st.y_mm, st.heading_deg, cmd))
        if cmd.arrived:
            break
        st = bike.step(st, cmd.throttle, cmd.steering)
        t += sim.dt_s
    return st, last_cmd, trajectory


class TestConvergence(unittest.TestCase):
    def _dist_cm(self, st: SimState, wp: Waypoint) -> float:
        return math.hypot(wp.x_mm - st.x_mm, wp.y_mm - st.y_mm) / 10.0

    def test_straight(self) -> None:
        start = SimState(0.0, 0.0, 0.0)  # 오른쪽 향함
        wp = Waypoint(x_mm=600.0, y_mm=0.0, position_tolerance_cm=8.0, speed_cm_s=12.0)
        st, cmd, _ = run_sim(start, wp)
        self.assertTrue(cmd.arrived, "직진 수렴 실패")
        self.assertLessEqual(self._dist_cm(st, wp), 8.0 + 3.0)

    def test_left_turn(self) -> None:
        # 오른쪽을 향한 채 시작, 목표는 왼쪽 위 → 좌회전으로 수렴해야 함
        start = SimState(0.0, 0.0, 0.0)
        wp = Waypoint(x_mm=400.0, y_mm=400.0, position_tolerance_cm=8.0, speed_cm_s=12.0)
        st, cmd, _ = run_sim(start, wp)
        self.assertTrue(cmd.arrived, "좌회전 수렴 실패")
        self.assertLessEqual(self._dist_cm(st, wp), 8.0 + 3.0)

    def test_right_turn(self) -> None:
        start = SimState(0.0, 0.0, 0.0)
        wp = Waypoint(x_mm=400.0, y_mm=-400.0, position_tolerance_cm=8.0, speed_cm_s=12.0)
        st, cmd, _ = run_sim(start, wp)
        self.assertTrue(cmd.arrived, "우회전 수렴 실패")
        self.assertLessEqual(self._dist_cm(st, wp), 8.0 + 3.0)

    def test_off_axis(self) -> None:
        # 위를 향한 채 시작(heading 90°), 목표는 오른쪽 → 우회전으로 수렴
        start = SimState(0.0, 0.0, 90.0)
        wp = Waypoint(x_mm=500.0, y_mm=200.0, position_tolerance_cm=8.0, speed_cm_s=12.0)
        st, cmd, _ = run_sim(start, wp)
        self.assertTrue(cmd.arrived, "off-axis 수렴 실패")
        self.assertLessEqual(self._dist_cm(st, wp), 8.0 + 3.0)

    def test_no_runaway_oscillation(self) -> None:
        # 어려운 초기조건(목표가 거의 뒤): 발산/진동/NaN 이 없어야 함.
        start = SimState(0.0, 0.0, 0.0)
        wp = Waypoint(x_mm=-300.0, y_mm=100.0, position_tolerance_cm=8.0, speed_cm_s=12.0)
        st, cmd, traj = run_sim(start, wp, max_steps=1500)
        # 모든 출력이 유한하고 범위 내
        for _, _, _, c in traj:
            self.assertFalse(math.isnan(c.steering) or math.isnan(c.throttle))
            self.assertGreaterEqual(c.steering, -1.0)
            self.assertLessEqual(c.steering, 1.0)
            self.assertGreaterEqual(c.throttle, 0.0)
            self.assertLessEqual(c.throttle, 1.0)
        # 궤적이 유한한 영역에 머무름(발산하지 않음): 맵(1200mm)의 여유 배수 이내
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        self.assertLess(max(abs(min(xs)), abs(max(xs))), 5000.0)
        self.assertLess(max(abs(min(ys)), abs(max(ys))), 5000.0)

    def test_heading_error_decreases_on_straight_approach(self) -> None:
        # 초기 heading 오차가 시간이 지나며 대체로 감소하는지(진동 발산 아님) 확인
        start = SimState(0.0, 0.0, 0.0)
        wp = Waypoint(x_mm=400.0, y_mm=300.0, position_tolerance_cm=8.0, speed_cm_s=12.0)
        _, _, traj = run_sim(start, wp)
        errs = [abs(c.heading_error_deg) for _, _, _, c in traj if c.mode == ControlMode.DRIVE]
        self.assertGreater(len(errs), 5)
        # 초반 평균보다 후반 평균이 작아야(수렴 경향)
        head = errs[: max(1, len(errs) // 4)]
        tail = errs[-max(1, len(errs) // 4):]
        self.assertLess(sum(tail) / len(tail), sum(head) / len(head))


if __name__ == "__main__":
    unittest.main()
