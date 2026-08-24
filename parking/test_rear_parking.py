"""후면주차 경로 생성기 검증 (B1 기준).

발렛 인계 모델(test_planner.py)과 달리 **슬롯 안까지 후진으로 넣는** 경로다.
ESP32 에 새 FSM 을 만들지 않고 phase + motion_direction 으로만 표현하므로,
여기서 보는 것은 "그 두 필드가 기하와 일치하는가"다.

실행: python -m unittest parking.test_rear_parking -v
"""

from __future__ import annotations

import math
import unittest

from parking.waypoints import (
    PHASE_DEFAULTS,
    AISLE_Y,
    CAR_LENGTH_MM,
    CAR_WIDTH_MM,
    LOT_SIZE_MM,
    MIN_TURN_RADIUS_MM,
    WIRE_PHASES,
    InfeasibleRouteError,
    build_rear_parking_waypoints,
    build_setup_recovery_waypoints,
    choose_rear_parking_plan,
    default_slot_specs,
    plan_rear_parking,
    plan_setup_recovery,
    _car_footprint,
)

SPECS = default_slot_specs()
# 08-12 실차에서 차를 놓았던 자리 (통로 왼쪽 끝, 통로 방향).
LEFT_START = (150.0, AISLE_Y)


class TestSetupRecovery(unittest.TestCase):
    def test_handoff_uses_setup_when_direct_candidate_is_infeasible(self) -> None:
        from parking.waypoints import choose_rear_candidate
        pose = (SPECS["B1"].center_x, AISLE_Y)
        direct, _ = choose_rear_candidate(SPECS["B1"], pose, 0.0)
        self.assertIsNone(direct)
        self.assertIsNotNone(plan_setup_recovery(SPECS["B1"], pose, 0.0))

    def test_reverse_setup_arc_carries_curvature(self) -> None:
        pose = (SPECS["B1"].center_x, AISLE_Y)
        wps = build_setup_recovery_waypoints(
            SPECS["B1"], route_id=1, from_pose=pose, from_heading_deg=0.0)
        self.assertTrue(wps)
        self.assertTrue(all(w.motion_direction == "REVERSE" for w in wps))
        self.assertTrue(all(w.curvature != 0.0 for w in wps))

    def test_two_segment_fallback_covers_previous_b1_dead_zone(self) -> None:
        pose = (SPECS["B1"].center_x - 30.0, AISLE_Y)
        rec = plan_setup_recovery(SPECS["B1"], pose, 5.0)
        self.assertIsNotNone(rec)
        self.assertEqual(len(rec.segments), 2)

    def test_representative_slots_survive_pose_perturbations(self) -> None:
        for slot_id in ("B1", "B2", "A1"):
            spec = SPECS[slot_id]
            for dx in (-30.0, 0.0, 30.0):
                for dy in (-30.0, 0.0, 30.0):
                    for dh in (-5.0, 0.0, 5.0):
                        rec = plan_setup_recovery(
                            spec, (spec.center_x + dx, AISLE_Y + dy), dh)
                        self.assertIsNotNone(
                            rec, f"{slot_id} dx={dx} dy={dy} dh={dh}")

    def test_curved_recovery_steers_but_straight_recovery_stays_locked(self) -> None:
        from controller.config import ControllerConfig
        from controller.models import MotionDirection, Pose, Waypoint
        from controller.pose_controller import PoseWaypointController
        cfg = ControllerConfig(allow_reverse=True, steer_kd=0.0)
        pose = Pose(0.0, 0.0, 0.0, timestamp=100.0,
                    heading_source="FRONT_CUSHION")
        base = dict(x_mm=-1000.0, y_mm=0.0, phase="RECOVERY",
                    speed_cm_s=5.0, position_tolerance_cm=4.0,
                    motion_direction=MotionDirection.REVERSE)
        straight = PoseWaypointController(cfg).compute(
            pose, Waypoint(**base, curvature=0.0), now=100.0)
        curved = PoseWaypointController(cfg).compute(
            pose, Waypoint(**base, curvature=1.0 / 900.0), now=100.0)
        self.assertEqual(straight.steering, 0.0)
        self.assertNotEqual(curved.steering, 0.0)


def route_b1(from_pose=LEFT_START):
    return build_rear_parking_waypoints(SPECS["B1"], route_id=1,
                                        from_pose=from_pose, from_heading_deg=0.0)


class TestRearRouteShape(unittest.TestCase):
    """① 순서 · ② 방향 — 요청받은 8종 중 1,2번."""

    def setUp(self) -> None:
        self.wps = route_b1()

    def test_phase_order_is_cruise_approach_align_entry_final(self) -> None:
        order = ["CRUISE", "APPROACH", "ALIGN", "ENTRY", "FINAL"]
        seen = []
        for w in self.wps:
            if not seen or seen[-1] != w.phase:
                seen.append(w.phase)
        self.assertEqual(seen, [p for p in order if p in seen])
        self.assertEqual(seen[-1], "FINAL")
        self.assertIn("ALIGN", seen)
        self.assertIn("ENTRY", seen)

    def test_only_firmware_known_phases(self) -> None:
        """펌웨어 parse_phase 가 아는 이름만 나간다 (TURN 같은 새 phase 금지)."""
        for w in self.wps:
            self.assertIn(w.phase, WIRE_PHASES, f"wire 에서 거절될 phase: {w.phase}")

    def test_motion_direction_per_phase(self) -> None:
        for w in self.wps:
            expected = "REVERSE" if w.phase in ("ENTRY", "FINAL") else "FORWARD"
            self.assertEqual(w.motion_direction, expected,
                             f"{w.phase} wp{w.waypoint_id} 방향이 {w.motion_direction}")

    def test_exactly_one_final_and_it_is_last(self) -> None:
        finals = [w for w in self.wps if w.is_final]
        self.assertEqual(len(finals), 1)
        self.assertIs(finals[0], self.wps[-1])

    def test_waypoint_ids_start_at_one_and_increase(self) -> None:
        # 펌웨어가 waypoint_id >= 1 을 요구한다
        self.assertEqual([w.waypoint_id for w in self.wps],
                         list(range(1, len(self.wps) + 1)))

    def test_heading_required_only_where_it_matters(self) -> None:
        """Arc entry, REVERSE_START, FINAL 에만 heading 을 요구한다.

        중간 원호점까지 요구하면 도착 반경 안에서 각도가 안 맞을 때
        HEADING_OUT_OF_TOLERANCE 로 원호 중간에 정지한다.
        """
        required = [w for w in self.wps if w.heading_required]
        self.assertEqual(len(required), 3)
        self.assertEqual([w.phase for w in required],
                         ["APPROACH", "ALIGN", "FINAL"])
        self.assertIs(required[1], [w for w in self.wps if w.phase == "ALIGN"][-1])

    def test_every_curved_waypoint_carries_path_tangent(self) -> None:
        curved = [w for w in self.wps if w.curvature]
        self.assertTrue(curved)
        self.assertTrue(all(w.target_heading_deg is not None for w in curved))
        align = [w for w in curved if w.phase == "ALIGN"]
        entry = [w for w in curved if w.phase == "ENTRY"]
        self.assertTrue(all(not w.heading_required for w in align[:-1]))
        self.assertTrue(align[-1].heading_required)
        self.assertTrue(all(not w.heading_required for w in entry))


class TestRearGeometry(unittest.TestCase):
    """③ rear heading · ④ 기하 — 요청받은 8종 중 3,4번."""

    def setUp(self) -> None:
        self.slot = SPECS["B1"]
        self.wps = route_b1()
        self.final = self.wps[-1]

    def test_final_is_slot_center(self) -> None:
        self.assertAlmostEqual(self.final.x, self.slot.center_x, places=6)
        self.assertAlmostEqual(self.final.y, self.slot.center_y, places=6)

    def test_rear_heading_is_forward_plus_180(self) -> None:
        expected = (self.slot.target_heading_deg + 180.0) % 360.0
        self.assertAlmostEqual(self.final.target_heading_deg, expected, places=6)
        self.assertAlmostEqual(expected, 270.0, places=6)   # B행은 코가 통로(아래)

    def test_a_row_and_b_row_rear_headings_are_mirrored(self) -> None:
        a_rear = (SPECS["A1"].target_heading_deg + 180.0) % 360.0
        b_rear = (SPECS["B1"].target_heading_deg + 180.0) % 360.0
        self.assertAlmostEqual(a_rear, 90.0, places=6)
        self.assertAlmostEqual(b_rear, 270.0, places=6)

    def test_reverse_start_is_outside_the_slot(self) -> None:
        """후진 시작 자세는 슬롯 밖(진입선 아래)에 있어야 한다."""
        align = [w for w in self.wps if w.phase == "ALIGN"][-1]
        slot_entry_y = self.slot.center_y - self.slot.length / 2.0   # B행 진입선 900
        self.assertLess(align.y, slot_entry_y,
                        "REVERSE_START 가 이미 슬롯 안이다")

    def test_entry_points_monotonically_approach_final(self) -> None:
        entries = [w for w in self.wps if w.phase == "ENTRY"]
        self.assertGreaterEqual(len(entries), 2)
        d = [math.hypot(w.x - self.final.x, w.y - self.final.y) for w in entries]
        self.assertEqual(d, sorted(d, reverse=True),
                         "ENTRY 가 FINAL 쪽으로 단조 접근하지 않는다")

    def test_entry_arc_respects_min_turn_radius(self) -> None:
        """연속 세 점의 외접원 반경이 최소 선회 반경 이상이어야 한다."""
        pts = [(w.x, w.y) for w in self.wps
               if w.phase in ("ENTRY", "FINAL")]
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            r = _circumradius(a, b, c)
            self.assertGreaterEqual(
                r, MIN_TURN_RADIUS_MM * 0.95,
                f"후진 원호 반경 {r:.0f}mm 가 계획 반경보다 작다")

    def test_every_pose_keeps_the_car_inside_the_map(self) -> None:
        plan, _ = choose_rear_parking_plan(SPECS["B1"], from_pose=LEFT_START)
        poses = ([plan.setup_start] + plan.align_poses + plan.entry_poses
                 + [plan.final_pose])
        for x, y, h in poses:
            for px, py in _car_footprint(x, y, h):
                self.assertGreaterEqual(px, 0.0, f"({x:.0f},{y:.0f},{h:.0f})")
                self.assertLessEqual(px, LOT_SIZE_MM, f"({x:.0f},{y:.0f},{h:.0f})")
                self.assertGreaterEqual(py, 0.0, f"({x:.0f},{y:.0f},{h:.0f})")
                self.assertLessEqual(py, LOT_SIZE_MM, f"({x:.0f},{y:.0f},{h:.0f})")

    def test_setup_start_sits_on_the_aisle(self) -> None:
        """통로를 달려온 차가 S자 합류 없이 바로 원호에 올라탈 수 있어야 한다."""
        plan, _ = choose_rear_parking_plan(SPECS["B1"], from_pose=LEFT_START)
        self.assertLess(plan.aisle_offset_mm, 80.0)

    def test_final_clearance_is_reported_not_hidden(self) -> None:
        """슬롯 깊이 300 - 차 길이 250 이라 FINAL 에서 여유는 25mm 뿐이다.

        이 값이 후보 선택을 지배하므로 계산에서 빠뜨리면 안 된다
        (분석 스크립트는 FINAL 자세를 경로에서 빼서 38mm 로 보고했다).
        """
        plan, _ = choose_rear_parking_plan(SPECS["B1"], from_pose=LEFT_START)
        self.assertLessEqual(plan.clearance_mm, 30.0)
        self.assertEqual(plan.overflow_mm, 0.0)


class TestRearPlanSelection(unittest.TestCase):
    def test_b1_selects_the_only_feasible_candidate(self) -> None:
        plan, cands = choose_rear_parking_plan(SPECS["B1"], from_pose=LEFT_START)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.side, +1)                 # 통로를 +x 로 접근
        self.assertAlmostEqual(plan.phi_deg, 60.0)
        self.assertAlmostEqual(plan.psi_deg, 30.0)
        self.assertEqual(sum(1 for c in cands if c.feasible), 1)

    def test_phi_45_is_rejected_for_b1_because_setup_leaves_the_aisle(self) -> None:
        """문서 07 이 고른 φ=45 는 setup 시작점이 통로에서 201mm 떨어진다.

        좌측 벽에 붙어 진입 활주로가 없어서 실제로는 갈 수 없다.
        """
        p45 = plan_rear_parking(SPECS["B1"], +1, 45.0)
        self.assertFalse(p45.feasible)
        self.assertGreater(p45.aisle_offset_mm, 80.0)

    def test_passed_start_pose_is_rejected(self) -> None:
        """setup 시작점을 지나친 자리에서는 전진으로 되잡을 수 없다."""
        with self.assertRaises(InfeasibleRouteError):
            route_b1(from_pose=(900.0, AISLE_Y))

    def test_plan_is_deterministic(self) -> None:
        a = [(w.phase, w.x, w.y, w.target_heading_deg, w.motion_direction)
             for w in route_b1()]
        b = [(w.phase, w.x, w.y, w.target_heading_deg, w.motion_direction)
             for w in route_b1()]
        self.assertEqual(a, b)


class TestRearAdapterContract(unittest.TestCase):
    """⑤ adapter — backend REVERSE 가 controller 로 그대로 넘어가는가."""

    def test_reverse_waypoints_survive_the_adapter(self) -> None:
        from controller.models import MotionDirection
        from integration.backend_adapter import waypoints_from_backend

        core = waypoints_from_backend(route_b1())
        backend = route_b1()
        self.assertEqual(len(core), len(backend))
        for c, b in zip(core, backend):
            self.assertEqual(c.phase, b.phase)
            self.assertAlmostEqual(c.x_mm, b.x, places=6)
            self.assertAlmostEqual(c.y_mm, b.y, places=6)
            expected = (MotionDirection.REVERSE if b.motion_direction == "REVERSE"
                        else MotionDirection.FORWARD)
            self.assertIs(c.motion_direction, expected)

    def test_entry_and_final_are_reverse_after_adapter(self) -> None:
        from controller.models import MotionDirection
        from integration.backend_adapter import waypoints_from_backend

        for w in waypoints_from_backend(route_b1()):
            if w.phase in ("ENTRY", "FINAL"):
                self.assertIs(w.motion_direction, MotionDirection.REVERSE)


class TestRearReverseGate(unittest.TestCase):
    """⑥ ENTRY/FINAL 후진이 제어기 게이트를 통과하고 조향이 살아 있는가."""

    def test_entry_and_final_pass_the_reverse_phase_gate(self) -> None:
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        for w in route_b1():
            if w.motion_direction == "REVERSE":
                self.assertIn(w.phase, cfg.reverse_allowed_phases)

    def test_entry_and_final_are_not_locked_to_straight_steering(self) -> None:
        """이게 막히면 계획한 후진 원호를 절대 못 탄다."""
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        for w in route_b1():
            if w.motion_direction == "REVERSE":
                self.assertFalse(cfg.reverse_steering_locked(w.phase),
                                 f"{w.phase} 후진이 11자로 묶여 있다")


class TestRearPlanTuningKnobs(unittest.TestCase):
    """실차에서 돌려야 할 값들이 실제로 경로에 반영되는가."""

    def test_larger_planning_radius_is_still_feasible_for_b1(self) -> None:
        """계획 반경 = 실측 최소값(610)이면 조향 여유가 0 이다.

        700mm 로 올려도 B1 이 성립해야 원호 추종에 여유를 줄 수 있다.
        """
        plan, _ = choose_rear_parking_plan(SPECS["B1"], from_pose=LEFT_START,
                                           min_radius_mm=700.0)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.overflow_mm, 0.0)
        self.assertLess(plan.aisle_offset_mm, 80.0)
        self.assertAlmostEqual(plan.setup_start[0], 425.0, delta=1.0)

    def test_entry_spacing_knob_changes_waypoint_count(self) -> None:
        few = build_rear_parking_waypoints(SPECS["B1"], route_id=1,
                                           from_pose=LEFT_START,
                                           from_heading_deg=0.0, step_deg=30.0)
        many = build_rear_parking_waypoints(SPECS["B1"], route_id=1,
                                            from_pose=LEFT_START,
                                            from_heading_deg=0.0, step_deg=12.0)
        n_few = sum(1 for w in few if w.phase == "ENTRY")
        n_many = sum(1 for w in many if w.phase == "ENTRY")
        self.assertLess(n_few, n_many)
        for wps in (few, many):
            self.assertEqual(wps[-1].phase, "FINAL")
            self.assertTrue(wps[-1].is_final)

    def test_route_is_identical_regardless_of_step_for_final(self) -> None:
        """샘플 간격을 바꿔도 FINAL 자세는 슬롯 중심 그대로여야 한다."""
        for step in (12.0, 20.0, 30.0):
            wps = build_rear_parking_waypoints(SPECS["B1"], route_id=1,
                                               from_pose=LEFT_START,
                                               from_heading_deg=0.0,
                                               step_deg=step)
            self.assertAlmostEqual(wps[-1].x, SPECS["B1"].center_x, places=6)
            self.assertAlmostEqual(wps[-1].y, SPECS["B1"].center_y, places=6)
            self.assertAlmostEqual(wps[-1].target_heading_deg, 270.0, places=6)


class TestPathCurvatureMetadata(unittest.TestCase):
    """planner 가 경로 기하를 waypoint 에 실어야 제어기가 feedforward 를 만든다."""

    def setUp(self) -> None:
        self.plan, _ = choose_rear_parking_plan(SPECS["B1"], from_pose=LEFT_START,
                                                min_radius_mm=700.0)
        self.wps = build_rear_parking_waypoints(
            SPECS["B1"], route_id=1, from_pose=LEFT_START,
            from_heading_deg=0.0, min_radius_mm=700.0)

    def test_arc_waypoints_carry_curvature(self) -> None:
        for w in self.wps:
            if w.phase in ("ALIGN", "ENTRY"):
                self.assertNotEqual(w.curvature, 0.0,
                                    f"{w.phase} wp{w.waypoint_id} 에 곡률이 없다")

    def test_curvature_magnitude_matches_planning_radius(self) -> None:
        for w in self.wps:
            if w.phase in ("ALIGN", "ENTRY"):
                self.assertAlmostEqual(abs(w.curvature), 1.0 / 700.0, places=9)

    def test_setup_and_entry_curve_opposite_ways(self) -> None:
        """전진 setup 과 후진 원호가 같은 방향이면 두 경로가 겹친다(퇴화)."""
        align = [w.curvature for w in self.wps if w.phase == "ALIGN"][0]
        entry = [w.curvature for w in self.wps if w.phase == "ENTRY"][0]
        self.assertLess(align * entry, 0.0)

    def test_final_has_no_curvature(self) -> None:
        """원호가 끝난 지점 — feedforward 가 남으면 과회전한다."""
        self.assertEqual(self.wps[-1].curvature, 0.0)

    def test_straight_routes_have_zero_curvature(self) -> None:
        """기존 인계 경로는 값이 0 이라 동작이 그대로다."""
        from parking.waypoints import build_waypoints
        for w in build_waypoints(SPECS["B1"], route_id=1, from_pose=LEFT_START,
                                 from_heading_deg=0.0, strict=True):
            self.assertEqual(w.curvature, 0.0)

    def test_curvature_survives_the_adapter(self) -> None:
        from integration.backend_adapter import waypoints_from_backend
        for core, backend in zip(waypoints_from_backend(self.wps), self.wps):
            self.assertAlmostEqual(core.curvature, backend.curvature, places=12)

    def test_reverse_start_tolerance_is_tighter_than_normal_align(self) -> None:
        aligns = [w for w in self.wps if w.phase == "ALIGN"]
        self.assertLess(aligns[-1].position_tolerance_cm,
                        PHASE_DEFAULTS["ALIGN"]["position_tolerance_cm"])
        self.assertLess(aligns[-1].heading_tolerance_deg,
                        PHASE_DEFAULTS["ALIGN"]["heading_tolerance_deg"])


class TestFeedforwardContract(unittest.TestCase):
    """제어기가 곡률을 논리 steering 으로 바꾸는 규약."""

    def test_feedforward_sign_follows_curvature(self) -> None:
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        self.assertGreater(cfg.feedforward_steering("ENTRY", +1 / 700.0), 0.0)
        self.assertLess(cfg.feedforward_steering("ENTRY", -1 / 700.0), 0.0)

    def test_feedforward_saturates_at_min_radius(self) -> None:
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        full = cfg.feedforward_steering("ENTRY", 1.0 / 610.0)
        self.assertAlmostEqual(full, 1.0, places=6)
        self.assertAlmostEqual(
            cfg.feedforward_steering("ENTRY", 1.0 / 300.0), 1.0, places=6)

    def test_final_gets_no_feedforward(self) -> None:
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        self.assertEqual(cfg.feedforward_steering("FINAL", 1 / 700.0), 0.0)

    def test_no_curvature_means_no_feedforward(self) -> None:
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        self.assertEqual(cfg.feedforward_steering("ENTRY", 0.0), 0.0)
        self.assertEqual(cfg.feedforward_steering("CRUISE", 1 / 700.0), 0.0)

    def test_controller_adds_feedforward_to_feedback(self) -> None:
        """같은 오차라도 곡률이 실린 waypoint 는 더 크게 꺾는다."""
        from controller.config import ControllerConfig
        from controller.models import MotionDirection, Pose, Waypoint
        from controller.pose_controller import PoseWaypointController
        cfg = ControllerConfig(allow_reverse=True, steer_kd=0.0)
        pose = Pose(0.0, 0.0, 0.0, timestamp=100.0,
                    heading_source="FRONT_CUSHION")
        plain = Waypoint(-1000.0, 300.0, phase="ENTRY", speed_cm_s=5.0,
                         position_tolerance_cm=4.0,
                         motion_direction=MotionDirection.REVERSE)
        curved = Waypoint(-1000.0, 300.0, phase="ENTRY", speed_cm_s=5.0,
                          position_tolerance_cm=4.0, curvature=1.0 / 700.0,
                          motion_direction=MotionDirection.REVERSE)
        a = PoseWaypointController(cfg).compute(pose, plain, now=100.0)
        b = PoseWaypointController(cfg).compute(pose, curved, now=100.0)
        self.assertNotAlmostEqual(a.logical_steering, b.logical_steering)
        self.assertGreater(b.logical_steering, a.logical_steering)


class TestReverseHeadingSafety(unittest.TestCase):
    """후진 중 궤적 heading 은 180° 뒤집힌다 — 그 값으로 조향하면 안 된다."""

    def _cmd(self, source, phase="ENTRY"):
        from controller.config import ControllerConfig
        from controller.models import MotionDirection, Pose, Waypoint
        from controller.pose_controller import PoseWaypointController
        ctl = PoseWaypointController(ControllerConfig(allow_reverse=True))
        wp = Waypoint(-1000.0, 300.0, phase=phase, speed_cm_s=5.0,
                      position_tolerance_cm=4.0,
                      motion_direction=MotionDirection.REVERSE)
        return ctl.compute(Pose(0.0, 0.0, 0.0, timestamp=100.0,
                                heading_source=source), wp, now=100.0)

    def test_trajectory_heading_halts_reverse_entry(self) -> None:
        cmd = self._cmd("TRAJECTORY")
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.reason, "REVERSE_HEADING_UNSAFE")

    def test_trajectory_heading_halts_reverse_final(self) -> None:
        self.assertEqual(self._cmd("TRAJECTORY", "FINAL").reason,
                         "REVERSE_HEADING_UNSAFE")

    def test_front_cushion_is_allowed(self) -> None:
        self.assertLess(self._cmd("FRONT_CUSHION").throttle, 0.0)

    def test_last_valid_halts_reverse_entry(self) -> None:
        """얼어붙은 heading은 후진 원호의 현재 차체 방향이 아니다."""
        cmd = self._cmd("LAST_VALID")
        self.assertEqual(cmd.throttle, 0.0)
        self.assertEqual(cmd.reason, "REVERSE_HEADING_UNSAFE")

    def test_unknown_source_is_allowed(self) -> None:
        """heading_source 를 안 주는 기존 호출부는 동작이 바뀌지 않는다."""
        self.assertLess(self._cmd(None).throttle, 0.0)

    def test_recovery_reverse_is_not_gated(self) -> None:
        """복구는 11자 후진이라 이 게이트 대상이 아니다 (별도 게이트가 있다)."""
        self.assertNotEqual(self._cmd("TRAJECTORY", "RECOVERY").reason,
                            "REVERSE_HEADING_UNSAFE")


class TestParkingThrottleInvariant(unittest.TestCase):
    """rear parking phase 에서는 어떤 조건이 와도 상한을 넘지 않는다.

    정지마찰 하한(0.70)은 전진 일반주행용 예외다. 그게 정밀 주차나 후진으로
    새면 조향이 포화되는 순간 throttle 이 7배로 튀어 차가 날아간다.
    """

    REAR_PHASES = ("APPROACH", "ALIGN", "ENTRY", "FINAL", "PARKING")

    def test_no_rear_phase_can_exceed_its_limit(self) -> None:
        from controller.config import ControllerConfig
        from controller.models import MotionDirection, Pose, Waypoint
        from controller.pose_controller import PoseWaypointController

        cfg = ControllerConfig(allow_reverse=True, max_throttle=0.40)
        worst = 0.0
        for phase in self.REAR_PHASES:
            for reverse in (False, True):
                limit = cfg.throttle_limit(phase, reverse=reverse)
                direction = (MotionDirection.REVERSE if reverse
                             else MotionDirection.FORWARD)
                for curvature in (0.0, 1 / 610.0, -1 / 610.0, 1 / 700.0):
                    for tx, ty in ((-2000.0, 0.0), (0.0, 2000.0),
                                   (1500.0, 1500.0), (-1500.0, 900.0),
                                   (300.0, -1800.0)):
                        for hdg in (0.0, 45.0, 120.0, 200.0, 300.0):
                            wp = Waypoint(tx, ty, phase=phase, speed_cm_s=12.0,
                                          position_tolerance_cm=4.0,
                                          curvature=curvature,
                                          motion_direction=direction)
                            ctl = PoseWaypointController(cfg)
                            cmd = ctl.compute(
                                Pose(0.0, 0.0, hdg, timestamp=100.0,
                                     heading_source="FRONT_CUSHION"),
                                wp, now=100.0)
                            worst = max(worst, abs(cmd.throttle))
                            self.assertLessEqual(
                                abs(cmd.throttle), limit + 1e-9,
                                f"{phase} reverse={reverse} k={curvature:.5f} "
                                f"hdg={hdg} → throttle {cmd.throttle} > {limit}")
        self.assertLessEqual(worst, 0.25 + 1e-9)
        self.assertNotAlmostEqual(worst, 0.70, places=2)

    def test_reverse_never_exceeds_reverse_cap_in_any_phase(self) -> None:
        """후진은 phase 를 불문하고 reverse 상한을 넘지 않는다 (RECOVERY 포함)."""
        from controller.config import ControllerConfig
        cfg = ControllerConfig(allow_reverse=True, max_throttle=0.40)
        for phase in ("CRUISE", "RECOVERY", "PARKING", "ENTRY", "FINAL",
                      "ALIGN", "APPROACH", None):
            ceiling = cfg.final_throttle_ceiling(phase, reverse=True)
            self.assertIsNotNone(ceiling, f"{phase} 후진에 상한이 없다")
            self.assertLessEqual(ceiling, cfg.reverse_max_throttle + 1e-9)

    def test_forward_cruise_keeps_the_stiction_override(self) -> None:
        """전진 일반주행의 정지마찰 예외는 그대로 살아 있어야 한다."""
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        self.assertIsNone(cfg.final_throttle_ceiling("CRUISE", reverse=False))
        self.assertAlmostEqual(cfg.stiction_floor_for("CRUISE", reverse=False),
                               0.70, places=6)

    def test_rear_route_contains_no_cruise_waypoint(self) -> None:
        """rear 경로에 CRUISE 가 섞이면 그 구간만 일반주행 상한이 걸린다."""
        for start in ((100.0, AISLE_Y), (125.0, AISLE_Y), LEFT_START):
            for w in build_rear_parking_waypoints(
                    SPECS["B1"], route_id=1, from_pose=start,
                    from_heading_deg=0.0, min_radius_mm=700.0):
                self.assertNotEqual(w.phase, "CRUISE",
                                    f"출발 {start} 경로에 CRUISE 가 있다")


class TestParkingThrottleCeiling(unittest.TestCase):
    """--max-throttle 은 주차 구간 상한을 못 올린다 — 실차에서 헷갈리기 쉽다."""

    def test_max_throttle_cannot_raise_parking_phases(self) -> None:
        from controller.config import ControllerConfig
        cfg = ControllerConfig(max_throttle=0.35)      # CLI --max-throttle 0.35
        for phase in ("APPROACH", "ALIGN", "ENTRY", "FINAL"):
            self.assertAlmostEqual(
                cfg.throttle_limit(phase, reverse=False), 0.25, places=6,
                msg=f"{phase} 상한이 parking_max_throttle 을 넘었다")
            self.assertAlmostEqual(
                cfg.throttle_limit(phase, reverse=True), 0.25, places=6)

    def test_parking_throttle_knob_actually_raises_the_ceiling(self) -> None:
        from controller.config import ControllerConfig
        cfg = ControllerConfig(max_throttle=0.40, parking_max_throttle=0.35,
                               reverse_max_throttle=0.35)
        self.assertAlmostEqual(cfg.throttle_limit("ENTRY", reverse=True), 0.35,
                               places=6)

    def test_strong_turn_floor_exceeds_the_reverse_ceiling(self) -> None:
        """최대 조향에서 stiction floor 가 후진 상한을 **의도적으로** 넘는다.

        전진 정지마찰용으로 넣은 값인데 후진 정밀 구간에도 그대로 걸린다.
        """
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        self.assertGreater(cfg.strong_turn_min_throttle,
                           cfg.throttle_limit("ENTRY", reverse=True))


class TestProvisionalCurvatureTable(unittest.TestCase):
    """실차 표(PROVISIONAL) 기반 steering↔curvature 모델.

    값 자체가 CALIBRATED 라는 뜻이 아니다. **모델의 구조적 성질**만 고정한다.
    """

    def test_table_points_are_reproduced(self) -> None:
        from controller.config import (STEERING_CURVATURE_TABLE_MM,
                                       curvature_for_steering)
        for (drive, side), pts in STEERING_CURVATURE_TABLE_MM.items():
            rev = drive == "REVERSE"
            sign = 1.0 if side == "LEFT" else -1.0
            for s, radius in pts:
                k = curvature_for_steering(sign * s, reverse=rev)
                self.assertAlmostEqual(abs(1.0 / k), radius, places=6,
                                       msg=f"{drive}/{side} |s|={s}")

    def test_left_right_asymmetry_is_preserved(self) -> None:
        """RIGHT 가 같은 |steering| 에서 더 조인다 — 실차 관측 경향."""
        from controller.config import curvature_for_steering
        for s in (0.70, 0.90):
            left = abs(1.0 / curvature_for_steering(+s, reverse=False))
            right = abs(1.0 / curvature_for_steering(-s, reverse=False))
            self.assertLess(right, left, f"|s|={s} 에서 비대칭이 사라졌다")

    def test_curvature_increases_with_steering(self) -> None:
        from controller.config import curvature_for_steering
        for rev in (False, True):
            for sign in (+1.0, -1.0):
                ks = [abs(curvature_for_steering(sign * s, reverse=rev))
                      for s in (0.15, 0.30, 0.45, 0.70, 0.90)]
                self.assertEqual(ks, sorted(ks), "곡률이 단조 증가하지 않는다")

    def test_near_straight_zone_is_linear_through_origin(self) -> None:
        """|s|<=0.30 은 반경을 못 잰 구간 — 원점 직선으로만 취급한다."""
        from controller.config import (NEAR_STRAIGHT_STEERING,
                                       curvature_for_steering)
        half = curvature_for_steering(NEAR_STRAIGHT_STEERING / 2, reverse=False)
        full = curvature_for_steering(NEAR_STRAIGHT_STEERING, reverse=False)
        self.assertAlmostEqual(full / 2.0, half, places=9)
        self.assertEqual(curvature_for_steering(0.0, reverse=False), 0.0)

    def test_sign_convention_left_is_positive_curvature(self) -> None:
        from controller.config import curvature_for_steering
        self.assertGreater(curvature_for_steering(+0.7, reverse=False), 0.0)
        self.assertLess(curvature_for_steering(-0.7, reverse=False), 0.0)

    def test_inverse_round_trips(self) -> None:
        from controller.config import (curvature_for_steering,
                                       steering_for_curvature)
        for rev in (False, True):
            for s in (-0.9, -0.45, -0.1, 0.1, 0.45, 0.9):
                k = curvature_for_steering(s, reverse=rev)
                self.assertAlmostEqual(steering_for_curvature(k, reverse=rev),
                                       s, places=6)

    def test_reverse_table_differs_from_forward(self) -> None:
        """후진은 initial guess 라도 전진과 별도 표를 가져야 한다."""
        from controller.config import STEERING_CURVATURE_TABLE_MM
        self.assertIn(("REVERSE", "LEFT"), STEERING_CURVATURE_TABLE_MM)
        self.assertIn(("REVERSE", "RIGHT"), STEERING_CURVATURE_TABLE_MM)

    def test_feedforward_uses_table_and_saturates(self) -> None:
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        # 970mm(LEFT 0.70 점)를 요구하면 feedforward 도 0.70 근처여야 한다
        ff = cfg.feedforward_steering("ENTRY", 1.0 / 970.0, reverse=False)
        self.assertAlmostEqual(ff, 0.70, places=2)
        # 물리 한계보다 조인 곡률은 1.0 으로 포화
        self.assertAlmostEqual(
            cfg.feedforward_steering("ENTRY", 1.0 / 300.0, reverse=False),
            1.0, places=6)

    def test_planner_radii_leave_steering_margin(self) -> None:
        """계획 반경은 |steering| 0.9 실측보다 커야 보정 여력이 남는다."""
        from parking.waypoints import REAR_RADIUS_CANDIDATES
        from controller.config import curvature_for_steering
        tightest = abs(1.0 / curvature_for_steering(-0.90, reverse=False))
        self.assertGreaterEqual(min(REAR_RADIUS_CANDIDATES), tightest)
        self.assertGreater(len(set(REAR_RADIUS_CANDIDATES)), 1,
                           "단일 반경에 의존하면 안 된다")


def _circumradius(a, b, c) -> float:
    """세 점의 외접원 반경. 일직선이면 inf."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return float("inf")
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return math.hypot(ax - ux, ay - uy)


if __name__ == "__main__":
    unittest.main()
