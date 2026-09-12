"""통로 주행 → 슬롯 앞 인계 경로 생성기 + 후진 복구 계획기 검증.

역할 분담이 전제다. **슬롯 안으로 넣는 주차 기동은 하드웨어팀 주차 공식**이
담당하고, backend 는 차를 슬롯 앞에 통로 방향(가로)으로 세워 인계하는
데까지만 책임진다. 그래서 경로에 슬롯 축으로 꺾어 들어가는 90° 선회가
있으면 안 된다 — 통로에서 진입선까지 300mm 인데 최소 선회 반경은 570mm 라
물리적으로 불가능하고, 애초에 우리 일도 아니다.

실행: python -m unittest parking.test_planner -v
"""

from __future__ import annotations

import math
import unittest

from parking.recovery import (
    MAX_BACKUP_MM,
    MIN_BACKUP_MM,
    REVERSE_TRIGGER_REASONS,
    decide,
    forward_unreachable,
    plan_reverse_recovery,
)
from parking.waypoints import (
    AISLE_Y,
    MIN_TURN_RADIUS_MM,
    WIRE_PHASES,
    InfeasibleRouteError,
    build_waypoints,
    default_slot_specs,
    plan_entry_turn,
    plan_handoff,
)

SPECS = default_slot_specs()
ALL_SLOTS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
LEFT_START = (150.0, AISLE_Y)
RIGHT_START = (1100.0, AISLE_Y)
# ─ 진입 우회전 (향후 확장) ─
# 지금 실차 운용은 "차를 통로 위에 놓고 시작"이다. 최소 선회 반경 610mm 로는
# 바닥 아래쪽(y≈150)에서 중앙 통로(y=600)로 90° 우회전을 끝낼 수 없기
# 때문이다 — 정렬 완료 지점이 y=760 으로 통로를 160mm 지나친다.
# plan_entry_turn 은 그 확장을 위해 남겨두고, 여기서는 실제로 성립하는
# 조합(통로 y=760)으로 검증한다.
ENTRY_START = (150.0, 150.0)
ENTRY_AISLE_Y = 760.0                 # = 150 + 610, 우회전이 딱 끝나는 높이
ENTRY_HEADING = 90.0                  # +y — 통로와 수직으로 서야 90° 선회다
# 합류점 x = 150 + 610 = 760 이후의 슬롯만 닿는다
ENTRY_REACHABLE = ["A3", "A4", "B3", "B4"]


class TestHandoffPoint(unittest.TestCase):
    def test_handoff_is_on_the_aisle_in_front_of_the_slot(self):
        for sid in ALL_SLOTS:
            with self.subTest(slot=sid):
                p = plan_handoff(SPECS[sid], from_pose=LEFT_START)
                self.assertAlmostEqual(p.point[1], AISLE_Y, places=6)
                self.assertAlmostEqual(p.point[0], SPECS[sid].center_x, places=6)

    def test_handoff_heading_is_along_the_aisle(self):
        """가로로 선다 — 슬롯 축(90°/270°)이 아니라 통로 축(0°/180°)."""
        for sid in ALL_SLOTS:
            with self.subTest(slot=sid):
                p = plan_handoff(SPECS[sid], from_pose=LEFT_START)
                self.assertIn(p.heading_deg, (0.0, 180.0))
                self.assertNotEqual(p.heading_deg, SPECS[sid].target_heading_deg)

    def test_every_slot_reachable_from_the_aisle(self):
        """이미 통로 위라면 8칸 전부 도달 가능하다 (선회가 없으므로)."""
        for sid in ALL_SLOTS:
            with self.subTest(slot=sid):
                self.assertTrue(plan_handoff(SPECS[sid], from_pose=LEFT_START).feasible)

    def test_approach_direction_follows_vehicle_position(self):
        """인계 지점을 지나치면 선회 반경 때문에 되돌아올 수 없다."""
        self.assertEqual(plan_handoff(SPECS["A4"], from_pose=LEFT_START).heading_deg, 0.0)
        self.assertEqual(plan_handoff(SPECS["A1"], from_pose=RIGHT_START).heading_deg, 180.0)

    def test_gap_to_slot_is_the_hw_working_space(self):
        """양쪽 슬롯행 모두 인계 지점에서 진입선까지 30cm 를 확보한다."""
        for sid in ALL_SLOTS:
            with self.subTest(slot=sid):
                self.assertAlmostEqual(
                    plan_handoff(SPECS[sid], from_pose=LEFT_START).gap_to_slot_mm,
                    300.0, places=6)

    def test_offset_shifts_along_travel_direction(self):
        """HW 주차 공식이 슬롯을 지나친 위치를 요구하면 offset 으로 맞춘다."""
        left = plan_handoff(SPECS["A3"], from_pose=LEFT_START, offset_mm=150.0)
        right = plan_handoff(SPECS["A3"], from_pose=RIGHT_START, offset_mm=150.0)
        self.assertGreater(left.point[0], SPECS["A3"].center_x)
        self.assertLess(right.point[0], SPECS["A3"].center_x)

    def test_slightly_off_aisle_merges_instead_of_turning(self):
        """통로에서 몇 cm 벗어난 건 S자 합류로 붙는다.

        실측으로 잡힌 버그다 — 9.5cm 벗어났을 뿐인데 90° 선회로 처리해
        (반경 61cm 필요) 전 슬롯이 거부됐다.
        """
        p = plan_handoff(SPECS["A3"], from_pose=(150.0, 505.0), from_heading_deg=0.0)
        self.assertTrue(p.feasible)
        self.assertFalse(p.lead_in_turn_required)
        self.assertIsNotNone(p.merge_point)

    def test_perpendicular_start_needs_a_turn_not_a_merge(self):
        """통로와 수직으로 서 있으면 합류가 아니라 90° 선회다."""
        p = plan_handoff(SPECS["A3"], from_pose=(150.0, 150.0),
                         from_heading_deg=90.0, aisle_y=760.0)
        self.assertTrue(p.lead_in_turn_required)
        self.assertIsNone(p.merge_point)

    def test_on_aisle_start_needs_neither(self):
        p = plan_handoff(SPECS["A3"], from_pose=LEFT_START, from_heading_deg=0.0)
        self.assertFalse(p.lead_in_turn_required)
        self.assertIsNone(p.merge_point)

    def test_merge_keeps_travel_direction(self):
        """합류 중에는 방향을 못 바꾼다 — 뒤쪽 슬롯은 거부해야 한다."""
        back = plan_handoff(SPECS["A1"], from_pose=(150.0, 505.0), from_heading_deg=0.0)
        self.assertFalse(back.feasible)
        self.assertIn("지나친다", back.reason)

    def test_out_of_bounds_handoff_raises_in_strict_mode(self):
        with self.assertRaises(InfeasibleRouteError):
            build_waypoints(SPECS["A3"], route_id=1, from_pose=LEFT_START,
                            aisle_y=20.0, strict=True)


class TestRouteShape(unittest.TestCase):
    def setUp(self) -> None:
        self.wps = build_waypoints(SPECS["A4"], route_id=7, from_pose=LEFT_START)

    def test_phases_are_wire_legal(self):
        """펌웨어 protocol.c 가 모르는 phase 를 만들면 WAYPOINT 가 거절된다."""
        for wp in self.wps:
            self.assertIn(wp.phase, WIRE_PHASES, f"wire 에 없는 phase: {wp.phase}")

    def test_no_waypoint_enters_the_slot(self):
        """슬롯 안 waypoint 를 만들면 우리가 주차까지 하겠다는 뜻이 된다.

        주차 기동은 HW 주차 공식 담당이다. 우리 경로는 통로에서 끝난다.
        """
        for sid in ALL_SLOTS:
            spec = SPECS[sid]
            lo, hi = sorted((spec.center_y - spec.length / 2,
                             spec.center_y + spec.length / 2))
            for wp in build_waypoints(spec, route_id=1, from_pose=LEFT_START):
                self.assertFalse(lo <= wp.y <= hi,
                                 f"{sid}: {wp.phase} 가 슬롯 안({wp.y:.0f}mm)에 있다")

    def test_route_stays_on_the_aisle(self):
        for wp in self.wps:
            self.assertAlmostEqual(wp.y, AISLE_Y, places=6)

    def test_route_has_no_turn(self):
        """모든 노드가 한 직선 위 = 90° 코너 없음."""
        ys = {round(w.y, 6) for w in self.wps}
        self.assertEqual(len(ys), 1)

    def test_ids_and_terminator(self):
        self.assertEqual([w.waypoint_id for w in self.wps],
                         list(range(1, len(self.wps) + 1)))
        self.assertTrue(all(w.route_id == 7 for w in self.wps))
        self.assertTrue(self.wps[-1].is_final)
        self.assertEqual(sum(w.is_final for w in self.wps), 1)

    def test_only_the_handoff_requires_heading(self):
        heading_wps = [w for w in self.wps if w.heading_required]
        self.assertEqual(len(heading_wps), 1)
        self.assertTrue(heading_wps[0].is_final)

    def test_consecutive_waypoints_are_far_enough_apart(self):
        """간격이 허용오차보다 좁으면 그 waypoint 는 없는 것과 같다.

        기존 생성기는 CRUISE(허용 8cm) 다음 APPROACH 를 3cm 뒤에 뒀다.
        CRUISE 도착이 나는 순간 이미 APPROACH 안이라 단계가 통째로
        건너뛰어졌다 — 구간 판정이 애매했던 원인이다.
        """
        for a, b in zip(self.wps, self.wps[1:]):
            gap_cm = math.hypot(b.x - a.x, b.y - a.y) / 10.0
            self.assertGreaterEqual(
                gap_cm, a.position_tolerance_cm,
                f"{a.phase}({a.waypoint_id}) → {b.phase}({b.waypoint_id}) "
                f"간격 {gap_cm:.1f}cm 가 허용오차 {a.position_tolerance_cm}cm 이하")

    def test_aisle_run_is_segmented_for_progress(self):
        """긴 직선을 노드 하나로 두면 중간 진행률을 알 수 없다."""
        self.assertGreaterEqual(sum(w.phase == "CRUISE" for w in self.wps), 2)

    def test_route_is_forward_only(self):
        self.assertTrue(all(w.motion_direction == "FORWARD" for w in self.wps))

    def test_approach_capture_is_coarser_than_completion(self):
        approach = next(w for w in self.wps if w.phase == "APPROACH")
        self.assertIsNotNone(approach.capture_tolerance_cm)
        self.assertGreater(approach.capture_tolerance_cm,
                           approach.position_tolerance_cm)

    def test_all_waypoints_inside_the_lot(self):
        for wp in self.wps:
            self.assertTrue(0.0 <= wp.x <= 1200.0 and 0.0 <= wp.y <= 1200.0)


class TestDeadZone(unittest.TestCase):
    """전진 도달 가능성 판정 (후진을 걸지 말지의 기준)."""

    def test_target_straight_ahead_is_reachable(self):
        self.assertFalse(forward_unreachable((0.0, 0.0), 0.0, (1000.0, 0.0)))

    def test_target_inside_turning_circle_is_unreachable(self):
        self.assertTrue(
            forward_unreachable((0.0, 0.0), 0.0, (0.0, MIN_TURN_RADIUS_MM)))

    def test_target_outside_the_circle_is_reachable(self):
        self.assertFalse(
            forward_unreachable((0.0, 0.0), 0.0, (0.0, 2 * MIN_TURN_RADIUS_MM + 1)))

    def test_boundary_is_treated_as_reachable(self):
        """원 밖이면 접선으로 닿을 수 있으므로 후진하지 않는다.

        정확히 원 위인 점은 부동소수점 반올림으로 안팎이 갈리므로 1mm 바깥을
        본다 — 실제 판정에서도 1mm 차이는 카메라 오차(9~11cm)에 묻힌다.
        """
        self.assertFalse(forward_unreachable(
            (0.0, 0.0), 0.0, (MIN_TURN_RADIUS_MM + 1.0, MIN_TURN_RADIUS_MM)))

    def test_overshot_handoff_is_unreachable(self):
        """인계 지점을 지나쳐 버린 실제 상황 — 되돌아 갈 수 없다.

        사각지대 판정만으로는 안 잡힌다. 등 뒤 목표는 "한 바퀴 돌면 도달"이라
        원 안에 안 들어가기 때문이다. 맵 크기를 줘야 그 한 바퀴가 실제로
        불가능하다는 걸 안다.
        """
        handoff = (1100.0, AISLE_Y)
        overshot = (1180.0, AISLE_Y)      # 8cm 지나침, 계속 +x 를 보고 있음
        self.assertFalse(forward_unreachable(overshot, 0.0, handoff),
                         "맵 제약이 없으면 이론상 도달 가능이다")
        self.assertTrue(forward_unreachable(overshot, 0.0, handoff,
                                            lot_mm=(1200.0, 1200.0)))

    def test_far_behind_with_room_to_loop_is_reachable(self):
        """맵 한가운데라 선회원이 통째로 들어가면 돌아서 갈 수 있다."""
        self.assertFalse(forward_unreachable(
            (600.0, 600.0), 0.0, (400.0, 600.0), radius_mm=200.0,
            lot_mm=(1200.0, 1200.0)))


class TestRecoveryDecision(unittest.TestCase):
    def test_no_heading_declines(self):
        d = decide((100.0, 100.0), None, (200.0, 200.0))
        self.assertFalse(d.needed)
        self.assertEqual(d.reason, "NO_HEADING")

    def test_reachable_target_declines(self):
        d = decide((0.0, 0.0), 0.0, (1000.0, 0.0))
        self.assertFalse(d.needed)
        self.assertEqual(d.reason, "FORWARD_REACHABLE")

    def test_heading_mismatch_backs_up_straight_along_current_heading(self):
        """11자 후진 — 목표 축이 아니라 **차가 보고 있는 방향** 반대로 물러난다.

        목표 heading 축으로 물러나면 후진 중에 조향이 들어간다. 뒤를 못 보는
        상태에서 궤적이 휘고, 궤적 기반 heading 이면 부호까지 뒤집힌다.
        """
        handoff = (875.0, AISLE_Y)
        d = decide(handoff, 25.0, handoff, target_heading_deg=0.0)
        self.assertTrue(d.needed)
        # 25° 반대 방향으로 backup_mm 만큼 이동한 자리여야 한다
        self.assertAlmostEqual(
            d.backup_point[0], handoff[0] - d.backup_mm * math.cos(math.radians(25.0)),
            places=6)
        self.assertAlmostEqual(
            d.backup_point[1], handoff[1] - d.backup_mm * math.sin(math.radians(25.0)),
            places=6)

    def test_backup_distance_scales_with_heading_error(self):
        h = (875.0, AISLE_Y)
        small = decide(h, 5.0, h, target_heading_deg=0.0)
        large = decide(h, 50.0, h, target_heading_deg=0.0)
        self.assertGreater(large.backup_mm, small.backup_mm)

    def test_backup_distance_is_bounded(self):
        h = (875.0, AISLE_Y)
        # heading 허용오차(FINAL 12°) 아래는 애초에 복구 대상이 아니다
        for err in (20.0, 45.0, 120.0, 179.0):
            d = decide(h, err, h, target_heading_deg=0.0)
            self.assertGreaterEqual(d.backup_mm, MIN_BACKUP_MM)
            self.assertLessEqual(d.backup_mm, MAX_BACKUP_MM)

    def test_aligned_arrival_does_not_trigger_reverse(self):
        h = (875.0, AISLE_Y)
        self.assertFalse(decide(h, 0.0, h, target_heading_deg=0.0).needed)


class TestRecoveryPlan(unittest.TestCase):
    def setUp(self) -> None:
        self.wps = build_waypoints(SPECS["A4"], route_id=1, from_pose=LEFT_START)
        self.handoff = self.wps[-1]

    def test_produces_single_reverse_waypoint(self):
        out = plan_reverse_recovery((self.handoff.x, self.handoff.y), 30.0,
                                    self.handoff, route_id=2)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].motion_direction, "REVERSE")
        self.assertEqual(out[0].phase, "RECOVERY")

    def test_recovery_waypoint_is_never_final(self):
        """is_final 이 붙으면 원래 route 로 복귀하기 전에 DONE 이 된다."""
        out = plan_reverse_recovery((self.handoff.x, self.handoff.y), 30.0,
                                    self.handoff, route_id=2)
        self.assertFalse(out[0].is_final)

    def test_recovery_backs_straight_from_the_car(self):
        """복구 지점은 차 뒤 일직선 — 후진 중 조향이 필요 없어야 한다."""
        out = plan_reverse_recovery((self.handoff.x, self.handoff.y), 30.0,
                                    self.handoff, route_id=2,
                                    bounds_mm=(1200.0, 1200.0))
        self.assertIsNotNone(out, "50cm 후진이 맵 안에 들어가야 한다")
        dx = self.handoff.x - out[0].x
        dy = self.handoff.y - out[0].y
        self.assertAlmostEqual(math.degrees(math.atan2(dy, dx)) % 360.0, 30.0,
                               places=4)

    def test_rejects_backup_outside_the_lot(self):
        """맵 밖으로 물러나는 경로는 만들지 않는다.

        통로 오른쪽 끝(x=110cm)에서 서쪽을 보고 있으면 곧게 물러날 때
        동쪽 벽을 넘는다 — 110 + 50 = 160cm > 120cm.
        """
        out = plan_reverse_recovery((self.handoff.x, self.handoff.y), 180.0,
                                    self.handoff, route_id=2,
                                    bounds_mm=(1200.0, 1200.0))
        self.assertIsNone(out)

    def test_declines_when_no_heading(self):
        out = plan_reverse_recovery((self.handoff.x, self.handoff.y), None,
                                    self.handoff, route_id=2)
        self.assertIsNone(out)

    def test_accepts_controller_schema_waypoint(self):
        """미션이 들고 있는 건 x_mm/y_mm 필드를 쓰는 controller Waypoint 다."""
        from integration.backend_adapter import waypoint_from_backend
        conv = waypoint_from_backend(self.handoff)
        out = plan_reverse_recovery((conv.x_mm, conv.y_mm), 30.0, conv, route_id=2)
        self.assertIsNotNone(out)


class TestTriggerContract(unittest.TestCase):
    """후진을 거는 사유가 실제로 발생하는 문자열과 일치하는가."""

    def test_reasons_match_mission_and_guard(self):
        from host_control.approach_guard import ApproachEvent
        self.assertIn(ApproachEvent.COARSE_MISSED.value, REVERSE_TRIGGER_REASONS)
        self.assertIn(ApproachEvent.FINE_MISSED.value, REVERSE_TRIGGER_REASONS)
        self.assertIn("HEADING_OUT_OF_TOLERANCE", REVERSE_TRIGGER_REASONS)

    def test_cruise_is_not_a_reverse_phase(self):
        """통로 주행 중 후진은 뒤를 못 보므로 허용하지 않는다."""
        from controller.config import ControllerConfig
        cfg = ControllerConfig()
        self.assertNotIn("CRUISE", cfg.reverse_allowed_phases)
        self.assertIn("RECOVERY", cfg.reverse_allowed_phases)
        self.assertIn("APPROACH", cfg.reverse_allowed_phases)

    def test_final_is_a_reverse_phase(self):
        """인계 지점(FINAL)에서 잘못 서면 후진으로 다시 잡아야 한다."""
        from controller.config import ControllerConfig
        self.assertIn("FINAL", ControllerConfig().reverse_allowed_phases)


if __name__ == "__main__":
    unittest.main()


class TestEntryTurn(unittest.TestCase):
    """출발점 → 통로 합류 우회전.

    반경은 고르는 값이 아니라 **출발 위치가 정한다**: R = 통로y − 출발y.
    출발점 (150,100) 이면 R=500mm 이고 합류점이 정확히 A2 앞(650,600)이다.
    이 관계가 계획 반경 500mm 의 근거다.
    """

    def test_radius_comes_from_start_position(self):
        t = plan_entry_turn(ENTRY_START, aisle_y=ENTRY_AISLE_Y)
        self.assertAlmostEqual(t.radius_mm, MIN_TURN_RADIUS_MM, places=6)
        self.assertAlmostEqual(t.join_point[0],
                               ENTRY_START[0] + MIN_TURN_RADIUS_MM, places=6)
        self.assertAlmostEqual(t.join_point[1], ENTRY_AISLE_Y, places=6)

    def test_corner_start_cannot_reach_the_centre_aisle(self):
        """왜 지금 통로 위에서 출발하는지 — 이 사실이 설계 근거다.

        실측 최소 선회 반경 610mm 인데 바닥에서 중앙 통로까지는 600mm 뿐이라,
        어디에 놓든 90° 우회전을 통로 안에서 끝낼 수 없다.
        """
        for start_y in (0.0, 100.0, 150.0, 200.0):
            with self.subTest(start_y=start_y):
                t = plan_entry_turn((150.0, start_y))     # 기본 통로 y=600
                self.assertFalse(t.feasible)
                self.assertIn("최소 선회 반경", t.reason)

    def test_starting_too_close_to_the_aisle_is_rejected(self):
        """통로에 가까이 놓으면 반경이 모자라 못 돈다 — 조용히 실패시키지 않는다."""
        t = plan_entry_turn((150.0, 300.0), aisle_y=ENTRY_AISLE_Y)   # R=460mm
        self.assertFalse(t.feasible)
        self.assertIn("최소 선회 반경", t.reason)

    def test_arc_starts_at_the_car_and_ends_on_the_aisle(self):
        from parking.waypoints import _entry_arc_points
        t = plan_entry_turn(ENTRY_START, aisle_y=ENTRY_AISLE_Y)
        pts = _entry_arc_points(t)
        self.assertAlmostEqual(pts[-1][0], t.join_point[0], places=6)
        self.assertAlmostEqual(pts[-1][1], t.join_point[1], places=6)
        for x, y in pts:                              # 원 위에 있는가
            self.assertAlmostEqual(
                math.hypot(x - t.center[0], y - t.center[1]), t.radius_mm, places=6)

    def test_arc_turns_right(self):
        """+y 로 출발해 +x 로 나온다 = 우회전. x 는 늘고 y 도 는다."""
        from parking.waypoints import _entry_arc_points
        pts = _entry_arc_points(plan_entry_turn(ENTRY_START, aisle_y=ENTRY_AISLE_Y))
        xs = [p[0] for p in pts]
        self.assertEqual(xs, sorted(xs))
        self.assertGreater(pts[-1][1], pts[0][1])

    def test_slots_left_of_the_join_point_are_rejected(self):
        """합류점보다 왼쪽 슬롯은 지나쳐 버린다 — 전진으로 못 돌아온다."""
        for sid in ("A1", "A2", "B1", "B2"):
            with self.subTest(slot=sid):
                p = plan_handoff(SPECS[sid], from_pose=ENTRY_START,
                                 from_heading_deg=ENTRY_HEADING,
                                 aisle_y=ENTRY_AISLE_Y)
                self.assertFalse(p.feasible)
                self.assertIn("지나친다", p.reason)

    def test_slots_right_of_the_join_point_are_reachable(self):
        for sid in ENTRY_REACHABLE:
            with self.subTest(slot=sid):
                self.assertTrue(plan_handoff(SPECS[sid], from_pose=ENTRY_START,
                                             from_heading_deg=ENTRY_HEADING,
                                             aisle_y=ENTRY_AISLE_Y).feasible)


class TestEntryRoute(unittest.TestCase):
    """출발점에서 만든 전체 경로 (우회전 + 통로 직진)."""

    def test_arc_curvature_never_tighter_than_the_plan_radius(self):
        for sid in ENTRY_REACHABLE:
            with self.subTest(slot=sid):
                pts = [(w.x, w.y) for w in
                       build_waypoints(SPECS[sid], route_id=1,
                                       from_pose=ENTRY_START,
                                       from_heading_deg=ENTRY_HEADING,
                                       aisle_y=ENTRY_AISLE_Y)]
                pts = [ENTRY_START] + pts
                for (x1, y1), (x2, y2), (x3, y3) in zip(pts, pts[1:], pts[2:]):
                    area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                    if area2 < 1e-6:
                        continue
                    a = math.hypot(x2 - x1, y2 - y1)
                    b = math.hypot(x3 - x2, y3 - y2)
                    c = math.hypot(x3 - x1, y3 - y1)
                    r = (a * b * c) / (2.0 * area2)
                    self.assertGreaterEqual(
                        r, MIN_TURN_RADIUS_MM * 0.95,
                        f"{sid}: ({x2:.0f},{y2:.0f}) 곡률 반경 {r:.0f}mm 가 "
                        f"계획 반경 {MIN_TURN_RADIUS_MM:.0f}mm 보다 작다")

    def test_approach_is_the_last_before_handoff(self):
        """감속 구간은 끊기지 않아야 COARSE→FINE 포착이 성립한다."""
        for sid in ENTRY_REACHABLE:
            with self.subTest(slot=sid):
                wps = build_waypoints(SPECS[sid], route_id=1, from_pose=ENTRY_START,
                                       aisle_y=ENTRY_AISLE_Y)
                self.assertEqual(wps[-1].phase, "FINAL")
                self.assertEqual(wps[-2].phase, "APPROACH")

    def test_handoff_heading_is_along_the_aisle(self):
        for sid in ENTRY_REACHABLE:
            with self.subTest(slot=sid):
                last = build_waypoints(SPECS[sid], route_id=1,
                                       from_pose=ENTRY_START)[-1]
                self.assertEqual(last.target_heading_deg, 0.0)

    def test_no_duplicate_waypoints(self):
        """A2 는 원호 끝이 곧 인계 지점이라 좌표가 겹치기 쉽다."""
        for sid in ENTRY_REACHABLE:
            with self.subTest(slot=sid):
                wps = build_waypoints(SPECS[sid], route_id=1, from_pose=ENTRY_START,
                                       aisle_y=ENTRY_AISLE_Y)
                for a, b in zip(wps, wps[1:]):
                    self.assertGreater(math.hypot(b.x - a.x, b.y - a.y), 1.0,
                                       f"{sid}: {a.waypoint_id}/{b.waypoint_id} 중복")

    def test_route_never_enters_a_slot(self):
        for sid in ENTRY_REACHABLE:
            spec = SPECS[sid]
            for other in SPECS.values():
                lo, hi = (other.center_y - other.length / 2,
                          other.center_y + other.length / 2)
                xlo, xhi = (other.center_x - other.width / 2,
                            other.center_x + other.width / 2)
                for wp in build_waypoints(spec, route_id=1, from_pose=ENTRY_START):
                    if lo <= wp.y <= hi and xlo <= wp.x <= xhi:
                        self.fail(f"{sid} 경로의 {wp.phase} 가 {other.slot_id} 안에 있다")


class TestTravelDirection(unittest.TestCase):
    """통로 위에서도 "가던 방향"이 접근 방향이다.

    위치만 보고 고르면 등 뒤 슬롯에 180° 접근하라는 답이 나오는데, 선회 지름이
    맵 한 변과 맞먹어 U턴이 불가능하다.
    """

    def test_slot_behind_the_car_is_rejected(self):
        # x=110cm 에서 +x 로 달리는 중 → 왼쪽 슬롯은 이미 지나쳤다
        p = plan_handoff(SPECS["A1"], from_pose=(1100.0, AISLE_Y), from_heading_deg=0.0)
        self.assertFalse(p.feasible)
        self.assertIn("뒤쪽", p.reason)

    def test_slot_ahead_is_accepted_in_both_directions(self):
        fwd = plan_handoff(SPECS["A4"], from_pose=(150.0, AISLE_Y), from_heading_deg=0.0)
        rev = plan_handoff(SPECS["A1"], from_pose=(1100.0, AISLE_Y), from_heading_deg=180.0)
        self.assertTrue(fwd.feasible)
        self.assertTrue(rev.feasible)
        self.assertEqual(fwd.heading_deg, 0.0)
        self.assertEqual(rev.heading_deg, 180.0)

    def test_every_slot_reachable_from_either_end(self):
        """통로 양 끝에서 출발하면 8칸 전부 닿는다."""
        for pose, hdg in (((150.0, AISLE_Y), 0.0), ((1100.0, AISLE_Y), 180.0)):
            for sid in ALL_SLOTS:
                with self.subTest(start=pose, slot=sid):
                    self.assertTrue(plan_handoff(SPECS[sid], from_pose=pose,
                                                 from_heading_deg=hdg).feasible)

    def test_deceleration_point_never_falls_behind_the_car(self):
        """감속점을 인계 앞 25cm 로 고정하면 차 뒤나 맵 밖에 찍힌다."""
        for sid in ALL_SLOTS:
            with self.subTest(slot=sid):
                p = plan_handoff(SPECS[sid], from_pose=(1100.0, AISLE_Y),
                                 from_heading_deg=180.0)
                self.assertTrue(p.feasible, p.reason)
                self.assertLessEqual(p.lead_point[0], 1100.0 + 1e-6)


class TestBackupDistance(unittest.TestCase):
    """후진 거리 계산 — 되짚은 자리에서 다시 전진으로 잡을 수 있어야 한다."""

    def test_target_directly_behind_is_solvable(self):
        """일직선 뒤 목표에 여유(margin)를 얹으면 해가 없어진다.

        목표가 거의 정면일 때 선회원 중심까지 거리는 sqrt(d^2+R^2) 라 R 을
        조금만 넘는다. 거기에 반경 여유를 더하면 아무리 물러나도 사각지대를
        못 벗어난다고 나온다 — 실측으로 잡힌 버그다.
        """
        d = decide((425.0, 600.0), 0.0, (175.0, 600.0), lot_mm=(1200.0, 1200.0))
        self.assertTrue(d.needed, d.reason)
        self.assertGreater(d.backup_mm, 250.0)          # 목표를 지나칠 만큼
        self.assertAlmostEqual(d.backup_point[1], 600.0, places=6)
        self.assertLess(d.backup_point[0], 175.0)       # 목표보다 뒤로

    def test_backup_leaves_target_ahead(self):
        """되짚은 자리에서 목표가 앞쪽에 있어야 다시 전진으로 간다."""
        pose, tgt = (425.0, 600.0), (175.0, 600.0)
        d = decide(pose, 0.0, tgt, lot_mm=(1200.0, 1200.0))
        ahead = (tgt[0] - d.backup_point[0])            # heading 0 → x 성분
        self.assertGreater(ahead, 0.0)

    def test_backup_point_stays_reachable(self):
        pose, tgt = (425.0, 600.0), (175.0, 600.0)
        d = decide(pose, 0.0, tgt, lot_mm=(1200.0, 1200.0))
        self.assertFalse(forward_unreachable(d.backup_point, 0.0, tgt))
