"""Gymnasium-compatible parking slot allocation RL environment.

Architecture
------------
Three distinct layers:

  RL Assignment Layer  (step / action_masks / _get_obs)
      The agent assigns one incoming vehicle to a slot each step.
      Uses the reservation table to enforce Safety Shield.
      Terminates when step_count >= MAX_STEPS or deadlock.

  Traffic Simulator Layer  (advance_time / _active_vehicles)
      Tracks every assigned vehicle as it physically moves through
      waypoints.  Position is linearly interpolated between nodes.
      advance_time(dt) advances the clock and updates vehicle physics.

  Departure Layer  (_process_departures / _parked_vehicles)
      Every successfully assigned vehicle is registered in _parked_vehicles
      with a scheduled depart_time.  advance_time() calls
      _process_departures() which frees slots and clears reservations.
      reset() pre-parks 0–INITIAL_OCCUPANCY_MAX vehicles so the agent
      starts with a partially-filled lot.

Observation (21-dim)  ← STATE_DIM unchanged
--------------------
[0  :8 ]  slot_statuses           — 0.0 free / 1.0 taken
[8  :17]  active_vehicle_features — 3 × (cur_node, nxt_node, eta)
[17 :20]  bottleneck_loads        — reservation density per bottleneck node
[20]      inter_arrival_time      — normalised gap to current vehicle

Reservation table schema
------------------------
self._reservations[node] = list of dicts:
  {
    "vehicle_id": int,
    "kind":       "entering" | "parked",
    "start":      float,   # wall-clock entry time  (with safety margin applied in check)
    "end":        float,   # wall-clock exit  time
  }

The conflict check uses start/end directly; kind is metadata only.
_remove_reservations_for_vehicle(vid) deletes every entry for that vehicle
from all nodes so departed vehicles leave no ghost reservations.

Episode termination
-------------------
  step_count >= MAX_STEPS (primary)
  deadlock: all slots TAKEN AND _parked_vehicles is empty (no future release)

Fast-forward (all-slots-full prevention)
-----------------------------------------
action_masks() calls _ensure_free_slot() which, when all slots are taken,
advances simulation time to the next scheduled departure so at least one
slot becomes available before the mask is returned to the policy.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import compute_reward

# ─── Layout Constants ─────────────────────────────────────────────────────────
NUM_SLOTS: int = 8
MAX_ACTIVE_VEHICLES: int = 3
NUM_BOTTLENECK_FEATURES: int = 3
STATE_DIM: int = (
    NUM_SLOTS + MAX_ACTIVE_VEHICLES * 3 + NUM_BOTTLENECK_FEATURES + 1
)  # 21  — must not change

SLOT_NAMES: list[str] = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
SLOT_INDEX: dict[str, int] = {name: i for i, name in enumerate(SLOT_NAMES)}

SLOT_COORDINATES: dict[str, tuple[float, float]] = {
    "A1": (425.0, 1050.0), "A2": (650.0, 1050.0),
    "A3": (875.0, 1050.0), "A4": (1100.0, 1050.0),
    "B1": (425.0,  150.0), "B2": (650.0,  150.0),
    "B3": (875.0,  150.0), "B4": (1100.0,  150.0),
}
ENTRY_POINT: tuple[float, float] = (150.0, 0.0)
MAX_DISTANCE: float = math.hypot(1200.0, 1200.0)

# ─── Waypoint Graph ───────────────────────────────────────────────────────────
SLOT_ROUTES: dict[str, list[str]] = {
    "A1": ["entrance", "A_corridor", "A_lane", "A1_front"],
    "A2": ["entrance", "A_corridor", "A_lane", "A2_front"],
    "A3": ["entrance", "A_corridor", "A_lane", "A3_front"],
    "A4": ["entrance", "A_corridor", "A_lane", "A4_front"],
    "B1": ["entrance", "B_corridor", "B_lane", "B1_front"],
    "B2": ["entrance", "B_corridor", "B_lane", "B2_front"],
    "B3": ["entrance", "B_corridor", "B_lane", "B3_front"],
    "B4": ["entrance", "B_corridor", "B_lane", "B4_front"],
}
BOTTLENECK_NODES: list[str] = ["entrance", "A_corridor", "B_corridor"]

_ALL_NODES: list[str] = sorted(
    {node for route in SLOT_ROUTES.values() for node in route}
)
NODE_INDEX: dict[str, int] = {node: i for i, node in enumerate(_ALL_NODES)}

# Node physical coordinates (mm).  Single source of truth for traffic simulator.
NODE_COORDINATES: dict[str, tuple[float, float]] = {
    "entrance":   (150.0,   20.0),
    "A_corridor": (150.0,  720.0),
    "A_lane":     (300.0, 1050.0),
    "A1_front":   (425.0, 1050.0),
    "A2_front":   (650.0, 1050.0),
    "A3_front":   (875.0, 1050.0),
    "A4_front":  (1100.0, 1050.0),
    "B_corridor": (150.0,  350.0),
    "B_lane":     (300.0,  150.0),
    "B1_front":   (425.0,  150.0),
    "B2_front":   (650.0,  150.0),
    "B3_front":   (875.0,  150.0),
    "B4_front":  (1100.0,  150.0),
}

_MAX_ROUTE_LEN: int = max(len(r) for r in SLOT_ROUTES.values())  # 4

# ─── Physics Constants ────────────────────────────────────────────────────────
NODE_TRAVEL_TIME: float = 1.0   # seconds a vehicle occupies one waypoint
SAFETY_MARGIN: float    = 0.5   # padding around reserved intervals

# ─── Arrival Model ────────────────────────────────────────────────────────────
ARRIVAL_MIN: float = 0.8   # minimum inter-arrival time (s)
ARRIVAL_MAX: float = 2.2   # maximum inter-arrival time (s)

# ─── Departure / Parking Model ────────────────────────────────────────────────
PARK_DURATION_MIN: float   =  8.0   # min time a vehicle stays parked (s)
PARK_DURATION_MAX: float   = 20.0   # max time a vehicle stays parked (s)
INITIAL_OCCUPANCY_MAX: int =    4   # max pre-parked vehicles at episode start

# ─── Episode Constants ────────────────────────────────────────────────────────
MAX_STEPS: int = 64   # increased from 16 to allow departure/reuse cycles

# ─── Slot Status ─────────────────────────────────────────────────────────────
STATUS_FREE:  float = 0.0
STATUS_TAKEN: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
class ParkingRoutingEnv(gym.Env):
    """Multi-step parking slot allocation + traffic + departure environment.

    RL interface (Gymnasium-compatible)
    ------------------------------------
    observation_space : Box(0, 1, shape=(21,), dtype=float32)  ← unchanged
    action_space      : Discrete(8)                             ← unchanged
    action_masks()    : bool ndarray(8,)  for MaskablePPO

    Traffic simulator interface
    ---------------------------
    advance_time(dt)  : advance clock, update physics, process departures
    active_vehicles   : list of moving vehicle dicts (full kinematic state)

    Departure interface
    -------------------
    _parked_vehicles  : [{id, slot, slot_idx, depart_time, status}]
    _departure_count  : cumulative departures this episode
    _fast_forward_count : times action_masks() fast-forwarded clock

    Moving vehicle dict schema
    ----------------------------
    id                 : int
    slot               : str
    route              : list[str]
    route_index        : int
    route_intervals    : list[(t_enter, t_exit)]
    segment_start_time : float
    segment_end_time   : float
    current_node       : str
    next_node          : str
    current_position   : (float, float)
    remaining_eta      : float
    status             : "moving"
    enter_time         : float
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_SLOTS)
        self._init_state()

    # ─── State Initialisation ────────────────────────────────────────────────

    def _init_state(self) -> None:
        """Zero-out all mutable episode state."""
        self._slot_statuses: np.ndarray = np.zeros(NUM_SLOTS, dtype=np.float32)
        # Moving vehicles (traffic simulator layer)
        self._active_vehicles: list[dict[str, Any]] = []
        # Parked vehicles awaiting departure (departure layer)
        self._parked_vehicles: list[dict[str, Any]] = []
        # Reservation table: node → list of reservation dicts
        self._reservations: dict[str, list[dict[str, Any]]] = {
            node: [] for node in _ALL_NODES
        }
        self._current_time: float = 0.0
        self._inter_arrival_time: float = (ARRIVAL_MIN + ARRIVAL_MAX) / 2.0
        self._step_count: int = 0
        self._terminated: bool = False
        self._vehicle_id_counter: int = 0
        # Diagnostics
        self.n_fallback_triggers: int = 0
        self._departure_count: int = 0
        self._fast_forward_count: int = 0

    # ─── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a fresh episode with optional pre-parked vehicles."""
        super().reset(seed=seed)
        self._init_state()
        self._inter_arrival_time = float(
            self.np_random.uniform(ARRIVAL_MIN, ARRIVAL_MAX)
        )

        # ── Pre-park 0–INITIAL_OCCUPANCY_MAX vehicles ─────────────────────────
        n_pre = int(self.np_random.integers(0, INITIAL_OCCUPANCY_MAX + 1))
        if n_pre > 0:
            chosen = self.np_random.choice(NUM_SLOTS, size=n_pre, replace=False)
            for idx in chosen.tolist():
                self._vehicle_id_counter += 1
                vid  = self._vehicle_id_counter
                slot = SLOT_NAMES[idx]

                depart_time = float(
                    self.np_random.uniform(PARK_DURATION_MIN, PARK_DURATION_MAX)
                )

                # Mark slot taken
                self._slot_statuses[idx] = STATUS_TAKEN

                # Register parked reservation on the slot's front node
                slot_node = SLOT_ROUTES[slot][-1]
                self._reservations[slot_node].append({
                    "vehicle_id": vid,
                    "kind":       "parked",
                    "start":      0.0,
                    "end":        depart_time,
                })

                # Register in departure list
                self._parked_vehicles.append({
                    "id":          vid,
                    "slot":        slot,
                    "slot_idx":    idx,
                    "depart_time": depart_time,
                    "status":      "parked",
                })

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Assign one incoming vehicle to a slot.

        Sequence
        --------
        1. Guard checks.
        2. If slot already taken → penalise and advance time.
        3. Build route intervals; check Safety Shield.
        4. Compute reward BEFORE state mutation.
        5a. Conflict → penalise; do NOT register or spawn.
        5b. No conflict → register reservation, spawn vehicle, schedule parking.
        6. Sample IAT, advance clock (physics + departures).
        7. Check termination; pack info.
        """
        if self._terminated:
            raise RuntimeError("Episode is over — call reset() first.")
        if not (0 <= action < NUM_SLOTS):
            raise ValueError(f"Action {action} out of range [0, {NUM_SLOTS}).")

        slot_name = SLOT_NAMES[action]
        info: dict[str, Any] = {
            "slot":               slot_name,
            "conflict":           False,
            "step":               self._step_count,
            "inter_arrival_time": self._inter_arrival_time,
        }

        # ── already taken ─────────────────────────────────────────────────────
        if self._slot_statuses[action] >= STATUS_TAKEN:
            iat = self._sample_iat()
            self._step_count += 1
            self.advance_time(iat)
            info["reason"] = "slot_already_taken"
            terminated = self._check_done()
            self._terminated = terminated
            info.update(self._build_info_metrics())
            return self._get_obs(), -10.0, terminated, False, info

        # ── build intervals & conflict check ──────────────────────────────────
        route    = SLOT_ROUTES[slot_name]
        t_arrive = self._current_time
        intervals = self._build_intervals(route, t_arrive)
        conflict  = self._check_conflict(route, intervals)

        # ── reward BEFORE state mutation ──────────────────────────────────────
        reward = compute_reward(
            slot_name=slot_name,
            route=route,
            conflict=conflict,
            slot_coordinates=SLOT_COORDINATES,
            entry_point=ENTRY_POINT,
            max_distance=MAX_DISTANCE,
            bottleneck_nodes=BOTTLENECK_NODES,
            reservations=self._reservations,
        )

        if conflict:
            info["conflict"] = True
        else:
            # ── commit ────────────────────────────────────────────────────────
            self._vehicle_id_counter += 1
            vid = self._vehicle_id_counter

            # Register entering reservations (all route nodes)
            self._register_reservation(route, intervals, vid, kind="entering")

            # Mark slot taken
            self._slot_statuses[action] = STATUS_TAKEN

            # Schedule parking + departure
            parking_complete_time = intervals[-1][1]
            depart_time = float(
                self.np_random.uniform(
                    parking_complete_time + PARK_DURATION_MIN,
                    parking_complete_time + PARK_DURATION_MAX,
                )
            )
            slot_node = route[-1]   # e.g. "A1_front"
            self._reservations[slot_node].append({
                "vehicle_id": vid,
                "kind":       "parked",
                "start":      parking_complete_time,
                "end":        depart_time,
            })
            self._parked_vehicles.append({
                "id":          vid,
                "slot":        slot_name,
                "slot_idx":    action,
                "depart_time": depart_time,
                "status":      "parked",
            })

            # Spawn moving vehicle (physics layer)
            self._spawn_vehicle(slot_name, route, intervals, t_arrive, vid)

        # ── advance clock ─────────────────────────────────────────────────────
        iat = self._sample_iat()
        self._step_count += 1
        self.advance_time(iat)

        terminated = self._check_done()
        self._terminated = terminated
        info.update(self._build_info_metrics())
        return self._get_obs(), reward, terminated, False, info

    # ─── Action Masking ───────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return bool mask: True = slot selectable this step.

        Side-effect: if all slots are currently TAKEN, fast-forwards the
        simulation clock to the next scheduled departure before computing
        masks (prevents all-False deadlock in MaskablePPO).

        Fallback: if all free slots would conflict, opens the nearest free
        slot (conflict penalty still applies in step()).
        """
        if not self._terminated:
            self._ensure_free_slot()

        masks = np.ones(NUM_SLOTS, dtype=bool)
        for i, name in enumerate(SLOT_NAMES):
            if self._slot_statuses[i] >= STATUS_TAKEN:
                masks[i] = False
            elif self._quick_conflict_check(name):
                masks[i] = False

        if not masks.any() and not self._terminated:
            self.n_fallback_triggers += 1
            free = [
                i for i in range(NUM_SLOTS)
                if self._slot_statuses[i] < STATUS_TAKEN
            ]
            if free:
                ex, ey = ENTRY_POINT
                best = min(
                    free,
                    key=lambda i: math.hypot(
                        SLOT_COORDINATES[SLOT_NAMES[i]][0] - ex,
                        SLOT_COORDINATES[SLOT_NAMES[i]][1] - ey,
                    ),
                )
                masks[best] = True

        return masks

    # ─── Traffic Simulator Layer ──────────────────────────────────────────────

    def advance_time(self, dt: float) -> None:
        """Advance simulation clock by *dt* seconds.

        1. Advances _current_time.
        2. Updates every active vehicle's physics (position, node, ETA, status).
        3. Removes vehicles that have reached their destination from
           _active_vehicles (they remain in _parked_vehicles).
        4. Calls _process_departures() to free slots whose depart_time has
           elapsed and purge their reservations.
        """
        self._current_time += dt
        t = self._current_time

        still_moving: list[dict[str, Any]] = []
        for v in self._active_vehicles:
            route     = v["route"]
            intervals = v["route_intervals"]
            ri        = v["route_index"]

            # Advance segment index
            while ri < len(intervals) - 1 and t >= intervals[ri][1]:
                ri += 1
            v["route_index"] = ri

            # Update node labels
            v["current_node"]       = route[ri]
            v["next_node"]          = (
                route[ri + 1] if ri + 1 < len(route) else route[ri]
            )
            v["segment_start_time"] = intervals[ri][0]
            v["segment_end_time"]   = intervals[ri][1]

            # Interpolate physical position
            t_enter, t_exit = intervals[ri]
            frac = float(
                np.clip((t - t_enter) / max(t_exit - t_enter, 1e-9), 0.0, 1.0)
            )
            p1 = NODE_COORDINATES.get(v["current_node"], ENTRY_POINT)
            p2 = NODE_COORDINATES.get(v["next_node"],    p1)
            v["current_position"] = (
                p1[0] + frac * (p2[0] - p1[0]),
                p1[1] + frac * (p2[1] - p1[1]),
            )

            # ETA and status
            v["remaining_eta"] = max(0.0, intervals[-1][1] - t)

            if t < intervals[-1][1]:
                v["status"] = "moving"
                still_moving.append(v)
            else:
                v["status"] = "parked"
                # Vehicle is now physically parked; stays in _parked_vehicles
                # until depart_time is reached by _process_departures().

        self._active_vehicles = still_moving

        # Process scheduled departures
        self._process_departures()

    def _spawn_vehicle(
        self,
        slot:       str,
        route:      list[str],
        intervals:  list[tuple[float, float]],
        t_arrive:   float,
        vehicle_id: int,
    ) -> None:
        """Create a vehicle with full kinematic state and add to simulator."""
        p0 = NODE_COORDINATES.get(route[0], ENTRY_POINT)
        vehicle: dict[str, Any] = {
            "id":                 vehicle_id,
            "slot":               slot,
            "route":              route,
            "route_intervals":    intervals,
            "route_index":        0,
            "segment_start_time": intervals[0][0],
            "segment_end_time":   intervals[0][1],
            "current_node":       route[0],
            "next_node":          route[1] if len(route) > 1 else route[0],
            "current_position":   p0,
            "remaining_eta":      intervals[-1][1] - t_arrive,
            "status":             "moving",
            "enter_time":         t_arrive,
        }
        self._active_vehicles.append(vehicle)

    # ─── Departure Layer ──────────────────────────────────────────────────────

    def _process_departures(self) -> int:
        """Free slots whose depart_time ≤ current_time.

        For each departing vehicle:
          - Sets _slot_statuses[slot_idx] = STATUS_FREE.
          - Calls _remove_reservations_for_vehicle() to purge all of that
            vehicle's entries from every node's reservation list.
          - Increments _departure_count.

        Returns the number of vehicles that departed this call.
        """
        t = self._current_time
        still_parked: list[dict[str, Any]] = []
        n_departed = 0

        for pv in self._parked_vehicles:
            if t >= pv["depart_time"]:
                self._slot_statuses[pv["slot_idx"]] = STATUS_FREE
                self._remove_reservations_for_vehicle(pv["id"])
                self._departure_count += 1
                n_departed += 1
            else:
                still_parked.append(pv)

        self._parked_vehicles = still_parked
        return n_departed

    def _remove_reservations_for_vehicle(self, vehicle_id: int) -> None:
        """Remove every reservation entry belonging to *vehicle_id*.

        Called on departure to prevent ghost reservations from blocking
        future assignments to the freed slot.
        """
        for node in self._reservations:
            self._reservations[node] = [
                r for r in self._reservations[node]
                if r["vehicle_id"] != vehicle_id
            ]

    def _ensure_free_slot(self) -> None:
        """Guarantee at least one slot is free before action_masks() returns.

        If all slots are TAKEN:
          - If parked vehicles exist → fast-forward clock to earliest
            depart_time and call advance_time(dt) (which triggers
            _process_departures and frees ≥1 slot).
          - If no parked vehicles → deadlock; do nothing (step() will
            return terminated=True via _check_done()).

        Increments _fast_forward_count each time a fast-forward occurs.
        """
        if np.any(self._slot_statuses < STATUS_TAKEN):
            return   # at least one free slot already

        if not self._parked_vehicles:
            return   # deadlock — _check_done() will handle termination

        earliest = min(pv["depart_time"] for pv in self._parked_vehicles)
        dt_ff    = max(0.0, earliest - self._current_time)
        # advance_time(0) is a no-op on clock but still triggers departures
        self.advance_time(dt_ff)
        if dt_ff > 0.0:
            self._fast_forward_count += 1

    # ─── Safety Shield ────────────────────────────────────────────────────────

    def _build_intervals(
        self, route: list[str], start_time: float
    ) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        t = start_time
        for _ in route:
            intervals.append((t, t + NODE_TRAVEL_TIME))
            t += NODE_TRAVEL_TIME
        return intervals

    def _quick_conflict_check(self, slot_name: str) -> bool:
        route     = SLOT_ROUTES[slot_name]
        intervals = self._build_intervals(route, self._current_time)
        return self._check_conflict(route, intervals)

    def _check_conflict(
        self,
        route:     list[str],
        intervals: list[tuple[float, float]],
    ) -> bool:
        """Return True if any route node reservation overlaps (with safety margin)."""
        for node, (t_start, t_end) in zip(route, intervals):
            ps = t_start - SAFETY_MARGIN
            pe = t_end   + SAFETY_MARGIN
            for res in self._reservations.get(node, []):
                r_start = res["start"]
                r_end   = res["end"]
                if max(ps, r_start) < min(pe, r_end):
                    return True
        return False

    def _register_reservation(
        self,
        route:      list[str],
        intervals:  list[tuple[float, float]],
        vehicle_id: int,
        kind:       str = "entering",
    ) -> None:
        """Append one reservation dict per node on the given route."""
        for node, (t_start, t_end) in zip(route, intervals):
            self._reservations[node].append({
                "vehicle_id": vehicle_id,
                "kind":       kind,
                "start":      t_start,
                "end":        t_end,
            })

    # ─── Arrival Sampling ─────────────────────────────────────────────────────

    def _sample_iat(self) -> float:
        iat = float(self.np_random.uniform(ARRIVAL_MIN, ARRIVAL_MAX))
        self._inter_arrival_time = iat
        return iat

    # ─── Termination ─────────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        """Episode ends when step_count hits MAX_STEPS, or deadlock occurs.

        Deadlock = all slots TAKEN and no parked vehicles left to depart.
        Note: the all-slots-full condition alone no longer terminates the
        episode; _ensure_free_slot() will fast-forward to the next departure.
        """
        if self._step_count >= MAX_STEPS:
            return True
        # Deadlock: every slot occupied, no future releases
        if (
            bool(np.all(self._slot_statuses >= STATUS_TAKEN))
            and not self._parked_vehicles
        ):
            return True
        return False

    # ─── Info Metrics ─────────────────────────────────────────────────────────

    def _build_info_metrics(self) -> dict[str, Any]:
        return {
            "parked_vehicle_count": len(self._parked_vehicles),
            "departure_count":      self._departure_count,
            "free_slot_count":      int(np.sum(self._slot_statuses < STATUS_TAKEN)),
            "fast_forward_count":   self._fast_forward_count,
            "current_time":         self._current_time,
        }

    # ─── Observation Builder ──────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build normalised 21-dim observation vector.

        STATE_DIM = 21 is fixed; no departure ETA is exposed to the agent.
        The agent infers slot availability from obs[0:8] (slot_statuses),
        which automatically reflects departures via STATUS_FREE.
        """
        obs = np.zeros(STATE_DIM, dtype=np.float32)

        # [0:8] slot statuses
        obs[:NUM_SLOTS] = self._slot_statuses

        # [8:17] active vehicle features (up to MAX_ACTIVE_VEHICLES)
        num_nodes = max(len(_ALL_NODES), 1)
        max_eta   = _MAX_ROUTE_LEN * NODE_TRAVEL_TIME   # 4.0 s

        for i, v in enumerate(self._active_vehicles[:MAX_ACTIVE_VEHICLES]):
            base = NUM_SLOTS + i * 3
            cur  = NODE_INDEX.get(v["current_node"], 0)
            nxt  = NODE_INDEX.get(v["next_node"],    0)
            eta  = v["remaining_eta"]
            obs[base]     = cur / num_nodes
            obs[base + 1] = nxt / num_nodes
            obs[base + 2] = float(np.clip(eta / max_eta, 0.0, 1.0))

        # [17:20] bottleneck reservation density
        base = NUM_SLOTS + MAX_ACTIVE_VEHICLES * 3
        for j, node in enumerate(BOTTLENECK_NODES):
            count = len(self._reservations.get(node, []))
            obs[base + j] = min(count / max(MAX_ACTIVE_VEHICLES, 1), 1.0)

        # [20] inter_arrival_time (normalised)
        obs[STATE_DIM - 1] = (self._inter_arrival_time - ARRIVAL_MIN) / (
            ARRIVAL_MAX - ARRIVAL_MIN
        )
        return obs
