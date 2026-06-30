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

# Slot physical positions (mm).  Match the *_front node coordinates exactly.
#
# Distance convention: A4 / B4 are CLOSEST to the entrance, A1 / B1 are the
# farthest.  The numbering within each row goes from far→near (1=far, 4=near).
# Slots are reached via the central lane; route length is proportional to
# distance from the junction:
#
#     A4 = B4   (4 nodes / 4.0 s,  x=425)   ← efficiency reward highest
#     A3 = B3   (5 nodes / 5.0 s,  x=650)
#     A2 = B2   (6 nodes / 6.0 s,  x=875)
#     A1 = B1   (7 nodes / 7.0 s,  x=1100)  ← efficiency reward lowest
SLOT_COORDINATES: dict[str, tuple[float, float]] = {
    "A1": (1100.0, 1050.0), "A2": (875.0, 1050.0),
    "A3": ( 650.0, 1050.0), "A4": (425.0, 1050.0),
    "B1": (1100.0,  150.0), "B2": (875.0,  150.0),
    "B3": ( 650.0,  150.0), "B4": (425.0,  150.0),
}
# ENTRY_POINT = efficiency-reward reference point.
# Set to the junction position (where slot-choice differentiation begins,
# AFTER the common entrance→junction segment).  This makes A and B rows
# symmetric in distance — A4 = B4, A3 = B3, etc.
# The physical entrance node lives at (150, 100); the agent is rewarded by
# proximity to the central-lane junction, not the raw entry point.
ENTRY_POINT: tuple[float, float] = (150.0, 600.0)
MAX_DISTANCE: float = math.hypot(1200.0, 1200.0)

# ─── Waypoint Graph ───────────────────────────────────────────────────────────
#
# Physical layout (matches the real lot):
#
#   exit (150, 1200)            ─────── A row (y=1050) ───────
#         ▲                     A4    A3    A2    A1
#         │                     ●     ●     ●     ●          (slot fronts)
#         │                     │     │     │     │
#         │                     │     │     │     │
#         └─── junction ────────●─────●─────●─────●           central lane (y=600)
#              (150, 600)      lane  lane  lane  lane
#                              pt_4  pt_3  pt_2  pt_1
#                               │     │     │     │
#                               │     │     │     │
#                               ●     ●     ●     ●          (slot fronts)
#         ▲                     B4    B3    B2    B1
#         │                     ─────── B row (y=150) ───────
#   entrance (150, 100)
#
# Entry path (enter): entrance → junction → central lane (RIGHT) → slot_front
# Exit  path (exit) : slot_front → central lane (LEFT) → junction → exit
#
# Route length varies by slot column:
#   A4 / B4 (closest):  4 nodes / 4.0 s travel
#   A3 / B3:            5 nodes / 5.0 s travel
#   A2 / B2:            6 nodes / 6.0 s travel
#   A1 / B1 (farthest): 7 nodes / 7.0 s travel
#
# All slots share entrance, junction, and lane_pt_4.  Farther slots
# additionally share lane_pt_3 / lane_pt_2 / lane_pt_1.  This creates the
# bidirectional contention modelled by the Safety Shield (entering and
# exiting vehicles compete for the central lane points).
SLOT_ROUTES: dict[str, list[str]] = {
    "A1": ["entrance", "junction", "lane_pt_4", "lane_pt_3", "lane_pt_2", "lane_pt_1", "A1_front"],
    "A2": ["entrance", "junction", "lane_pt_4", "lane_pt_3", "lane_pt_2",              "A2_front"],
    "A3": ["entrance", "junction", "lane_pt_4", "lane_pt_3",                           "A3_front"],
    "A4": ["entrance", "junction", "lane_pt_4",                                        "A4_front"],
    "B1": ["entrance", "junction", "lane_pt_4", "lane_pt_3", "lane_pt_2", "lane_pt_1", "B1_front"],
    "B2": ["entrance", "junction", "lane_pt_4", "lane_pt_3", "lane_pt_2",              "B2_front"],
    "B3": ["entrance", "junction", "lane_pt_4", "lane_pt_3",                           "B3_front"],
    "B4": ["entrance", "junction", "lane_pt_4",                                        "B4_front"],
}

# Bottlenecks: the three highest-traffic nodes (every vehicle's entering AND
# exiting path passes through these).  STATE_DIM=21 reserves 3 obs slots
# (obs[17:20]) so we expose the 3 most contended nodes to the agent.
BOTTLENECK_NODES: list[str] = ["junction", "lane_pt_4", "lane_pt_3"]

# Exit routes: physically separate path — slot_front → back along central
# lane (LEFT) → junction → up vertical lane → exit at top-left.
# NOT just reversed(SLOT_ROUTES) because the exit endpoint ("exit" node) is
# physically distinct from the entry endpoint ("entrance" node).
EXIT_ROUTES: dict[str, list[str]] = {
    "A1": ["A1_front", "lane_pt_1", "lane_pt_2", "lane_pt_3", "lane_pt_4", "junction", "exit"],
    "A2": ["A2_front",              "lane_pt_2", "lane_pt_3", "lane_pt_4", "junction", "exit"],
    "A3": ["A3_front",                           "lane_pt_3", "lane_pt_4", "junction", "exit"],
    "A4": ["A4_front",                                        "lane_pt_4", "junction", "exit"],
    "B1": ["B1_front", "lane_pt_1", "lane_pt_2", "lane_pt_3", "lane_pt_4", "junction", "exit"],
    "B2": ["B2_front",              "lane_pt_2", "lane_pt_3", "lane_pt_4", "junction", "exit"],
    "B3": ["B3_front",                           "lane_pt_3", "lane_pt_4", "junction", "exit"],
    "B4": ["B4_front",                                        "lane_pt_4", "junction", "exit"],
}

# All graph nodes — must include both entering and exiting waypoints because
# the "exit" node only appears in EXIT_ROUTES (not in SLOT_ROUTES) but
# reservations are registered on it for exiting vehicles.
_ALL_NODES: list[str] = sorted(
    {node for route in SLOT_ROUTES.values() for node in route}
    | {node for route in EXIT_ROUTES.values() for node in route}
)
NODE_INDEX: dict[str, int] = {node: i for i, node in enumerate(_ALL_NODES)}

# Node physical coordinates (mm).  Single source of truth for traffic simulator.
#
# Layout:
#   - Left vertical lane: entrance (y=100) → junction (y=600) → exit (y=1200)
#   - Central horizontal lane at y=600: junction → lane_pt_4 → … → lane_pt_1
#   - A row at y=1050 (above central lane), B row at y=150 (below)
#   - A4 / B4 sit at the leftmost slot column (closest), A1 / B1 at the
#     rightmost column (farthest from junction along the lane).
NODE_COORDINATES: dict[str, tuple[float, float]] = {
    # Left vertical lane
    "entrance":  ( 150.0,  100.0),
    "junction":  ( 150.0,  600.0),
    "exit":      ( 150.0, 1200.0),
    # Central horizontal lane (y=600), shared by all entering and exiting traffic
    "lane_pt_4": ( 425.0,  600.0),
    "lane_pt_3": ( 650.0,  600.0),
    "lane_pt_2": ( 875.0,  600.0),
    "lane_pt_1": (1100.0,  600.0),
    # A row (top, y=1050)
    "A1_front":  (1100.0, 1050.0),
    "A2_front":  ( 875.0, 1050.0),
    "A3_front":  ( 650.0, 1050.0),
    "A4_front":  ( 425.0, 1050.0),
    # B row (bottom, y=150)
    "B1_front":  (1100.0,  150.0),
    "B2_front":  ( 875.0,  150.0),
    "B3_front":  ( 650.0,  150.0),
    "B4_front":  ( 425.0,  150.0),
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

# ─── WAIT Action ─────────────────────────────────────────────────────────────
WAIT_ACTION: int              = NUM_SLOTS   # action index 8  (0-7 = slots)
WAIT_TIME: float              = 1.0         # seconds to advance on WAIT
WAIT_PENALTY_BASE: float      = -0.2        # penalty for 1st consecutive wait
WAIT_PENALTY_INCREMENT: float =  0.1        # escalation per consecutive wait
MAX_CONSECUTIVE_WAITS: int    = 5           # WAIT masked after this many in a row


# ─────────────────────────────────────────────────────────────────────────────
class ParkingRoutingEnv(gym.Env):
    """Multi-step parking slot allocation + traffic + departure environment.

    RL interface (Gymnasium-compatible)
    ------------------------------------
    observation_space : Box(0, 1, shape=(21,), dtype=float32)  ← STATE_DIM=21 fixed
    action_space      : Discrete(9)   — 0-7 slot assignment, 8=WAIT
    action_masks()    : bool ndarray(9,)  for MaskablePPO

    WAIT action (index 8)
    ---------------------
    Holds the pending vehicle at the entrance and advances the simulation
    clock by WAIT_TIME=1.0 s.  No reservation is registered; physics and
    departures proceed normally.  This lets the agent wait for an in-flight
    vehicle to clear a shared waypoint before committing to a slot.

    Penalty escalates with consecutive waits to prevent policy collapse:
        penalty = WAIT_PENALTY_BASE − WAIT_PENALTY_INCREMENT × consecutive_waits
    WAIT is masked (action_masks()[8] = False) after MAX_CONSECUTIVE_WAITS=5
    consecutive WAIT actions to guarantee forward progress.

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

    def __init__(
        self,
        *,
        wait_penalty_base:       float | None = None,
        wait_penalty_increment:  float | None = None,
        wait_time:               float | None = None,
        max_consecutive_waits:   int   | None = None,
        assignment_bonus:        float = 0.0,
    ) -> None:
        """Construct env with optional WAIT-policy hyperparameter overrides.

        Parameters
        ----------
        wait_penalty_base       : Base penalty for the 1st consecutive WAIT.
                                  Defaults to module constant WAIT_PENALTY_BASE.
        wait_penalty_increment  : Additive penalty per extra consecutive WAIT.
                                  Defaults to WAIT_PENALTY_INCREMENT.
        wait_time               : Seconds to advance on each WAIT action.
                                  Defaults to WAIT_TIME (1.0 s).
        max_consecutive_waits   : Cap on consecutive WAITs before masking.
                                  Defaults to MAX_CONSECUTIVE_WAITS (5).
        assignment_bonus        : Additional reward added on a successful
                                  (non-conflict) slot assignment.  0.0 = no
                                  throughput bonus (default).  Useful for
                                  sweeping the conflict-vs-throughput tradeoff.
        """
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        # action 0-7 = slot assignment, action 8 = WAIT
        self.action_space = spaces.Discrete(NUM_SLOTS + 1)

        # WAIT-policy hyperparameters (instance-level for sweep experiments)
        self.wait_penalty_base      = (
            WAIT_PENALTY_BASE      if wait_penalty_base      is None else float(wait_penalty_base)
        )
        self.wait_penalty_increment = (
            WAIT_PENALTY_INCREMENT if wait_penalty_increment is None else float(wait_penalty_increment)
        )
        self.wait_time              = (
            WAIT_TIME              if wait_time              is None else float(wait_time)
        )
        self.max_consecutive_waits  = (
            MAX_CONSECUTIVE_WAITS  if max_consecutive_waits  is None else int(max_consecutive_waits)
        )
        self.assignment_bonus       = float(assignment_bonus)

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
        # WAIT-action state
        self._consecutive_waits: int = 0   # reset on each slot assignment
        self._wait_count: int = 0          # total WAITs this episode
        # Flow tracking — counts how many times each slot has been assigned
        # this episode.  ≥1 means a departure happened and the slot is cycling.
        self._slot_use_counts: list[int] = [0] * NUM_SLOTS
        # Exiting vehicle log: accumulated per-episode for visualisation.
        # Each entry: {id, slot, slot_idx, route, route_intervals, enter_time}
        self._exiting_log: list[dict[str, Any]] = []

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
        """Execute one central-control decision tick.

        action 0-7 : Assign the pending vehicle to the chosen slot.
        action 8   : WAIT — hold the vehicle; advance clock by WAIT_TIME.

        Sequence (slot action)
        ----------------------
        1. Guard checks.
        2. If slot already taken → penalise and advance time.
        3. Build route intervals; check Safety Shield.
        4. Compute reward BEFORE state mutation.
        5a. Conflict → penalise; do NOT register or spawn.
        5b. No conflict → register reservation, spawn vehicle, schedule parking.
        6. Sample IAT, advance clock (physics + departures).
        7. Check termination; pack info.

        Sequence (WAIT action)
        ----------------------
        1. Compute escalating penalty: WAIT_PENALTY_BASE - WAIT_PENALTY_INCREMENT
           × consecutive_waits.
        2. Increment consecutive_waits and wait_count.
        3. advance_time(WAIT_TIME) — physics + departures proceed normally.
        4. No new IAT sampled; same vehicle remains pending.
        """
        if self._terminated:
            raise RuntimeError("Episode is over — call reset() first.")
        if not (0 <= action < NUM_SLOTS + 1):
            raise ValueError(f"Action {action} out of range [0, {NUM_SLOTS}].")

        # ── WAIT action ───────────────────────────────────────────────────────
        if action == WAIT_ACTION:
            penalty = (
                self.wait_penalty_base
                - self.wait_penalty_increment * self._consecutive_waits
            )
            self._consecutive_waits += 1
            self._wait_count += 1
            self._step_count += 1
            self.advance_time(self.wait_time)
            terminated = self._check_done()
            self._terminated = terminated
            info: dict[str, Any] = {
                "slot":               None,
                "conflict":           False,
                "wait":               True,
                "step":               self._step_count,
                "inter_arrival_time": self._inter_arrival_time,
            }
            info.update(self._build_info_metrics())
            return self._get_obs(), float(penalty), terminated, False, info

        # ── slot action (0-7) ─────────────────────────────────────────────────
        # Any slot action resets the consecutive-wait counter.
        self._consecutive_waits = 0

        slot_name = SLOT_NAMES[action]
        info = {
            "slot":               slot_name,
            "conflict":           False,
            "wait":               False,
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
            # flow context (v2)
            slot_use_count=self._slot_use_counts[action],
            free_slot_count=int(np.sum(self._slot_statuses < STATUS_TAKEN)),
            num_slots=NUM_SLOTS,
        )

        if conflict:
            info["conflict"] = True
        else:
            # Throughput bonus on successful (non-conflict) assignment.
            # 0.0 = disabled (default); used for sweep experiments to
            # incentivise the agent to assign rather than WAIT.
            if self.assignment_bonus != 0.0:
                reward += self.assignment_bonus
            # Track slot use count for flow reward (v2)
            self._slot_use_counts[action] += 1
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
        """Return bool mask of shape (9,): True = action selectable this step.

        Indices 0-7  slot assignment: True when slot is free AND no Safety-
                     Shield conflict at current time.
        Index   8    WAIT:            True when consecutive_waits <
                     MAX_CONSECUTIVE_WAITS.

        Masking policy
        --------------
        1. Compute slot masks (0-7).
        2. Set WAIT mask (8).
        3. Normal case — at least one action valid → return masks.
        4. Safety net (WAIT disabled AND all slots conflict/taken):
           a. _ensure_free_slot() fast-forwards to the next departure.
           b. Recompute slot masks.
           c. If still all False → forced-fallback to nearest free slot
              (conflict penalty still applies in step()).

        The forced fallback is now only reached when both WAIT is exhausted
        (MAX_CONSECUTIVE_WAITS exceeded) and every free slot has a conflict —
        a rare edge case that prevents MaskablePPO from seeing an all-False mask.
        """
        # ── slot masks ────────────────────────────────────────────────────────
        masks = np.zeros(NUM_SLOTS + 1, dtype=bool)
        for i, name in enumerate(SLOT_NAMES):
            if (self._slot_statuses[i] < STATUS_TAKEN
                    and not self._quick_conflict_check(name)):
                masks[i] = True

        # ── WAIT mask ─────────────────────────────────────────────────────────
        if not self._terminated:
            masks[WAIT_ACTION] = (
                self._consecutive_waits < self.max_consecutive_waits
            )

        # ── safety net: all 9 actions False ───────────────────────────────────
        if not masks.any() and not self._terminated:
            # WAIT is disabled; ensure at least one slot is physically free
            self._ensure_free_slot()
            for i, name in enumerate(SLOT_NAMES):
                if (self._slot_statuses[i] < STATUS_TAKEN
                        and not self._quick_conflict_check(name)):
                    masks[i] = True
            # Still nothing valid → forced fallback to nearest free slot
            if not masks.any():
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
                # Still in transit — preserve "exiting" status, mark others "moving"
                if v["status"] != "exiting":
                    v["status"] = "moving"
                still_moving.append(v)
            else:
                if v["status"] == "exiting":
                    # Physical exit complete: purge exit-route reservations
                    self._remove_reservations_for_vehicle(v["id"])
                    # Not added to still_moving → removed from active_vehicles
                else:
                    v["status"] = "parked"
                    # Entering vehicle physically parked; _parked_vehicles
                    # handles the scheduled departure.

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
        """Initiate physical exit for vehicles whose depart_time has elapsed.

        For each vehicle whose depart_time has elapsed:
          1. Build the exit route intervals starting at the current time.
          2. Run the Safety Shield against the exit route, *excluding* this
             vehicle's own (still-active) parked reservation.  If the lane is
             busy (e.g. an entering vehicle is currently transiting junction
             or lane_pt_4), defer the departure — the vehicle stays parked
             and we retry on the next advance_time tick.  This mirrors real
             world behaviour: a car cannot pull out of its slot until the
             central lane is clear.
          3. If the exit route is clear:
               - Set _slot_statuses[slot_idx] = STATUS_FREE
               - Remove the parked-phase reservation
               - _spawn_exiting_vehicle() builds the physical exit movement
                 and registers exit-route reservations (kind="exiting")
               - Increment _departure_count

        departure_count is incremented at exit *start* (not completion).

        Returns the number of exits initiated this call.
        """
        t = self._current_time
        still_parked: list[dict[str, Any]] = []
        n_departed = 0

        for pv in self._parked_vehicles:
            if t < pv["depart_time"]:
                still_parked.append(pv)
                continue

            # depart_time reached — Safety-Shield-check the exit route.
            exit_route = EXIT_ROUTES[pv["slot"]]
            intervals  = self._build_intervals(exit_route, t)

            if self._check_conflict_excluding(exit_route, intervals, pv["id"]):
                # Central lane is currently occupied → defer this departure.
                # Slot stays TAKEN; we will retry on the next tick.
                still_parked.append(pv)
                continue

            # Exit route is clear — proceed.
            self._slot_statuses[pv["slot_idx"]] = STATUS_FREE
            self._remove_reservations_for_vehicle(pv["id"])
            self._spawn_exiting_vehicle(pv)
            self._departure_count += 1
            n_departed += 1

        self._parked_vehicles = still_parked
        return n_departed

    def _check_conflict_excluding(
        self,
        route:      list[str],
        intervals:  list[tuple[float, float]],
        vehicle_id: int,
    ) -> bool:
        """Conflict check that ignores reservations belonging to *vehicle_id*.

        Used by `_process_departures()` so the departing vehicle's own
        parked-phase reservation on the slot front node does not count as
        a self-conflict when probing the exit route.
        """
        for node, (t_start, t_end) in zip(route, intervals):
            ps = t_start - SAFETY_MARGIN
            pe = t_end   + SAFETY_MARGIN
            for res in self._reservations.get(node, []):
                if res["vehicle_id"] == vehicle_id:
                    continue
                if max(ps, res["start"]) < min(pe, res["end"]):
                    return True
        return False

    def _spawn_exiting_vehicle(self, parked_vehicle: dict[str, Any]) -> None:
        """Create a physically-moving exit vehicle from a parked vehicle record.

        Exit route = reversed entry route (slot_front → … → entrance).
        Exit-route reservations are registered with kind="exiting" so the
        Safety Shield detects conflicts between entering and exiting traffic.
        The vehicle is added to _active_vehicles with status="exiting".
        """
        t_start   = self._current_time
        slot      = parked_vehicle["slot"]
        exit_route = EXIT_ROUTES[slot]
        intervals  = self._build_intervals(exit_route, t_start)

        # Register exit-route reservations (Safety Shield sees these)
        self._register_reservation(
            exit_route, intervals, parked_vehicle["id"], kind="exiting"
        )

        p0 = NODE_COORDINATES.get(exit_route[0], ENTRY_POINT)
        vehicle: dict[str, Any] = {
            "id":                 parked_vehicle["id"],
            "slot":               slot,
            "slot_idx":           parked_vehicle["slot_idx"],
            "route":              exit_route,
            "route_intervals":    intervals,
            "route_index":        0,
            "segment_start_time": intervals[0][0],
            "segment_end_time":   intervals[0][1],
            "current_node":       exit_route[0],
            "next_node":          exit_route[1] if len(exit_route) > 1 else exit_route[0],
            "current_position":   p0,
            "remaining_eta":      intervals[-1][1] - t_start,
            "status":             "exiting",
            "enter_time":         t_start,
        }
        self._active_vehicles.append(vehicle)

        # Append to episode-level log for visualisation
        self._exiting_log.append({
            "id":             parked_vehicle["id"],
            "slot":           slot,
            "slot_idx":       parked_vehicle["slot_idx"],
            "route":          exit_route,
            "route_intervals": intervals,
            "enter_time":     t_start,
        })

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
            "wait_count":           self._wait_count,
            "consecutive_waits":    self._consecutive_waits,
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

        # Only entering (moving) vehicles contribute to the obs.
        # Exiting vehicles are hidden dynamics — exposing their ETA would
        # give the agent an oracle on departure timing.
        entering_only = [v for v in self._active_vehicles if v["status"] == "moving"]
        for i, v in enumerate(entering_only[:MAX_ACTIVE_VEHICLES]):
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
