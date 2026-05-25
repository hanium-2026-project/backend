"""Gymnasium-compatible parking slot allocation RL environment.

The RL agent performs high-level slot assignment only (Discrete(8) action).
Actual collision avoidance is delegated to a waypoint-based Safety Shield
that checks time-interval overlaps on a shared reservation table.

Episode structure
-----------------
Multi-step: one episode = one full parking day / simulation run.
Each step assigns one incoming vehicle to a slot.
Terminated when all slots are taken OR max_steps is reached.
State (slot_statuses, reservations, active_vehicles) persists across steps
and is only reset in reset().

Reward timing
-------------
compute_reward() is called BEFORE _register_reservation() so the congestion
term reflects *prior* bottleneck load, not including the vehicle being assigned.
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
STATE_DIM: int = NUM_SLOTS + MAX_ACTIVE_VEHICLES * 3 + NUM_BOTTLENECK_FEATURES  # 20

SLOT_NAMES: list[str] = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
SLOT_INDEX: dict[str, int] = {name: i for i, name in enumerate(SLOT_NAMES)}

# Actual mm coordinates from services.py (1200 × 1200 mm lot)
SLOT_COORDINATES: dict[str, tuple[float, float]] = {
    "A1": (425.0, 1050.0),
    "A2": (650.0, 1050.0),
    "A3": (875.0, 1050.0),
    "A4": (1100.0, 1050.0),
    "B1": (425.0,  150.0),
    "B2": (650.0,  150.0),
    "B3": (875.0,  150.0),
    "B4": (1100.0, 150.0),
}
ENTRY_POINT: tuple[float, float] = (150.0, 0.0)
MAX_DISTANCE: float = math.hypot(1200.0, 1200.0)  # lot diagonal, for normalisation

# ─── Waypoint Graph ───────────────────────────────────────────────────────────
SLOT_ROUTES: dict[str, list[str]] = {
    "A1": ["entrance", "main_corridor", "intersection", "A_lane", "A1_front"],
    "A2": ["entrance", "main_corridor", "intersection", "A_lane", "A2_front"],
    "A3": ["entrance", "main_corridor", "intersection", "A_lane", "A3_front"],
    "A4": ["entrance", "main_corridor", "intersection", "A_lane", "A4_front"],
    "B1": ["entrance", "main_corridor", "intersection", "B_lane", "B1_front"],
    "B2": ["entrance", "main_corridor", "intersection", "B_lane", "B2_front"],
    "B3": ["entrance", "main_corridor", "intersection", "B_lane", "B3_front"],
    "B4": ["entrance", "main_corridor", "intersection", "B_lane", "B4_front"],
}

BOTTLENECK_NODES: list[str] = ["entrance", "main_corridor", "intersection"]

# Sorted list of all unique nodes (stable order for observation encoding)
_ALL_NODES: list[str] = sorted(
    {node for route in SLOT_ROUTES.values() for node in route}
)
NODE_INDEX: dict[str, int] = {node: i for i, node in enumerate(_ALL_NODES)}

# ─── Physics Constants ────────────────────────────────────────────────────────
NODE_TRAVEL_TIME: float = 1.0  # seconds per waypoint hop
SAFETY_MARGIN: float = 0.5     # seconds of padding around each reserved interval

# ─── Episode Constants ────────────────────────────────────────────────────────
MAX_STEPS: int = 16  # max slot-assignment steps per episode

# ─── Slot Status ─────────────────────────────────────────────────────────────
STATUS_FREE: float = 0.0
STATUS_TAKEN: float = 1.0  # occupied or reserved


class ParkingRoutingEnv(gym.Env):
    """Multi-step episode environment for parking slot allocation.

    One episode = one full parking day / simulation run.
    Each step assigns one incoming vehicle to a slot; the episode ends when
    all 8 slots are filled or MAX_STEPS assignments have been attempted.
    State (slot_statuses, reservations, active_vehicles) persists across
    steps within an episode and is only cleared by reset().

    Observation space : Box(low=0, high=1, shape=(20,), dtype=float32)
    Action space      : Discrete(8)  —  index maps to SLOT_NAMES
    Action masking    : action_masks() → ndarray(shape=(8,), dtype=bool)
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_SLOTS)

        # Mutable state (reset on each episode)
        self._slot_statuses: np.ndarray = np.zeros(NUM_SLOTS, dtype=np.float32)
        self._active_vehicles: list[dict[str, Any]] = []
        self._reservations: dict[str, list[tuple[float, float]]] = {
            node: [] for node in _ALL_NODES
        }
        self._current_time: float = 0.0
        self._step_count: int = 0
        self._terminated: bool = False

    # ─── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._slot_statuses = np.zeros(NUM_SLOTS, dtype=np.float32)
        self._active_vehicles = []
        self._reservations = {node: [] for node in _ALL_NODES}
        self._current_time = 0.0
        self._step_count = 0
        self._terminated = False

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Assign the selected slot to the next incoming vehicle.

        Flow
        ----
        1. Guard: raise if episode is already terminated.
        2. Validate action index.
        3. Penalise and continue if slot is already taken (no termination).
        4. Build per-node time intervals for the chosen route.
        5. Safety Shield conflict check.
        6a. Conflict  → reward computed on current reservations, NO registration,
                        episode continues (agent learns to avoid conflicts).
        6b. No conflict → reward computed BEFORE registration (so congestion term
                          excludes this vehicle's own reservation), then commit.
        7. Increment step counter and advance simulation clock.
        8. Terminate when all slots are filled OR max_steps is reached.
        """
        if self._terminated:
            raise RuntimeError("Episode is over. Call reset() before step().")

        if not (0 <= action < NUM_SLOTS):
            raise ValueError(f"Action {action} out of range [0, {NUM_SLOTS}).")

        slot_name = SLOT_NAMES[action]
        info: dict[str, Any] = {"slot": slot_name, "conflict": False, "step": self._step_count}

        # Step 3 — already-occupied check: penalise but do NOT terminate
        if self._slot_statuses[action] >= STATUS_TAKEN:
            self._step_count += 1
            self._current_time += NODE_TRAVEL_TIME
            self._tick_active_vehicles()
            info["reason"] = "slot_already_taken"
            terminated = bool(
                np.all(self._slot_statuses >= STATUS_TAKEN)
                or self._step_count >= MAX_STEPS
            )
            self._terminated = terminated
            return self._get_obs(), -10.0, terminated, False, info

        # Step 4 — build time intervals
        route = SLOT_ROUTES[slot_name]
        intervals = self._build_intervals(route, self._current_time)

        # Step 5 — Safety Shield conflict check
        conflict = self._check_conflict(route, intervals)

        # Step 6 — compute reward BEFORE any state mutation so congestion
        #           reflects only pre-existing reservations
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
            # 6a — do NOT register; episode continues
            info["conflict"] = True
        else:
            # 6b — commit reservation and update state
            self._register_reservation(route, intervals)
            self._slot_statuses[action] = STATUS_TAKEN
            self._add_active_vehicle(route)

        # Step 7 — advance counters and decay vehicle ETAs
        self._step_count += 1
        self._current_time += NODE_TRAVEL_TIME
        self._tick_active_vehicles()  # remove vehicles that have arrived

        # Step 8 — check termination
        terminated = bool(
            np.all(self._slot_statuses >= STATUS_TAKEN)
            or self._step_count >= MAX_STEPS
        )
        self._terminated = terminated
        return self._get_obs(), reward, terminated, False, info

    # ─── Action Masking ───────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return boolean mask compatible with sb3-contrib MaskablePPO.

        Returns
        -------
        ndarray of shape (NUM_SLOTS,) with dtype bool.
            True  = slot is selectable
            False = slot is occupied, reserved, or would cause a conflict

        Fallback
        --------
        If every free slot would cause a conflict AND the episode is not yet
        terminated, the nearest free slot (by Euclidean distance from
        ENTRY_POINT) is forced True so MaskablePPO always has at least one
        legal action.  The conflict penalty is still applied in step().
        """
        masks = np.ones(NUM_SLOTS, dtype=bool)
        for i, name in enumerate(SLOT_NAMES):
            if self._slot_statuses[i] >= STATUS_TAKEN:
                masks[i] = False
            elif self._quick_conflict_check(name):
                masks[i] = False

        # All-False guard: open the least-bad slot when the episode is live
        if not masks.any() and not self._terminated:
            free_indices = [
                i for i in range(NUM_SLOTS)
                if self._slot_statuses[i] < STATUS_TAKEN
            ]
            if free_indices:
                ex, ey = ENTRY_POINT
                best = min(
                    free_indices,
                    key=lambda i: math.hypot(
                        SLOT_COORDINATES[SLOT_NAMES[i]][0] - ex,
                        SLOT_COORDINATES[SLOT_NAMES[i]][1] - ey,
                    ),
                )
                masks[best] = True  # conflict penalty still fires in step()

        return masks

    # ─── Safety Shield ────────────────────────────────────────────────────────

    def _build_intervals(
        self, route: list[str], start_time: float
    ) -> list[tuple[float, float]]:
        """Generate (enter_time, exit_time) for each node in route."""
        intervals: list[tuple[float, float]] = []
        t = start_time
        for _ in route:
            intervals.append((t, t + NODE_TRAVEL_TIME))
            t += NODE_TRAVEL_TIME
        return intervals

    def _quick_conflict_check(self, slot_name: str) -> bool:
        """Side-effect-free conflict probe used by action_masks()."""
        route = SLOT_ROUTES[slot_name]
        intervals = self._build_intervals(route, self._current_time)
        return self._check_conflict(route, intervals)

    def _check_conflict(
        self,
        route: list[str],
        intervals: list[tuple[float, float]],
    ) -> bool:
        """Return True if any node in route conflicts with existing reservations.

        Two intervals conflict when:
            max(new_start, r_start) < min(new_end, r_end)

        A SAFETY_MARGIN is applied to the new interval before checking.
        """
        for node, (t_start, t_end) in zip(route, intervals):
            padded_start = t_start - SAFETY_MARGIN
            padded_end = t_end + SAFETY_MARGIN
            for r_start, r_end in self._reservations.get(node, []):
                if max(padded_start, r_start) < min(padded_end, r_end):
                    return True
        return False

    def _register_reservation(
        self,
        route: list[str],
        intervals: list[tuple[float, float]],
    ) -> None:
        """Append time intervals to the reservation table."""
        for node, interval in zip(route, intervals):
            self._reservations[node].append(interval)

    # ─── Active Vehicle Tracking ──────────────────────────────────────────────

    def _add_active_vehicle(self, route: list[str]) -> None:
        """Record a new vehicle in transit; evict the oldest if at capacity."""
        if len(self._active_vehicles) >= MAX_ACTIVE_VEHICLES:
            self._active_vehicles.pop(0)
        self._active_vehicles.append({
            "current_node": route[0],
            "next_node": route[1] if len(route) > 1 else route[0],
            "eta": len(route) * NODE_TRAVEL_TIME,
        })

    def _tick_active_vehicles(self) -> None:
        """Advance simulation clock for all tracked vehicles.

        Decrements every vehicle's eta by NODE_TRAVEL_TIME and removes
        vehicles that have reached their destination (eta <= 0).
        Called once per step after the clock is advanced.
        """
        self._active_vehicles = [
            {**v, "eta": v["eta"] - NODE_TRAVEL_TIME}
            for v in self._active_vehicles
        ]
        self._active_vehicles = [
            v for v in self._active_vehicles if v["eta"] > 0
        ]

    # ─── Observation Builder ──────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the normalised 20-dim observation vector.

        Layout
        ------
        [0 : 8]    slot_statuses          — 0.0 free / 1.0 taken
        [8 : 17]   active_vehicle_info    — up to 3 × (current_node, next_node, eta)
        [17 : 20]  bottleneck_summary     — reservation load on each bottleneck node
        """
        obs = np.zeros(STATE_DIM, dtype=np.float32)

        # Slot statuses (already 0.0 / 1.0)
        obs[:NUM_SLOTS] = self._slot_statuses

        # Active vehicle features (zero-padded if fewer than MAX_ACTIVE_VEHICLES)
        num_nodes = max(len(_ALL_NODES), 1)
        max_eta = len(_ALL_NODES) * NODE_TRAVEL_TIME

        for i, v in enumerate(self._active_vehicles[:MAX_ACTIVE_VEHICLES]):
            base = NUM_SLOTS + i * 3
            cur = NODE_INDEX.get(v["current_node"], 0)
            nxt = NODE_INDEX.get(v["next_node"], 0)
            obs[base]     = cur / num_nodes
            obs[base + 1] = nxt / num_nodes
            obs[base + 2] = min(v["eta"] / max_eta, 1.0)

        # Bottleneck summary
        base = NUM_SLOTS + MAX_ACTIVE_VEHICLES * 3
        for j, node in enumerate(BOTTLENECK_NODES):
            count = len(self._reservations.get(node, []))
            obs[base + j] = min(count / max(MAX_ACTIVE_VEHICLES, 1), 1.0)

        return obs
