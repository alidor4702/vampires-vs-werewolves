# core/simple_heuristic_agent.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from core.agent_base import Agent
from core.state import GameState, Cell

Move = Tuple[int, int, int, int, int]  # (r1, c1, r2, c2, num)

@dataclass
class SimpleParams:
    lambda_K: float = 0.25    # distance smoothing
    weight_H: float = 0.5     # human value weight
    weight_M: float = 0.3     # combat advantage weight
    weight_J: float = 0.2     # joining (merge) weight


class SimpleHeuristicAgent(Agent):
    """
    A clean, faithful implementation of the 3-value heuristic algorithm:

    V = w_H * U^H + w_M * U^M + w_J * U^J

    Movement:
        1) Compute V over the grid.
        2) Choose the global best target cell.
        3) For each stack, move one step in the direction that reduces Chebyshev distance to target.

    No oscillation hacks. No memory tricks. Just the algorithm as written.
    """

    def __init__(self, params: Optional[SimpleParams] = None):
        self.p = params or SimpleParams()
        self.side = "V"
        self._log = []
        self._last_heatmap = None

    def name(self): 
        return "SimpleHeuristicAgent"

    def log(self, msg: str):
        self._log.append(msg)

    def debug_messages(self):
        return self._log

    def get_heatmap(self):
        return self._last_heatmap

    def begin_turn(self, state: GameState, side: str):
        self.side = side
        self._log.clear()

    # --------------------------------------------------------------
    def select_action(self, state: GameState) -> List[Move]:
        self.begin_turn(state, "W")  # agent controls Werewolves

        UH, UM, UJ = self._compute_maps(state)
        V = self.p.weight_H * UH + self.p.weight_M * UM + self.p.weight_J * UJ
        self._last_heatmap = self._normalize(V)

        # Choose best global target cell
        tr, tc = np.unravel_index(np.argmax(V), V.shape)
        self.log(f"Target = ({tr},{tc}), V={V[tr,tc]:.3f}")

        moves = []
        for (r, c), s in self._our_stacks(state):
            r2, c2 = self._step_toward(r, c, tr, tc, state)
            if r2 is not None:
                moves.append((r, c, r2, c2, s))
                self.log(f"Move ({r},{c}) → ({r2},{c2}) with s={s}")

        return moves

    # --------------------------------------------------------------
    def _compute_maps(self, state: GameState):
        R, C = state.rows, state.cols
        UH = np.zeros((R, C))
        UM = np.zeros((R, C))
        UJ = np.zeros((R, C))

        def K(d): return math.exp(-self.p.lambda_K * d)

        for r in range(R):
            for c in range(C):
                cell = state.grid[r, c]
                n = cell.werewolves if self.side == "W" else cell.vampires
                n = max(1, n)

                uh = um = 0.0

                # --- Human & Combat Terms ---
                for i in range(R):
                    for j in range(C):
                        d = max(abs(r - i), abs(c - j))
                        k = K(d)
                        tgt = state.grid[i, j]

                        # Human value
                        if tgt.humans > 0:
                            H = tgt.humans
                            if n == H: P = 0.5
                            elif n < H: P = n / (2*H)
                            else: P = n/H - 0.5
                            uh += k * P

                        # Combat value
                        e = tgt.vampires if self.side == "W" else tgt.werewolves
                        if e > 0:
                            if n >= 1.5 * e: um += k
                            elif e >= 1.5 * n: um -= k
                            else:
                                P = n/e - 0.5 if n > e else n / (2*e)
                                um += k * (2*P - 1)

                # --- Joining Term (simple N8 merge potential) ---
                uj = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0: continue
                        rr, cc = r + dr, c + dc
                        if not state.in_bounds(rr, cc): continue
                        ally = state.grid[rr, cc]
                        a = ally.werewolves if self.side == "W" else ally.vampires
                        if a > 0:
                            uj += 1     # simply reward cluster potential

                UH[r, c], UM[r, c], UJ[r, c] = uh, um, uj

        return UH, UM, UJ

    # --------------------------------------------------------------
    def _our_stacks(self, state: GameState):
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                val = cell.werewolves if self.side == "W" else cell.vampires
                if val > 0:
                    yield (r, c), val

    def _step_toward(self, r, c, tr, tc, state: GameState):
        best = None
        best_dist = 1e9

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                rr, cc = r + dr, c + dc
                if not state.in_bounds(rr, cc): continue
                if not state.is_adjacent(r, c, rr, cc): continue

                d = max(abs(rr - tr), abs(cc - tc))  # Chebyshev distance
                if d < best_dist:
                    best_dist = d
                    best = (rr, cc)

        return best if best else (None, None)

    # --------------------------------------------------------------
    def _normalize(self, V):
        vmin, vmax = V.min(), V.max()
        if abs(vmax - vmin) < 1e-9:
            return np.zeros_like(V)
        return (V - vmin) / (vmax - vmin)
