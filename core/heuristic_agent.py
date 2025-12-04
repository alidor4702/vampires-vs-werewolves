# core/heuristic_agent.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import math

from core.agent_base import Agent
from .state import GameState, Cell

Move = Tuple[int,int,int,int,int]

@dataclass
class HeuristicParams:
    lambda_K: float = 0.25   # distance decay
    join_bonus: float = 1.2  # weight of merging when no good attack available
    human_weight: float = 1.0
    enemy_weight: float = 1.0
    backtrack_penalty: float = 1.5     # discourage ping-pong
    lock_weight: float = 1.25          # softly bias towards previous target
    dir_to_human_bias: float = 0.15    # small push toward nearest human

class HeuristicAgent(Agent):
    def __init__(self, params: Optional[HeuristicParams]=None):
        self.p = params or HeuristicParams()
        self.side = "V"
        self._log = []
        # memory to reduce oscillations and keep directionality
        self._last_dest_by_src: dict[Tuple[int,int], Tuple[int,int]] = {}
        self._target_lock: dict[Tuple[int,int], Tuple[int,int]] = {}

    def name(self):
        return "HeuristicAgent-Simple"

    def begin_turn(self, state: GameState, side: str):
        self.side = side
        self._log = []

    # ----------------- MAIN ENTRY -----------------
    def propose_moves(self, state: GameState) -> List[Move]:
        UH = self._score_humans(state)
        UM = self._score_enemies(state)
        UJ = self._score_joining(state)

        # Combined board value
        V = self.p.human_weight * UH + self.p.enemy_weight * UM + UJ

        # Light normalization to avoid flat maps on sparse boards
        abs_mean = float(np.mean(np.abs(V))) + 1e-9
        V = V / abs_mean

        # Apply gentle behavioral nudges
        V = self._apply_backtrack_penalty(state, V)
        V = self._apply_target_lock(V)
        V = self._apply_dir_to_nearest_human(state, V)

        moves = []
        used = set()
        for (r,c), s in self._our_stacks(state):
            if s < 2:
                continue
            if state.movable[r,c] <= 0:
                continue

            best = self._best_step(state, r, c, s, V, used)
            if best:
                r2, c2 = best
                moves.append((r,c,s,r2,c2))
                used.add((r2,c2))
                # update memories
                self._last_dest_by_src[(r,c)] = (r2,c2)
                self._target_lock[(r,c)] = (r2,c2)

        self._log_state(state, UH, UM, UJ, V, moves)
        self._last_heatmap = V
        return moves

    # ----------------- SCORING -----------------
    def _score_humans(self, state: GameState):
        R,C = state.rows, state.cols
        Hscore = np.zeros((R,C))
        for (r,c), s in self._our_stacks(state):
            for i in range(R):
                for j in range(C):
                    h = state.grid[i,j].humans
                    if h <= 0: 
                        continue
                    d = max(abs(r-i), abs(c-j))
                    gain = h if s >= h else s / (2*h)
                    Hscore[i,j] += gain * math.exp(-self.p.lambda_K * d)
        return Hscore

    def _score_enemies(self, state: GameState):
        R,C = state.rows, state.cols
        Escore = np.zeros((R,C))
        for (r,c), s in self._our_stacks(state):
            for i in range(R):
                for j in range(C):
                    e = self._enemy_at(state, i, j)
                    if e <= 0:
                        continue
                    d = max(abs(r-i), abs(c-j))
                    if s >= 1.5 * e:
                        # attractive target
                        Escore[i,j] += 1.0 * math.exp(-self.p.lambda_K * d)
                    elif e >= 1.5 * s:
                        # local danger at our origin
                        Escore[r,c] -= 1.0 * math.exp(-self.p.lambda_K * d)
        return Escore

    def _score_joining(self, state: GameState):
        R,C = state.rows, state.cols
        J = np.zeros((R,C))
        for (r,c), s in self._our_stacks(state):
            # only encourage merging if no free human/enemy capture nearby
            if self._has_easy_target_near(state, r, c, s):
                continue
            for (r2,c2), s2 in self._our_stacks(state):
                if (r,c) == (r2,c2):
                    continue
                if max(abs(r-r2), abs(c-c2)) == 1:
                    J[r2,c2] += self.p.join_bonus
        return J

    # ----------------- BEHAVIORAL NUDGES -----------------
    def _apply_backtrack_penalty(self, state: GameState, V: np.ndarray) -> np.ndarray:
        """Penalize stepping back into the exact last-destination cell for that source."""
        V2 = V.copy()
        for (src_r, src_c), (dst_r, dst_c) in self._last_dest_by_src.items():
            if state.in_bounds(dst_r, dst_c):
                V2[dst_r, dst_c] -= self.p.backtrack_penalty
        return V2

    def _apply_target_lock(self, V: np.ndarray) -> np.ndarray:
        """Slightly lift previously chosen targets to preserve direction."""
        V2 = V.copy()
        for _, (tr, tc) in self._target_lock.items():
            V2[tr, tc] *= self.p.lock_weight
        return V2

    def _apply_dir_to_nearest_human(self, state: GameState, V: np.ndarray) -> np.ndarray:
        """Small boost on cells that move closer to the nearest human cluster."""
        if self.p.dir_to_human_bias <= 0:
            return V
        R, C = state.rows, state.cols
        # precompute nearest human chebyshev distance per cell
        human_cells = [(i,j) for i in range(R) for j in range(C) if state.grid[i,j].humans > 0]
        if not human_cells:
            return V
        V2 = V.copy()
        # simple potential: lower distance → higher bonus
        def nearest_human_dist(i,j):
            dmin = 1e9
            for (hr,hc) in human_cells:
                d = max(abs(i-hr), abs(j-hc))
                if d < dmin:
                    dmin = d
            return dmin
        # compute normalization scale
        # use a coarse sample to keep it cheap
        sample = []
        for i in range(0, R, max(1, R//6)):
            for j in range(0, C, max(1, C//6)):
                sample.append(nearest_human_dist(i,j))
        denom = max(1.0, float(max(sample) - min(sample)))
        for i in range(R):
            for j in range(C):
                d = nearest_human_dist(i,j)
                # invert and normalize into roughly [0,1]
                bonus = (max(sample) - d) / denom
                V2[i,j] += self.p.dir_to_human_bias * bonus
        return V2

    # ----------------- MOVE SELECTION -----------------
    def _best_step(self, state: GameState, r, c, s, V, used):
        best = None
        bestv = -1e9
        last_dst = self._last_dest_by_src.get((r,c))
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if not state.in_bounds(rr, cc):
                    continue
                if (rr, cc) in used:
                    continue
                # avoid immediate backtrack to previous dst
                if last_dst is not None and (rr, cc) == last_dst:
                    continue
                val = V[rr, cc]
                if val > bestv:
                    bestv = val
                    best = (rr, cc)
        # if everything filtered out, allow the best neighbor anyway
        if best is None:
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if state.in_bounds(rr, cc) and (rr, cc) not in used:
                        val = V[rr, cc]
                        if val > bestv:
                            bestv = val
                            best = (rr, cc)
        return best

    # ----------------- HELPERS -----------------
    def _our_stacks(self, state: GameState):
        for r in range(state.rows):
            for c in range(state.cols):
                s = state.grid[r,c].vampires if self.side == "V" else state.grid[r,c].werewolves
                if s > 0:
                    yield (r,c), s

    def _enemy_at(self, state, r, c):
        cell = state.grid[r,c]
        return cell.werewolves if self.side == "V" else cell.vampires

    def _has_easy_target_near(self, state, r, c, s):
        # easy humans or overpowering enemy in N8
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr, cc = r + dr, c + dc
                if state.in_bounds(rr, cc):
                    h = state.grid[rr, cc].humans
                    if h > 0 and s >= h:
                        return True
                    e = self._enemy_at(state, rr, cc)
                    if e > 0 and s >= 1.5 * e:
                        return True
        return False

    # ----------------- LOGGING -----------------
    def _log_state(self, state, UH, UM, UJ, V, moves):
        self._log.append("=== Turn Log ===")
        self._log.append(
            f"means |UH|={np.mean(np.abs(UH)):.3f} |UM|={np.mean(np.abs(UM)):.3f} |UJ|={np.mean(np.abs(UJ)):.3f}"
        )
        self._log.append("Top valued cells:")
        flat = [((r,c), float(V[r,c])) for r in range(state.rows) for c in range(state.cols)]
        flat.sort(key=lambda x: x[1], reverse=True)
        for ((r,c), v) in flat[:6]:
            self._log.append(f"({r},{c}) = {v:.2f}")

        self._log.append("Moves:")
        for m in moves:
            self._log.append(str(m))

    def debug_messages(self):
        return self._log

    def get_heatmap(self):
        return getattr(self, "_last_heatmap", None)

    def select_action(self, state: GameState):
        self.begin_turn(state, state.turn)
        moves = self.propose_moves(state)
        # keep tuple ordering consistent with board runner: (r,c,r2,c2,num)
        return [(r, c, r2, c2, num) for (r, c, num, r2, c2) in moves]
    
    def evaluate_state(self, state: GameState, side: str) -> float:
        """
        Static evaluation: how good is 'state' for 'side' (V or W).

        Uses the same human/enemy/join/immediate-opportunity scores as the
        heuristic policy, but without path-dependent nudges.
        Positive values are good for 'side', negative bad.
        """
        self.side = side

        UH = self._score_humans(state)
        UM = self._score_enemies(state)
        UJ = self._score_joining(state)
        UI = self._score_immediate_opportunities(state)

        V = self.p.human_weight * UH + self.p.enemy_weight * UM + UJ

        abs_mean = float(np.mean(np.abs(V))) + 1e-9
        V = V / abs_mean
        V = V + UI

        # Scalar score: sum over board
        return float(np.sum(V))

