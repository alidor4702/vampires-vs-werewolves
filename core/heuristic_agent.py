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
    # New params from improved algorithm
    kappa: float = 3.0       # sigmoid sharpness for human favorability
    epsilon: float = 1.0     # sigmoid offset to avoid division issues
    threat_gamma: float = 0.5  # weight of adjacent enemy threat

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
        """
        U^H from PDF eq (4): Score human cells accounting for enemy competition.

        For each human cell, compute:
          - Our favorability: g_H(our_units, humans) weighted by distance
          - Enemy favorability: g_H(enemy_units, humans) weighted by their distance
          - Net value = our_gain - enemy_gain (we want cells where we have advantage)
        """
        R, C = state.rows, state.cols
        Hscore = np.zeros((R, C))

        # Precompute enemy stack info for efficiency
        enemy_stacks = list(self._enemy_stacks(state))

        # Find all human cells
        human_cells = []
        for i in range(R):
            for j in range(C):
                h = state.grid[i, j].humans
                if h > 0:
                    human_cells.append((i, j, h))

        if not human_cells:
            return Hscore

        # For each human cell, compute competitive value
        for (hi, hj, h) in human_cells:
            # Our potential gain from this human cell
            our_value = 0.0
            for (r, c), s in self._our_stacks(state):
                d = max(abs(r - hi), abs(c - hj))  # Chebyshev distance
                K = math.exp(-self.p.lambda_K * d)
                gain = self._human_favorability(s, h)
                our_value += gain * K

            # Enemy's potential gain from this human cell
            enemy_value = 0.0
            for (er, ec), e in enemy_stacks:
                d = max(abs(er - hi), abs(ec - hj))
                K = math.exp(-self.p.lambda_K * d)
                gain = self._human_favorability(e, h)
                enemy_value += gain * K

            # Net value: positive if we have advantage, negative if enemy does
            # This makes the agent prefer humans it can reach before the enemy
            Hscore[hi, hj] = our_value - enemy_value

        return Hscore

    def _score_enemies(self, state: GameState):
        """
        U^M from PDF eq (7): Score enemy cells and apply threat penalty.

        For each enemy cell:
          - Compute h_M(our_units, enemy) weighted by distance
          - Subtract enemy's h_M(enemy_units, our_units) (their view of attacking us)
          - Apply threat penalty for adjacent dangerous enemies
        """
        R, C = state.rows, state.cols
        Escore = np.zeros((R, C))

        our_stacks = list(self._our_stacks(state))
        enemy_stacks = list(self._enemy_stacks(state))

        if not enemy_stacks:
            return Escore

        # Compute median enemy stack size for reference
        enemy_sizes = [e for _, e in enemy_stacks]
        median_enemy = sorted(enemy_sizes)[len(enemy_sizes) // 2] if enemy_sizes else 1

        for (ei, ej), e in enemy_stacks:
            # Our combat value against this enemy
            our_combat_value = 0.0
            for (r, c), s in our_stacks:
                d = max(abs(r - ei), abs(c - ej))
                K = math.exp(-self.p.lambda_K * d)
                utility = self._combat_utility(s, e)
                our_combat_value += utility * K

            # Enemy's combat value against us (how much they want to attack us)
            enemy_combat_value = 0.0
            for (r, c), s in our_stacks:
                d = max(abs(ei - r), abs(ej - c))
                K = math.exp(-self.p.lambda_K * d)
                # From enemy's perspective: they attack us
                utility = self._combat_utility(e, s)
                enemy_combat_value += utility * K

            # Net value: positive if we should attack, negative if dangerous
            Escore[ei, ej] = our_combat_value - enemy_combat_value

        # Apply threat penalty: dangerous enemies adjacent to our stacks
        # This makes us want to move AWAY from cells near strong enemies
        for (r, c), s in our_stacks:
            threat = self._compute_threat(state, r, c, s)
            # Reduce value of our current position if threatened
            Escore[r, c] -= self.p.threat_gamma * threat

        return Escore

    def _compute_threat(self, state: GameState, r: int, c: int, s: int) -> float:
        """
        Threat from adjacent enemies (PDF eq 6).
        Enemies in N8 can attack us next turn - penalize if they overpower us.
        """
        threat = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if not state.in_bounds(rr, cc):
                    continue
                e = self._enemy_at(state, rr, cc)
                if e > 0:
                    # h_M from enemy's perspective (negative for us if they're stronger)
                    utility = self._combat_utility(e, s)
                    # Only count as threat if enemy has advantage (utility > 0 for them)
                    if utility > 0:
                        threat += utility
        return threat

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
        """
        Select best adjacent cell to move into.
        CRITICAL: Avoid cells where we'd lose (humans > us, or enemy overpowers us).
        """
        best = None
        bestv = -1e9
        last_dst = self._last_dest_by_src.get((r, c))

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
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

                # SAFETY CHECK: Don't step into cells where we'd likely lose
                if not self._is_safe_to_enter(state, rr, cc, s):
                    continue

                val = V[rr, cc]
                if val > bestv:
                    bestv = val
                    best = (rr, cc)

        # if everything filtered out due to safety, find safest option
        if best is None:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if not state.in_bounds(rr, cc):
                        continue
                    if (rr, cc) in used:
                        continue
                    # In fallback, still prefer safe cells but accept any if none safe
                    safety_bonus = 100.0 if self._is_safe_to_enter(state, rr, cc, s) else 0.0
                    val = V[rr, cc] + safety_bonus
                    if val > bestv:
                        bestv = val
                        best = (rr, cc)
        return best

    def _is_safe_to_enter(self, state: GameState, r: int, c: int, our_count: int) -> bool:
        """
        Check if it's safe for our_count units to enter cell (r, c).
        Safe means:
          - Empty cell: always safe
          - Humans: safe if we outnumber them (guaranteed conversion)
          - Enemy: safe if we have 1.5x advantage (guaranteed win)
        """
        cell = state.grid[r, c]

        # Check humans
        if cell.humans > 0:
            # Only safe if we can guarantee conversion (need >= humans)
            if our_count < cell.humans:
                return False

        # Check enemy
        enemy = self._enemy_at(state, r, c)
        if enemy > 0:
            # Only safe if we overpower them (need >= 1.5x)
            if our_count < 1.5 * enemy:
                return False

        return True

    # ----------------- HELPERS -----------------
    def _our_stacks(self, state: GameState):
        for r in range(state.rows):
            for c in range(state.cols):
                s = state.grid[r,c].vampires if self.side == "V" else state.grid[r,c].werewolves
                if s > 0:
                    yield (r,c), s

    def _enemy_stacks(self, state: GameState):
        """Iterate over enemy positions and counts."""
        for r in range(state.rows):
            for c in range(state.cols):
                e = self._enemy_at(state, r, c)
                if e > 0:
                    yield (r, c), e

    def _enemy_at(self, state, r, c):
        cell = state.grid[r,c]
        return cell.werewolves if self.side == "V" else cell.vampires

    def _battle_prob(self, n: int, m: int) -> float:
        """Combat probability P(n, m) from game rules."""
        if n == m:
            return 0.5
        elif n < m:
            return n / (2 * m)
        else:
            return (n / m) - 0.5

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function σ(x) = 1/(1+e^-x)."""
        return 1.0 / (1.0 + math.exp(-x))

    def _human_favorability(self, n: int, h: int) -> float:
        """
        g_H(n, H) from PDF eq (3): P(n,H) * σ(κ * min(n,H)/(H+ε))

        This peaks when n ≈ H (efficient use of units).
        Penalizes overkill (n >> H, wasting units) and underkill (n << H, risky).
        """
        if h <= 0:
            return 0.0
        p = self._battle_prob(n, h)
        # Sigmoid term penalizes inefficiency
        ratio = min(n, h) / (h + self.p.epsilon)
        sig = self._sigmoid(self.p.kappa * ratio)
        return p * sig

    def _combat_utility(self, n: int, e: int) -> float:
        """
        h_M(n, e) from PDF eq (5):
        +1 if n >= 1.5*e (we overpower)
        -1 if e >= 1.5*n (they overpower us)
        2*P(n,e) - 1 otherwise (maps P ∈ [0,1] to [-1,1])
        """
        if e <= 0:
            return 0.0
        if n >= 1.5 * e:
            return 1.0
        elif e >= 1.5 * n:
            return -1.0
        else:
            p = self._battle_prob(n, e)
            return 2 * p - 1  # maps [0,1] -> [-1,1]

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
