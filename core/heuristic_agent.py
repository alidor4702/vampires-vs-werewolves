# core/heuristic_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import numpy as np

from core.agent_base import Agent
from .state import GameState, Cell

Move = Tuple[int, int, int, int, int]  # (r1, c1, num, r2, c2)

@dataclass
class HeuristicParams:
    # distance kernels
    lambda_K: float = 0.22        # K(d) = exp(-lambda_K * d)
    # human favorability (sigmoid)
    kappa: float = 4.0
    eps: float = 1e-6
    # threat / combat
    gamma_threat: float = 1.0
    # area ring
    alpha_A: float = 0.07          # L_A = floor(alpha_A * max(R,C))
    mu_J: float = 0.9             # J(Δ) = exp(-mu_J * Δ)
    betaM_loc: float = 0.6
    lambda_A: float = 0.18
    omega_A: float = 0.9
    # time/context weights a_, b_ (we’ll normalize each turn)
    aH: float = 0.5; bH: float = 0.3
    aM: float = 0.8; bM: float = 0.5
    aJ: float = 0.5; bJ: float = 0.6
    # path scoring
    delta: float = 0.92           # discount for steps
    alpha_gain: float = 0.6       # weight for immediate gain G
    detour_slack: int = 1         # allow shortest path + {0,1} steps
    topK: int = 5                 # candidate targets to consider
    keep_rho: float = 0.93        # how much value we keep at origin

class HeuristicAgent(Agent):
    r"""
    Bare-bones, testable heuristic agent:
    1) builds value maps U^H, U^M, U^J
    2) adds area augmentation A to get \widetilde{V}
    3) picks up to one detour-limited path per friendly stack (no splitting yet)
    4) returns a set of legal adjacent moves for this turn
    """
    def __init__(self, params: Optional[HeuristicParams] = None):
        self.p = params or HeuristicParams()
        self.side = "V"  # updated in begin_turn
        self._last_debug = []      # store messages you can print in board log
        self._recent_cells = []

    def log(self, msg: str, level: str = "INFO", indent: int = 0):
        """Append a formatted debug message to the internal log buffer."""
        prefix = " " * indent + f"[{level}] "
        self._last_debug.append(prefix + msg)
   
    def name(self) -> str:
        return "HeuristicAgent-v0"

    # --- lifecycle -------------------------------------------------
    def begin_turn(self, state: GameState, side: str):
        self.side = side
        self._last_debug.clear()

    # --- main API --------------------------------------------------
    def propose_moves(self, state: GameState) -> List[Move]:
        """
        Return a list of legal (r1,c1,num,r2,c2) moves for this turn.
        This version normalizes U-terms, avoids fragmentation, and adds safety penalties.
        """
        rows, cols = state.rows, state.cols
        ours_mask = self._ours_mask(state)
        if not np.any(ours_mask):
            return []

        # 1️⃣ compute U^H, U^M, U^J (for n = stack size as a proxy)
        UH, UM, UJ = self._compute_U_terms(state)

        # 2️⃣ normalize each to prevent UH domination
        UHn = (UH - UH.mean()) / (abs(UH).mean() + 1e-9)
        UMn = (UM - UM.mean()) / (abs(UM).mean() + 1e-9)
        UJn = (UJ - UJ.mean()) / (abs(UJ).mean() + 1e-9)


        # 3️⃣ combine with time/context weights
        tauH, tauM, tauJ = self._time_weights(state)
        self._last_tauH, self._last_tauM, self._last_tauJ = tauH, tauM, tauJ
        V = 0.5 * tauH * UHn + tauM * UMn + tauJ * UJn

        # 4️⃣ area augmentation -> \widetilde V
        Vtil = self._area_augmented(state, V, UHn, UMn, UJn)

        # 7️⃣ choose top-K targets and per-stack next step
        moves: List[Move] = []
        used_targets = set(state.targets_used_this_turn)

        # merge skip: ignore tiny fragments
        for (r, c), s in self._our_stacks(state):
            if s < 2:  # skip lone units to reduce fragmentation
                continue
            if state.movable[r, c] <= 0:
                continue

            tgt_list = self._topK_targets(Vtil, self.p.topK)
            best_step = self._best_step_towards(state, r, c, s, tgt_list, used_targets, Vtil)
            if best_step is not None:
                r2, c2, k = best_step
                k = min(s, state.movable[r, c])  # move whole stack, no splitting
                moves.append((r, c, k, r2, c2))
                used_targets.add((r2, c2))

        # --- deduplicate & finalize ---
        legal = []
        for r1, c1, num, r2, c2 in moves:
            if self._legal(state, r1, c1, num, r2, c2):
                legal.append((r1, c1, num, r2, c2))

        self._last_heatmap = Vtil
        self.debug_summary(state, UH, UM, UJ, Vtil, legal)
        return legal

    # --- helpers: masks, stacks -----------------------------------
    def _ours_mask(self, state: GameState) -> np.ndarray:
        m = np.zeros((state.rows, state.cols), dtype=bool)
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                v = cell.vampires if self.side == "V" else cell.werewolves
                if v > 0:
                    m[r, c] = True
        return m

    def _our_stacks(self, state: GameState):
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                s = cell.vampires if self.side == "V" else cell.werewolves
                if s > 0:
                    yield (r, c), s

    def _enemy_count(self, cell: Cell) -> int:
        return cell.werewolves if self.side == "V" else cell.vampires

    # --- U^H, U^M, U^J ---------------------------------------------
    def _compute_U_terms(self, state: GameState):
        R, C = state.rows, state.cols
        UH = np.zeros((R, C), dtype=float)
        UM = np.zeros((R, C), dtype=float)
        UJ = np.zeros((R, C), dtype=float)

        # pre-read counts
        H = np.zeros((R, C), dtype=int)
        O = np.zeros((R, C), dtype=int)   # ours
        E = np.zeros((R, C), dtype=int)   # enemy
        for r in range(R):
            for c in range(C):
                cell = state.grid[r, c]
                H[r, c] = cell.humans
                O[r, c] = cell.vampires if self.side == "V" else cell.werewolves
                E[r, c] = self._enemy_count(cell)

        # distance kernel
        def Kdist(d: int) -> float:
            return math.exp(-self.p.lambda_K * d)

        # -------------------------------
        # UH fix: enemy pressure uses real enemy stacks and distances to EACH human
        # -------------------------------
        # collect enemy stacks
        Epos = [(re, ce, E[re, ce]) for re in range(R) for ce in range(C) if E[re, ce] > 0]

        # precompute enemy pressure to humans (independent of our candidate cell)
        enemy_H_pressure = 0.0
        if Epos:
            for i in range(R):
                for j in range(C):
                    Hij = H[i, j]
                    if Hij <= 0:
                        continue
                    best_e = 0.0
                    for (re, ce, ne) in Epos:
                        de = max(abs(re - i), abs(ce - j))
                        ke = Kdist(de)
                        # same human capture rule: equal or more → certain
                        Pe = 1.0 if ne >= Hij else (ne / (2.0 * Hij))
                        best_e = max(best_e, ke * Pe)
                    enemy_H_pressure += best_e
        # -------------------------------

        # nominal enemy packet size for UM symmetric (kept as-is for now)
        nz = E[E > 0]
        nbar = int(np.median(nz)) if nz.size else 1

        # main loops
        for r in range(R):
            for c in range(C):
                n = max(1, O[r, c])  # evaluate terms as if sending current stack

                # U^H: sum of our discounted human capture values, minus global enemy pressure
                uh = 0.0
                for i in range(R):
                    for j in range(C):
                        Hij = H[i, j]
                        if Hij <= 0:
                            continue
                        d = max(abs(r - i), abs(c - j))
                        k = Kdist(d)
                        uh += k * self._gH(n, Hij)

                UH[r, c] = uh - (enemy_H_pressure if Epos else 0.0)
                # --- U^M COMBAT UTILITY (REAL ENEMY PRESSURE) -------------------
                # Our advantage over all enemy stacks
                um = 0.0
                for (re, ce, ne) in Epos:
                    d = max(abs(r - re), abs(c - ce))
                    k = Kdist(d)
                    um += k * self._hM(n, ne)

                # Enemy advantage over us (enemy perspective)
                um_enemy = 0.0
                for (oe_r, oe_c, oe_s) in [(rr, cc, O[rr, cc]) for rr in range(R) for cc in range(C) if O[rr, cc] > 0]:
                    for (re, ce, ne) in Epos:
                        d = max(abs(oe_r - re), abs(oe_c - ce))
                        k = Kdist(d)
                        um_enemy += k * self._hM(ne, oe_s)

                UM[r, c] = um - um_enemy - self._Threat_local(state, r, c, n)

                # U^J (unchanged)
                UJ[r, c] = self._U_join(state, r, c, n)

        # clip UH after filling it
        UH = np.clip(UH, -10.0, 10.0)
        return UH, UM, UJ


    def _gH(self, n: int, H: int) -> float:
        if H <= 0 or n <= 0: return 0.0
        # success value
        if n >= H: 
            P = 1.0                      # equal or more → guaranteed capture per rules
        else:
            P = n / (2.0 * H)            # same scaling you used for underpowered
        return P                          # remove the sigmoid here (see Q3)


    def _hM(self, n: int, e: int) -> float:
        if e <= 0 or n <= 0:
            return 0.0
        if n >= 1.5 * e:
            return 1.0
        if e >= 1.5 * n:
            return -1.0
        # otherwise map P to [-1,1]
        if n == e:
            P = 0.5
        elif n < e:
            P = n / (2.0 * e)
        else:
            P = n / e - 0.5
        return 2.0 * P - 1.0

    def _Threat_local(self, state: GameState, r: int, c: int, n: int) -> float:
        s = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: 
                    continue
                rr, cc = r + dr, c + dc
                if not state.in_bounds(rr, cc):
                    continue
                e = self._enemy_count(state.grid[rr, cc])
                if e > 0:
                    s += self.p.gamma_threat * min(self._hM(n, e), 0.0)
        return s
    
    def _has_easy_human(self, state, r, c, n):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr, cc = r+dr, c+dc
                if state.in_bounds(rr,cc):
                    H = state.grid[rr,cc].humans
                    if H>0 and n >= H:
                        return True
        return False

    def _has_easy_enemy(self, state, r, c, n):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr, cc = r+dr, c+dc
                if state.in_bounds(rr,cc):
                    e = self._enemy_count(state.grid[rr,cc])
                    if e>0 and n >= 1.5*e:
                        return True
        return False


    def _U_join(self, state: GameState, r: int, c: int, s_rc: int) -> float:
        # UJ applies only when we have no easy capture or overpowering attack available
        if self._has_easy_human(state, r, c, s_rc) or self._has_easy_enemy(state, r, c, s_rc):
            return 0.0

        def pow_val(s: int) -> int:
            cnt = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if not state.in_bounds(rr, cc):
                        continue
                    e = self._enemy_count(state.grid[rr, cc])
                    if s >= 1.5 * e:
                        cnt += 1
            return cnt

        base = pow_val(s_rc)
        gain = 0.0
        # consider merging from N8 allies
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: 
                    continue
                rr, cc = r + dr, c + dc
                if not state.in_bounds(rr, cc): 
                    continue
                cell = state.grid[rr, cc]
                s_uv = cell.vampires if self.side == "V" else cell.werewolves
                if s_uv <= 0: 
                    continue
                delta = pow_val(s_rc + s_uv) - base
                # distance kernel within small radius (Δ = 1 here)
                gain += math.exp(-self.p.lambda_K * 1) * max(0, delta)
        return gain

    # --- time/context τ --------------------------------------------
    def _time_weights(self, state: GameState):
    # Compute remaining human proportion
        Htot = 0
        for r in range(state.rows):
            for c in range(state.cols):
                Htot += state.grid[r,c].humans

        if not hasattr(self, "_H0"):
            self._H0 = max(1, Htot)

        qH = Htot / self._H0   # 1.0 early, → 0 late

        # Strong early human priority
        xH = 0.70 * qH + 0.35 * (1 - qH)
        xM = 0.20 * (1 - qH) + 0.35 * qH
        xJ = 0.10 + 0.15 * (1 - qH)

        s = xH + xM + xJ + 1e-9
        tauH, tauM, tauJ = xH/s, xM/s, xJ/s
        return tauH, tauM, tauJ


    # --- area augmentation -----------------------------------------
    def _area_augmented(self, state: GameState, V, UH, UM, UJ):
        R, C = state.rows, state.cols
        D = max(R, C)
        LA = max(1, int(self.p.alpha_A * D))

        # reach scaling S_reach per cell
        # distance from cell to nearest of OUR stacks
        ours = [(r, c) for (r, c), s in self._our_stacks(state)]
        if not ours:
            return V.copy()
        Sreach = np.zeros((R, C), dtype=float)
        for r in range(R):
            for c in range(C):
                dmin = min(max(abs(r - rr), abs(c - cc)) for rr, cc in ours)
                Sreach[r, c] = math.exp(-self.p.lambda_A * dmin)

        # local ring J(Δ)
        def J(delta: int) -> float:
            return math.exp(-self.p.mu_J * delta)

        Vtil = V.copy()
        for r in range(R):
            for c in range(C):
                acc_pos = 0.0
                acc_neg = 0.0
                # sweep square window; keep it simple (Δ as Chebyshev)
                r0 = max(0, r - LA); r1 = min(R - 1, r + LA)
                c0 = max(0, c - LA); c1 = min(C - 1, c + LA)
                for i in range(r0, r1 + 1):
                    for j in range(c0, c1 + 1):
                        delta = max(abs(r - i), abs(c - j))
                        if delta == 0: 
                            continue
                        w = J(delta)
                        # we don't have enemy U maps explicitly; approximate by flipping side weights:
                        # treat negative part of UM as enemy-favor—keep it simple for v0.
                        acc_pos += w * (UH[i, j] + UJ[i, j] + self.p.betaM_loc * max(UM[i, j], 0.0))
                        acc_neg += w * (self.p.betaM_loc * max(-UM[i, j], 0.0))
                A = acc_pos - acc_neg
                Vtil[r, c] = V[r, c] + self.p.omega_A * Sreach[r, c] * A
                if hasattr(self, "_prev_UH"):
                    UH = 0.7 * self._prev_UH + 0.3 * UH
                self._prev_UH = UH.copy()
        return Vtil

    # --- target and step selection --------------------------------
    def _topK_targets(self, Vtil: np.ndarray, K: int):
        flat = [((r, c), Vtil[r, c]) for r in range(Vtil.shape[0]) for c in range(Vtil.shape[1])]
        flat.sort(key=lambda x: x[1], reverse=True)
        return [pos for (pos, _) in flat[:K]]

    def _best_step_towards(self, state: GameState, r: int, c: int, s: int,
                           targets, used_targets, Vtil: np.ndarray):
        """
        Find one adjacent step (8-dir) that moves toward the best detour-limited
        target’s shortest (or +1) route and yields the highest discounted Vtil.
        """
        if not targets:
            return None

        # choose the target with max Vtil minus simple Manhattan distance penalty
        def score_tgt(pos):
            (tr, tc) = pos
            d = max(abs(tr - r), abs(tc - c))
            return Vtil[tr, tc] - 0.05 * d
        targets = sorted(targets, key=score_tgt, reverse=True)

        # candidate adjacent moves
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if state.in_bounds(rr, cc) and state.is_adjacent(r, c, rr, cc):
                    neigh.append((rr, cc))

        if not neigh:
            return None

        # score each adjacent step by discounted Vtil at that step (one-step lookahead)
        best = None; best_val = -1e18
        for rr, cc in neigh:
            if (rr, cc) in used_targets:
                continue
            # incorporate distance toward target estimate
            (tr, tc) = targets[0]   # best target
            d_now = max(abs(tr - r), abs(tc - c))
            d_next = max(abs(tr - rr), abs(tc - cc))
            step_val = Vtil[rr, cc] + 0.3 * (d_now - d_next)
            if step_val > best_val:
                best_val = step_val
                best = (rr, cc)
        if best is None:
            return None

        move_count = min(s, state.movable[r, c])  # move full stack
        return best[0], best[1], move_count


    def _legal(self, state: GameState, r1, c1, num, r2, c2) -> bool:
        return (
            state.in_bounds(r1, c1)
            and state.in_bounds(r2, c2)
            and state.is_adjacent(r1, c1, r2, c2)
            and num > 0
            and state.movable[r1, c1] >= num
            and (r2, c2) not in state.targets_used_this_turn
        )
    
    # --- unified debugging summary ---------------------------------
    def debug_summary(self, state, UH, UM, UJ, Vtil, moves):
        """
        Collect and log a full diagnostic snapshot of the agent’s internal state
        for one turn — ranges, means, best cells, per-stack context, and selected moves.
        """
        self._last_debug.clear()
        self.log(f"--- Turn debug summary for side={self.side} ---", "INFO")

        # 1️⃣ Basic ranges and means
        self.log(
            f"RANGE UH=({UH.min():.2f},{UH.max():.2f}) UM=({UM.min():.2f},{UM.max():.2f}) "
            f"UJ=({UJ.min():.2f},{UJ.max():.2f}) Ṽ=({Vtil.min():.2f},{Vtil.max():.2f})",
            "RANGE"
        )
        self.log(
            f"MEAN UH={UH.mean():.3f}, UM={UM.mean():.3f}, UJ={UJ.mean():.3f}, Ṽ={Vtil.mean():.3f}",
            "MEAN"
        )
        self.log(
            f"τ weights: tauH={getattr(self,'_last_tauH',None)}, "
            f"tauM={getattr(self,'_last_tauM',None)}, tauJ={getattr(self,'_last_tauJ',None)}",
            "PARAMS"
        )
        self.log(
            f"Scale factors: |UH|mean={abs(UH).mean():.3f}, |UM|mean={abs(UM).mean():.3f}, |UJ|mean={abs(UJ).mean():.3f}",
            "SCALE"
        )


        # 2️⃣ Top and bottom 5 cells
        flat = [((r, c), Vtil[r, c]) for r in range(Vtil.shape[0]) for c in range(Vtil.shape[1])]
        flat.sort(key=lambda x: x[1], reverse=True)
        top5 = ", ".join(f"({r},{c},{v:.2f})" for (r,c),v in flat[:5])
        bot5 = ", ".join(f"({r},{c},{v:.2f})" for (r,c),v in flat[-5:])
        self.log(f"TOP5 {top5}", "CELLS")
        self.log(f"BOT5 {bot5}", "CELLS")

        # 3️⃣ Per-stack local environment summary
        for (r, c), s in self._our_stacks(state):
            cell = state.grid[r, c]
            H_here = cell.humans
            E_near = sum(
                self._enemy_count(state.grid[r+dr, c+dc]) > 0
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if state.in_bounds(r+dr, c+dc)
            )
            H_near = sum(
                state.grid[r+dr, c+dc].humans
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if state.in_bounds(r+dr, c+dc)
            )
            self.log(
                f"STACK ({r},{c}) s={s}, H_here={H_here}, H_near={H_near}, enemies_around={E_near}, Vtil_here={Vtil[r,c]:.2f}",
                "STACK",
                indent=2
            )

        # 4️⃣ Final move decisions
        self.log(f"Selected {len(moves)} moves this turn:", "DECISION")
        for (r1, c1, num, r2, c2) in moves:
            self.log(f"({r1},{c1}) → ({r2},{c2}) with {num}", "MOVE", indent=4)


    # --- optional: expose short debug strings for GUI logs ----------
    def debug_messages(self) -> List[str]:
        return list(self._last_debug)

    def get_heatmap(self):
        return getattr(self, "_last_heatmap", None)

    def _best_path_heat(self, r, c):
        """
        Optional: build a dummy 2-D array highlighting a path
        from (r,c) toward its top target for visualization.
        """
        if not hasattr(self, "_last_heatmap"):
            return None
        import numpy as np
        heat = np.copy(self._last_heatmap)
        R, C = heat.shape
        rr, cc = r, c
        for _ in range(8):
            # move greedily toward global max
            best = None; bestv = -1e9
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < R and 0 <= c2 < C and heat[r2, c2] > bestv:
                        bestv = heat[r2, c2]; best = (r2, c2)
            if best is None:
                break
            rr, cc = best
            heat[rr, cc] = heat.max()  # highlight path
        return heat
    
    def select_action(self, state: GameState):
        """
        Compatibility wrapper so that board.py can use this agent
        exactly like MCTSAgent.
        """
        self.begin_turn(state, "W")  # assuming AI controls Werewolves
        moves = self.propose_moves(state)

        # 🔹 reorder (r, c, num, r2, c2) → (r, c, r2, c2, num)
        moves = [(r, c, r2, c2, num) for (r, c, num, r2, c2) in moves]
        return moves


