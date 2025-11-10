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

class HeuristicAgent(Agent):
    def __init__(self, params: Optional[HeuristicParams]=None):
        self.p = params or HeuristicParams()
        self.side = "V"
        self._log = []

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

        moves = []
        used = set()
        for (r,c), s in self._our_stacks(state):
            if s < 2: continue
            if state.movable[r,c] <= 0: continue

            best = self._best_step(state, r, c, s, V, used)
            if best:
                r2, c2 = best
                moves.append((r,c,s,r2,c2))
                used.add((r2,c2))

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
                    if h <= 0: continue
                    d = max(abs(r-i), abs(c-j))
                    if s >= h:
                        gain = h
                    else:
                        gain = s / (2*h)
                    Hscore[i,j] += gain * math.exp(-self.p.lambda_K * d)
        return Hscore

    def _score_enemies(self, state: GameState):
        R,C = state.rows, state.cols
        Escore = np.zeros((R,C))
        for (r,c), s in self._our_stacks(state):
            for i in range(R):
                for j in range(C):
                    e = self._enemy_at(state, i,j)
                    if e <= 0: continue
                    d = max(abs(r-i), abs(c-j))
                    if s >= 1.5*e: 
                        Escore[i,j] += 1.0 * math.exp(-self.p.lambda_K * d) # good attack
                    elif e >= 1.5*s:
                        Escore[r,c] -= 1.0 * math.exp(-self.p.lambda_K * d) # danger at our own cell
        return Escore

    def _score_joining(self, state: GameState):
        R,C = state.rows, state.cols
        J = np.zeros((R,C))
        for (r,c), s in self._our_stacks(state):
            # only encourage merging if no free human/enemy capture nearby
            if self._has_easy_target_near(state,r,c,s): 
                continue
            for (r2,c2), s2 in self._our_stacks(state):
                if (r,c)==(r2,c2): continue
                d = max(abs(r-r2),abs(c-c2))
                if d == 1:
                    J[r2,c2] += self.p.join_bonus
        return J

    # ----------------- MOVE SELECTION -----------------
    def _best_step(self, state: GameState, r,c,s,V,used):
        best=None; bestv=-1e9
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                rr,cc=r+dr,c+dc
                if not state.in_bounds(rr,cc): continue
                if (rr,cc) in used: continue
                val = V[rr,cc]
                if val>bestv:
                    bestv=val; best=(rr,cc)
        return best

    # ----------------- HELPERS -----------------
    def _our_stacks(self, state:GameState):
        for r in range(state.rows):
            for c in range(state.cols):
                s = state.grid[r,c].vampires if self.side=="V" else state.grid[r,c].werewolves
                if s>0: yield (r,c),s

    def _enemy_at(self, state,r,c):
        cell = state.grid[r,c]
        return cell.werewolves if self.side=="V" else cell.vampires

    def _has_easy_target_near(self,state,r,c,s):
        # easy humans
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr,cc=r+dr,c+dc
                if state.in_bounds(rr,cc):
                    h=state.grid[rr,cc].humans
                    if h>0 and s>=h: return True
                    e=self._enemy_at(state,rr,cc)
                    if e>0 and s>=1.5*e: return True
        return False

    # ----------------- LOGGING -----------------
    def _log_state(self,state,UH,UM,UJ,V,moves):
        self._log.append("=== Turn Log ===")
        self._log.append("Top valued cells:")
        flat=[((r,c),V[r,c]) for r in range(state.rows) for c in range(state.cols)]
        flat.sort(key=lambda x:x[1],reverse=True)
        for ((r,c),v) in flat[:6]:
            self._log.append(f"({r},{c}) = {v:.2f}")

        self._log.append("Moves:")
        for m in moves:
            self._log.append(str(m))

    def debug_messages(self):
        return self._log

    def get_heatmap(self):
        return getattr(self,"_last_heatmap",None)

    def select_action(self,state:GameState):
        self.begin_turn(state,state.turn)
        moves = self.propose_moves(state)
        return [(r,c,r2,c2,num) for (r,c,num,r2,c2) in moves]
