# core/alphabeta_agent_v5.py
"""
Alpha-Beta Agent V5 for Vampires vs Werewolves.

BASED ON V3 with one addition:
- ADJACENT SPLIT: If 2+ human groups at distance 1 in different directions,
  enemy at distance >= 3, and we can capture each safely (our units >= humans),
  then split to capture all adjacent humans in one turn.

Strategy (from V3):
- Each group acts INDEPENDENTLY with proportional time budget
- ATTACK ENEMY when we can kill them safely (our_count >= 1.5 * enemies)
- Capture humans safely (our_count >= humans)
- Flee from dangerous enemies (enemies >= 1.5 * our_count)
- Split based on nearby human groups in different directions
- Merge only when threatened or groups have no targets
"""

from __future__ import annotations
import copy
import time
from typing import List, Tuple, Optional, Dict
from core.agent_base import Agent

# Type aliases
Move = Tuple[int, int, int, int, int]  # (r1, c1, r2, c2, num) - external format
InternalMove = Tuple[int, int, int, int, int]  # (r1, c1, num, r2, c2) for state.move_group


def chebyshev_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """Chebyshev distance (king moves)."""
    return max(abs(r1 - r2), abs(c1 - c2))


class AlphaBetaAgentV5(Agent):
    """Alpha-Beta agent V5 with V3 base + adjacent split."""

    def __init__(self, time_limit: float = 1.9, max_depth: int = 6):
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.log: List[str] = []
        
        # Time management
        self._start_time = 0.0
        self._group_time_limit = 0.0
        self._group_start_time = 0.0
        self._nodes_evaluated = 0
        
        # Anti-oscillation per group
        self._group_history: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        
        # Group objectives (for persistent targeting)
        self._group_objectives: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Map analysis
        self._map_analyzed = False
        self._is_corridor = False
        self._prefer_split = False

    def name(self) -> str:
        return "AlphaBetaAgentV5"

    # ==================== TIME MANAGEMENT ====================

    def _total_time_remaining(self) -> float:
        """Time remaining for entire turn."""
        return self.time_limit - (time.time() - self._start_time)

    def _group_time_remaining(self) -> float:
        """Time remaining for current group."""
        return self._group_time_limit - (time.time() - self._group_start_time)

    def _is_timeout(self) -> bool:
        """Check if current group has run out of time."""
        return self._group_time_remaining() <= 0.03

    # ==================== UTILITY FUNCTIONS ====================

    def _get_positions(self, state) -> Tuple[List, List, List]:
        """Get all positions: (my_pos, enemy_pos, human_pos)."""
        my_pos, enemy_pos, human_pos = [], [], []
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                if state.turn == "V":
                    if cell.vampires > 0:
                        my_pos.append((r, c, cell.vampires))
                    if cell.werewolves > 0:
                        enemy_pos.append((r, c, cell.werewolves))
                else:
                    if cell.werewolves > 0:
                        my_pos.append((r, c, cell.werewolves))
                    if cell.vampires > 0:
                        enemy_pos.append((r, c, cell.vampires))
                if cell.humans > 0:
                    human_pos.append((r, c, cell.humans))
        
        return my_pos, enemy_pos, human_pos

    def _get_totals(self, state) -> Tuple[int, int, int]:
        """Get total counts: (humans, my_total, enemy_total)."""
        humans, my_total, enemy_total = 0, 0, 0
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                humans += cell.humans
                if state.turn == "V":
                    my_total += cell.vampires
                    enemy_total += cell.werewolves
                else:
                    my_total += cell.werewolves
                    enemy_total += cell.vampires
        
        return humans, my_total, enemy_total

    def _can_kill_enemy_safely(self, my_count: int, enemy_count: int) -> bool:
        """Can we kill the enemy safely? (guaranteed win)"""
        return my_count >= 1.5 * enemy_count

    def _enemy_can_kill_us(self, my_count: int, enemy_count: int) -> bool:
        """Can enemy kill us? (we'd lose for sure)"""
        return enemy_count >= 1.5 * my_count

    def _can_capture_humans(self, my_count: int, human_count: int) -> bool:
        """Can we capture humans safely? (guaranteed conversion)"""
        return my_count >= human_count

    # ==================== MAP ANALYSIS ====================

    def _analyze_map(self, state, human_pos: List) -> None:
        """Analyze map to determine strategy preferences."""
        if self._map_analyzed:
            return
        
        # Check if map is corridor-like (narrow)
        self._is_corridor = min(state.rows, state.cols) <= 5
        
        # Prefer split on larger maps with spread humans
        if len(human_pos) >= 4 and not self._is_corridor:
            self._prefer_split = True
        
        self._map_analyzed = True

    # ==================== SAFETY CHECKS ====================

    def _is_position_safe(self, r: int, c: int, my_count: int, enemy_pos: List) -> bool:
        """Check if position is safe (no dangerous enemy nearby)."""
        for er, ec, ecount in enemy_pos:
            dist = chebyshev_distance(r, c, er, ec)
            if dist <= 2 and self._enemy_can_kill_us(my_count, ecount):
                return False
        return True

    def _min_enemy_distance(self, r: int, c: int, enemy_pos: List) -> int:
        """Minimum distance to any enemy."""
        if not enemy_pos:
            return 999
        return min(chebyshev_distance(r, c, er, ec) for er, ec, _ in enemy_pos)

    # ==================== ADJACENT SPLIT (NEW IN V5) ====================

    def _calculate_adjacent_split(self, state, mr: int, mc: int, mcount: int,
                                   enemy_pos: List, human_pos: List) -> Optional[List[Move]]:
        """
        V5 FEATURE: Split to capture 2+ adjacent human groups in one turn.
        
        Conditions:
        - 2+ human groups at distance 1 (adjacent)
        - Enemy at distance >= 3
        - We have exact minimum units to capture each group safely
        """
        if mcount < 2:
            return None  # Need at least 2 units to split
        
        # Check enemy distance
        min_enemy_dist = self._min_enemy_distance(mr, mc, enemy_pos)
        if min_enemy_dist < 3:
            return None  # Enemy too close, don't split
        
        # Find adjacent human groups (distance 1)
        adjacent_humans = []
        for hr, hc, hcount in human_pos:
            dist = chebyshev_distance(mr, mc, hr, hc)
            if dist == 1:  # Adjacent
                adjacent_humans.append((hr, hc, hcount))
        
        if len(adjacent_humans) < 2:
            return None  # Need at least 2 adjacent groups
        
        # Sort by human count (smallest first - easier to capture)
        adjacent_humans.sort(key=lambda x: x[2])
        
        # Check if we can capture all adjacent humans with exact minimum units
        total_needed = sum(h[2] for h in adjacent_humans)
        if mcount < total_needed:
            return None  # Not enough units to capture all
        
        # Build split moves: each group gets exactly the humans at target
        moves = []
        remaining = mcount
        
        for hr, hc, hcount in adjacent_humans:
            if remaining < hcount:
                break  # Can't capture this group
            
            # Send exactly hcount units (minimum needed for safe capture)
            units_to_send = hcount
            moves.append((mr, mc, hr, hc, units_to_send))
            remaining -= units_to_send
        
        if len(moves) < 2:
            return None  # Couldn't make at least 2 valid captures
        
        # If we have remaining units, add them to the largest capture group
        if remaining > 0:
            # Find the move with the largest target and add remaining units there
            max_idx = 0
            max_humans = 0
            for i, (_, _, hr, hc, _) in enumerate(moves):
                for ah, ac, acount in adjacent_humans:
                    if ah == hr and ac == hc and acount > max_humans:
                        max_humans = acount
                        max_idx = i
                        break
            
            r1, c1, r2, c2, num = moves[max_idx]
            moves[max_idx] = (r1, c1, r2, c2, num + remaining)
        
        self.log.append(f"[ADJACENT SPLIT] {mcount} units at ({mr},{mc}) -> {len(moves)} groups to capture {len(moves)} adjacent human groups")
        for move in moves:
            self.log.append(f"  -> ({move[2]},{move[3]}) with {move[4]} units")
        
        return moves

    # ==================== SPLIT STRATEGY (FROM V3) ====================

    def _calculate_split(self, state, mr: int, mc: int, mcount: int,
                         enemy_pos: List, human_pos: List) -> Optional[List[Move]]:
        """
        Calculate if we should split based on direction and human groups.
        Returns list of moves if split is beneficial, None otherwise.
        """
        # Need enough units to split meaningfully
        if mcount < 6:
            return None
        
        # Check enemy distance - only split if safe
        min_enemy_dist = self._min_enemy_distance(mr, mc, enemy_pos)
        
        # If enemy is close, need more advantage to split
        if min_enemy_dist <= 3:
            # Only split if we have 2x the total enemy
            total_enemy = sum(e[2] for e in enemy_pos) if enemy_pos else 0
            if mcount < 2 * total_enemy:
                return None
        elif min_enemy_dist <= 5:
            # Moderate distance - need 1.5x advantage
            total_enemy = sum(e[2] for e in enemy_pos) if enemy_pos else 0
            if mcount < 1.5 * total_enemy:
                return None
        
        # Group humans by direction from us
        directions = {}  # direction -> list of (dist, r, c, count)
        
        for hr, hc, hcount in human_pos:
            dist = chebyshev_distance(mr, mc, hr, hc)
            if dist > 8:  # Too far, don't consider for split
                continue
            
            # Determine direction (simplified: 4 quadrants + center)
            dr = 0 if hr == mr else (1 if hr > mr else -1)
            dc = 0 if hc == mc else (1 if hc > mc else -1)
            direction = (dr, dc)
            
            if direction not in directions:
                directions[direction] = []
            directions[direction].append((dist, hr, hc, hcount))
        
        # Need at least 2 different directions to split
        if len(directions) < 2:
            return None
        
        # Sort directions by total humans (most valuable first)
        dir_values = []
        for direction, targets in directions.items():
            total_humans = sum(t[3] for t in targets)
            min_dist = min(t[0] for t in targets)
            dir_values.append((total_humans, min_dist, direction, targets))
        
        dir_values.sort(key=lambda x: (-x[0], x[1]))  # Most humans, then closest
        
        # Take top 2 directions
        if len(dir_values) < 2:
            return None
        
        # Calculate split based on human counts
        dir1_humans = dir_values[0][0]
        dir2_humans = dir_values[1][0]
        total_target = dir1_humans + dir2_humans
        
        if total_target == 0:
            return None
        
        # Allocate units proportionally
        units1 = max(2, int(mcount * dir1_humans / total_target))
        units2 = mcount - units1
        
        if units2 < 2:
            return None
        
        # Get target positions (closest in each direction)
        target1 = min(dir_values[0][3], key=lambda x: x[0])
        target2 = min(dir_values[1][3], key=lambda x: x[0])
        
        # Check if each split group can actually capture their target
        if not self._can_capture_humans(units1, target1[3]):
            return None
        if not self._can_capture_humans(units2, target2[3]):
            return None
        
        # Move towards targets (one step)
        def step_towards(r1, c1, r2, c2):
            dr = 0 if r2 == r1 else (1 if r2 > r1 else -1)
            dc = 0 if c2 == c1 else (1 if c2 > c1 else -1)
            return r1 + dr, c1 + dc
        
        nr1, nc1 = step_towards(mr, mc, target1[1], target1[2])
        nr2, nc2 = step_towards(mr, mc, target2[1], target2[2])
        
        # Verify positions are valid and different
        if (nr1, nc1) == (nr2, nc2):
            return None
        if not (0 <= nr1 < state.rows and 0 <= nc1 < state.cols):
            return None
        if not (0 <= nr2 < state.rows and 0 <= nc2 < state.cols):
            return None
        
        # Check safety of new positions
        if not self._is_position_safe(nr1, nc1, units1, enemy_pos):
            return None
        if not self._is_position_safe(nr2, nc2, units2, enemy_pos):
            return None
        
        self.log.append(f"[SPLIT] {mcount} at ({mr},{mc}) -> ({nr1},{nc1}) x{units1}, ({nr2},{nc2}) x{units2}")
        
        moves = [
            (mr, mc, nr1, nc1, units1),
            (mr, mc, nr2, nc2, units2)
        ]
        
        return moves

    # ==================== MERGE STRATEGY ====================

    def _should_groups_merge(self, state, my_pos: List, enemy_pos: List, 
                              human_pos: List) -> Optional[List[Move]]:
        """
        Check if groups should merge (threatened or no targets).
        Returns list of merge moves if merging, None otherwise.
        """
        if len(my_pos) < 2:
            return None
        
        # Check if any group is threatened
        threatened = []
        for mr, mc, mcount in my_pos:
            for er, ec, ecount in enemy_pos:
                dist = chebyshev_distance(mr, mc, er, ec)
                if dist <= 3 and self._enemy_can_kill_us(mcount, ecount):
                    threatened.append((mr, mc, mcount))
                    break
        
        # If any group is threatened, merge all to strongest position
        if threatened:
            # Find safest position (furthest from dangerous enemies)
            best_pos = None
            best_safety = -1
            
            for mr, mc, mcount in my_pos:
                min_danger_dist = 999
                for er, ec, ecount in enemy_pos:
                    if self._enemy_can_kill_us(sum(g[2] for g in my_pos), ecount):
                        dist = chebyshev_distance(mr, mc, er, ec)
                        min_danger_dist = min(min_danger_dist, dist)
                
                if min_danger_dist > best_safety:
                    best_safety = min_danger_dist
                    best_pos = (mr, mc, mcount)
            
            if best_pos is None:
                best_pos = my_pos[0]
            
            # All groups move to best position
            moves = []
            target_r, target_c, _ = best_pos
            
            for mr, mc, mcount in my_pos:
                if (mr, mc) == (target_r, target_c):
                    # This is the target, it can stay or move away from danger
                    if enemy_pos:
                        # Move away from nearest enemy
                        nearest_enemy = min(enemy_pos, key=lambda e: chebyshev_distance(mr, mc, e[0], e[1]))
                        er, ec = nearest_enemy[0], nearest_enemy[1]
                        dr = -1 if er > mr else (1 if er < mr else 0)
                        dc = -1 if ec > mc else (1 if ec < mc else 0)
                        nr, nc = mr + dr, mc + dc
                        nr = max(0, min(state.rows - 1, nr))
                        nc = max(0, min(state.cols - 1, nc))
                        moves.append((mr, mc, nr, nc, mcount))
                    else:
                        moves.append((mr, mc, mr, mc, mcount))  # Stay
                else:
                    # Move towards target
                    dr = 0 if target_r == mr else (1 if target_r > mr else -1)
                    dc = 0 if target_c == mc else (1 if target_c > mc else -1)
                    nr, nc = mr + dr, mc + dc
                    moves.append((mr, mc, nr, nc, mcount))
            
            self.log.append(f"[MERGE] Threatened groups merging to ({target_r},{target_c})")
            return moves
        
        # Check if groups have no nearby targets - merge to hunt together
        groups_without_targets = 0
        for mr, mc, mcount in my_pos:
            has_target = False
            
            # Check for capturable humans nearby
            for hr, hc, hcount in human_pos:
                if chebyshev_distance(mr, mc, hr, hc) <= 5 and self._can_capture_humans(mcount, hcount):
                    has_target = True
                    break
            
            # Check for killable enemies nearby
            if not has_target:
                for er, ec, ecount in enemy_pos:
                    if chebyshev_distance(mr, mc, er, ec) <= 5 and self._can_kill_enemy_safely(mcount, ecount):
                        has_target = True
                        break
            
            if not has_target:
                groups_without_targets += 1
        
        # If most groups have no targets, merge
        if groups_without_targets >= len(my_pos) * 0.6:
            # Merge to largest group position
            largest = max(my_pos, key=lambda x: x[2])
            target_r, target_c, _ = largest
            
            moves = []
            for mr, mc, mcount in my_pos:
                if (mr, mc) == (target_r, target_c):
                    # Largest stays or moves towards nearest target
                    best_target = None
                    best_dist = 999
                    
                    for hr, hc, hcount in human_pos:
                        dist = chebyshev_distance(mr, mc, hr, hc)
                        if dist < best_dist:
                            best_dist = dist
                            best_target = (hr, hc)
                    
                    for er, ec, ecount in enemy_pos:
                        dist = chebyshev_distance(mr, mc, er, ec)
                        if dist < best_dist:
                            best_dist = dist
                            best_target = (er, ec)
                    
                    if best_target:
                        tr, tc = best_target
                        dr = 0 if tr == mr else (1 if tr > mr else -1)
                        dc = 0 if tc == mc else (1 if tc > mc else -1)
                        nr, nc = mr + dr, mc + dc
                        moves.append((mr, mc, nr, nc, mcount))
                    else:
                        moves.append((mr, mc, mr, mc, mcount))
                else:
                    dr = 0 if target_r == mr else (1 if target_r > mr else -1)
                    dc = 0 if target_c == mc else (1 if target_c > mc else -1)
                    nr, nc = mr + dr, mc + dc
                    moves.append((mr, mc, nr, nc, mcount))
            
            self.log.append(f"[MERGE] No targets, merging to ({target_r},{target_c})")
            return moves
        
        return None

    # ==================== MOVE GENERATION ====================

    def _get_possible_moves(self, state, mr: int, mc: int, mcount: int) -> List[Move]:
        """Get all possible moves for a group."""
        moves = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = mr + dr, mc + dc
                if 0 <= nr < state.rows and 0 <= nc < state.cols:
                    if dr == 0 and dc == 0:
                        # Stay in place
                        moves.append((mr, mc, nr, nc, mcount))
                    else:
                        # Move all units
                        moves.append((mr, mc, nr, nc, mcount))
        
        return moves

    def _score_move_for_group(self, state, move: Move, 
                               enemy_pos: List, human_pos: List,
                               my_total: int, enemy_total: int) -> float:
        """
        Score a move for a single group.
        AGGRESSIVE: Prioritizes killing enemy when safe.
        """
        r1, c1, r2, c2, num = move
        dst = state.grid[r2, c2]
        
        enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
        humans_at = dst.humans
        
        # CRITICAL: Check for suicide FIRST - reject any move that kills us
        if enemy_at > 0 and self._enemy_can_kill_us(num, enemy_at):
            return -50000  # Very bad - would die
        
        # Priority 1: Kill enemy if we can do it safely (HIGHEST)
        if enemy_at > 0 and self._can_kill_enemy_safely(num, enemy_at):
            return 25000 + enemy_at * 200  # Huge bonus for safe kills
        
        # Priority 2: Capture humans safely
        if humans_at > 0 and self._can_capture_humans(num, humans_at):
            return 20000 + humans_at * 100
        
        # Priority 3: Flee from dangerous enemies
        for er, ec, ecount in enemy_pos:
            if self._enemy_can_kill_us(num, ecount):
                current_dist = chebyshev_distance(r1, c1, er, ec)
                new_dist = chebyshev_distance(r2, c2, er, ec)
                if new_dist > current_dist:
                    return 12000 + new_dist * 10
        
        # Priority 4: Hunt enemy we can kill (move towards them)
        for er, ec, ecount in enemy_pos:
            if self._can_kill_enemy_safely(num, ecount):
                current_dist = chebyshev_distance(r1, c1, er, ec)
                new_dist = chebyshev_distance(r2, c2, er, ec)
                if new_dist < current_dist:
                    return 10000 + (10 - new_dist) * 50  # Closer = better
        
        # Priority 5: Move towards capturable humans
        best_human_score = 0
        for hr, hc, hcount in human_pos:
            if self._can_capture_humans(num, hcount):
                current_dist = chebyshev_distance(r1, c1, hr, hc)
                new_dist = chebyshev_distance(r2, c2, hr, hc)
                if new_dist < current_dist:
                    score = 5000 + hcount * 50 + (10 - new_dist) * 20
                    best_human_score = max(best_human_score, score)
        
        if best_human_score > 0:
            return best_human_score
        
        # Fallback: Move towards center (general exploration)
        center_r, center_c = state.rows // 2, state.cols // 2
        current_center_dist = chebyshev_distance(r1, c1, center_r, center_c)
        new_center_dist = chebyshev_distance(r2, c2, center_r, center_c)
        
        return 100 + (current_center_dist - new_center_dist) * 10

    def _get_fallback_move(self, state, mr: int, mc: int, mcount: int,
                           enemy_pos: List, human_pos: List) -> Optional[Move]:
        """Get a quick fallback move using simple heuristics."""
        moves = self._get_possible_moves(state, mr, mc, mcount)
        
        if not moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        _, my_total, enemy_total = self._get_totals(state)
        
        for move in moves:
            score = self._score_move_for_group(state, move, enemy_pos, human_pos, my_total, enemy_total)
            
            # Anti-oscillation penalty
            _, _, nr, nc, _ = move
            history = self._group_history.get((mr, mc), [])
            if (nr, nc) in history[-3:]:
                score -= 1000
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    # ==================== EVALUATION FUNCTION ====================

    def _evaluate_state(self, state, root_player: str, my_total: int, enemy_total: int) -> float:
        """Evaluate state from perspective of root_player."""
        
        # Count current state
        my_count = 0
        enemy_count = 0
        humans = 0
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                humans += cell.humans
                if root_player == "V":
                    my_count += cell.vampires
                    enemy_count += cell.werewolves
                else:
                    my_count += cell.werewolves
                    enemy_count += cell.vampires
        
        # Victory conditions
        if enemy_count == 0 and my_count > 0:
            return 100000  # Win
        if my_count == 0 and enemy_count > 0:
            return -100000  # Loss
        if my_count == 0 and enemy_count == 0:
            return 0  # Draw (both eliminated)
        
        # Advantage based on unit counts
        score = (my_count - enemy_count) * 100
        
        # Bonus for having more than enemy
        if my_count > enemy_count:
            score += 500
        elif enemy_count > my_count:
            score -= 500
        
        # Consider humans as potential gain
        score += humans * 10  # Humans are valuable
        
        return score

    # ==================== ALPHA-BETA SEARCH ====================

    def _alphabeta_for_group(self, state, mr: int, mc: int, mcount: int,
                              depth: int, alpha: float, beta: float,
                              maximizing: bool, root_player: str,
                              enemy_pos: List, human_pos: List,
                              my_total: int, enemy_total: int) -> Tuple[float, Optional[InternalMove]]:
        """Alpha-Beta for a single group's decision."""
        self._nodes_evaluated += 1
        
        if self._is_timeout():
            return self._evaluate_state(state, root_player, my_total, enemy_total), None
        
        if depth == 0:
            return self._evaluate_state(state, root_player, my_total, enemy_total), None
        
        # Generate moves for this group
        moves = self._get_possible_moves(state, mr, mc, mcount)
        
        if not moves:
            return self._evaluate_state(state, root_player, my_total, enemy_total), None
        
        # Score and sort moves for better pruning
        scored_moves = []
        for move in moves:
            r1, c1, r2, c2, num = move
            score = self._score_move_for_group(state, move, enemy_pos, human_pos, my_total, enemy_total)
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=maximizing, key=lambda x: x[0])
        
        if maximizing:
            max_eval = float('-inf')
            best_move = None
            
            for _, move in scored_moves:
                if self._is_timeout():
                    break
                
                r1, c1, r2, c2, num = move
                
                # Try move
                child = copy.deepcopy(state)
                internal_move = (r1, c1, num, r2, c2)
                child.move_group(*internal_move)
                
                # Don't swap turns - we're searching for this group only
                eval_score, _ = self._alphabeta_for_group(
                    child, r2, c2, num,  # Updated position
                    depth - 1, alpha, beta, False, root_player,
                    enemy_pos, human_pos, my_total, enemy_total
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = internal_move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        
        else:
            min_eval = float('inf')
            best_move = None
            
            for _, move in scored_moves:
                if self._is_timeout():
                    break
                
                r1, c1, r2, c2, num = move
                
                child = copy.deepcopy(state)
                internal_move = (r1, c1, num, r2, c2)
                child.move_group(*internal_move)
                
                eval_score, _ = self._alphabeta_for_group(
                    child, r2, c2, num,
                    depth - 1, alpha, beta, True, root_player,
                    enemy_pos, human_pos, my_total, enemy_total
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move

    def _search_best_move_for_group(self, state, mr: int, mc: int, mcount: int,
                                     time_budget: float, enemy_pos, human_pos,
                                     my_total: int, enemy_total: int) -> Optional[Move]:
        """Find best move for a single group with given time budget."""
        self._group_start_time = time.time()
        self._group_time_limit = time_budget
        self._nodes_evaluated = 0
        
        root_player = state.turn
        
        # Get fallback immediately
        fallback = self._get_fallback_move(state, mr, mc, mcount, enemy_pos, human_pos)
        if fallback is None:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        # Iterative deepening with time budget
        for depth in range(1, self.max_depth + 1):
            if self._group_time_remaining() < 0.1:
                break
            
            score, move = self._alphabeta_for_group(
                copy.deepcopy(state), mr, mc, mcount,
                depth, float('-inf'), float('inf'),
                True, root_player,
                enemy_pos, human_pos, my_total, enemy_total
            )
            
            if move is not None and not self._is_timeout():
                best_move = move
                best_score = score
        
        if best_move is None:
            return fallback
        
        # Convert internal format to external
        r1, c1, num, r2, c2 = best_move
        
        # Verify the move is actually from this group
        if r1 != mr or c1 != mc:
            self.log.append(f"[G({mr},{mc})] WARNING: Got move from ({r1},{c1}), using fallback")
            return fallback
        
        # Safety check
        dst = state.grid[r2, c2]
        enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
        if enemy_at > 0 and self._enemy_can_kill_us(num, enemy_at):
            return fallback
        
        # Log attack if we're attacking enemy
        if enemy_at > 0 and self._can_kill_enemy_safely(num, enemy_at):
            self.log.append(f"[ATTACK] Killing {enemy_at} enemies at ({r2},{c2})!")
        
        elapsed = time.time() - self._group_start_time
        self.log.append(f"[G({mr},{mc}) x{mcount}] Best: ({r1},{c1})->({r2},{c2}) score={best_score:.3f} time={elapsed:.2f}s nodes={self._nodes_evaluated}")
        
        return (r1, c1, r2, c2, num)

    # ==================== MAIN ENTRY POINT ====================

    def select_action(self, state) -> List[Move]:
        """Select best action(s) for current turn - each group gets proportional time."""
        self._start_time = time.time()
        self.log.clear()
        
        root_player = state.turn
        my_pos, enemy_pos, human_pos = self._get_positions(state)
        _, my_total, enemy_total = self._get_totals(state)
        
        if not my_pos:
            self.log.append("[AI] No units!")
            return []
        
        # Analyze map once
        self._analyze_map(state, human_pos)
        
        # Sort groups by size (biggest first = more important)
        my_pos.sort(key=lambda x: x[2], reverse=True)
        total_units = sum(g[2] for g in my_pos)
        
        # === STRATEGIC DECISIONS ===
        
        # 1. Check for emergency MERGE
        if len(my_pos) >= 2:
            merge_moves = self._should_groups_merge(state, my_pos, enemy_pos, human_pos)
            if merge_moves:
                return merge_moves
        
        # 2. V5 NEW: Check for ADJACENT SPLIT (priority over normal split)
        if len(my_pos) == 1 and human_pos:
            main = my_pos[0]
            mr, mc, mcount = main
            adjacent_split = self._calculate_adjacent_split(state, mr, mc, mcount, enemy_pos, human_pos)
            if adjacent_split:
                # Update history for new positions
                for move in adjacent_split:
                    _, _, nr, nc, _ = move
                    self._group_history[(nr, nc)] = []
                return adjacent_split
        
        # 3. Check for SPLIT (only if we have 1-2 groups)
        if len(my_pos) <= 2 and human_pos:
            main = my_pos[0]
            mr, mc, mcount = main
            split_moves = self._calculate_split(state, mr, mc, mcount, enemy_pos, human_pos)
            if split_moves:
                # Update history for new positions
                for move in split_moves:
                    _, _, nr, nc, _ = move
                    self._group_history[(nr, nc)] = []
                return split_moves
        
        # === PER-GROUP ALPHA-BETA ===
        
        all_moves = []
        time_used = 0.0
        
        for i, (mr, mc, mcount) in enumerate(my_pos):
            # Calculate time budget proportional to group size
            remaining_time = self.time_limit * 0.95 - time_used
            if remaining_time <= 0.1:
                # Use quick fallback for remaining groups
                fallback = self._get_fallback_move(state, mr, mc, mcount, enemy_pos, human_pos)
                if fallback:
                    all_moves.append(fallback)
                    self.log.append(f"[G({mr},{mc}) x{mcount}] Fallback (time exhausted)")
                continue
            
            # Proportional time based on remaining units
            remaining_units = sum(g[2] for g in my_pos[i:])
            time_budget = (mcount / remaining_units) * remaining_time
            time_budget = max(0.15, min(time_budget, remaining_time - 0.1))  # At least 150ms, leave buffer
            
            self.log.append(f"[G({mr},{mc}) x{mcount}] Time budget: {time_budget:.2f}s")
            
            # Search for best move
            start = time.time()
            best_move = self._search_best_move_for_group(
                state, mr, mc, mcount, time_budget,
                enemy_pos, human_pos, my_total, enemy_total
            )
            time_used += time.time() - start
            
            if best_move:
                all_moves.append(best_move)
                # Update history
                r1, c1, r2, c2, num = best_move
                if (r1, c1) not in self._group_history:
                    self._group_history[(r1, c1)] = []
                self._group_history[(r1, c1)].append((r2, c2))
                if len(self._group_history[(r1, c1)]) > 6:
                    self._group_history[(r1, c1)].pop(0)
        
        if not all_moves:
            self.log.append("[AI] No valid moves found!")
            return []
        
        total_elapsed = time.time() - self._start_time
        self.log.append(f"[AI] Total: {len(all_moves)} moves in {total_elapsed:.2f}s")
        
        return all_moves
