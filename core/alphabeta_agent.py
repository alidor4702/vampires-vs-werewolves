# core/alphabeta_agent_v2.py
"""
Alpha-Beta Agent V2 for Vampires vs Werewolves.

FIX: Each group now only considers its OWN moves in alpha-beta search,
preventing the bug where multiple groups would suggest the same move.

Strategy:
- Each group acts INDEPENDENTLY with proportional time budget
- Capture humans safely (our_count >= humans)
- Attack enemies only when guaranteed (our_count >= 1.5 * enemies)
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


class AlphaBetaAgentV2(Agent):
    """Alpha-Beta agent V2 with fixed per-group move selection."""

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
        return "AlphaBetaAgentV2"

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
        """Get (total_humans, my_total, enemy_total)."""
        H, V, W = state.population_counts()
        if state.turn == "V":
            return H, V, W
        else:
            return H, W, V

    # ==================== MAP ANALYSIS ====================

    def _analyze_map(self, state, human_pos):
        """Analyze map characteristics once at the start."""
        if self._map_analyzed:
            return
        
        self._map_analyzed = True
        map_size = state.rows * state.cols
        
        # Corridor detection
        self._is_corridor = min(state.rows, state.cols) <= 5
        
        # Large maps with many humans -> prefer split
        if human_pos:
            self._prefer_split = len(human_pos) >= 6 and map_size >= 64
        
        self.log.append(f"[MAP] Size: {state.rows}x{state.cols}, Humans: {len(human_pos)}, Split pref: {self._prefer_split}")

    # ==================== SAFETY CHECKS ====================

    def _can_capture_humans_safely(self, my_count: int, human_count: int) -> bool:
        """Can we capture humans without risk?"""
        return my_count >= human_count

    def _can_kill_enemy_safely(self, my_count: int, enemy_count: int) -> bool:
        """Can we kill enemy guaranteed?"""
        return my_count >= 1.5 * enemy_count

    def _enemy_can_kill_us(self, my_count: int, enemy_count: int) -> bool:
        """Can enemy kill us guaranteed?"""
        return enemy_count >= 1.5 * my_count

    def _is_oscillating(self, group_pos: Tuple[int, int], r2: int, c2: int) -> bool:
        """Check if move creates oscillation pattern for this group."""
        history = self._group_history.get(group_pos, [])
        if (r2, c2) in history[-4:]:
            return True
        return False

    # ==================== SPLIT STRATEGY ====================

    def _calculate_split(self, state, mr: int, mc: int, mcount: int, 
                         enemy_pos, human_pos) -> Optional[List[Move]]:
        """
        Calculate optimal split based on nearby human groups.
        Returns split moves or None if split not advisable.
        """
        # Minimum size to split
        if mcount < 6:
            return None
        
        # Check enemy distance - don't split if enemy is close
        min_enemy_dist = float('inf')
        closest_enemy_count = 0
        if enemy_pos:
            for er, ec, ecount in enemy_pos:
                d = chebyshev_distance(mr, mc, er, ec)
                if d < min_enemy_dist:
                    min_enemy_dist = d
                    closest_enemy_count = ecount
        
        # Safety checks
        if min_enemy_dist <= 3:
            # Enemy very close - don't split unless we have overwhelming advantage
            if mcount < closest_enemy_count * 2:
                return None
        elif min_enemy_dist <= 5:
            # Enemy moderately close - only split if we're stronger
            if mcount < closest_enemy_count * 1.5:
                return None
        
        # Find capturable human groups and group by direction
        direction_targets: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        
        for hr, hc, hcount in human_pos:
            dist = chebyshev_distance(mr, mc, hr, hc)
            if dist > 8:  # Too far
                continue
            
            # Direction vector (normalized to -1, 0, 1)
            dr = 0 if hr == mr else (1 if hr > mr else -1)
            dc = 0 if hc == mc else (1 if hc > mc else -1)
            key = (dr, dc)
            
            if key not in direction_targets:
                direction_targets[key] = []
            direction_targets[key].append((hr, hc, hcount, dist))
        
        if len(direction_targets) < 2:
            return None  # Not enough distinct directions
        
        # Calculate value of each direction
        direction_values = []
        for direction, targets in direction_targets.items():
            # Total humans in this direction
            total_humans = sum(h for _, _, h, _ in targets)
            # Closest target
            closest = min(targets, key=lambda x: x[3])
            closest_dist = closest[3]
            closest_count = closest[2]
            
            # Value = humans we can reach / distance
            value = total_humans / (closest_dist + 1)
            
            direction_values.append({
                'direction': direction,
                'targets': targets,
                'total_humans': total_humans,
                'closest_dist': closest_dist,
                'closest_count': closest_count,
                'value': value
            })
        
        # Sort by value (best directions first)
        direction_values.sort(key=lambda x: x['value'], reverse=True)
        
        # Take top 2 directions
        if len(direction_values) < 2:
            return None
        
        dir1 = direction_values[0]
        dir2 = direction_values[1]
        
        # Calculate split ratio based on human values
        total_value = dir1['total_humans'] + dir2['total_humans']
        if total_value == 0:
            return None
        
        # Allocate units proportionally to human group sizes
        # But ensure each group can capture its closest target
        min_for_dir1 = max(dir1['closest_count'], 3)
        min_for_dir2 = max(dir2['closest_count'], 3)
        
        if min_for_dir1 + min_for_dir2 > mcount:
            # Not enough units to split safely
            return None
        
        # Proportional allocation
        ratio1 = dir1['total_humans'] / total_value
        split1 = int(mcount * ratio1)
        
        # Ensure minimums
        split1 = max(min_for_dir1, min(split1, mcount - min_for_dir2))
        split2 = mcount - split1
        
        # Verify both can capture their targets
        if split1 < dir1['closest_count'] or split2 < dir2['closest_count']:
            return None
        
        # Create moves
        moves = []
        
        dr1, dc1 = dir1['direction']
        nr1, nc1 = mr + dr1, mc + dc1
        if state.in_bounds(nr1, nc1):
            moves.append((mr, mc, nr1, nc1, split1))
        
        dr2, dc2 = dir2['direction']
        nr2, nc2 = mr + dr2, mc + dc2
        if state.in_bounds(nr2, nc2):
            moves.append((mr, mc, nr2, nc2, split2))
        
        if len(moves) == 2:
            # Store objectives for each new group
            self._group_objectives[(nr1, nc1)] = (dir1['targets'][0][0], dir1['targets'][0][1])
            self._group_objectives[(nr2, nc2)] = (dir2['targets'][0][0], dir2['targets'][0][1])
            
            self.log.append(f"[SPLIT] {mcount} units -> {split1} (dir {dir1['direction']}, {dir1['total_humans']}H) + {split2} (dir {dir2['direction']}, {dir2['total_humans']}H)")
            return moves
        
        return None

    # ==================== MERGE STRATEGY ====================

    def _should_groups_merge(self, state, my_pos, enemy_pos, human_pos) -> Optional[List[Move]]:
        """
        Determine if groups should merge due to emergency.
        Only merge if:
        1. A group is threatened (enemy close and can kill it)
        2. Groups have no more reachable targets
        """
        if len(my_pos) < 2:
            return None
        
        _, my_total, enemy_total = self._get_totals(state)
        
        # Find the largest group (merge target)
        main_group = max(my_pos, key=lambda x: x[2])
        main_r, main_c, main_count = main_group
        
        merge_moves = []
        
        for mr, mc, mcount in my_pos:
            if (mr, mc) == (main_r, main_c):
                continue
            
            should_merge_this = False
            reason = ""
            
            # Check 1: Is this group threatened?
            for er, ec, ecount in enemy_pos:
                dist = chebyshev_distance(mr, mc, er, ec)
                if dist <= 2 and self._enemy_can_kill_us(mcount, ecount):
                    should_merge_this = True
                    reason = f"threatened at ({mr},{mc})"
                    break
            
            # Check 2: Does this group have any reachable targets?
            if not should_merge_this:
                has_target = False
                for hr, hc, hcount in human_pos:
                    if self._can_capture_humans_safely(mcount, hcount):
                        dist = chebyshev_distance(mr, mc, hr, hc)
                        if dist <= 5:  # Reasonable distance
                            has_target = True
                            break
                
                if not has_target:
                    should_merge_this = True
                    reason = f"no targets for ({mr},{mc})"
            
            if should_merge_this:
                # Move towards main group
                dr = 0 if main_r == mr else (1 if main_r > mr else -1)
                dc = 0 if main_c == mc else (1 if main_c > mc else -1)
                nr, nc = mr + dr, mc + dc
                
                if state.in_bounds(nr, nc):
                    # Check if this move is safe
                    dst = state.grid[nr, nc]
                    enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
                    if enemy_at == 0 or not self._enemy_can_kill_us(mcount, enemy_at):
                        merge_moves.append((mr, mc, nr, nc, mcount))
                        self.log.append(f"[MERGE] {reason}")
        
        return merge_moves if merge_moves else None

    # ==================== MOVE GENERATION FOR SINGLE GROUP ====================

    def _generate_moves_for_group(self, state, mr: int, mc: int, mcount: int,
                                   enemy_pos, human_pos, my_total: int, enemy_total: int) -> List[Tuple[float, InternalMove]]:
        """Generate all legal moves for a single group with priority scores."""
        moves = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = mr + dr, mc + dc
                if not state.in_bounds(nr, nc):
                    continue
                
                priority = self._score_move_for_group(
                    state, mr, mc, nr, nc, mcount,
                    enemy_pos, human_pos, my_total, enemy_total
                )
                
                # Skip guaranteed death moves
                if priority <= -100000:
                    continue
                
                moves.append((priority, (mr, mc, mcount, nr, nc)))
        
        # Sort by priority (best first)
        moves.sort(reverse=True, key=lambda x: x[0])
        return moves[:12]  # Limit for performance

    def _score_move_for_group(self, state, r1: int, c1: int, r2: int, c2: int, num: int,
                               enemy_pos, human_pos, my_total: int, enemy_total: int) -> float:
        """Score a move for a single group. Higher = better."""
        dst = state.grid[r2, c2]
        humans_at = dst.humans
        enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
        
        # === ABSOLUTE RULES ===
        
        # NEVER: Attack into guaranteed death
        if enemy_at > 0 and self._enemy_can_kill_us(num, enemy_at):
            return -200000
        
        # NEVER: Attack humans we can't capture safely (except desperate)
        if humans_at > 0 and not self._can_capture_humans_safely(num, humans_at):
            return -150000
        
        # NEVER: Attack enemy in risky zone
        if enemy_at > 0 and not self._can_kill_enemy_safely(num, enemy_at):
            if human_pos:  # Still humans on map
                return -100000
        
        # === PRIORITY SCORING ===
        score = 0.0
        
        # Priority 1: Capture humans safely (HIGHEST)
        if humans_at > 0 and self._can_capture_humans_safely(num, humans_at):
            score = 20000 + humans_at * 100
            # Bonus if enemy is racing for same target
            if enemy_pos:
                for er, ec, _ in enemy_pos:
                    if chebyshev_distance(r2, c2, er, ec) <= 2:
                        score += 5000  # Urgent!
            return score
        
        # Priority 2: Kill enemy safely
        if enemy_at > 0 and self._can_kill_enemy_safely(num, enemy_at):
            return 15000 + enemy_at * 50
        
        # Priority 3: Flee from immediate danger
        for er, ec, ecount in enemy_pos:
            dist_from_src = chebyshev_distance(r1, c1, er, ec)
            if dist_from_src <= 2 and self._enemy_can_kill_us(num, ecount):
                dist_from_dst = chebyshev_distance(r2, c2, er, ec)
                if dist_from_dst > dist_from_src:
                    return 12000 + dist_from_dst * 100  # Fleeing is good
        
        # Priority 4: Move towards capturable humans
        capturable = [(hr, hc, hcount) for hr, hc, hcount in human_pos 
                      if self._can_capture_humans_safely(num, hcount)]
        
        if capturable:
            # Find best target considering enemy race
            best_score = float('-inf')
            
            for hr, hc, hcount in capturable:
                my_new_dist = chebyshev_distance(r2, c2, hr, hc)
                my_old_dist = chebyshev_distance(r1, c1, hr, hc)
                
                # Enemy distance to this target
                enemy_dist = float('inf')
                if enemy_pos:
                    enemy_dist = min(chebyshev_distance(hr, hc, er, ec) for er, ec, _ in enemy_pos)
                
                # Base value from human count
                target_score = hcount * 10
                
                # Racing bonus
                if my_new_dist < enemy_dist:
                    target_score += 500 + (enemy_dist - my_new_dist) * 30
                elif my_new_dist == enemy_dist:
                    target_score += 200
                
                # Distance penalty
                target_score -= my_new_dist * 15
                
                # Movement bonus/penalty
                if my_new_dist < my_old_dist:
                    target_score += 100
                elif my_new_dist > my_old_dist:
                    target_score -= 200
                
                if target_score > best_score:
                    best_score = target_score
            
            if best_score > float('-inf'):
                score = 3000 + best_score
        
        # Priority 5: Endgame (no humans)
        if not human_pos and enemy_pos:
            closest = min(enemy_pos, key=lambda e: chebyshev_distance(r1, c1, e[0], e[1]))
            er, ec, ecount = closest
            old_dist = chebyshev_distance(r1, c1, er, ec)
            new_dist = chebyshev_distance(r2, c2, er, ec)
            
            if my_total >= enemy_total:
                # Hunt
                if new_dist < old_dist:
                    score = 2000 + (10 - new_dist) * 50
            else:
                # Flee
                if new_dist > old_dist:
                    score = 2000 + new_dist * 50
        
        # Oscillation penalty
        if self._is_oscillating((r1, c1), r2, c2):
            score -= 3000
        
        return score

    def _get_fallback_move(self, state, mr: int, mc: int, mcount: int, 
                           enemy_pos, human_pos) -> Optional[Move]:
        """Fast fallback move for a single group."""
        best_move = None
        best_score = float('-inf')
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = mr + dr, mc + dc
                if not state.in_bounds(nr, nc):
                    continue
                
                dst = state.grid[nr, nc]
                enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
                
                # Never move into death
                if enemy_at > 0 and self._enemy_can_kill_us(mcount, enemy_at):
                    continue
                
                score = 0
                
                # Capture humans
                if dst.humans > 0 and self._can_capture_humans_safely(mcount, dst.humans):
                    score = 10000 + dst.humans
                # Kill enemy
                elif enemy_at > 0 and self._can_kill_enemy_safely(mcount, enemy_at):
                    score = 8000 + enemy_at
                # Move towards humans
                elif human_pos:
                    capturable = [h for h in human_pos if self._can_capture_humans_safely(mcount, h[2])]
                    if capturable:
                        closest = min(capturable, key=lambda h: chebyshev_distance(nr, nc, h[0], h[1]))
                        score = 1000 - chebyshev_distance(nr, nc, closest[0], closest[1])
                
                if score > best_score:
                    best_score = score
                    best_move = (mr, mc, nr, nc, mcount)
        
        # Fallback: any safe move
        if best_move is None:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = mr + dr, mc + dc
                    if state.in_bounds(nr, nc):
                        dst = state.grid[nr, nc]
                        enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
                        if enemy_at == 0 or not self._enemy_can_kill_us(mcount, enemy_at):
                            return (mr, mc, nr, nc, mcount)
        
        return best_move

    # ==================== EVALUATION ====================

    def _evaluate(self, state, root_player: str) -> float:
        """Evaluate state from root_player's perspective. Returns [-1, 1]."""
        H, V, W = state.population_counts()
        
        # Terminal states
        if V == 0 and W == 0:
            return 0.0
        if V == 0:
            return -1.0 if root_player == "V" else 1.0
        if W == 0:
            return 1.0 if root_player == "V" else -1.0
        
        my_count = V if root_player == "V" else W
        enemy_count = W if root_player == "V" else V
        
        # Population score (main factor)
        total = my_count + enemy_count
        pop_score = (my_count - enemy_count) / total
        
        return max(-1.0, min(1.0, pop_score))

    # ==================== ALPHA-BETA FOR SINGLE GROUP (FIXED) ====================

    def _alphabeta_for_group(self, state, mr: int, mc: int, mcount: int,
                              depth: int, alpha: float, beta: float,
                              maximizing: bool, root_player: str,
                              enemy_pos, human_pos, my_total: int, enemy_total: int) -> Tuple[float, Optional[InternalMove]]:
        """
        Alpha-Beta search focused on a single group.
        
        FIX: At the root level (maximizing=True, first call), we ONLY consider
        moves from the specified group (mr, mc). Other groups' moves are used
        for simulation but not returned as the best move.
        """
        self._nodes_evaluated += 1
        
        if self._is_timeout():
            return self._evaluate(state, root_player), None
        
        # Terminal check
        _, V, W = state.population_counts()
        if V == 0 or W == 0:
            return self._evaluate(state, root_player), None
        
        if depth == 0:
            return self._evaluate(state, root_player), None
        
        # Generate moves for current player
        current_my_pos, current_enemy_pos, current_human_pos = self._get_positions(state)
        
        if maximizing:
            current_groups = current_my_pos
        else:
            current_groups = current_enemy_pos
        
        if not current_groups:
            return self._evaluate(state, root_player), None
        
        _, curr_my_total, curr_enemy_total = self._get_totals(state)
        
        # ===== FIX: Filter moves to only include this group's moves at root =====
        # At non-root levels, we still consider all moves for realistic simulation
        all_moves = []
        
        for gr, gc, gcount in current_groups:
            group_moves = self._generate_moves_for_group(
                state, gr, gc, gcount,
                current_enemy_pos if maximizing else current_my_pos,
                current_human_pos,
                curr_my_total, curr_enemy_total
            )
            all_moves.extend(group_moves)
        
        if not all_moves:
            return self._evaluate(state, root_player), None
        
        # Sort all moves by priority
        all_moves.sort(reverse=True, key=lambda x: x[0])
        all_moves = all_moves[:15]  # Limit branching
        
        # ===== FIX: Find best move ONLY from this group's moves =====
        # Get moves that belong to this group (mr, mc)
        this_group_moves = [(p, m) for p, m in all_moves if m[0] == mr and m[1] == mc]
        
        # Default best move: first move from this group, or None
        best_move = this_group_moves[0][1] if this_group_moves else None
        
        if maximizing:
            max_eval = float('-inf')
            
            for _, move in all_moves:
                if self._is_timeout():
                    break
                
                new_state = copy.deepcopy(state)
                new_state.move_group(*move)
                new_state.next_turn()
                
                eval_score, _ = self._alphabeta_for_group(
                    new_state, mr, mc, mcount,
                    depth - 1, alpha, beta, False, root_player,
                    enemy_pos, human_pos, my_total, enemy_total
                )
                
                # ===== FIX: Only update best_move if this move is from OUR group =====
                if eval_score > max_eval:
                    max_eval = eval_score
                    # Only set best_move if this move is from the group we're searching for
                    if move[0] == mr and move[1] == mc:
                        best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            
            for _, move in all_moves:
                if self._is_timeout():
                    break
                
                new_state = copy.deepcopy(state)
                new_state.move_group(*move)
                new_state.next_turn()
                
                eval_score, _ = self._alphabeta_for_group(
                    new_state, mr, mc, mcount,
                    depth - 1, alpha, beta, True, root_player,
                    enemy_pos, human_pos, my_total, enemy_total
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    # For minimizing (enemy), we don't care about group filtering
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
        
        # ===== EXTRA SAFETY: Verify the move is actually from this group =====
        if r1 != mr or c1 != mc:
            self.log.append(f"[G({mr},{mc})] WARNING: Got move from ({r1},{c1}), using fallback")
            return fallback
        
        # Safety check
        dst = state.grid[r2, c2]
        enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
        if enemy_at > 0 and self._enemy_can_kill_us(num, enemy_at):
            return fallback
        
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
        
        # 2. Check for SPLIT (only if we have 1-2 groups)
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
