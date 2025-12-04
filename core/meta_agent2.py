# core/meta_agent.py
"""
Meta Agent v3 - Hybrid Heuristic + Alpha-Beta Search Agent

CRITICAL: ALL groups move EVERY turn. No group sits idle.
TIME LIMIT: Must return moves within 2 seconds (uses 1.8s safety margin)

This agent combines:
1. Fast heuristic scoring for move ordering
2. Iterative deepening alpha-beta (guaranteed move within time)
3. Smart merging - merge when being hunted
4. Quality splitting - minimum 5 units per split (3 in early game)
5. Predictive enemy movement
6. Endgame detection and strategy
7. Anti-corner/edge awareness
8. Race detection for human captures

v3 CRITICAL FIX - Rally Point Merge System:
- All groups converge to the SAME fixed point (centroid)
- Prevents groups from "chasing each other" and never merging
- Small vulnerable groups (<15 units) get extra merge urgency
- Attack mode: ALL groups move toward rally point, then hunt together

Key features:
- Iterative deepening ensures we ALWAYS have a move
- Predictive scoring anticipates enemy moves
- Endgame mode when humans < 10% of initial
- Position quality (avoid corners)
- Flee-toward-allies logic
- Futility pruning (don't chase hopeless goals)
- Rally point system for proper group consolidation
"""

from __future__ import annotations
import numpy as np
import time
import math
from typing import List, Tuple, Optional, Dict, Set
from core.agent_base import Agent
from core.state import GameState

# Type alias
Move = Tuple[int, int, int, int, int]  # (r1, c1, r2, c2, num)


def chebyshev_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """Chebyshev distance (king moves)."""
    return max(abs(r1 - r2), abs(c1 - c2))


def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """Manhattan distance."""
    return abs(r1 - r2) + abs(c1 - c2)


class MetaAgent(Agent):
    """
    Hybrid agent: Heuristics + Alpha-Beta search with iterative deepening.
    Uses heuristics for move ordering, alpha-beta for deeper evaluation.
    Guaranteed to return a move within time limit.
    """

    # Configuration
    MAX_GROUPS = 5
    MIN_SPLIT_UNITS_EARLY = 3  # Early game: can split smaller (race for humans!)
    MIN_SPLIT_UNITS_LATE = 5   # Late game: need bigger groups to survive
    EARLY_GAME_TURNS = 8       # First 8 turns = aggressive human capture
    ENDGAME_HUMAN_THRESHOLD = 0.15  # Enter endgame when humans < 15% of initial
    
    # Alpha-Beta settings  
    MAX_DEPTH = 6  # Maximum depth for iterative deepening
    
    # Time settings (CRITICAL: must be under 2 seconds!)
    TIME_LIMIT_HARD = 1.8  # Hard limit - stop everything
    TIME_LIMIT_SOFT = 1.5  # Soft limit - stop deepening
    TIME_RESERVE = 0.1    # Reserve for final move generation

    def __init__(self, time_limit: float = 1.8, max_depth: int = 6):
        self.time_limit = min(time_limit, self.TIME_LIMIT_HARD)
        self.max_depth = max_depth
        self.log: List[str] = []
        self.turn_count = 0
        
        # Time management
        self._start_time = 0.0
        self._nodes_searched = 0
        self._depth_reached = 0
        
        # Anti-oscillation
        self._move_history: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
        
        # Heatmap for visualization
        self._last_heatmap: Optional[np.ndarray] = None
        
        # Cluster cache
        self._human_clusters: List[Dict] = []
        
        # Game state tracking
        self._initial_humans: Optional[int] = None
        self._map_rows = 0
        self._map_cols = 0
        
        # Transposition table
        self._tt: Dict[str, Tuple[float, int, List[Move]]] = {}
        self._tt_hits = 0

    def name(self) -> str:
        return "MetaAgent"

    def debug_messages(self) -> List[str]:
        return self.log
    
    def _log(self, msg: str, indent: int = 0):
        """Add a log message with optional indentation."""
        prefix = "  " * indent
        self.log.append(f"{prefix}{msg}")
    
    def _time_remaining(self) -> float:
        """Time left in seconds."""
        return self.time_limit - (time.time() - self._start_time)
    
    def _time_critical(self) -> bool:
        """Are we running low on time?"""
        return self._time_remaining() < self.TIME_RESERVE
    
    def _should_stop_deepening(self) -> bool:
        """Should we stop increasing search depth?"""
        return self._time_remaining() < 0.3  # Stop deepening with < 0.3s left
    
    def _get_min_split_units(self) -> int:
        """Get minimum split units based on game phase."""
        if self.turn_count <= self.EARLY_GAME_TURNS:
            return self.MIN_SPLIT_UNITS_EARLY  # 3 in early game
        return self.MIN_SPLIT_UNITS_LATE  # 5 in late game
    
    def _is_early_game(self) -> bool:
        """Are we in early game (aggressive human capture)?"""
        return self.turn_count <= self.EARLY_GAME_TURNS

    # ==================== MAP AWARENESS ====================
    
    def _is_corner(self, r: int, c: int) -> bool:
        """Check if position is a corner."""
        return ((r == 0 or r == self._map_rows - 1) and 
                (c == 0 or c == self._map_cols - 1))
    
    def _is_edge(self, r: int, c: int) -> bool:
        """Check if position is on the edge."""
        return (r == 0 or r == self._map_rows - 1 or 
                c == 0 or c == self._map_cols - 1)
    
    def _position_quality(self, r: int, c: int) -> float:
        """
        Rate position quality (0-1). Center is best, corners are worst.
        Important for larger maps to avoid getting trapped.
        """
        if self._map_rows <= 1 or self._map_cols <= 1:
            return 0.5
        
        # Distance from center (normalized)
        center_r = (self._map_rows - 1) / 2
        center_c = (self._map_cols - 1) / 2
        max_dist = chebyshev_distance(0, 0, int(center_r), int(center_c))
        
        if max_dist == 0:
            return 0.5
        
        dist = chebyshev_distance(r, c, int(center_r), int(center_c))
        quality = 1.0 - (dist / max_dist)
        
        # Extra penalty for corners
        if self._is_corner(r, c):
            quality *= 0.5
        elif self._is_edge(r, c):
            quality *= 0.8
        
        return quality
    
    def _count_escape_routes(self, state: GameState, r: int, c: int, enemy_pos: List) -> int:
        """Count safe adjacent cells (not blocked by enemy)."""
        safe = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not state.in_bounds(nr, nc):
                    continue
                # Check if enemy threatens this cell
                threatened = False
                for er, ec, _ in enemy_pos:
                    if chebyshev_distance(nr, nc, er, ec) <= 1:
                        threatened = True
                        break
                if not threatened:
                    safe += 1
        return safe

    # ==================== GAME PHASE DETECTION ====================
    
    def _is_endgame(self, total_humans: int) -> bool:
        """Detect if we're in endgame (few humans left)."""
        if self._initial_humans is None or self._initial_humans == 0:
            return total_humans <= 5
        return total_humans < self._initial_humans * self.ENDGAME_HUMAN_THRESHOLD
    
    def _is_losing(self, my_total: int, enemy_total: int) -> bool:
        """Are we significantly behind?"""
        return enemy_total > my_total * 1.5
    
    def _is_winning(self, my_total: int, enemy_total: int) -> bool:
        """Are we significantly ahead?"""
        return my_total > enemy_total * 1.5

    # ==================== POSITION HELPERS ====================
    
    def _get_positions(self, state: GameState) -> Tuple[List, List, List]:
        """Get positions of our units, enemies, and humans."""
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

    def _get_totals(self, state: GameState) -> Tuple[int, int, int]:
        """Get total counts: (humans, our_units, enemy_units)."""
        my_pos, enemy_pos, human_pos = self._get_positions(state)
        return (
            sum(h[2] for h in human_pos),
            sum(m[2] for m in my_pos),
            sum(e[2] for e in enemy_pos)
        )

    # ==================== COMBAT HELPERS ====================
    
    def _can_capture_humans(self, our_count: int, human_count: int) -> bool:
        """Can we deterministically capture humans?"""
        return our_count >= human_count
    
    def _can_kill_enemy(self, our_count: int, enemy_count: int) -> bool:
        """Can we deterministically kill enemy? (1.5x rule)"""
        return our_count >= 1.5 * enemy_count
    
    def _enemy_can_kill_us(self, our_count: int, enemy_count: int) -> bool:
        """Can enemy deterministically kill us?"""
        return enemy_count >= 1.5 * our_count

    # ==================== PREDICTIVE ANALYSIS ====================
    
    def _predict_enemy_targets(self, enemy_pos: List, human_pos: List, my_pos: List) -> Dict[Tuple[int,int], List]:
        """
        Predict where enemy groups will likely move.
        Returns dict: enemy_pos -> list of likely target positions
        """
        predictions = {}
        
        for er, ec, ecount in enemy_pos:
            targets = []
            
            # Priority 1: Humans they can capture
            for hr, hc, hcount in human_pos:
                if ecount >= hcount:
                    dist = chebyshev_distance(er, ec, hr, hc)
                    targets.append((hr, hc, 'human', dist, hcount))
            
            # Priority 2: Our groups they can kill
            for mr, mc, mcount in my_pos:
                if ecount >= 1.5 * mcount:
                    dist = chebyshev_distance(er, ec, mr, mc)
                    targets.append((mr, mc, 'kill_us', dist, mcount))
            
            # Sort by distance (closest first)
            targets.sort(key=lambda x: x[3])
            predictions[(er, ec)] = targets[:3]  # Top 3 likely targets
        
        return predictions
    
    def _enemy_threatening_human(self, hr: int, hc: int, enemy_pos: List, my_dist: int) -> Tuple[bool, int]:
        """
        Check if enemy is threatening a human position.
        Returns (is_threatened, enemy_distance)
        """
        for er, ec, ecount in enemy_pos:
            enemy_dist = chebyshev_distance(hr, hc, er, ec)
            if enemy_dist <= my_dist + 1:  # Enemy can get there same time or sooner
                return True, enemy_dist
        return False, 999
    
    def _is_race_lost(self, my_pos: Tuple[int,int,int], target_pos: Tuple[int,int,int], 
                      enemy_pos: List) -> bool:
        """Check if we've already lost the race to a target."""
        mr, mc, mcount = my_pos
        tr, tc, tcount = target_pos
        my_dist = chebyshev_distance(mr, mc, tr, tc)
        
        # Can we even capture it?
        if mcount < tcount:
            return True
        
        # Check if any enemy is closer
        for er, ec, ecount in enemy_pos:
            if ecount >= tcount:  # Enemy can capture
                enemy_dist = chebyshev_distance(er, ec, tr, tc)
                if enemy_dist < my_dist:
                    return True
        
        return False

    # ==================== CLUSTER ANALYSIS ====================
    
    def _analyze_human_clusters(self, state: GameState, human_pos: List) -> List[Dict]:
        """Identify clusters of humans (groups within 3 cells of each other)."""
        if not human_pos:
            self._human_clusters = []
            return []
        
        # Union-find
        parent = {(h[0], h[1]): (h[0], h[1]) for h in human_pos}
        
        def find(p):
            if parent[p] != p:
                parent[p] = find(parent[p])
            return parent[p]
        
        def union(p1, p2):
            r1, r2 = find(p1), find(p2)
            if r1 != r2:
                parent[r1] = r2
        
        for i, (r1, c1, _) in enumerate(human_pos):
            for j, (r2, c2, _) in enumerate(human_pos):
                if i < j and chebyshev_distance(r1, c1, r2, c2) <= 3:
                    union((r1, c1), (r2, c2))
        
        cluster_members: Dict[Tuple, List] = {}
        for r, c, count in human_pos:
            root = find((r, c))
            if root not in cluster_members:
                cluster_members[root] = []
            cluster_members[root].append((r, c, count))
        
        clusters = []
        for root, members in cluster_members.items():
            total = sum(m[2] for m in members)
            centroid_r = sum(m[0] for m in members) / len(members)
            centroid_c = sum(m[1] for m in members) / len(members)
            density = total / (len(members) + 1)
            value = total * (1 + 0.1 * density)
            
            clusters.append({
                'cells': members,
                'total': total,
                'centroid': (centroid_r, centroid_c),
                'value': value,
                'size': len(members)
            })
        
        clusters.sort(key=lambda c: c['value'], reverse=True)
        self._human_clusters = clusters
        return clusters
    
    def _get_cluster_for_cell(self, r: int, c: int) -> Optional[Dict]:
        """Get cluster containing cell (r, c)."""
        for cluster in self._human_clusters:
            for cr, cc, _ in cluster['cells']:
                if cr == r and cc == c:
                    return cluster
        return None

    # ==================== EVALUATION FUNCTION ====================
    
    def _evaluate(self, state: GameState) -> float:
        """
        Evaluate board position. Returns score in [-1, 1].
        Positive = good for current player at root.
        
        Considers:
        - Material advantage (weighted heavily)
        - Human proximity advantage
        - Position quality (center vs corner)
        - Group consolidation (fewer groups when no humans)
        - Safety (threatened groups)
        - Escape routes (mobility)
        """
        my_pos, enemy_pos, human_pos = self._get_positions(state)
        total_humans, my_total, enemy_total = self._get_totals(state)
        
        # Terminal states
        if my_total == 0:
            return -1.0
        if enemy_total == 0:
            return 1.0
        
        total_units = my_total + enemy_total + total_humans
        if total_units == 0:
            return 0.0
        
        # === MATERIAL SCORE (50%) ===
        material_score = (my_total - enemy_total) / total_units
        
        # === HUMAN PROXIMITY (20%) ===
        human_score = 0.0
        if human_pos and my_pos and enemy_pos:
            # Find closest human for each side
            my_closest = []
            enemy_closest = []
            
            for hr, hc, hcount in human_pos:
                my_dist = min(chebyshev_distance(mr, mc, hr, hc) for mr, mc, _ in my_pos)
                enemy_dist = min(chebyshev_distance(er, ec, hr, hc) for er, ec, _ in enemy_pos)
                my_closest.append((my_dist, hcount))
                enemy_closest.append((enemy_dist, hcount))
            
            # Weight by human count
            for i, (hr, hc, hcount) in enumerate(human_pos):
                my_dist = my_closest[i][0]
                enemy_dist = enemy_closest[i][0]
                weight = hcount / total_humans if total_humans > 0 else 0
                
                if my_dist < enemy_dist:
                    human_score += weight * 0.3 * (enemy_dist - my_dist) / max(state.rows, state.cols)
                elif enemy_dist < my_dist:
                    human_score -= weight * 0.3 * (my_dist - enemy_dist) / max(state.rows, state.cols)
        
        # === POSITION QUALITY (10%) ===
        position_score = 0.0
        for mr, mc, mcount in my_pos:
            weight = mcount / my_total if my_total > 0 else 0
            position_score += weight * self._position_quality(mr, mc)
        for er, ec, ecount in enemy_pos:
            weight = ecount / enemy_total if enemy_total > 0 else 0
            position_score -= weight * self._position_quality(er, ec)
        position_score *= 0.5  # Scale to reasonable range
        
        # === CONSOLIDATION (10%) ===
        consolidation_score = 0.0
        if total_humans == 0 or self._is_endgame(total_humans):
            # Prefer fewer, larger groups for final battle
            my_consolidation = 1.0 / len(my_pos) if my_pos else 0
            enemy_consolidation = 1.0 / len(enemy_pos) if enemy_pos else 0
            consolidation_score = (my_consolidation - enemy_consolidation) * 0.5
        
        # === SAFETY (10%) ===
        safety_score = 0.0
        for mr, mc, mcount in my_pos:
            for er, ec, ecount in enemy_pos:
                dist = chebyshev_distance(mr, mc, er, ec)
                if dist <= 2:
                    if self._enemy_can_kill_us(mcount, ecount):
                        # Penalize threatened groups proportionally
                        safety_score -= 0.3 * (mcount / my_total) * (3 - dist) / 3
                    elif self._can_kill_enemy(mcount, ecount):
                        # Bonus for threatening enemy
                        safety_score += 0.1 * (ecount / enemy_total) * (3 - dist) / 3
        
        # Combine scores
        score = (0.50 * material_score + 
                 0.20 * human_score + 
                 0.10 * position_score + 
                 0.10 * consolidation_score + 
                 0.10 * safety_score)
        
        return max(-1.0, min(1.0, score))

    # ==================== MOVE GENERATION ====================
    
    def _generate_moves_for_group(self, state: GameState, r: int, c: int, count: int) -> List[Move]:
        """Generate all legal moves for a single group."""
        moves = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if state.in_bounds(nr, nc):
                    moves.append((r, c, nr, nc, count))
        return moves
    
    def _generate_all_move_combinations(self, state: GameState, my_pos: List) -> List[List[Move]]:
        """
        Generate move combinations for all groups.
        For efficiency, we limit combinations.
        """
        if not my_pos:
            return [[]]
        
        # For each group, get its moves
        group_moves = []
        for r, c, count in my_pos:
            moves = self._generate_moves_for_group(state, r, c, count)
            # Sort by heuristic score for better pruning
            scored = []
            for move in moves:
                score = self._quick_score_move(state, move, my_pos)
                scored.append((score, move))
            scored.sort(reverse=True)
            # Keep top moves for efficiency
            top_moves = [m for _, m in scored[:5]]
            if not top_moves:
                top_moves = moves[:3]
            group_moves.append(top_moves)
        
        # Generate combinations (limit to avoid explosion)
        if len(my_pos) == 1:
            return [[m] for m in group_moves[0]]
        
        # For multiple groups, sample combinations
        combinations = []
        max_combos = 30
        
        if len(my_pos) == 2:
            for m1 in group_moves[0][:5]:
                for m2 in group_moves[1][:5]:
                    if (m1[2], m1[3]) != (m2[2], m2[3]):  # Different destinations
                        combinations.append([m1, m2])
                        if len(combinations) >= max_combos:
                            break
                if len(combinations) >= max_combos:
                    break
        else:
            # For 3+ groups, use greedy combination
            for i in range(min(max_combos, len(group_moves[0]))):
                combo = [group_moves[0][min(i, len(group_moves[0])-1)]]
                claimed = {(combo[0][2], combo[0][3])}
                
                for g in range(1, len(my_pos)):
                    for move in group_moves[g]:
                        if (move[2], move[3]) not in claimed:
                            combo.append(move)
                            claimed.add((move[2], move[3]))
                            break
                    else:
                        # Fallback
                        if group_moves[g]:
                            combo.append(group_moves[g][0])
                
                if len(combo) == len(my_pos):
                    combinations.append(combo)
        
        return combinations if combinations else [[]]
    
    def _quick_score_move(self, state: GameState, move: Move, my_pos: List) -> float:
        """Quick heuristic score for move ordering."""
        r1, c1, r2, c2, count = move
        dst = state.grid[r2, c2]
        
        enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
        
        # Immediate capture/kill bonuses
        if dst.humans > 0 and count >= dst.humans:
            return 100 + dst.humans
        if enemy_at > 0 and count >= 1.5 * enemy_at:
            return 90 + enemy_at
        
        # Death penalty
        if enemy_at > 0 and enemy_at >= 1.5 * count:
            return -100
        
        # Risky attack penalty
        if enemy_at > 0 and count < 1.5 * enemy_at:
            return -50
        
        return 0

    # ==================== ALPHA-BETA SEARCH ====================
    
    def _alphabeta(self, state: GameState, depth: int, alpha: float, beta: float, 
                   maximizing: bool, root_turn: str) -> float:
        """Alpha-beta search with pruning."""
        self._nodes_searched += 1
        
        # Time check - critical!
        if self._time_critical():
            return self._evaluate(state) if state.turn == root_turn else -self._evaluate(state)
        
        # Terminal or depth limit
        my_pos, enemy_pos, _ = self._get_positions(state)
        
        if not my_pos:
            return -1.0 if maximizing else 1.0
        if not enemy_pos:
            return 1.0 if maximizing else -1.0
        if depth == 0:
            score = self._evaluate(state)
            return score if state.turn == root_turn else -score
        
        # Generate moves
        move_combos = self._generate_all_move_combinations(state, my_pos)
        
        if maximizing:
            value = -float('inf')
            for moves in move_combos:
                if self._time_critical():
                    break
                child_state = self._apply_moves(state, moves)
                if child_state:
                    child_value = self._alphabeta(child_state, depth - 1, alpha, beta, False, root_turn)
                    value = max(value, child_value)
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            return value if value > -float('inf') else self._evaluate(state)
        else:
            value = float('inf')
            for moves in move_combos:
                if self._time_critical():
                    break
                child_state = self._apply_moves(state, moves)
                if child_state:
                    child_value = self._alphabeta(child_state, depth - 1, alpha, beta, True, root_turn)
                    value = min(value, child_value)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
            return value if value < float('inf') else -self._evaluate(state)
    
    def _iterative_deepening_search(self, state: GameState, my_pos: List, 
                                     heuristic_moves: List[Move]) -> Tuple[List[Move], float, int]:
        """
        Iterative deepening alpha-beta search.
        GUARANTEES a move is returned within time limit.
        Returns: (best_moves, best_score, depth_reached)
        """
        best_moves = heuristic_moves
        best_score = float('-inf')
        depth_reached = 0
        
        # Generate alternatives - ALWAYS generate some even with heuristic
        alternatives = []
        
        # For each group, generate top moves
        for r, c, count in my_pos:
            group_moves = self._generate_moves_for_group(state, r, c, count)
            scored = [(self._quick_score_move(state, m, my_pos), m) for m in group_moves]
            scored.sort(reverse=True, key=lambda x: x[0])
            
            # Add top 3 moves for this group
            for _, move in scored[:3]:
                alternatives.append([move])
        
        # Add the heuristic choice as first option
        if heuristic_moves and heuristic_moves not in alternatives:
            alternatives.insert(0, heuristic_moves)
        
        # Limit alternatives
        alternatives = alternatives[:8]
        
        if not alternatives:
            return heuristic_moves, 0.0, 0
        
        # Iterative deepening
        for depth in range(1, self.MAX_DEPTH + 1):
            if self._should_stop_deepening():
                self._log(f"  [ID] Stopped at depth {depth} (time)", 1)
                break
            
            depth_best_moves = best_moves
            depth_best_score = float('-inf')
            searched_any = False
            
            for moves in alternatives:
                if self._time_critical():
                    break
                
                child_state = self._apply_moves(state, moves)
                if child_state:
                    searched_any = True
                    # Search with full depth (not depth-1)
                    score = self._alphabeta(
                        child_state, depth, 
                        float('-inf'), float('inf'), 
                        False, state.turn
                    )
                    
                    if score > depth_best_score:
                        depth_best_score = score
                        depth_best_moves = moves
            
            # Only update if we completed this depth
            if searched_any and not self._time_critical():
                best_moves = depth_best_moves
                best_score = depth_best_score
                depth_reached = depth
                self._depth_reached = depth
        
        return best_moves, best_score, depth_reached
    
    def _apply_moves(self, state: GameState, moves: List[Move]) -> Optional[GameState]:
        """Apply moves and return new state. Returns None if invalid."""
        try:
            new_state = state.apply_moves(moves)
            return new_state
        except:
            return None

    # ==================== STRATEGIC ANALYSIS ====================
    
    def _should_merge_urgently(self, my_pos: List, enemy_pos: List, my_total: int, enemy_total: int) -> bool:
        """
        Determine if we should urgently merge groups.
        True if:
        - Enemy is larger than any single group but we could fight if merged
        - We're being hunted (enemy approaching our groups)
        - We're in endgame and need to consolidate
        """
        if len(my_pos) < 2:
            return False
        
        if not enemy_pos:
            return False
        
        largest_enemy = max(e[2] for e in enemy_pos)
        largest_ours = max(m[2] for m in my_pos)
        
        # Case 1: Enemy's largest can kill our largest, but combined we can survive
        if largest_enemy >= 1.5 * largest_ours:
            # Can we fight or at least survive if merged?
            if my_total >= largest_enemy or my_total >= 0.67 * largest_enemy:
                # Check if enemy is close to any of our groups
                for mr, mc, _ in my_pos:
                    for er, ec, ecount in enemy_pos:
                        if chebyshev_distance(mr, mc, er, ec) <= 4 and ecount >= 1.5 * largest_ours:
                            return True
        
        # Case 2: In endgame with multiple groups - consolidate
        if self._initial_humans and self._is_endgame(sum(1 for _ in [])):  # Placeholder, will be set properly
            if len(my_pos) > 2 and my_total >= enemy_total * 0.8:
                return True
        
        return False
    
    def _should_fight_or_flee(self, my_total: int, enemy_total: int, total_humans: int) -> str:
        """
        Determine overall strategy: 'fight', 'flee', or 'harvest'.
        """
        # Endgame with few humans
        if total_humans <= 5:
            if my_total >= 1.5 * enemy_total:
                return 'fight'
            elif enemy_total >= 1.5 * my_total:
                return 'flee'
            else:
                return 'fight'  # Close fight, be aggressive
        
        # Normal game
        if enemy_total >= 2 * my_total:
            return 'flee'
        
        return 'harvest'
    
    def _find_ally_direction(self, r: int, c: int, my_pos: List) -> Optional[Tuple[int, int]]:
        """Find direction toward nearest friendly group."""
        other_groups = [(gr, gc, gc_count) for gr, gc, gc_count in my_pos if (gr, gc) != (r, c)]
        if not other_groups:
            return None
        
        nearest = min(other_groups, key=lambda g: chebyshev_distance(r, c, g[0], g[1]))
        nr, nc, _ = nearest
        
        # Direction toward ally
        dr = 0 if nr == r else (1 if nr > r else -1)
        dc = 0 if nc == c else (1 if nc > c else -1)
        
        return (dr, dc)
    
    def _find_center_direction(self, r: int, c: int) -> Tuple[int, int]:
        """Find direction toward map center."""
        center_r = self._map_rows // 2
        center_c = self._map_cols // 2
        
        dr = 0 if center_r == r else (1 if center_r > r else -1)
        dc = 0 if center_c == c else (1 if center_c > c else -1)
        
        return (dr, dc)
    
    def _calculate_human_sparsity(self, state: GameState, mr: int, mc: int, human_pos: List) -> Tuple[float, Dict]:
        """Calculate how spread out humans are from position (mr, mc)."""
        analysis = {
            'directions': {},
            'num_directions': 0,
            'total_humans': 0,
            'total_humans_nearby': 0,  # Within 5 cells
            'num_clusters': len(self._human_clusters),
            'direction_score': 0,
            'evenness_score': 0,
            'cluster_spread_score': 0
        }
        
        if not human_pos:
            return 0.0, analysis
        
        direction_counts: Dict[Tuple[int,int], int] = {}
        total_humans = 0
        total_nearby = 0
        
        for hr, hc, hcount in human_pos:
            dist = chebyshev_distance(mr, mc, hr, hc)
            dr = 0 if hr == mr else (1 if hr > mr else -1)
            dc = 0 if hc == mc else (1 if hc > mc else -1)
            key = (dr, dc)
            
            if key not in direction_counts:
                direction_counts[key] = 0
            direction_counts[key] += hcount
            total_humans += hcount
            
            if dist <= 5:
                total_nearby += hcount
        
        analysis['directions'] = dict(direction_counts)
        analysis['num_directions'] = len(direction_counts)
        analysis['total_humans'] = total_humans
        analysis['total_humans_nearby'] = total_nearby
        
        if total_humans == 0 or len(direction_counts) <= 1:
            return 0.0, analysis
        
        num_directions = len(direction_counts)
        values = list(direction_counts.values())
        mean_val = total_humans / num_directions
        variance = sum((v - mean_val) ** 2 for v in values) / num_directions
        max_variance = mean_val ** 2 if mean_val > 0 else 1
        
        direction_score = min(1.0, num_directions / 4.0)
        evenness_score = 1.0 - (variance / max_variance if max_variance > 0 else 0)
        
        cluster_directions = set()
        for cluster in self._human_clusters:
            cr, cc = cluster['centroid']
            dr = 0 if cr == mr else (1 if cr > mr else -1)
            dc = 0 if cc == mc else (1 if cc > mc else -1)
            cluster_directions.add((int(dr), int(dc)))
        cluster_spread_score = min(1.0, len(cluster_directions) / 3.0)
        
        analysis['direction_score'] = direction_score
        analysis['evenness_score'] = evenness_score
        analysis['cluster_spread_score'] = cluster_spread_score
        
        sparsity = 0.3 * direction_score + 0.3 * evenness_score + 0.4 * cluster_spread_score
        return sparsity, analysis

    # ==================== SPLIT STRATEGY ====================

    def _try_split(self, state: GameState, my_pos: List, enemy_pos: List, 
                   human_pos: List) -> Optional[List[Move]]:
        """
        Quality-focused splitting with early game aggression.
        Early game: Split aggressively to capture multiple human groups
        Late game: More conservative, need bigger groups
        """
        if len(my_pos) >= self.MAX_GROUPS:
            self._log(f"[SPLIT] Skipped: at MAX_GROUPS ({self.MAX_GROUPS})", 1)
            return None
        
        main = max(my_pos, key=lambda x: x[2])
        mr, mc, mcount = main
        
        # Get min split units based on game phase
        min_split = self._get_min_split_units()
        early_game = self._is_early_game()
        
        # Need enough units for split
        min_total = 2 * min_split
        if mcount < min_total:
            self._log(f"[SPLIT] Skipped: only {mcount} units (need {min_total}+)", 1)
            return None
        
        sparsity, analysis = self._calculate_human_sparsity(state, mr, mc, human_pos)
        
        self._log(f"[SPLIT ANALYSIS] Position: ({mr},{mc}) Units: {mcount}", 1)
        self._log(f"  Sparsity: {sparsity:.3f} | Directions: {analysis['num_directions']} | Clusters: {analysis['num_clusters']}", 1)
        
        # Early game = MUCH more aggressive splitting
        if early_game:
            # In early game, split if humans are spread in multiple directions
            should_split = (analysis['num_directions'] >= 2 and 
                           analysis['total_humans_nearby'] >= 6 and
                           mcount >= 2 * min_split)
            # Also split if sparsity is decent
            if not should_split:
                should_split = sparsity >= 0.25 and mcount >= 2 * min_split
        else:
            # Late game: more conservative
            should_split = sparsity >= 0.4 and mcount >= 2 * min_split
        
        if not should_split:
            self._log(f"[SPLIT] Skipped: conditions not met (sparsity={sparsity:.2f}, dirs={analysis['num_directions']})", 1)
            return None
        
        # Enemy safety check - RELAXED in early game
        if enemy_pos:
            min_enemy_dist = min(chebyshev_distance(mr, mc, er, ec) for er, ec, _ in enemy_pos)
            closest_enemy = min(enemy_pos, key=lambda e: chebyshev_distance(mr, mc, e[0], e[1]))
            
            if early_game:
                # Early game: only avoid if enemy is RIGHT NEXT to us AND can kill us
                if min_enemy_dist <= 1 and closest_enemy[2] >= 1.5 * mcount:
                    self._log(f"[SPLIT] Skipped: enemy adjacent and dangerous", 1)
                    return None
            else:
                # Late game: more cautious
                if min_enemy_dist <= 2 and mcount < closest_enemy[2] * 2:
                    self._log(f"[SPLIT] Skipped: enemy too close ({min_enemy_dist})", 1)
                    return None
        
        # Find best directions for split
        direction_targets: Dict[Tuple[int,int], List] = {}
        
        for hr, hc, hcount in human_pos:
            dist = chebyshev_distance(mr, mc, hr, hc)
            if dist > 10:  # Reduced from 12 to focus on closer targets
                continue
            
            dr = 0 if hr == mr else (1 if hr > mr else -1)
            dc = 0 if hc == mc else (1 if hc > mc else -1)
            key = (dr, dc)
            
            if key not in direction_targets:
                direction_targets[key] = []
            direction_targets[key].append((hr, hc, hcount, dist))
        
        if len(direction_targets) < 2:
            self._log(f"[SPLIT] Skipped: not enough directions ({len(direction_targets)})", 1)
            return None
        
        dir_values = []
        for direction, targets in direction_targets.items():
            total_h = sum(h for _, _, h, _ in targets)
            closest = min(targets, key=lambda x: x[3])
            
            # Value considers: total humans, distance, and whether we can capture closest
            can_capture = mcount // 2 >= closest[2]  # Can half our force capture it?
            capture_bonus = 2.0 if can_capture else 1.0
            
            dir_values.append({
                'dir': direction,
                'total': total_h,
                'closest_h': closest[2],
                'closest_d': closest[3],
                'value': (total_h * capture_bonus) / (closest[3] + 1),
                'can_capture': can_capture
            })
        
        dir_values.sort(key=lambda x: x['value'], reverse=True)
        d1, d2 = dir_values[0], dir_values[1]
        
        # Calculate split - ensure each piece can capture its target
        if early_game:
            # Early game: min is just enough to capture
            min1 = max(d1['closest_h'], min_split)
            min2 = max(d2['closest_h'], min_split)
        else:
            # Late game: need more buffer
            min1 = max(d1['closest_h'] + 2, min_split)
            min2 = max(d2['closest_h'] + 2, min_split)
        
        if min1 + min2 > mcount:
            # Try with absolute minimums
            min1 = max(d1['closest_h'], min_split)
            min2 = max(d2['closest_h'], min_split)
            if min1 + min2 > mcount:
                self._log(f"[SPLIT] Skipped: need {min1}+{min2}={min1+min2} but only have {mcount}", 1)
                return None
        
        # Proportional split based on value
        total_value = d1['value'] + d2['value']
        if total_value == 0:
            split1 = mcount // 2
        else:
            split1 = int(mcount * d1['value'] / total_value)
        
        split1 = max(min1, min(split1, mcount - min2))
        split2 = mcount - split1
        
        # Ensure both pieces meet minimum
        if split1 < min_split or split2 < min_split:
            self._log(f"[SPLIT] Skipped: split too uneven ({split1}, {split2})", 1)
            return None
        
        dr1, dc1 = d1['dir']
        dr2, dc2 = d2['dir']
        nr1, nc1 = mr + dr1, mc + dc1
        nr2, nc2 = mr + dr2, mc + dc2
        
        if state.in_bounds(nr1, nc1) and state.in_bounds(nr2, nc2):
            self._log(f"[SPLIT] EXECUTING: {mcount} -> {split1} to ({nr1},{nc1}) [toward {d1['total']}H] + {split2} to ({nr2},{nc2}) [toward {d2['total']}H]", 0)
            return [
                (mr, mc, nr1, nc1, split1),
                (mr, mc, nr2, nc2, split2)
            ]
        
        return None

    # ==================== MOVE SCORING ====================
    
    def _score_move(self, state: GameState, r1: int, c1: int, mcount: int, r2: int, c2: int,
                    enemy_pos: List, human_pos: List, my_pos: List, my_total: int, enemy_total: int,
                    attack_mode: bool, urgent_merge: bool, endgame: bool) -> Tuple[float, str]:
        """Score a move with comprehensive strategic awareness."""
        dst = state.grid[r2, c2]
        humans_at = dst.humans
        enemy_at = dst.werewolves if state.turn == "V" else dst.vampires
        
        total_humans = sum(h[2] for h in human_pos) if human_pos else 0
        
        # === ABSOLUTE PENALTIES ===
        if enemy_at > 0 and self._enemy_can_kill_us(mcount, enemy_at):
            return -50000, "SUICIDE"
        
        if humans_at > 0 and not self._can_capture_humans(mcount, humans_at):
            return -40000, "CANT_CAPTURE"
        
        if enemy_at > 0 and not self._can_kill_enemy(mcount, enemy_at):
            if not attack_mode:
                return -30000, "RISKY_ATTACK"
            return -20000, "RISKY_ATTACK_AM"
        
        # === POSITION QUALITY ADJUSTMENT ===
        pos_quality = self._position_quality(r2, c2)
        quality_bonus = int(pos_quality * 50)  # Up to +50 for center
        
        # Penalty for moving into corner when enemy is near
        if self._is_corner(r2, c2):
            for er, ec, ecount in enemy_pos:
                if chebyshev_distance(r2, c2, er, ec) <= 4:
                    quality_bonus -= 500  # Strong penalty
                    break
        
        # === URGENT MERGE - Use RALLY POINT (centroid) not nearest ally ===
        # This ensures all groups converge to the SAME point!
        if urgent_merge and len(my_pos) > 1:
            # Calculate rally point (weighted centroid of all groups)
            total_weight = sum(g[2] for g in my_pos)
            rally_r = sum(g[0] * g[2] for g in my_pos) / total_weight
            rally_c = sum(g[1] * g[2] for g in my_pos) / total_weight
            rally_r = int(round(rally_r))
            rally_c = int(round(rally_c))
            rally_r = max(0, min(self._map_rows - 1, rally_r))
            rally_c = max(0, min(self._map_cols - 1, rally_c))
            
            old_rally_dist = chebyshev_distance(r1, c1, rally_r, rally_c)
            new_rally_dist = chebyshev_distance(r2, c2, rally_r, rally_c)
            
            # Also check nearest ally for immediate merge opportunity
            other_groups = [(gr, gc, gc_count) for gr, gc, gc_count in my_pos if (gr, gc) != (r1, c1)]
            nearest = min(other_groups, key=lambda g: chebyshev_distance(r1, c1, g[0], g[1])) if other_groups else None
            
            if nearest:
                new_ally_dist = chebyshev_distance(r2, c2, nearest[0], nearest[1])
                
                # Immediate merge takes highest priority!
                if new_ally_dist == 0:
                    combined = mcount + nearest[2]
                    largest_enemy = max(e[2] for e in enemy_pos) if enemy_pos else 0
                    survival_bonus = 1000 if combined >= 0.67 * largest_enemy else 0
                    fight_bonus = 2000 if combined >= 1.5 * largest_enemy else 0
                    return 85000 + survival_bonus + fight_bonus + quality_bonus, "MERGE_NOW"
            
            # Move toward rally point
            if new_rally_dist < old_rally_dist:
                combined = my_total  # After merge
                largest_enemy = max(e[2] for e in enemy_pos) if enemy_pos else 0
                survival_bonus = 1000 if combined >= 0.67 * largest_enemy else 0
                fight_bonus = 2000 if combined >= 1.5 * largest_enemy else 0
                
                return 76000 + (10 - new_rally_dist) * 100 + survival_bonus + fight_bonus + quality_bonus, f"URGENT_MERGE(d={new_rally_dist})"
            elif new_rally_dist == old_rally_dist and old_rally_dist <= 1:
                # At rally point, move toward nearest ally
                if nearest:
                    old_ally_dist = chebyshev_distance(r1, c1, nearest[0], nearest[1])
                    new_ally_dist = chebyshev_distance(r2, c2, nearest[0], nearest[1])
                    if new_ally_dist < old_ally_dist:
                        return 74000 + (10 - new_ally_dist) * 100 + quality_bonus, f"URGENT_TO_ALLY(d={new_ally_dist})"
        
        # === SMALL VULNERABLE GROUP - MUST MERGE! ===
        # If we're a small group that can be killed by enemy, prioritize merging VERY highly
        if len(my_pos) > 1 and enemy_pos:
            largest_enemy = max(e[2] for e in enemy_pos)
            # Small group: < 15 units and can be killed by largest enemy
            if mcount < 15 and largest_enemy >= 1.5 * mcount:
                # Find rally point
                total_weight = sum(g[2] for g in my_pos)
                rally_r = int(round(sum(g[0] * g[2] for g in my_pos) / total_weight))
                rally_c = int(round(sum(g[1] * g[2] for g in my_pos) / total_weight))
                rally_r = max(0, min(self._map_rows - 1, rally_r))
                rally_c = max(0, min(self._map_cols - 1, rally_c))
                
                old_rally_dist = chebyshev_distance(r1, c1, rally_r, rally_c)
                new_rally_dist = chebyshev_distance(r2, c2, rally_r, rally_c)
                
                # Check for immediate merge
                other_groups = [(gr, gc, gc_count) for gr, gc, gc_count in my_pos if (gr, gc) != (r1, c1)]
                if other_groups:
                    nearest = min(other_groups, key=lambda g: chebyshev_distance(r1, c1, g[0], g[1]))
                    new_ally_dist = chebyshev_distance(r2, c2, nearest[0], nearest[1])
                    
                    # Immediate merge with ally!
                    if new_ally_dist == 0:
                        return 82000 + quality_bonus, "MERGE_SMALL"
                
                # Move toward rally point with high priority
                if new_rally_dist < old_rally_dist:
                    size_urgency = 3000 if mcount < 10 else 1500  # Smaller = more urgent
                    return 50000 + (10 - new_rally_dist) * 100 + size_urgency + quality_bonus, f"MERGE_SMALL(d={new_rally_dist})"
        
        # === ATTACK MODE ===
        if attack_mode:
            return self._score_attack_mode(
                state, r1, c1, mcount, r2, c2,
                enemy_pos, enemy_at, my_pos, my_total, enemy_total, quality_bonus
            )
        
        # === ENDGAME MODE (few humans left) ===
        if endgame and total_humans <= 5:
            # Don't chase far-away humans if we're close in score
            return self._score_endgame_move(
                state, r1, c1, mcount, r2, c2,
                enemy_pos, human_pos, my_pos, my_total, enemy_total, 
                humans_at, enemy_at, quality_bonus
            )
        
        # === HUMAN EXTRACTION MODE ===
        enemy_significance = (enemy_at / enemy_total) if enemy_total > 0 else 0
        
        # Cluster bonus
        target_cluster = self._get_cluster_for_cell(r2, c2)
        cluster_bonus = target_cluster['total'] * 5 if target_cluster else 0
        
        # Priority 1: KILL SIGNIFICANT ENEMY
        if enemy_at > 0 and self._can_kill_enemy(mcount, enemy_at):
            if enemy_significance >= 0.5:
                return 70000 + enemy_at * 100 + quality_bonus, f"KILL_MAJOR({enemy_at}, {enemy_significance*100:.0f}%)"
            elif enemy_significance >= 0.3:
                return 60000 + enemy_at * 80 + quality_bonus, f"KILL_SIG({enemy_at})"
            return 40000 + enemy_at * 50 + quality_bonus, f"KILL({enemy_at})"
        
        # Priority 2: CAPTURE HUMANS (check for race)
        if humans_at > 0 and self._can_capture_humans(mcount, humans_at):
            # Check if we're losing this race
            is_losing_race = self._is_race_lost((r1, c1, mcount), (r2, c2, humans_at), enemy_pos)
            if is_losing_race:
                # Still capture if it's right next to us
                if chebyshev_distance(r1, c1, r2, c2) == 1:
                    return 45000 + humans_at * 100 + cluster_bonus + quality_bonus, f"CAPTURE_CONTESTED({humans_at}H)"
                # Otherwise deprioritize
                return 15000 + humans_at * 50, f"RACE_LOST({humans_at}H)"
            
            score = 50000 + humans_at * 100 + cluster_bonus + quality_bonus
            
            racing = 0
            if enemy_pos:
                for er, ec, _ in enemy_pos:
                    enemy_dist = chebyshev_distance(r2, c2, er, ec)
                    if enemy_dist <= 2:
                        racing = max(racing, 5000)
                    elif enemy_dist <= 4:
                        racing = max(racing, 2000)
            score += racing
            
            parts = [f"{humans_at}H"]
            if cluster_bonus > 0:
                parts.append(f"c+{cluster_bonus:.0f}")
            if racing > 0:
                parts.append(f"race+{racing}")
            return score, f"CAPTURE({', '.join(parts)})"
        
        # Priority 3: FLEE FROM DANGER (toward allies if possible)
        for er, ec, ecount in enemy_pos:
            old_dist = chebyshev_distance(r1, c1, er, ec)
            if old_dist <= 2 and self._enemy_can_kill_us(mcount, ecount):
                new_dist = chebyshev_distance(r2, c2, er, ec)
                if new_dist > old_dist:
                    # Bonus for fleeing toward allies
                    ally_dir = self._find_ally_direction(r1, c1, my_pos)
                    move_dir = (r2 - r1, c2 - c1)
                    
                    ally_bonus = 0
                    if ally_dir and ally_dir == move_dir:
                        ally_bonus = 500
                    elif ally_dir:
                        # Partial credit for moving generally toward allies
                        if (ally_dir[0] == move_dir[0] or ally_dir[0] == 0 or move_dir[0] == 0):
                            ally_bonus = 200
                    
                    # Bonus for fleeing toward center (more escape routes)
                    center_bonus = quality_bonus
                    
                    return 30000 + new_dist * 100 + ally_bonus + center_bonus, f"FLEE(d:{old_dist}->{new_dist})"
        
        # Priority 4: MOVE TOWARD CAPTURABLE HUMANS
        capturable = [(hr, hc, hcount) for hr, hc, hcount in human_pos
                      if self._can_capture_humans(mcount, hcount)]
        
        if capturable:
            best_score = float('-inf')
            best_info = ""
            
            for hr, hc, hcount in capturable:
                # Skip if we're losing the race badly
                if self._is_race_lost((r1, c1, mcount), (hr, hc, hcount), enemy_pos):
                    my_dist = chebyshev_distance(r1, c1, hr, hc)
                    if my_dist > 3:  # Only skip if it's far
                        continue
                
                my_old_dist = chebyshev_distance(r1, c1, hr, hc)
                my_new_dist = chebyshev_distance(r2, c2, hr, hc)
                
                if my_new_dist >= my_old_dist:
                    continue
                
                tscore = hcount * 10 - my_new_dist * 5
                h_cluster = self._get_cluster_for_cell(hr, hc)
                if h_cluster:
                    tscore += h_cluster['total'] * 3
                
                if enemy_pos:
                    enemy_dist = min(chebyshev_distance(hr, hc, er, ec) for er, ec, _ in enemy_pos)
                    if my_new_dist < enemy_dist:
                        tscore += 500
                
                if tscore > best_score:
                    best_score = tscore
                    best_info = f"{hcount}H,d={my_new_dist}"
            
            if best_score > float('-inf'):
                return 20000 + best_score + quality_bonus, f"TOWARD_H({best_info})"
        
        # Priority 5: MOVE TOWARD BIGGEST CLUSTER
        if self._human_clusters:
            best_cluster = self._human_clusters[0]
            cr, cc = best_cluster['centroid']
            old_dist = chebyshev_distance(r1, c1, int(cr), int(cc))
            new_dist = chebyshev_distance(r2, c2, int(cr), int(cc))
            
            if new_dist < old_dist:
                return 12000 + best_cluster['total'] * 5 - new_dist * 10 + quality_bonus, f"TOWARD_CLUSTER({best_cluster['total']}H)"
        
        # Priority 6: APPROACH KILLABLE ENEMY
        for er, ec, ecount in enemy_pos:
            if self._can_kill_enemy(mcount, ecount):
                old_dist = chebyshev_distance(r1, c1, er, ec)
                new_dist = chebyshev_distance(r2, c2, er, ec)
                if new_dist < old_dist:
                    return 5000 + ecount * 10 - new_dist * 5 + quality_bonus, f"APPROACH_E({ecount})"
        
        # Priority 7: MERGE toward largest group
        if len(my_pos) > 1:
            main = max(my_pos, key=lambda x: x[2])
            main_r, main_c, _ = main
            if (r1, c1) != (main_r, main_c):
                old_dist = chebyshev_distance(r1, c1, main_r, main_c)
                new_dist = chebyshev_distance(r2, c2, main_r, main_c)
                if new_dist < old_dist:
                    return 1000 + (10 - new_dist) * 10 + quality_bonus, f"MERGE(d={new_dist})"
        
        if self._is_oscillating((r1, c1), (r2, c2)):
            return -5000, "OSCILLATION"
        
        return quality_bonus, "NEUTRAL"
    
    def _score_endgame_move(self, state: GameState, r1: int, c1: int, mcount: int, r2: int, c2: int,
                            enemy_pos: List, human_pos: List, my_pos: List, my_total: int, enemy_total: int,
                            humans_at: int, enemy_at: int, quality_bonus: int) -> Tuple[float, str]:
        """
        Scoring for endgame (few humans left).
        Focus on fighting/surviving rather than chasing distant humans.
        """
        # Kill enemy if possible
        if enemy_at > 0 and self._can_kill_enemy(mcount, enemy_at):
            return 80000 + enemy_at * 100 + quality_bonus, f"KILL_ENDGAME({enemy_at})"
        
        # Capture humans ONLY if close
        if humans_at > 0 and self._can_capture_humans(mcount, humans_at):
            # Only worth it if very close
            dist = chebyshev_distance(r1, c1, r2, c2)
            if dist == 1:
                return 60000 + humans_at * 100 + quality_bonus, f"CAPTURE_CLOSE({humans_at}H)"
        
        # Flee from danger - but toward rally point!
        for er, ec, ecount in enemy_pos:
            old_dist = chebyshev_distance(r1, c1, er, ec)
            if old_dist <= 2 and self._enemy_can_kill_us(mcount, ecount):
                new_dist = chebyshev_distance(r2, c2, er, ec)
                if new_dist > old_dist:
                    # Check if fleeing toward rally/allies
                    flee_bonus = 0
                    if len(my_pos) > 1:
                        total_weight = sum(g[2] for g in my_pos)
                        rally_r = int(round(sum(g[0] * g[2] for g in my_pos) / total_weight))
                        rally_c = int(round(sum(g[1] * g[2] for g in my_pos) / total_weight))
                        old_rally = chebyshev_distance(r1, c1, rally_r, rally_c)
                        new_rally = chebyshev_distance(r2, c2, rally_r, rally_c)
                        if new_rally < old_rally:
                            flee_bonus = 3000
                        elif new_rally > old_rally:
                            flee_bonus = -2000
                    return 55000 + new_dist * 100 + flee_bonus + quality_bonus, f"FLEE_ENDGAME(d:{old_dist}->{new_dist})"
        
        # MERGE is very important in endgame - use RALLY POINT (centroid)
        if len(my_pos) > 1:
            # Calculate rally point
            total_weight = sum(g[2] for g in my_pos)
            rally_r = int(round(sum(g[0] * g[2] for g in my_pos) / total_weight))
            rally_c = int(round(sum(g[1] * g[2] for g in my_pos) / total_weight))
            rally_r = max(0, min(self._map_rows - 1, rally_r))
            rally_c = max(0, min(self._map_cols - 1, rally_c))
            
            old_rally_dist = chebyshev_distance(r1, c1, rally_r, rally_c)
            new_rally_dist = chebyshev_distance(r2, c2, rally_r, rally_c)
            
            # Also check for immediate merge with ally
            other_groups = [(gr, gc, gc_count) for gr, gc, gc_count in my_pos if (gr, gc) != (r1, c1)]
            if other_groups:
                nearest = min(other_groups, key=lambda g: chebyshev_distance(r1, c1, g[0], g[1]))
                new_ally_dist = chebyshev_distance(r2, c2, nearest[0], nearest[1])
                
                # Immediate merge highest priority
                if new_ally_dist == 0:
                    return 70000 + quality_bonus, "MERGE_ENDGAME_NOW"
            
            # Move toward rally point
            if new_rally_dist < old_rally_dist:
                return 50000 + (10 - new_rally_dist) * 50 + quality_bonus, f"MERGE_ENDGAME(d={new_rally_dist})"
            elif new_rally_dist == old_rally_dist and old_rally_dist <= 1:
                # At rally, move toward nearest ally
                if other_groups:
                    old_ally_dist = chebyshev_distance(r1, c1, nearest[0], nearest[1])
                    new_ally_dist = chebyshev_distance(r2, c2, nearest[0], nearest[1])
                    if new_ally_dist < old_ally_dist:
                        return 48000 + (10 - new_ally_dist) * 50 + quality_bonus, f"TO_ALLY_ENDGAME(d={new_ally_dist})"
        
        # Hunt enemy if we can win combined
        if enemy_pos and my_total >= 1.5 * enemy_total:
            for er, ec, ecount in enemy_pos:
                old_dist = chebyshev_distance(r1, c1, er, ec)
                new_dist = chebyshev_distance(r2, c2, er, ec)
                if new_dist < old_dist:
                    return 40000 + (10 - new_dist) * 20 + quality_bonus, f"HUNT_ENDGAME({ecount}E)"
        
        # Move toward center for better position
        center_dir = self._find_center_direction(r1, c1)
        move_dir = (r2 - r1, c2 - c1)
        if center_dir == move_dir:
            return 10000 + quality_bonus, "CENTER"
        
        return quality_bonus, "ENDGAME_NEUTRAL"
    
    def _score_attack_mode(self, state: GameState, r1: int, c1: int, mcount: int, r2: int, c2: int,
                           enemy_pos: List, enemy_at: int, my_pos: List, 
                           my_total: int, enemy_total: int, quality_bonus: int) -> Tuple[float, str]:
        """
        Scoring for attack mode (no humans).
        KEY STRATEGY: MERGE ALL GROUPS INTO ONE before engaging!
        
        CRITICAL FIX: Use rally point system where ALL groups converge to a FIXED point.
        The rally point is the weighted centroid of all our groups.
        """
        
        # Priority 1: KILL ENEMY (only if we can definitely win)
        if enemy_at > 0 and self._can_kill_enemy(mcount, enemy_at):
            score = 90000 + enemy_at * 100 + quality_bonus
            enemy_significance = enemy_at / enemy_total if enemy_total > 0 else 0
            if enemy_significance >= 0.5:
                score += 10000
            return score, f"KILL_AM({enemy_at})"
        
        # Calculate if we need to consolidate (we're weaker or have multiple groups)
        we_are_weaker = my_total < enemy_total * 0.9
        need_to_merge = len(my_pos) > 1
        
        # FIXED RALLY POINT: Compute centroid of all our groups (weighted by count)
        # This ensures ALL groups move to the SAME point, including the largest one!
        if len(my_pos) > 1:
            total_weight = sum(g[2] for g in my_pos)
            rally_r = sum(g[0] * g[2] for g in my_pos) / total_weight
            rally_c = sum(g[1] * g[2] for g in my_pos) / total_weight
            # Round to nearest cell
            rally_r = int(round(rally_r))
            rally_c = int(round(rally_c))
            # Clamp to bounds
            rally_r = max(0, min(self._map_rows - 1, rally_r))
            rally_c = max(0, min(self._map_cols - 1, rally_c))
        else:
            rally_r, rally_c = r1, c1
        
        # Priority 2: MERGE IS CRITICAL in attack mode!
        # ALL groups should move toward the rally point (centroid)
        if need_to_merge:
            old_rally_dist = chebyshev_distance(r1, c1, rally_r, rally_c)
            new_rally_dist = chebyshev_distance(r2, c2, rally_r, rally_c)
            
            # Check if this move gets us closer to rally point
            if new_rally_dist < old_rally_dist:
                # Moving toward rally - VERY HIGH PRIORITY
                urgency_bonus = 5000 if we_are_weaker else 2000
                return 80000 + (10 - new_rally_dist) * 100 + urgency_bonus + quality_bonus, f"RALLY(d={new_rally_dist})"
            elif new_rally_dist == old_rally_dist and old_rally_dist <= 1:
                # We're at or very near rally point, move toward nearest ally
                other_groups = [(gr, gc, gcount) for gr, gc, gcount in my_pos if (gr, gc) != (r1, c1)]
                if other_groups:
                    nearest = min(other_groups, key=lambda g: chebyshev_distance(r1, c1, g[0], g[1]))
                    old_ally_dist = chebyshev_distance(r1, c1, nearest[0], nearest[1])
                    new_ally_dist = chebyshev_distance(r2, c2, nearest[0], nearest[1])
                    if new_ally_dist < old_ally_dist:
                        return 78000 + (10 - new_ally_dist) * 100 + quality_bonus, f"RALLY_TO_ALLY(d={new_ally_dist})"
                    elif new_ally_dist == 0:  # Will merge!
                        return 85000 + quality_bonus, "MERGE_COMPLETE"
            elif new_rally_dist == old_rally_dist:
                # Lateral move, still score it but lower
                return 30000 + quality_bonus, "RALLY_LATERAL"
            # Moving away from rally - will fall through to lower priority
        
        # Priority 3: Small groups should always try to merge with nearest ally
        if need_to_merge:
            other_groups = [(gr, gc, gcount) for gr, gc, gcount in my_pos if (gr, gc) != (r1, c1)]
            if other_groups:
                nearest = min(other_groups, key=lambda g: chebyshev_distance(r1, c1, g[0], g[1]))
                old_ally_dist = chebyshev_distance(r1, c1, nearest[0], nearest[1])
                new_ally_dist = chebyshev_distance(r2, c2, nearest[0], nearest[1])
                
                if new_ally_dist == 0:  # Will merge with ally!
                    return 85000 + quality_bonus, "MERGE_NOW"
                elif new_ally_dist < old_ally_dist:
                    # Smaller groups get higher bonus for merging
                    size_bonus = 3000 if mcount < 15 else 1000
                    return 75000 + (10 - new_ally_dist) * 100 + size_bonus + quality_bonus, f"MERGE_ALLY(d={new_ally_dist})"
        
        # Priority 4: FLEE if in danger - but ALWAYS toward rally point or allies!
        for er, ec, ecount in enemy_pos:
            old_dist = chebyshev_distance(r1, c1, er, ec)
            if old_dist <= 2 and self._enemy_can_kill_us(mcount, ecount):
                new_dist = chebyshev_distance(r2, c2, er, ec)
                if new_dist > old_dist:
                    # CRITICAL: When fleeing, MUST go toward rally/allies
                    ally_bonus = 0
                    if len(my_pos) > 1:
                        # Check if this move brings us closer to rally point
                        old_rally_dist = chebyshev_distance(r1, c1, rally_r, rally_c)
                        new_rally_dist = chebyshev_distance(r2, c2, rally_r, rally_c)
                        
                        if new_rally_dist < old_rally_dist:
                            ally_bonus = 5000  # STRONG bonus for fleeing TOWARD rally
                        elif new_rally_dist == old_rally_dist:
                            ally_bonus = 2000  # Some bonus for not getting further
                        else:
                            ally_bonus = -3000  # PENALTY for fleeing AWAY from rally
                    
                    return 60000 + new_dist * 100 + ally_bonus + quality_bonus, f"FLEE_AM(d:{old_dist}->{new_dist})"
        
        # Priority 5: HUNT (only if we're consolidated AND can win)
        if enemy_pos and my_total >= 1.5 * enemy_total and len(my_pos) == 1:
            for er, ec, ecount in enemy_pos:
                old_dist = chebyshev_distance(r1, c1, er, ec)
                new_dist = chebyshev_distance(r2, c2, er, ec)
                if new_dist < old_dist:
                    return 50000 + (10 - new_dist) * 20 + quality_bonus, f"HUNT({ecount}E)"
        
        # Priority 6: Move toward center (only if already consolidated)
        if len(my_pos) == 1:
            center_dir = self._find_center_direction(r1, c1)
            move_dir = (r2 - r1, c2 - c1)
            if center_dir == move_dir:
                return 5000 + quality_bonus, "CENTER_AM"
        
        if self._is_oscillating((r1, c1), (r2, c2)):
            return -5000, "OSCILLATION"
        
        return quality_bonus, "NEUTRAL_AM"

    # ==================== ANTI-OSCILLATION ====================
    
    def _is_oscillating(self, pos: Tuple[int,int], target: Tuple[int,int]) -> bool:
        """Check if this move creates back-and-forth oscillation."""
        history = self._move_history.get(pos, [])
        if len(history) >= 2:
            if history[-1] == target and history[-2] == pos:
                return True
        return False
    
    def _update_history(self, pos: Tuple[int,int], target: Tuple[int,int]):
        """Update move history for anti-oscillation."""
        if pos not in self._move_history:
            self._move_history[pos] = []
        self._move_history[pos].append(target)
        if len(self._move_history[pos]) > 4:
            self._move_history[pos] = self._move_history[pos][-4:]

    # ==================== BEST MOVE FOR GROUP ====================
    
    def _get_best_move_for_group(self, state: GameState, mr: int, mc: int, mcount: int,
                                  enemy_pos: List, human_pos: List, claimed_targets: set,
                                  my_pos: List, my_total: int, enemy_total: int,
                                  attack_mode: bool, urgent_merge: bool, endgame: bool) -> Optional[Move]:
        """Find best move for a single group using heuristics."""
        candidates = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = mr + dr, mc + dc
                if not state.in_bounds(nr, nc):
                    continue
                if (nr, nc) in claimed_targets:
                    continue
                
                score, reason = self._score_move(
                    state, mr, mc, mcount, nr, nc,
                    enemy_pos, human_pos, my_pos, my_total, enemy_total,
                    attack_mode, urgent_merge, endgame
                )
                
                if score <= -10000:
                    continue
                
                candidates.append((score, nr, nc, reason))
        
        # Log candidates
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            self._log(f"  Candidates ({len(candidates)}):", 1)
            for i, (score, nr, nc, reason) in enumerate(candidates[:5]):
                marker = ">>>" if i == 0 else "   "
                self._log(f"    {marker} ({nr},{nc}): {score:.0f} [{reason}]", 1)
        
        if not candidates:
            # Fallback: any safe move
            self._log(f"  No scored candidates, trying fallback...", 1)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = mr + dr, mc + dc
                    if not state.in_bounds(nr, nc):
                        continue
                    if (nr, nc) in claimed_targets:
                        continue
                    
                    dst = state.grid[nr, nc]
                    dst_enemy = dst.werewolves if state.turn == "V" else dst.vampires
                    if dst_enemy == 0 or not self._enemy_can_kill_us(mcount, dst_enemy):
                        self._log(f"  FALLBACK -> ({nr},{nc})", 1)
                        return (mr, mc, nr, nc, mcount)
            self._log(f"  NO VALID MOVES!", 1)
            return None
        
        best_score, best_r, best_c, best_reason = candidates[0]
        self._log(f"  DECISION: ({mr},{mc}) -> ({best_r},{best_c}) | {best_reason}", 1)
        
        return (mr, mc, best_r, best_c, mcount)

    # ==================== HEATMAP ====================
    
    def _compute_heatmap(self, state: GameState) -> np.ndarray:
        """Compute heatmap for visualization."""
        heatmap = np.zeros((state.rows, state.cols), dtype=float)
        my_pos, enemy_pos, human_pos = self._get_positions(state)
        
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                value = 0.0
                
                if cell.humans > 0:
                    value += cell.humans * 10
                    cluster = self._get_cluster_for_cell(r, c)
                    if cluster:
                        value += cluster['total'] * 2
                    if my_pos:
                        max_our = max(g[2] for g in my_pos)
                        if max_our >= cell.humans:
                            value += 50
                
                enemy_at = cell.werewolves if state.turn == "V" else cell.vampires
                if enemy_at > 0:
                    if my_pos:
                        max_our = max(g[2] for g in my_pos)
                        if max_our >= 1.5 * enemy_at:
                            value += enemy_at * 15
                        elif enemy_at >= 1.5 * max_our:
                            value -= enemy_at * 20
                
                if my_pos:
                    min_dist = min(chebyshev_distance(r, c, mr, mc) for mr, mc, _ in my_pos)
                    value -= min_dist * 2
                
                heatmap[r, c] = value
        
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        self._last_heatmap = heatmap
        return heatmap

    def get_heatmap(self) -> Optional[np.ndarray]:
        """Return last computed heatmap."""
        return self._last_heatmap

    # ==================== MAIN ENTRY ====================

    def select_action(self, state: GameState) -> List[Move]:
        """
        Main entry point - hybrid heuristic + iterative deepening alpha-beta.
        GUARANTEED to return within time limit.
        """
        self._start_time = time.time()
        self.log.clear()
        self.turn_count += 1
        self._nodes_searched = 0
        self._depth_reached = 0
        self._tt_hits = 0
        
        # Store map dimensions
        self._map_rows = state.rows
        self._map_cols = state.cols
        
        my_pos, enemy_pos, human_pos = self._get_positions(state)
        total_humans, my_total, enemy_total = self._get_totals(state)
        
        # Track initial humans for endgame detection
        if self._initial_humans is None:
            self._initial_humans = total_humans
        
        # === DETAILED STATE LOG ===
        self._log(f"{'='*60}")
        self._log(f"TURN {self.turn_count} | {state.turn}'s move | Map: {state.rows}x{state.cols}")
        self._log(f"{'='*60}")
        self._log(f"POPULATION: Us={my_total} | Enemy={enemy_total} | Humans={total_humans}")
        
        if self._initial_humans:
            human_pct = (total_humans / self._initial_humans) * 100 if self._initial_humans > 0 else 0
            self._log(f"  Initial humans: {self._initial_humans} | Remaining: {human_pct:.0f}%")
        
        self._log(f"OUR GROUPS ({len(my_pos)}):")
        for r, c, count in sorted(my_pos, key=lambda x: x[2], reverse=True):
            pos_qual = self._position_quality(r, c)
            escape_routes = self._count_escape_routes(state, r, c, enemy_pos)
            self._log(f"  ({r},{c}): {count} units [pos={pos_qual:.2f}, esc={escape_routes}]", 1)
        
        self._log(f"ENEMY GROUPS ({len(enemy_pos)}):")
        for r, c, count in sorted(enemy_pos, key=lambda x: x[2], reverse=True):
            self._log(f"  ({r},{c}): {count} units", 1)
        
        # Analyze clusters
        self._analyze_human_clusters(state, human_pos)
        
        if self._human_clusters:
            self._log(f"HUMAN CLUSTERS ({len(self._human_clusters)}):")
            for i, cluster in enumerate(self._human_clusters[:5]):
                cr, cc = cluster['centroid']
                self._log(f"  #{i+1}: {cluster['total']}H at ({cr:.1f},{cc:.1f}), value={cluster['value']:.1f}", 1)
        
        # Compute heatmap
        self._compute_heatmap(state)
        
        if not my_pos:
            self._log("[ERROR] No units!")
            return []
        
        # === STRATEGIC ANALYSIS ===
        attack_mode = (total_humans == 0)
        endgame = self._is_endgame(total_humans)
        urgent_merge = self._should_merge_urgently(my_pos, enemy_pos, my_total, enemy_total)
        strategy = self._should_fight_or_flee(my_total, enemy_total, total_humans)
        
        # Predict enemy movement
        enemy_predictions = self._predict_enemy_targets(enemy_pos, human_pos, my_pos)
        
        self._log(f"MODE: {'ATTACK' if attack_mode else ('ENDGAME' if endgame else 'HARVEST')}")
        self._log(f"  Strategy: {strategy} | UrgentMerge: {urgent_merge} | Groups: {len(my_pos)}/{self.MAX_GROUPS}")
        
        # Log rally point when merging is needed (attack mode or urgent merge)
        if (attack_mode or urgent_merge) and len(my_pos) > 1:
            total_weight = sum(g[2] for g in my_pos)
            rally_r = int(round(sum(g[0] * g[2] for g in my_pos) / total_weight))
            rally_c = int(round(sum(g[1] * g[2] for g in my_pos) / total_weight))
            rally_r = max(0, min(self._map_rows - 1, rally_r))
            rally_c = max(0, min(self._map_cols - 1, rally_c))
            self._log(f"  RALLY POINT: ({rally_r},{rally_c}) - all groups converging here")
        
        if enemy_predictions:
            self._log(f"ENEMY PREDICTIONS:")
            for (er, ec), targets in enemy_predictions.items():
                if targets:
                    t = targets[0]
                    self._log(f"  ({er},{ec}) -> likely {t[2]} at ({t[0]},{t[1]}) d={t[3]}", 1)
        
        all_moves = []
        claimed_targets: set = set()  # Cells that will have units after moves
        source_positions: set = set()  # Original positions of groups (blocked for others)
        groups_that_split: set = set()
        
        # Track all source positions - other groups can't move here if the group is moving elsewhere
        for r, c, _ in my_pos:
            source_positions.add((r, c))
        
        sorted_groups = sorted(my_pos, key=lambda x: x[2], reverse=True)
        
        # === PHASE 1: SPLIT CHECK (only if not urgent merge or endgame) ===
        # In early game, be MORE aggressive about splitting
        early_game = self._is_early_game()
        should_try_split = (human_pos and 
                           len(my_pos) < self.MAX_GROUPS and 
                           not urgent_merge and 
                           (early_game or not endgame))
        
        if should_try_split:
            self._log(f"\n--- SPLIT EVALUATION ---")
            main_group = sorted_groups[0]
            mr, mc, mcount = main_group
            
            split_moves = self._try_split(state, my_pos, enemy_pos, human_pos)
            if split_moves:
                for move in split_moves:
                    all_moves.append(move)
                    claimed_targets.add((move[2], move[3]))
                groups_that_split.add((mr, mc))
                # Remove source from blocked (it's being split)
                source_positions.discard((mr, mc))
        
        # === PHASE 2: ALL GROUPS GET BEST MOVE ===
        self._log(f"\n--- MOVE DECISIONS ---")
        moves_by_source = {}  # Track which source is moving where
        
        for mr, mc, mcount in sorted_groups:
            if (mr, mc) in groups_that_split:
                self._log(f"Group ({mr},{mc})x{mcount}: SPLIT (handled above)")
                continue
            
            self._log(f"Group ({mr},{mc})x{mcount}:")
            
            # Combine blocked positions: claimed targets + other sources still there
            blocked = claimed_targets.copy()
            for sr, sc in source_positions:
                if (sr, sc) != (mr, mc):  # Don't block our own position
                    # Only block if that source hasn't moved yet or is staying
                    if (sr, sc) not in moves_by_source:
                        blocked.add((sr, sc))
            
            best_move = self._get_best_move_for_group(
                state, mr, mc, mcount,
                enemy_pos, human_pos, blocked,  # Use combined blocked set
                my_pos, my_total, enemy_total,
                attack_mode, urgent_merge, endgame
            )
            
            if best_move:
                all_moves.append(best_move)
                claimed_targets.add((best_move[2], best_move[3]))
                moves_by_source[(mr, mc)] = (best_move[2], best_move[3])
                self._update_history((best_move[0], best_move[1]), (best_move[2], best_move[3]))
        
        # === PHASE 3: ITERATIVE DEEPENING ALPHA-BETA ===
        if all_moves and self._time_remaining() > 0.3 and not self._time_critical():
            self._log(f"\n--- ALPHA-BETA SEARCH ---")
            refined_moves, ab_score, depth = self._iterative_deepening_search(state, my_pos, all_moves)
            
            if refined_moves != all_moves:
                all_moves = refined_moves
                self._log(f"  Refined by AB search (depth={depth}, score={ab_score:.3f})")
            else:
                self._log(f"  AB confirmed heuristic (depth={depth})")
        
        # === SUMMARY ===
        self._log(f"\n--- TURN SUMMARY ---")
        if not all_moves:
            self._log("[ERROR] No valid moves!")
        else:
            self._log(f"Executing {len(all_moves)} moves:")
            for r1, c1, r2, c2, num in all_moves:
                self._log(f"  ({r1},{c1}) -> ({r2},{c2}) x{num}", 1)
        
        elapsed = time.time() - self._start_time
        self._log(f"Time: {elapsed*1000:.1f}ms | Nodes: {self._nodes_searched} | Depth: {self._depth_reached}")
        self._log(f"Time remaining: {self._time_remaining()*1000:.0f}ms")
        self._log(f"{'='*60}\n")
        
        return all_moves