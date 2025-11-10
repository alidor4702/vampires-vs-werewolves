import copy
import math
import time
import numpy as np
from core.agent_base import Agent


def get_game_state_info(state):
    """Extract all positions from game state."""
    my_positions = []
    enemy_positions = []
    human_positions = []
    
    for r in range(state.rows):
        for c in range(state.cols):
            cell = state.grid[r, c]
            if state.turn == "V":
                if cell.vampires > 0:
                    my_positions.append((r, c, cell.vampires))
                if cell.werewolves > 0:
                    enemy_positions.append((r, c, cell.werewolves))
            else:
                if cell.werewolves > 0:
                    my_positions.append((r, c, cell.werewolves))
                if cell.vampires > 0:
                    enemy_positions.append((r, c, cell.vampires))
            if cell.humans > 0:
                human_positions.append((r, c, cell.humans))
    
    return my_positions, enemy_positions, human_positions


def evaluate_move_priority(state, move, my_pos, enemy_pos, human_pos):
    """Evaluate move priority score - SIMPLE and FAST."""
    r1, c1, r2, c2, num = move
    dst = state.grid[r2, c2]
    
    my_total = sum(c for _, _, c in my_pos)
    enemy_total = sum(c for _, _, c in enemy_pos)
    
    # PRIORITY 1: EATING HUMANS (nothing beats this!)
    if dst.humans > 0:
        score = 10000 + dst.humans * 100
        if my_total <= enemy_total:
            score *= 2  # More urgent when behind
        return score
    
    # PRIORITY 2: WINNING FIGHTS
    enemy_here = dst.werewolves if state.turn == "V" else dst.vampires
    if enemy_here > 0:
        if num >= enemy_here * 1.5:
            return 5000  # Overpower win
        elif num < enemy_here * 0.7:
            return -100000  # Suicide - NEVER
        else:
            return 1000  # Fair fight
    
    # PRIORITY 3: MOVING TOWARDS HUMANS
    if human_pos:
        # Distance from destination to closest human
        min_dist_dest = min(abs(r2-hr)+abs(c2-hc) for hr,hc,_ in human_pos)
        # Distance from source to closest human
        min_dist_src = min(abs(r1-hr)+abs(c1-hc) for hr,hc,_ in human_pos)
        
        if min_dist_dest < min_dist_src:
            # Moving closer - GOOD
            # Find which human
            closest = min(human_pos, key=lambda h: abs(r2-h[0])+abs(c2-h[1]))
            hr, hc, hcount = closest
            new_dist = abs(r2-hr) + abs(c2-hc)
            
            # Check if enemy is closer
            if enemy_pos:
                enemy_dist = min(abs(hr-er)+abs(hc-ec) for er,ec,_ in enemy_pos)
                if new_dist < enemy_dist:
                    # We'll reach first!
                    return 500 + hcount * 20 / (new_dist + 1)
                else:
                    # Enemy closer - less valuable
                    return 50
            return 300 + hcount * 10 / (new_dist + 1)
        else:
            # Moving away - BAD
            return -500
    
    # PRIORITY 4: CONSOLIDATE (only if nothing better)
    ally_here = dst.vampires if state.turn == "V" else dst.werewolves
    if ally_here > 0:
        return 100
    
    # PRIORITY 5: AVOID EMPTY CELLS
    return -200


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.get_legal_moves_by_distance(state)
    
    def get_legal_moves_by_distance(self, state):
        """Generate moves sorted by strategic priority."""
        my_pos, enemy_pos, human_pos = get_game_state_info(state)
        
        moves = []
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                num = cell.vampires if state.turn == "V" else cell.werewolves
                if num > 0:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            r2, c2 = r + dr, c + dc
                            if state.in_bounds(r2, c2):
                                move = (r, c, r2, c2, num)
                                priority = evaluate_move_priority(state, move, my_pos, enemy_pos, human_pos)
                                moves.append((priority, move))
        
        # Sort by priority (highest first) and return moves only
        moves.sort(reverse=True, key=lambda x: x[0])
        return [move for priority, move in moves[:30]]  # Keep top 30 moves only
    
    def best_child(self, c_param=1.4):
        """UCB1 selection."""
        best_score, best_node = -float("inf"), None
        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.value / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score, best_node = score, child
        return best_node
    
    def expand(self):
        """Expand one untried action."""
        if not self.untried_actions:
            return None
        move = self.untried_actions.pop(0)  # Take highest priority
        new_state = copy.deepcopy(self.state)
        new_state.move_group(*move)
        new_state.next_turn()
        child = Node(new_state, parent=self, action=move)
        self.children.append(child)
        return child


def simple_rollout(state, root_player, max_depth=10):
    """Quick simulation using greedy moves."""
    s = copy.deepcopy(state)
    
    for _ in range(max_depth):
        _, v, w = s.population_counts()
        if v == 0 or w == 0:
            break
        
        my_pos, enemy_pos, human_pos = get_game_state_info(s)
        
        # Generate all legal moves
        moves = []
        for r in range(s.rows):
            for c in range(s.cols):
                cell = s.grid[r, c]
                num = cell.vampires if s.turn == "V" else cell.werewolves
                if num > 0:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            r2, c2 = r + dr, c + dc
                            if s.in_bounds(r2, c2):
                                move = (r, c, r2, c2, num)
                                priority = evaluate_move_priority(s, move, my_pos, enemy_pos, human_pos)
                                moves.append((priority, move))
        
        if not moves:
            break
        
        # Pick best move (greedy)
        moves.sort(reverse=True, key=lambda x: x[0])
        _, best_move = moves[0]
        s.move_group(*best_move)
        s.next_turn()
    
    # Evaluate final state
    _, v, w = s.population_counts()
    if root_player == "V":
        return (v - w) / (v + w + 1)
    else:
        return (w - v) / (v + w + 1)


def backpropagate(node, result):
    """Backpropagate simulation result."""
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent


class MCTSAgent(Agent):
    def __init__(self, time_limit=1.7):
        self.time_limit = time_limit
        self.log = []
    
    def select_action(self, state):
        start_time = time.time()
        self.log.clear()
        
        root = Node(copy.deepcopy(state))
        root_player = state.turn
        
        sims_done = 0
        while time.time() - start_time < self.time_limit:
            node = root
            
            # Selection
            while node.untried_actions == [] and node.children:
                node = node.best_child()
            
            # Expansion
            if node.untried_actions:
                node = node.expand()
                if node is None:
                    break
            
            # Simulation
            result = simple_rollout(node.state, root_player)
            
            # Backpropagation
            backpropagate(node, result)
            sims_done += 1
        
        elapsed = time.time() - start_time
        self.log.append(f"[AI] MCTS: {sims_done} simulations in {elapsed:.2f}s")
        
        if not root.children:
            self.log.append("[AI] No valid moves")
            return []
        
        # Select most visited child
        best_move = max(root.children, key=lambda n: n.visits)
        best_q = best_move.value / best_move.visits if best_move.visits > 0 else 0
        
        # Log top 3 moves
        sorted_children = sorted(root.children, key=lambda n: n.visits, reverse=True)
        for i, child in enumerate(sorted_children[:3]):
            q = child.value / child.visits if child.visits > 0 else 0
            self.log.append(f"  #{i+1}: {child.action} visits={child.visits} Q={q:.3f}")
        
        self.log.append(f"[AI] Selected: {best_move.action} (Q={best_q:.3f})")
        
        return [best_move.action]
