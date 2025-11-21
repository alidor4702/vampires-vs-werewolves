import time
from client import ClientSocket
from argparse import ArgumentParser
import math
from collections import defaultdict
import random
import numpy as np

inf = float('inf')

GAME_STATE = None
TURN_COUNT = 0

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class GameState:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.grid = {}
        self.home = None
        self.species = 1  # 1 for vamp, 2 for wolf
        self.human_houses = []
        self.human_counts = {}

    def update(self, message):
        cmd = message[0]
        data = message[1]
        if cmd == 'set':
            self.m, self.n = data  # m = columns, n = rows
        elif cmd == 'hum':
            self.human_houses = [tuple(pos) for pos in data]
            self.human_counts = {house: 0 for house in self.human_houses}
        elif cmd == 'hme':
            self.home = tuple(data)
        elif cmd == 'map':
            self.grid = {}
            for x, y, h, v, w in data:
                if h or v or w:
                    self.grid[(x, y)] = (h, v, w)
                    if (x, y) in self.human_counts:
                        self.human_counts[(x, y)] = h
            # set species
            h, v, w = self.grid.get(self.home, (0, 0, 0))
            if v > 0:
                self.species = 1
            elif w > 0:
                self.species = 2
            else:
                raise ValueError("No species at home")
        elif cmd == 'upd':
            for x, y, h, v, w in data:
                if h or v or w:
                    self.grid[(x, y)] = (h, v, w)
                    if (x, y) in self.human_counts:
                        self.human_counts[(x, y)] = h
                else:
                    self.grid.pop((x, y), None)
                    if (x, y) in self.human_counts:
                        self.human_counts[(x, y)] = 0

    def copy(self):
        new = GameState()
        new.n = self.n
        new.m = self.m
        new.grid = {pos: val for pos, val in self.grid.items()}
        new.home = self.home
        new.species = self.species
        new.human_houses = self.human_houses.copy()
        new.human_counts = self.human_counts.copy()
        return new

    def get_species_count(self, pos, species):
        h, v, w = self.grid.get(pos, (0, 0, 0))
        if species == 1:
            return v
        elif species == 2:
            return w
        else:
            raise ValueError("Invalid species")

    def get_our(self, pos):
        return self.get_species_count(pos, self.species)

    def get_opp(self, pos):
        return self.get_species_count(pos, 3 - self.species)

    def get_hum(self, pos):
        return self.grid.get(pos, (0, 0, 0))[0]

    def get_humans_at(self, house):
        return self.human_counts.get(house, 0)

    def set_cell(self, pos, h, v, w):
        if h or v or w:
            self.grid[pos] = (h, v, w)
        else:
            self.grid.pop(pos, None)

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.n and 0 <= y < self.m

    def is_adjacent(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return dx <= 1 and dy <= 1 and not (dx == 0 and dy == 0)

    def is_terminal(self):
        our = sum(self.get_our(pos) for pos in self.grid)
        opp = sum(self.get_opp(pos) for pos in self.grid)
        return our == 0 or opp == 0

    def evaluate(self):
        our = sum(self.get_our(pos) for pos in self.grid)
        opp = sum(self.get_opp(pos) for pos in self.grid)
        humans = sum(self.get_humans_at(h) for h in self.human_houses)
        if our == 0:
            return -inf
        if opp == 0:
            return inf
        num_groups = len([p for p in self.grid if self.get_our(p) > 0])
        small_penalty = sum(1 for p in self.grid if 1 <= self.get_our(p) < 6)
        return our - opp + 2 * humans - 0.8 * num_groups - 2 * small_penalty

    def generate_possible_individual_moves(self, species):
        moves = []
        our_positions = [pos for pos in self.grid if self.get_species_count(pos, species) > 0]
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for pos in our_positions:
            sx, sy = pos
            count = self.get_species_count(pos, species)
            for dx, dy in directions:
                tx, ty = sx + dx, sy + dy
                if not self.in_bounds((tx, ty)):
                    continue
                h = self.get_hum((tx, ty))
                opp_count = self.get_species_count((tx, ty), 3 - species)
                if opp_count > 0:
                    required = math.ceil(1.5 * opp_count)
                    if count >= required:
                        moves.append((sx, sy, required, tx, ty))
                elif h > 0:
                    if count >= h:
                        moves.append((sx, sy, h, tx, ty))
                else:
                    moves.append((sx, sy, count, tx, ty))
        return moves

    def generate_actions(self, species):
        global TURN_COUNT
        individual = self.generate_possible_individual_moves(species)
        def single_heur(m):
            return action_heuristic(self, [m], species)
        individual.sort(key=single_heur, reverse=True)
        top_individual = individual[:12]  # Reduced for MCTS performance
        actions = [[m] for m in top_individual]
        
        # Limited combinations for early game
        if TURN_COUNT < 30:
            n = len(top_individual)
            for i in range(min(4, n)):
                for j in range(i + 1, min(i+4, n)):
                    m1 = top_individual[i]
                    m2 = top_individual[j]
                    s1 = (m1[0], m1[1])
                    s2 = (m2[0], m2[1])
                    if s1 == s2:
                        continue
                    actions.append([m1, m2])
        
        return actions[:20]  # Limit total actions for MCTS

    def simulate_moves(self, moves, species):
        if not moves:
            return [(self.copy(), 1.0)]
        
        # [Keep your original simulate_moves implementation here - unchanged]
        # For brevity, I'll include a simplified version, but use your full version
        new_state = self.copy()
        # Simplified deterministic simulation for MCTS
        for sx, sy, num, tx, ty in moves:
            if not self.is_adjacent((sx, sy), (tx, ty)) or not self.in_bounds((tx, ty)):
                continue
            source_count = new_state.get_species_count((sx, sy), species)
            actual_num = min(num, source_count)
            
            # Remove from source
            h, v, w = new_state.grid.get((sx, sy), (0, 0, 0))
            if species == 1:
                v = max(0, v - actual_num)
            else:
                w = max(0, w - actual_num)
            if h or v or w:
                new_state.grid[(sx, sy)] = (h, v, w)
            else:
                new_state.grid.pop((sx, sy), None)
            
            # Add to target and resolve combat
            h, v, w = new_state.grid.get((tx, ty), (0, 0, 0))
            if species == 1:
                v += actual_num
            else:
                w += actual_num
            
            opp_count = w if species == 1 else v
            h_target = h
            
            if opp_count > 0:
                required = math.ceil(1.5 * opp_count)
                if actual_num >= required:
                    if species == 1:
                        w = 0
                    else:
                        v = 0
                else:
                    if species == 1:
                        v = max(0, v - actual_num)
                    else:
                        w = max(0, w - actual_num)
            elif h_target > 0:
                if actual_num >= h_target:
                    h = 0
                    if species == 1:
                        v += h_target
                    else:
                        w += h_target
            
            if h or v or w:
                new_state.grid[(tx, ty)] = (h, v, w)
            else:
                new_state.grid.pop((tx, ty), None)
        
        return [(new_state, 1.0)]

def action_heuristic(state, action, species):
    score = 0
    humans_left = any(state.get_humans_at(pos) > 0 for pos in state.human_houses)
    for m in action:
        sx, sy, num, tx, ty = m
        h = state.get_hum((tx, ty))
        opp = state.get_opp((tx, ty))
        if h > 0:
            if num >= h:
                score += h * 3
            else:
                score -= h * 2
        if opp > 0:
            required = math.ceil(1.5 * opp)
            if num >= required:
                score += opp * 4
            else:
                score -= opp * 2
        if not h and not opp:
            # Move toward goals
            goals = [p for p in state.human_houses if state.get_humans_at(p) > 0]
            if goals:
                old_dist = min(manhattan((sx,sy), g) for g in goals)
                new_dist = min(manhattan((tx,ty), g) for g in goals)
                score += (old_dist - new_dist) * 2
    return score

def get_sorted_actions(state, species):
    actions = state.generate_actions(species)
    actions.sort(key=lambda a: action_heuristic(state, a, species), reverse=True)
    return actions[:15]  # Limit for MCTS

# ========================================================
# MCTS IMPLEMENTATION
# ========================================================

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = None
        self.rollout_depth = 10

    def is_leaf(self):
        return len(self.children) == 0 and self.untried_actions is None

    def is_terminal(self):
        return self.state.is_terminal()

    def best_child(self, c_param=1.4):
        if not self.children:
            return None
        choices = []
        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.total_value / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
            choices.append(exploit + explore)
        return self.children[np.argmax(choices)]

    def expand(self):
        if self.untried_actions is None:
            self.untried_actions = get_sorted_actions(self.state, self.state.species)
        
        if not self.untried_actions:
            return None

        # Pick first untried action
        action = self.untried_actions.pop(0)
        
        
        outcomes = self.state.simulate_moves(action, self.state.species)
        if not outcomes:
            return None
        
       
        chosen_state, _ = random.choices(outcomes, weights=[prob for _, prob in outcomes])[0]
        chosen_state.species = 3 - self.state.species  
        
        child = MCTSNode(chosen_state, parent=self, action=action)
        self.children.append(child)
        return child

    def rollout(self):
        current = self.state.copy()
        depth = 0
        while not current.is_terminal() and depth < self.rollout_depth:
            actions = get_sorted_actions(current, current.species)[:5]  # Limited random rollout
            if not actions:
                break
            action = random.choice(actions)
            outcomes = current.simulate_moves(action, current.species)
            if outcomes:
                next_state, _ = random.choices(outcomes, weights=[prob for _, prob in outcomes])[0]
                next_state.species = 3 - current.species
                current = next_state
            depth += 1
        return current.evaluate()

    def backpropagate(self, value):
        self.visits += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)  # Negate for opponent

def mcts_search(root, max_time=3.0):
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < max_time:
        # Selection
        node = root
        while not node.is_terminal() and not node.is_leaf():
            node = node.best_child()
            if not node:
                break
        
        # Expansion
        if not node.is_terminal():
            node = node.expand()
            if not node:
                continue
        
        # Simulation
        value = node.rollout()
        
        # Backpropagation
        node.backpropagate(value)
        iterations += 1
    
    print(f"MCTS completed {iterations} iterations in {time.time()-start_time:.2f}s")
    
    # Return best child (most visits)
    if not root.children:
        return None
    return max(root.children, key=lambda c: c.visits)


def compute_next_move(gstate):
    global TURN_COUNT
    TURN_COUNT += 1
    species = gstate.species
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    humans_left = any(gstate.get_humans_at(pos) > 0 for pos in gstate.human_houses)
    our_groups = [(p, gstate.get_our(p)) for p in gstate.grid if gstate.get_our(p) > 0]
    if not our_groups:
        return 0, []

   
    if humans_left:
        for house in gstate.human_houses:
            h = gstate.get_humans_at(house)
            if h == 0: continue
            candidates = []
            for src, count in our_groups:
                if gstate.is_adjacent(src, house) and count >= h:
                    candidates.append((src, count, h))
            if candidates:
                src, _, need = min(candidates, key=lambda x: x[1] - x[2])
                sx, sy = src
                print(f"CAPTURING HUMANS at {house} → +{need} troops")
                return 1, [[sx, sy, need, house[0], house[1]]]

    # PRIORITY 2: MOVE TOWARD HUMANS
    if humans_left:
        best_dist = float('inf')
        best_src = None
        best_target = None
        for src, count in our_groups:
            sx, sy = src
            for house in gstate.human_houses:
                if gstate.get_humans_at(house) == 0: continue
                dist = manhattan(src, house)
                if dist < best_dist:
                    best_dist = dist
                    best_src = src
                    dx = 0
                    dy = 0
                    if house[0] > sx: dx = 1
                    elif house[0] < sx: dx = -1
                    if house[1] > sy: dy = 1
                    elif house[1] < sy: dy = -1
                    tx, ty = sx + dx, sy + dy
                    if gstate.in_bounds((tx, ty)) and gstate.get_opp((tx, ty)) == 0:
                        best_target = (tx, ty)
        if best_target:
            sx, sy = best_src
            count = gstate.get_our(best_src)
            print(f"MOVING TOWARD HUMANS: {count} → {best_target}")
            return 1, [[sx, sy, count, best_target[0], best_target[1]]]

    
    enemy_groups = [(p, gstate.get_opp(p)) for p in gstate.grid if gstate.get_opp(p) > 0]
    for pos, opp_count in enemy_groups:
        need = math.ceil(1.5 * opp_count)
        for src, count in our_groups:
            if gstate.is_adjacent(src, pos) and count >= need:
                sx, sy = src
                print(f"CRUSHING ENEMY at {pos} with {count}")
                return 1, [[sx, sy, need, pos[0], pos[1]]]

    
    if len(our_groups) > 1:
        largest_group = max(our_groups, key=lambda x: x[1])
        largest_pos, largest_count = largest_group
        for src, count in our_groups:
            if src == largest_pos or count >= largest_count: continue
            sx, sy = src
            dx = 0
            dy = 0
            if largest_pos[0] > sx: dx = 1
            elif largest_pos[0] < sx: dx = -1
            if largest_pos[1] > sy: dy = 1
            elif largest_pos[1] < sy: dy = -1
            tx, ty = sx + dx, sy + dy
            if gstate.in_bounds((tx, ty)) and gstate.get_opp((tx, ty)) == 0:
                print(f"CONSOLIDATING: {count} from {src}")
                return 1, [[sx, sy, count, tx, ty]]

    # MCTS DECISION (replaces minimax)
    print("Running MCTS...")
    root = MCTSNode(gstate.copy())
    root.rollout_depth = 8
    best_node = mcts_search(root, 3.8)
    
    if best_node and best_node.action:
        nb_moves = len(best_node.action)
        moves = [[sx, sy, num, tx, ty] for sx, sy, num, tx, ty in best_node.action]
        print(f"MCTS MOVE: {nb_moves} moves {moves}")
        return nb_moves, moves

    # FALLBACK: Expand
    largest_pos, largest_count = max(our_groups, key=lambda x: x[1])
    sx, sy = largest_pos
    for dx, dy in directions:
        tx, ty = sx + dx, sy + dy
        if gstate.in_bounds((tx, ty)) and gstate.get_hum((tx, ty)) == 0 and gstate.get_opp((tx, ty)) == 0:
            print(f"FALLBACK expand {largest_count} → ({tx},{ty})")
            return 1, [[sx, sy, largest_count, tx, ty]]

    return 0, []

def play_game(args):
    global GAME_STATE, TURN_COUNT
    GAME_STATE = GameState()
    client_socket = ClientSocket(args.ip, args.port)
    client_socket.send_nme("LavaMaster MCTS")
    
    for _ in range(4):
        message = client_socket.get_message()
        print("Received from server:", message)
        GAME_STATE.update(message)

    while True:
        message = client_socket.get_message()
        GAME_STATE.update(message)
        print("Received from server upd:", message)
        if message[0] == "upd":
            nb_moves, moves = compute_next_move(GAME_STATE)
            client_socket.send_mov(nb_moves, moves)

if __name__ == '__main__':
    parser = ArgumentParser(description='Twilight AI client - MCTS Edition')
    parser.add_argument('--ip', dest='ip', default='127.0.0.1', type=str)
    parser.add_argument('--port', dest='port', default=5555, type=int)
    args = parser.parse_args()
    play_game(args)