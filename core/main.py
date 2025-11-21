import time
from client import ClientSocket
from argparse import ArgumentParser
import math
from collections import defaultdict

inf = float('inf')

GAME_STATE = None
TURN_COUNT = 0

class GameState:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.grid = {}
        self.home = None
        self.species = 2  # 1 for vamp, 2 for wolf
        self.human_houses = []

    def update(self, message):
        cmd = message[0]
        data = message[1]
        if cmd == 'set':
            self.m, self.n = data  # m = columns, n = rows
        elif cmd == 'hum':
            self.human_houses = [tuple(pos) for pos in data]
        elif cmd == 'hme':
            self.home = tuple(data)
        elif cmd == 'map':
            self.grid = {}
            for x, y, h, v, w in data:
                if h or v or w:
                    self.grid[(x, y)] = (h, v, w)
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
                else:
                    self.grid.pop((x, y), None)

    def copy(self):
        new = GameState()
        new.n = self.n
        new.m = self.m
        new.grid = {pos: val for pos, val in self.grid.items()}
        new.home = self.home
        new.species = self.species
        new.human_houses = self.human_houses.copy()
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
        humans = sum(self.get_hum(pos) for pos in self.grid)
        if our == 0:
            return -inf
        if opp == 0:
            return inf
        return our - opp + 2 * humans - 0.5 * len([p for p in self.grid if self.get_our(p) > 0])  # Penalize scattering

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
                    if count >= h:  # Only move if sufficient for humans
                        moves.append((sx, sy, h, tx, ty))
                else:
                    # empty or own
                    moves.append((sx, sy, count, tx, ty))
        return moves

    def generate_actions(self, species):
        global TURN_COUNT
        individual = self.generate_possible_individual_moves(species)
        def single_heur(m):
            return action_heuristic(self, [m], species)
        individual.sort(key=single_heur, reverse=True)
        top_individual = individual[:8]  # limit to top 8 individual moves
        actions = [[m] for m in top_individual]
        # Allow splitting only early game and for large groups
        if TURN_COUNT < 50:  # Early game
            n = len(top_individual)
            for i in range(n):
                for j in range(i + 1, n):
                    m1 = top_individual[i]
                    m2 = top_individual[j]
                    s1 = (m1[0], m1[1])
                    s2 = (m2[0], m2[1])
                    t1 = (m1[3], m1[4])
                    t2 = (m2[3], m2[4])
                    if s1 == s2 and m1[2] + m2[2] > self.get_species_count(s1, species):
                        continue
                    if m1[2] < 5 or m2[2] < 5:  # Minimum split size
                        continue
                    if t1 == s2 or t2 == s1:
                        continue
                    actions.append([m1, m2])
        return actions

    def simulate_moves(self, moves, species):
        if not moves:
            return [(self.copy(), 1.0)]
        outgoing = defaultdict(int)
        move_groups = defaultdict(int)
        targets = set()
        sources = set()
        for sx, sy, num, tx, ty in moves:
            source = (sx, sy)
            target = (tx, ty)
            sources.add(source)
            if not self.is_adjacent(source, target) or not self.in_bounds(target):
                continue
            if num <= 0:
                continue
            outgoing[source] += num
            move_groups[target] += num
            targets.add(target)
        if sources & targets:
            pass
        for source, out in outgoing.items():
            count = self.get_species_count(source, species)
            if out > count:
                outgoing[source] = count
        new_grid = {pos: val for pos, val in self.grid.items()}
        for source, out in outgoing.items():
            h, v, w = new_grid.get(source, (0, 0, 0))
            if species == 1:
                v = max(0, v - out)
            else:
                w = max(0, w - out)
            if h == 0 and v == 0 and w == 0:
                new_grid.pop(source, None)
            else:
                new_grid[source] = (h, v, w)
        outcomes = [(new_grid, 1.0)]
        for target, attackers in move_groups.items():
            new_outcomes = []
            for curr_grid, prob in outcomes:
                curr_grid = {p: v for p, v in curr_grid.items()}
                h, v, w = curr_grid.get(target, (0, 0, 0))
                if species == 1:
                    v += attackers
                else:
                    w += attackers
                opp = w if species == 1 else v
                is_human = False
                defenders = 0
                if opp > 0:
                    defenders = opp
                elif h > 0:
                    defenders = h
                    is_human = True
                else:
                    curr_grid[target] = (h, v, w) if h or v or w else curr_grid.pop(target, None)
                    new_outcomes.append((curr_grid, prob))
                    continue
                sure_win = False
                sure_lose = False
                if is_human:
                    if attackers >= defenders:
                        sure_win = True
                else:
                    if attackers >= math.ceil(1.5 * defenders):
                        sure_win = True
                    elif defenders >= math.ceil(1.5 * attackers):
                        sure_lose = True
                if sure_win:
                    if is_human:
                        h = 0
                        if species == 1:
                            v += defenders
                        else:
                            w += defenders
                    else:
                        if species == 1:
                            w = 0
                        else:
                            v = 0
                    curr_grid[target] = (h, v, w) if h or v or w else curr_grid.pop(target, None)
                    new_outcomes.append((curr_grid, prob))
                elif sure_lose:
                    if species == 1:
                        v = max(0, v - attackers)
                    else:
                        w = max(0, w - attackers)
                    curr_grid[target] = (h, v, w) if h or v or w else curr_grid.pop(target, None)
                    new_outcomes.append((curr_grid, prob))
                else:
                    p_win = attackers / (2 * defenders) if is_human else (attackers / defenders - 0.5)
                    p_win = max(0, min(1, p_win))  # Clamp probability
                    win_grid = {p: vals for p, vals in curr_grid.items()}
                    h, v, w = win_grid.get(target, (0, 0, 0))
                    if is_human:
                        h = 0
                        if species == 1:
                            v += defenders
                        else:
                            w += defenders
                    else:
                        if species == 1:
                            w = 0
                        else:
                            v = 0
                    win_grid[target] = (h, v, w) if h or v or w else win_grid.pop(target, None)
                    new_outcomes.append((win_grid, prob * p_win))
                    lose_grid = {p: vals for p, vals in curr_grid.items()}
                    h, v, w = lose_grid.get(target, (0, 0, 0))
                    if species == 1:
                        v = max(0, v - attackers)
                    else:
                        w = max(0, w - attackers)
                    lose_grid[target] = (h, v, w) if h or v or w else lose_grid.pop(target, None)
                    new_outcomes.append((lose_grid, prob * (1 - p_win)))
            outcomes = new_outcomes
        result = []
        for g, p in outcomes:
            new_state = self.copy()
            new_state.grid = g
            result.append((new_state, p))
        return result if result else [(self.copy(), 1.0)]

def action_heuristic(state, action, species):
    score = 0
    humans_left = any(state.get_hum(pos) > 0 for pos in state.human_houses)
    for m in action:
        sx, sy, num, tx, ty = m
        h = state.get_hum((tx, ty))
        opp = state.get_species_count((tx, ty), 3 - species)
        if h > 0:
            if num >= h:
                score += h * 2  # Reward human capture
            else:
                score -= h * 2  # Penalize risky human attack
        if opp > 0:
            if num >= math.ceil(1.5 * opp):
                score += opp * 3  # Reward enemy kill
            else:
                score -= opp * 2  # Penalize risky enemy attack
        if not humans_left and not opp and not h:
            # Reward consolidation by moving toward largest group
            largest = max([(p, state.get_our(p)) for p in state.grid if state.get_our(p) > 0], key=lambda x: x[1], default=((0, 0), 0))
            dist = abs(tx - largest[0][0]) + abs(ty - largest[0][1])
            score -= dist * 0.5  # Prefer moves closer to largest group
    return score

def get_sorted_actions(state, species):
    actions = state.generate_actions(species)
    def heur(a):
        return action_heuristic(state, a, species)
    actions.sort(key=heur, reverse=True)
    return actions[:10]  # Limit to top 10 actions

def minimax(state, depth, alpha, beta, maximizing, our_species, start_time):
    if time.time() - start_time > 4.0:
        return state.evaluate() if maximizing else -state.evaluate()
    if depth == 0 or state.is_terminal():
        eval = state.evaluate()
        if our_species != state.species:
            eval = -eval
        return eval
    species = our_species if maximizing else 3 - our_species
    actions = get_sorted_actions(state, species)
    if maximizing:
        max_eval = -inf
        for action in actions:
            outcomes = state.simulate_moves(action, species)
            expected = 0.0
            for new_state, prob in outcomes:
                new_state.species = our_species
                ev = minimax(new_state, depth - 1, alpha, beta, False, our_species, start_time)
                expected += prob * ev
            max_eval = max(max_eval, expected)
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = inf
        for action in actions:
            outcomes = state.simulate_moves(action, species)
            expected = 0.0
            for new_state, prob in outcomes:
                new_state.species = our_species
                ev = minimax(new_state, depth - 1, alpha, beta, True, our_species, start_time)
                expected += prob * ev
            min_eval = min(min_eval, expected)
            beta = min(beta, min_eval)
            if beta <= alpha:
                break
        return min_eval

def compute_next_move(gstate):
    global TURN_COUNT
    TURN_COUNT += 1
    species = gstate.species
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    # Check if humans are depleted
    humans_left = any(gstate.get_hum(pos) > 0 for pos in gstate.human_houses)

    # === 1. FIND ALL OUR GROUPS ===
    our_groups = [(p, gstate.get_our(p)) for p in gstate.grid if gstate.get_our(p) > 0]
    if not our_groups:
        return 0, []

    # === 2. PRIORITY 1: CAPTURE HUMANS ===
    if humans_left:
        for house in gstate.human_houses:
            h = gstate.get_hum(house)
            if h == 0:
                continue
            candidates = []
            for src, count in our_groups:
                if gstate.is_adjacent(src, house) and count >= h:
                    candidates.append((src, count, h))
            if candidates:
                src, _, need = min(candidates, key=lambda x: x[1] - x[2])
                sx, sy = src
                print(f"CAPTURING HUMANS at {house} → +{need} troops")
                return 1, [[sx, sy, need, house[0], house[1]]]

        # === 3. PRIORITY 2: MOVE TOWARD NEAREST HUMAN ===
        best_dist = float('inf')
        best_src = None
        best_target = None
        for src, count in our_groups:
            sx, sy = src
            for house in gstate.human_houses:
                if gstate.get_hum(house) == 0:
                    continue
                dist = abs(sx - house[0]) + abs(sy - house[1])
                if dist < best_dist:
                    best_dist = dist
                    best_src = src
                    dx = dy = 0
                    if house[0] < sx:
                        dx = -1
                    elif house[0] > sx:
                        dx = 1
                    if house[1] < sy:
                        dy = -1
                    elif house[1] > sy:
                        dy = 1
                    tx, ty = sx + dx, sy + dy
                    if gstate.in_bounds((tx, ty)) and gstate.get_opp((tx, ty)) == 0:
                        best_target = (tx, ty)
        if best_target:
            sx, sy = best_src
            count = gstate.get_our(best_src)
            print(f"MOVING TOWARD HUMANS: {count} → {best_target}")
            return 1, [[sx, sy, count, best_target[0], best_target[1]]]

    # === 4. PRIORITY 3: ATTACK ENEMY ===
    enemy_groups = [(p, gstate.get_opp(p)) for p in gstate.grid if gstate.get_opp(p) > 0]
    for pos, opp_count in enemy_groups:
        need = math.ceil(1.5 * opp_count)
        for src, count in our_groups:
            if gstate.is_adjacent(src, pos) and count >= need:
                sx, sy = src
                print(f"CRUSHING ENEMY at {pos} with {count} ≥ {need}")
                return 1, [[sx, sy, count, pos[0], pos[1]]]

    # === 5. PRIORITY 4: CONSOLIDATE ===
    largest_group = max(our_groups, key=lambda x: x[1])
    largest_pos, largest_count = largest_group
    if len(our_groups) > 1:
        for src, count in our_groups:
            if src == largest_pos or count >= largest_count:
                continue
            sx, sy = src
            dx = dy = 0
            if largest_pos[0] < sx:
                dx = -1
            elif largest_pos[0] > sx:
                dx = 1
            if largest_pos[1] < sy:
                dy = -1
            elif largest_pos[1] > sy:
                dy = 1
            tx, ty = sx + dx, sy + dy
            if gstate.in_bounds((tx, ty)) and gstate.get_opp((tx, ty)) == 0:
                print(f"CONSOLIDATING: {count} from {src} → ({tx},{ty})")
                return 1, [[sx, sy, count, tx, ty]]
    depth = 4
    start_time = time.time()
    best_value = -inf
    best_action = None
    actions = get_sorted_actions(gstate, species)
    for action in actions:
        if time.time() - start_time > 4.0:
            break
        outcomes = gstate.simulate_moves(action, species)
        expected = 0.0
        for new_state, prob in outcomes:
            new_state.species = species
            v = minimax(new_state, depth - 1, -inf, inf, False, species, start_time)
            expected += prob * v
        if expected > best_value:
            best_value = expected
            best_action = action

    if best_action:
        nb_moves = len(best_action)
        moves = [[sx, sy, num, tx, ty] for sx, sy, num, tx, ty in best_action]
        print(f"MINIMAX MOVE: {nb_moves} moves {moves}")
        return nb_moves, moves

    # === 7. PRIORITY 6: EXPAND ===
    if TURN_COUNT < 50 and largest_count >= 10:
        sx, sy = largest_pos
        valid = [(sx+dx, sy+dy) for dx, dy in directions if gstate.in_bounds((sx+dx, sy+dy))]
        empty_valid = [t for t in valid if gstate.get_hum(t) == 0 and gstate.get_opp(t) == 0]
        if len(empty_valid) >= 2:
            half = largest_count // 2
            t1, t2 = empty_valid[0], empty_valid[1]
            print(f"EXPANDING: splitting {largest_count} → {half} + {largest_count-half}")
            return 2, [
                [sx, sy, half, t1[0], t1[1]],
                [sx, sy, largest_count - half, t2[0], t2[1]]
            ]

    # === 8. FALLBACK ===
    sx, sy = largest_pos
    valid = [(sx+dx, sy+dy) for dx, dy in directions if gstate.in_bounds((sx+dx, sy+dy))]
    empty_valid = [t for t in valid if gstate.get_hum(t) == 0 and gstate.get_opp(t) == 0]
    if empty_valid:
        tx, ty = empty_valid[0]
        print(f"FALLBACK: moving {largest_count} → ({tx},{ty})")
        return 1, [[sx, sy, largest_count, tx, ty]]

    return 0, []

def play_game(args):
    global GAME_STATE, TURN_COUNT
    GAME_STATE = GameState()
    client_socket = ClientSocket(args.ip, args.port)
    client_socket.send_nme("LavaMasterweak")
    message = client_socket.get_message()
    print("Received from serverdtdtdt:", message)
    GAME_STATE.update(message)
    message = client_socket.get_message()
    GAME_STATE.update(message)
    message = client_socket.get_message()
    GAME_STATE.update(message)
    message = client_socket.get_message()
    GAME_STATE.update(message)

    while True:
        message = client_socket.get_message()
        time_message_received = time.time()
        GAME_STATE.update(message)
        print("Received from server upddd:", message)
        if message[0] == "upd":
            nb_moves, moves = compute_next_move(GAME_STATE)
            client_socket.send_mov(nb_moves, moves)

if __name__ == '__main__':
    parser = ArgumentParser(description='Twilight AI client')
    parser.add_argument('--ip', dest='ip', default='127.0.0.1', type=str,
                        help='IP address of the game server')
    parser.add_argument('--port', dest='port', default=5555, type=int,
                        help='Port the server is listening on')
    args = parser.parse_args()
    play_game(args)