# core/state.py
import random
from dataclasses import dataclass
import numpy as np

@dataclass
class Cell:
    humans: int = 0
    vampires: int = 0
    werewolves: int = 0


class GameState:
    def __init__(self, rows, cols, human_density=0.3, start_v=10, start_w=10):
        self.rows, self.cols = rows, cols
        self.turn = "V"
        self.grid = np.empty((rows, cols), dtype=object)

        # --- Step 1: fill grid with humans or empty cells
        for r in range(rows):
            for c in range(cols):
                if random.random() < human_density:
                    self.grid[r, c] = Cell(humans=random.randint(1, 12))
                else:
                    self.grid[r, c] = Cell()

        # --- Step 2: pick distinct spawn cells that are guaranteed empty
        empty_cells = [(r, c) for r in range(rows) for c in range(cols)
                       if self.grid[r, c].humans == 0]
        if len(empty_cells) < 2:
            all_cells = [(r, c) for r in range(rows) for c in range(cols)]
            random.shuffle(all_cells)
            for r, c in all_cells[:2]:
                self.grid[r, c].humans = 0
            empty_cells = [(r, c) for r, c in all_cells if self.grid[r, c].humans == 0]

        (vr, vc), (wr, wc) = random.sample(empty_cells, 2)
        self.grid[vr, vc].vampires = start_v
        self.grid[wr, wc].werewolves = start_w
        self.grid[vr, vc].humans = 0
        self.grid[wr, wc].humans = 0

        self.log: list[str] = []
        self._reset_turn_movement()

    # --------------------------------------------------------------
    def add_log(self, msg: str):
        self.log.append(msg)

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_adjacent(self, r1, c1, r2, c2):
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 and not (r1 == r2 and c1 == c2)

    def get_counts(self, r, c):
        cell = self.grid[r, c]
        return cell.humans, cell.vampires, cell.werewolves

    # --------------------------------------------------------------
    def _reset_turn_movement(self):
        self.targets_used_this_turn = set()
        self.movable = np.zeros((self.rows, self.cols), dtype=int)
        if self.turn == "V":
            for r in range(self.rows):
                for c in range(self.cols):
                    self.movable[r, c] = self.grid[r, c].vampires
        else:
            for r in range(self.rows):
                for c in range(self.cols):
                    self.movable[r, c] = self.grid[r, c].werewolves

    # --------------------------------------------------------------
    def _battle_prob(self, E1, E2):
        if E1 == E2:
            return 0.5
        if E1 < E2:
            return E1 / (2 * E2)
        return (E1 / E2) - 0.5

    def _binom(self, n, p):
        s = 0
        for _ in range(n):
            if random.random() < p:
                s += 1
        return s

    # --------------------------------------------------------------
    def _resolve_vs_humans(self, attackers, dst, species):
        H = dst.humans
        if attackers >= H:
            dst.humans = 0
            converted = H
            self.add_log(f"{species} converted all humans (H{H}).")
            return attackers + converted

        P = self._battle_prob(attackers, H)
        attacker_marks = []
        human_marks = []
        win = random.random() < P

        if win:
            for _ in range(attackers):
                alive = random.random() < P
                attacker_marks.append("✓" if alive else "✗")
            att_survive = attacker_marks.count("✓")

            for _ in range(H):
                converted = random.random() < P
                human_marks.append("✓" if converted else "✗")
            converted_n = human_marks.count("✓")

            dst.humans = 0  # clear humans completely
            total_attackers = att_survive + converted_n
            self.add_log(
                f"{species} WON vs humans  (P={P:.2f}) → "
                f"{att_survive}/{attackers} attackers survived, "
                f"{converted_n} humans converted. Total now {total_attackers}."
            )
            self.add_log(f"    Attackers: {''.join(attacker_marks)} (✓=alive)")
            self.add_log(f"    Humans   : {''.join(human_marks)} (✓=converted)")
            return total_attackers
        else:
            dst.humans = 0
            human_survivors = []
            for _ in range(H):
                alive = random.random() < (1 - P)
                human_survivors.append("✓" if alive else "✗")
            dst.humans = human_survivors.count("✓")
            self.add_log(
                f"{species} LOST vs humans  (P={P:.2f}) → "
                f"attackers wiped, humans survive {dst.humans}/{H}."
            )
            self.add_log(f"    Humans: {''.join(human_survivors)} (✓=alive)")
            return 0

    # --------------------------------------------------------------
    def _resolve_vs_enemy(self, attackers, defenders, dst, species):
        if attackers >= 1.5 * defenders:
            self.add_log(
                f"{species} OVERPOWER win vs enemy ({attackers} ≥ 1.5×{defenders}). Defenders wiped."
            )
            return attackers, 0
        if defenders >= 1.5 * attackers:
            self.add_log(
                f"{species} OVERPOWER loss vs enemy ({defenders} ≥ 1.5×{attackers}). Attackers wiped."
            )
            return 0, defenders

        P = self._battle_prob(attackers, defenders)
        attacker_marks, defender_marks = [], []
        win_roll = random.random() < P

        if win_roll:
            for _ in range(attackers):
                alive = random.random() < P
                attacker_marks.append("✓" if alive else "✗")
            att_survive = attacker_marks.count("✓")
            for _ in range(defenders):
                defender_marks.append("✗")
            self.add_log(
                f"{species} WON vs enemy  (P={P:.2f}) → "
                f"{att_survive}/{attackers} attackers survived, defenders wiped."
            )
            self.add_log(f"    Attackers: {''.join(attacker_marks)} (✓=alive)")
            return att_survive, 0
        else:
            for _ in range(attackers):
                attacker_marks.append("✗")
            for _ in range(defenders):
                alive = random.random() < (1 - P)
                defender_marks.append("✓" if alive else "✗")
            def_survive = defender_marks.count("✓")
            self.add_log(
                f"{species} LOST vs enemy  (P={P:.2f}) → "
                f"attackers wiped, defenders survive {def_survive}/{defenders}."
            )
            self.add_log(f"    Defenders: {''.join(defender_marks)} (✓=alive)")
            return 0, def_survive

    # --------------------------------------------------------------
    def move_group(self, r1, c1, num, r2, c2):
        if not (self.in_bounds(r1, c1) and self.in_bounds(r2, c2)):
            return False
        if not self.is_adjacent(r1, c1, r2, c2):
            return False
        if (r1, c1) in self.targets_used_this_turn:
            return False
        if num <= 0 or self.movable[r1, c1] < num:
            return False

        src = self.grid[r1, c1]
        dst = self.grid[r2, c2]
        species = "Vampires" if self.turn == "V" else "Werewolves"

        if self.turn == "V":
            src.vampires -= num
        else:
            src.werewolves -= num
        self.movable[r1, c1] -= num
        self.add_log(f"{species} moved {num} from ({r1},{c1}) → ({r2},{c2}).")

        if dst.humans > 0:
            result = self._resolve_vs_humans(num, dst, species)
            if self.turn == "V":
                dst.vampires += result
            else:
                dst.werewolves += result
        elif (self.turn == "V" and dst.werewolves > 0) or (self.turn == "W" and dst.vampires > 0):
            defenders = dst.werewolves if self.turn == "V" else dst.vampires
            att, deff = self._resolve_vs_enemy(num, defenders, dst, species)
            if self.turn == "V":
                dst.vampires += att
                dst.werewolves = deff
            else:
                dst.werewolves += att
                dst.vampires = deff
        else:
            if self.turn == "V":
                dst.vampires += num
            else:
                dst.werewolves += num

        self.targets_used_this_turn.add((r2, c2))
        return True

    # --------------------------------------------------------------
    def next_turn(self):
        self.turn = "W" if self.turn == "V" else "V"
        self.add_log(f"--- Next turn: {'Werewolves' if self.turn == 'W' else 'Vampires'} ---")
        self._reset_turn_movement()

    # --------------------------------------------------------------
    # --- End-of-game detection helpers ---
    def population_counts(self):
        H = V = W = 0
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r, c]
                H += cell.humans
                V += cell.vampires
                W += cell.werewolves
        return H, V, W

    def check_end_condition(self):
        _, V, W = self.population_counts()
        if V == 0 and W == 0:
            return "Draw! Both species perished."
        elif V == 0:
            return "Werewolves win! All vampires eliminated."
        elif W == 0:
            return "Vampires win! All werewolves eliminated."
        return None
    
    def clone(self) -> "GameState":
        """Lightweight deep copy of the game state for search."""
        new = GameState.__new__(GameState)  # bypass __init__
        new.rows = self.rows
        new.cols = self.cols
        new.turn = self.turn

        # Deep copy grid contents
        new.grid = np.empty((self.rows, self.cols), dtype=object)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r, c]
                new.grid[r, c] = Cell(
                    humans=cell.humans,
                    vampires=cell.vampires,
                    werewolves=cell.werewolves
                )

        # Movement bookkeeping
        new.log = []  # no need to copy logs in internal search
        new.targets_used_this_turn = set(self.targets_used_this_turn)
        new.movable = self.movable.copy()

        return new

