# core/random_agent.py
import random
from core.agent_base import Agent

class RandomAgent(Agent):
    """Random AI that may perform multiple random moves per turn."""

    def select_action(self, state):
        # 20% chance to do nothing
        if random.random() < 0.2:
            return []

        moves = []
        possible = []

        # Gather all legal source-target pairs
        for r in range(state.rows):
            for c in range(state.cols):
                cell = state.grid[r, c]
                amount = 0
                if state.turn == "W" and cell.werewolves > 0:
                    amount = cell.werewolves
                elif state.turn == "V" and cell.vampires > 0:
                    amount = cell.vampires
                else:
                    continue

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r2, c2 = r + dr, c + dc
                        if not state.in_bounds(r2, c2):
                            continue
                        if state.is_adjacent(r, c, r2, c2):
                            possible.append((r, c, r2, c2, amount))

        if not possible:
            return []

        # Choose 1–3 random moves per turn
        n_moves = random.randint(1, min(3, len(possible)))
        chosen = random.sample(possible, n_moves)

        result = []
        for (r1, c1, r2, c2, total_amt) in chosen:
            if total_amt <= 1:
                num = 1
            else:
                # move a random fraction (up to half)
                num = random.randint(1, max(1, total_amt // 2))
            result.append((r1, c1, r2, c2, num))

        return result
