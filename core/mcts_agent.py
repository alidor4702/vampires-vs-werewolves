
# core/mcts_agent.py
import copy
import random
import math
import numpy as np
from core.agent_base import Agent


# -----------------------------
# Node used by MCTS
# -----------------------------
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.get_legal_moves(state)

    def get_legal_moves(self, state):
        """Generate all possible single moves for the current player."""
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
                                moves.append((r, c, r2, c2, num))
        return moves

    def best_child(self, c_param=1.4):
        """Choose child using UCB1 formula."""
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
        """Expand by trying one untried move."""
        move = self.untried_actions.pop()
        new_state = copy.deepcopy(self.state)
        new_state.move_group(*move)
        new_state.next_turn()
        child = Node(new_state, parent=self, action=move)
        self.children.append(child)
        return child


# -----------------------------
# Rollout and Backprop helpers
# -----------------------------
def rollout(state, root_player, max_depth=10):
    """Play random moves for a few turns and return simple reward."""
    s = copy.deepcopy(state)
    for _ in range(max_depth):
        moves = Node(s).get_legal_moves(s)
        if not moves:
            break
        move = random.choice(moves)
        s.move_group(*move)
        s.next_turn()
    vamps = sum(cell.vampires for row in s.grid for cell in row)
    weres = sum(cell.werewolves for row in s.grid for cell in row)
    if root_player == "V":
        return (vamps - weres) / (vamps + weres + 1e-6)
    else:
        return (weres - vamps) / (vamps + weres + 1e-6)

def backpropagate(node, result):
    """Propagate result up the tree."""
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent


# -----------------------------
# The MCTS Agent itself
# -----------------------------
class MCTSAgent(Agent):
    def __init__(self, sims=200):
        self.sims = sims
        self.log = []

    def select_action(self, state):
        self.log.clear()  # reset previous reasoning log
        root = Node(copy.deepcopy(state))
        root_player = state.turn

        for i in range(self.sims):
            node = root
            # Selection
            while node.untried_actions == [] and node.children:
                node = node.best_child()
            # Expansion
            if node.untried_actions:
                node = node.expand()
            # Simulation
            result = rollout(node.state, root_player)
            # Backprop
            backpropagate(node, result)

        # Logging reasoning summary
        self.log.append(f"[AI] Ran {self.sims} MCTS simulations.")
        for child in root.children:
            q = child.value / child.visits if child.visits else 0
            self.log.append(
                f"  Move {child.action} → visits={child.visits}, Q={q:.3f}"
            )

        if not root.children:
            self.log.append("[AI] No valid moves found.")
            return []

        best_move = max(root.children, key=lambda n: n.visits)
        best_q = best_move.value / best_move.visits
        self.log.append(
            f"[AI] Selected move {best_move.action} (visits={best_move.visits}, Q={best_q:.3f})"
        )

        return [best_move.action]
