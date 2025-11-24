# Vampires vs Werewolves — AI Battle Arena

A turn-based strategy environment designed to train, test, and visualize AI decision-making models in a controlled grid-based simulation.
Built in Python (Tkinter + NumPy), the project supports human vs human, human vs AI, and AI vs AI modes.

---

## Overview

Vampires and Werewolves compete for territorial control on a grid map.
Each cell may contain:
- Humans (neutral, convertible units)
- Vampires
- Werewolves

Players take turns moving their units to adjacent cells (eight directions).
Combat, conversion, and survival follow deterministic and probabilistic rules inspired by resource-driven strategy systems.

---

## Features

| Category | Description |
|---------|-------------|
| Grid Engine | Scales up to 256×256 tiles |
| Turn System | Vampires (V) and Werewolves (W) alternate turns |
| Adjacency Movement | Move in eight directions (orthogonal and diagonal) |
| Battle Rules | Probabilistic outcomes based on stack sizes |
| Multi-Move Turns | Split stacks and perform several moves per turn |
| Skip Turn Option | Allows a player or agent to skip |
| Pan & Zoom | Scroll, drag, or use hotkeys to navigate |
| Live Game Log | Displays all moves, attacks, conversions, and results |
| Quick Restart | Restart with identical or randomized parameters |
| AI Port | Supports any AI model (MCTS, RL, policy networks, etc.) |
| Random Agent | Included baseline agent |

---

## Project Structure

project/
├── core/
│   ├── state.py           # Game logic and rules
│   ├── config.py          # Configuration dataclass
│   ├── agent_base.py      # Abstract AI interface
│   ├── random_agent.py    # Baseline random AI agent
│   └── mcts_agent.py      # Placeholder for future MCTS
│
├── gui/
│   ├── menu.py            # Main menu and mode selection
│   ├── board.py           # Game board, events, AI integration
│
├── main.py                # Program entry point
├── testmap2.xml           # Example map
├── thetrap.xml            # Example map
└── README.md

---

## Game Rules

### Movement
- Units may move to any adjacent tile (eight directions).
- Stacks may be split arbitrarily (for example, move part of a group while leaving the rest).
- A unit or sub-unit that has moved cannot move again during the same turn.

### Combat

Situation | Outcome
--------- | --------
Empty Cell | Units enter with no resistance.
Humans | If attackers ≥ humans: all humans convert. If attackers < humans: probabilistic resolution using P. P = E1/(2E2) when weaker, or (E1/E2) - 0.5 when stronger.
Enemy Units | If attackers ≥ 1.5× defenders: defenders are eliminated. If defenders ≥ 1.5× attackers: attackers are eliminated. Otherwise: probabilistic per-unit resolution using P.

### Turn End
- Press Spacebar or click “Next Turn”.
- The game automatically detects end conditions when one or both species are eliminated.

---

## Installation

### 1. Clone Repository
git clone https://github.com/yourusername/vampires-vs-werewolves.git
cd vampires-vs-werewolves

### 2. Create Environment and Install Dependencies
python3 -m venv venv
source venv/bin/activate
pip install numpy

### 3. Run
python main.py

---

## Controls

Action | Key / Mouse
-------|-------------
Select cell | Left click
Pan map | Middle drag or arrow keys
Zoom | Mouse wheel or keyboard shortcuts
Next Turn | Spacebar
Restart Game | Restart button
Back to Menu | Back button

---

## AI Integration

### Base Agent Interface

All AI models must inherit from Agent in core/agent_base.py and implement:

def select_action(self, state) -> list[tuple[int, int, int, int, int]]:
    '''
    Decide moves for a given GameState.
    Return [(r1, c1, r2, c2, num), ...] or [] to skip the turn.
    '''

---

## What Models Receive as Input

Each invocation of select_action(state) provides a GameState instance.

Attribute | Type | Description
----------|------|------------
rows, cols | int | Board dimensions
turn | str | "V" or "W"
grid[r][c] | Cell | Contains humans, vampires, werewolves
in_bounds(r, c) | bool | Checks coordinate validity
is_adjacent(r1, c1, r2, c2) | bool | Validates adjacency

---

## Model Output Requirements

Agents must return a list of moves, each defined as:

[(r1, c1, r2, c2, num), ...]

- Each tuple represents one movement of a specific number of units.
- An empty list [] means the agent skips the turn.
- All moves must target adjacent cells (validated by the engine).

---

## Headless Training Mode

Example for running AI vs AI without the GUI:

from core.state import GameState
from core.random_agent import RandomAgent
from core.my_model_agent import MyModelAgent

state = GameState(20, 20, 0.3)
vamp = MyModelAgent("weights.pth")
wolf = RandomAgent()

while not state.check_end_condition():
    agent = vamp if state.turn == "V" else wolf
    actions = agent.select_action(state)
    for a in actions:
        state.move_group(*a)
    state.next_turn()

print(state.check_end_condition())

---

## Summary

- Modular, extensible AI strategy environment
- Compatible with any Python machine learning model
- Supports both graphical and headless execution
- Provides a clear and consistent agent interface

To integrate your model:
1. Inherit from Agent
2. Implement select_action(state)
3. Return a list of valid moves
4. Run the environment to begin evaluation
