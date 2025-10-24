# 🧛‍♂️ Vampires vs Werewolves — AI Battle Arena

A turn-based strategy game designed to train, test, and visualize **AI decision-making models** in a controlled environment.  
Built entirely in **Python (Tkinter + NumPy)**, this project supports **human vs human**, **human vs AI**, and **AI vs AI** modes.  

---

## 🎮 Overview

Vampires and Werewolves compete for domination on a grid-based map.  
Each cell can contain:
- **Humans** (neutral, convertible units)
- **Vampires**
- **Werewolves**

Players take turns moving their creatures to adjacent cells (8 possible directions).  
Combat and conversion follow probabilistic and deterministic rules inspired by resource-based strategy games.

---

## ⚙️ Features

| Category | Description |
|-----------|-------------|
| 🧱 **Grid Engine** | Supports grids up to **256×256** |
| 🔁 **Turn System** | Vampires (V) and Werewolves (W) alternate turns |
| 🧩 **Adjacency Movement** | Move in 8 directions (orthogonal + diagonal) |
| ⚔️ **Battle Rules** | Probabilistic outcomes based on stack sizes |
| ➗ **Multi-Move Turns** | Split stacks and perform multiple actions |
| ⏸️ **Skip Turn Option** | Agents can skip a turn |
| 🔍 **Pan & Zoom** | Scroll, drag, and use ⌘+/⌘− or arrows to navigate |
| 📜 **Live Game Log** | Displays all moves, attacks, conversions, results |
| 🔁 **Quick Restart** | Restart with same or randomized parameters |
| 🧠 **AI Port** | Plug any AI model (MCTS, RL, policy network, etc.) |
| 🎲 **Random Agent** | Built-in random baseline AI |

---

## 🧩 Project Structure

```
project/
├── core/
│   ├── state.py           # Game logic & rules
│   ├── config.py          # Configuration dataclass
│   ├── agent_base.py      # Abstract AI interface
│   ├── random_agent.py    # Baseline random AI agent
│   └── mcts_agent.py      # Placeholder for future MCTS
│
├── gui/
│   ├── menu.py            # Main menu (parameters, random toggles, play modes)
│   ├── board.py           # Game board, event handling, AI execution
│
├── main.py                # Entry point
├── testmap2.xml           # Example map
├── thetrap.xml            # Example map
└── README.md
```

---

## 🧠 Game Rules

### 🎯 Movement
- Move to any of the 8 adjacent cells.
- Split stacks freely (e.g., move 3 left, 4 up, 3 stay).
- Once a stack or sub-stack moves, it cannot move again that turn.

### ⚔️ Combat
| Situation | Outcome |
|------------|----------|
| **Empty Cell** | Units simply move in. |
| **Humans** | If attackers ≥ humans → all convert.<br>Otherwise, probability `P` decides per-unit outcomes.<br>`P = E1/(2E2)` if weaker, or `(E1/E2) - 0.5` if stronger. |
| **Enemy Units** | If attackers ≥ 1.5× defenders → defenders die.<br>If defenders ≥ 1.5× attackers → attackers die.<br>Otherwise, each unit’s fate decided using `P`. |

### 🏁 Turn End
- Press **Spacebar** or click **Next Turn**.
- Game automatically detects **win/draw** when one or both species are extinct.

---

## 🖥️ Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/vampires-vs-werewolves.git
cd vampires-vs-werewolves
```

### 2️⃣ Create Environment & Install
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy
```

### 3️⃣ Run
```bash
python main.py
```

---

## 🎮 Controls

| Action | Key / Mouse |
|--------|--------------|
| **Select cell** | Left click |
| **Pan map** | Middle drag / Arrow keys |
| **Zoom** | Mouse wheel / ⌘ + / ⌘ − |
| **Next Turn** | Spacebar |
| **Restart Game** | Restart button |
| **Back to Menu** | Back button |

---

## 🧩 AI Integration

### Base Agent Interface
All AI models must inherit from `Agent` in `core/agent_base.py` and implement:
```python
def select_action(self, state) -> list[tuple[int,int,int,int,int]]:
    '''
    Decide moves given current GameState.
    Return [(r1,c1,r2,c2,num), ...] or [] to skip turn.
    '''
```

---

## 📊 What Models Receive as Input

Each time `.select_action(state)` is called, your model receives a **GameState** object.

| Attribute | Type | Description |
|------------|------|-------------|
| `state.rows`, `state.cols` | int | Board size |
| `state.turn` | str | `"V"` or `"W"` |
| `state.grid[r][c]` | Cell | Holds `humans`, `vampires`, `werewolves` |
| `state.in_bounds(r,c)` | bool | Valid coordinate check |
| `state.is_adjacent(r1,c1,r2,c2)` | bool | Checks move validity |

---

## 🧠 Model Output Requirements

The agent must output a list of moves:
```python
[(r1, c1, r2, c2, num), (r3, c3, r4, c4, num2), ...]
```

- Each tuple = one move  
- Return an empty list `[]` to skip turn  
- Moves must be adjacent (engine revalidates)

---

## 🧪 Headless Training Mode

To simulate AI vs AI without GUI:
```python
from core.state import GameState
from core.random_agent import RandomAgent
from core.my_model_agent import MyModelAgent

state = GameState(20, 20, 0.3)
vamp, wolf = MyModelAgent("weights.pth"), RandomAgent()

while not state.check_end_condition():
    agent = vamp if state.turn == "V" else wolf
    actions = agent.select_action(state)
    for a in actions:
        state.move_group(*a)
    state.next_turn()

print(state.check_end_condition())
```

---

## 🧱 Summary

✅ Fully functional, modular, expandable AI strategy environment  
✅ Works with any Python ML model  
✅ Supports visual + headless play modes  
✅ Clear agent interface for consistent integration  

To plug in your model:
1. Inherit from `Agent`  
2. Implement `select_action(state)`  
3. Return a list of legal moves  
4. Run the game — you’re live 🎮
