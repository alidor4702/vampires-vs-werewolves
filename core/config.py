# core/config.py
from dataclasses import dataclass

@dataclass
class GameConfig:
    grid_rows: int = 10
    grid_cols: int = 10
    cell_size: int = 50
    start_vampires: int = 10
    start_werewolves: int = 10
    human_density: float = 0.3
    max_turns: int = 100
    random_rows: bool = False
    random_cols: bool = False
    random_hum: bool = False
    mode: str = "human"  # "human" or "ai"
