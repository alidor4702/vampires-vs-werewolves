# core/agent_base.py
from abc import ABC, abstractmethod

class Agent(ABC):
    """Base interface for all AI agents."""

    @abstractmethod
    def select_action(self, state):
        """
        Decide one or more moves for the current turn.

        Parameters
        ----------
        state : GameState
            The current game state (read-only; copy if simulating).

        Returns
        -------
        list[tuple[int,int,int,int,int]] or []
            A list of moves, each (r1, c1, r2, c2, num).
            Return an empty list [] to skip the turn.
        """
        pass
