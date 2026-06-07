"""Representacao das acoes possiveis no jogo."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SetupAction:
    """Acao na fase de setup: colocar professor numa celula."""
    row: int
    col: int


@dataclass(frozen=True)
class TurnAction:
    """Acao na fase de turno: mover professor e opcionalmente orientar.

    mentor_at = None significa jogada de vitoria (chegou em celula nivel 4).
    """
    professor: str
    move_to_row: int
    move_to_col: int
    mentor_at_row: Optional[int] = None
    mentor_at_col: Optional[int] = None

    @property
    def is_winning_move(self) -> bool:
        return self.mentor_at_row is None and self.mentor_at_col is None
