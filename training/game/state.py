"""Representacao do estado do jogo.

Usa estruturas leves (listas, numpy arrays) para performance em treino.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy

from training.game.constants import (
    BOARD_SIZE,
    PROFESSORS_BY_TEAM,
    Phase,
    TEAM_TURING,
    TEAM_LOVELACE,
)


@dataclass
class Cell:
    level: int = 0
    professor: Optional[str] = None


@dataclass
class GameState:
    """Estado completo de uma partida."""

    # Tabuleiro: matriz BOARD_SIZE x BOARD_SIZE de Cells
    board: List[List[Cell]]

    # Fase atual: setup ou turn
    phase: Phase

    # De quem e o turno (1 = Turing, 2 = Lovelace)
    current_team: int

    # Contador de turnos
    turn_number: int = 1

    # Vencedor: 1, 2 ou None (ainda jogando ou empate)
    winner: Optional[int] = None

    # Lista de professores que ainda precisam ser posicionados (ordem alternada)
    setup_queue: List[str] = field(default_factory=list)

    @classmethod
    def initial(cls, first_team: int = TEAM_TURING) -> "GameState":
        """Cria estado inicial do jogo."""
        board = [
            [Cell(level=0, professor=None) for _ in range(BOARD_SIZE)]
            for _ in range(BOARD_SIZE)
        ]

        # Ordem de setup: alterna entre os times comecando pelo first_team
        opponent = TEAM_LOVELACE if first_team == TEAM_TURING else TEAM_TURING
        setup_queue = [
            PROFESSORS_BY_TEAM[first_team][0],
            PROFESSORS_BY_TEAM[opponent][0],
            PROFESSORS_BY_TEAM[first_team][1],
            PROFESSORS_BY_TEAM[opponent][1],
        ]

        return cls(
            board=board,
            phase=Phase.SETUP_PLACEMENT,
            current_team=first_team,
            turn_number=1,
            winner=None,
            setup_queue=setup_queue,
        )

    def copy(self) -> "GameState":
        """Copia profunda do estado (usado em busca de arvore)."""
        return copy.deepcopy(self)

    def get_professor_position(self, professor: str) -> Optional[Tuple[int, int]]:
        """Retorna (row, col) do professor ou None se nao esta no tabuleiro."""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c].professor == professor:
                    return (r, c)
        return None

    def get_team_professors_on_board(self, team: int) -> List[Tuple[str, int, int]]:
        """Retorna lista de (professor, row, col) do time no tabuleiro."""
        result = []
        for prof in PROFESSORS_BY_TEAM[team]:
            pos = self.get_professor_position(prof)
            if pos is not None:
                result.append((prof, pos[0], pos[1]))
        return result

    def professor_to_place(self) -> Optional[str]:
        """Retorna o proximo professor a ser posicionado, se houver."""
        if self.setup_queue:
            return self.setup_queue[0]
        return None

    def is_finished(self) -> bool:
        return self.winner is not None

    def to_dto(self) -> Dict:
        """Converte para formato similar ao que a API real usa (para debug/log)."""
        return {
            "board": [
                [{"level": c.level, "professor": c.professor} for c in row]
                for row in self.board
            ],
            "phase": self.phase.value,
            "current_team": self.current_team,
            "turn_number": self.turn_number,
            "winner": self.winner,
        }
