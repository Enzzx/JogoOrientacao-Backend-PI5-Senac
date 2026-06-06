"""Regras do Jogo da Orientação."""

from typing import List, Optional, Tuple
from app.schemas import Cell, Position, TeamID
from app.logic.config import (
    TEAM_TURING_PROFESSORS,
    TEAM_LOVELACE_PROFESSORS,
    WINNING_LEVEL,
)


def get_team_professors(team_id: TeamID) -> List[str]:
    """Retorna os nomes dos professores do time."""
    if team_id == TeamID.TURING:
        return TEAM_TURING_PROFESSORS
    return TEAM_LOVELACE_PROFESSORS


def get_opponent_professors(team_id: TeamID) -> List[str]:
    """Retorna os nomes dos professores do time adversário."""
    if team_id == TeamID.TURING:
        return TEAM_LOVELACE_PROFESSORS
    return TEAM_TURING_PROFESSORS


def find_professor_position(
    board: List[List[Cell]], professor: str
) -> Optional[Position]:
    """Encontra a posição de um professor no tabuleiro."""
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell.professor == professor:
                return Position(row=r, col=c)
    return None


def get_my_professors_positions(
    board: List[List[Cell]], team_id: TeamID
) -> List[Tuple[str, Position]]:
    """Retorna lista de (nome, posição) dos meus professores no tabuleiro."""
    result = []
    for prof in get_team_professors(team_id):
        pos = find_professor_position(board, prof)
        if pos is not None:
            result.append((prof, pos))
    return result


def is_within_board(board: List[List[Cell]], row: int, col: int) -> bool:
    """Verifica se a coordenada está dentro do tabuleiro."""
    if row < 0 or row >= len(board):
        return False
    if col < 0 or col >= len(board[0]):
        return False
    return True


def get_adjacent_positions(
    board: List[List[Cell]], pos: Position
) -> List[Position]:
    """Retorna todas as posições adjacentes (8 direções) dentro do tabuleiro."""
    deltas = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]
    result = []
    for dr, dc in deltas:
        nr, nc = pos.row + dr, pos.col + dc
        if is_within_board(board, nr, nc):
            result.append(Position(row=nr, col=nc))
    return result


def can_professor_move_to(
    board: List[List[Cell]],
    from_pos: Position,
    to_pos: Position,
) -> bool:
    """Verifica se um professor pode se mover de from_pos para to_pos.

    Regras:
    - to_pos deve estar adjacente a from_pos
    - to_pos não pode estar ocupada por outro professor
    - diferença de nível entre as células deve ser <= 1
    """
    if from_pos.row == to_pos.row and from_pos.col == to_pos.col:
        return False

    if abs(from_pos.row - to_pos.row) > 1 or abs(from_pos.col - to_pos.col) > 1:
        return False

    from_cell = board[from_pos.row][from_pos.col]
    to_cell = board[to_pos.row][to_pos.col]

    if to_cell.professor is not None:
        return False

    if abs(from_cell.level - to_cell.level) > 1:
        return False

    return True


def get_valid_moves_for_professor(
    board: List[List[Cell]], professor_pos: Position
) -> List[Position]:
    """Retorna todas as posições válidas de movimento para o professor."""
    return [
        p for p in get_adjacent_positions(board, professor_pos)
        if can_professor_move_to(board, professor_pos, p)
    ]


def get_valid_mentor_positions(
    board: List[List[Cell]], professor_pos: Position
) -> List[Position]:
    """Retorna as posições onde o professor pode orientar (aumentar nível).

    Regras:
    - Adjacente ao professor (depois do movimento)
    - Não pode ter professor na célula
    - Nível atual < MAX_LEVEL (não dá pra aumentar acima de 4)
    """
    result = []
    for pos in get_adjacent_positions(board, professor_pos):
        cell = board[pos.row][pos.col]
        if cell.professor is not None:
            continue
        if cell.level >= WINNING_LEVEL:
            continue
        result.append(pos)
    return result


def is_winning_move(
    board: List[List[Cell]], move_to: Position
) -> bool:
    """Verifica se mover para essa posição é uma jogada de vitória.

    Vitória = professor chega em célula de nível MAX (4).
    """
    cell = board[move_to.row][move_to.col]
    return cell.level == WINNING_LEVEL


def get_empty_low_level_cells(
    board: List[List[Cell]],
) -> List[Position]:
    """Retorna células vazias e de nível 0 (candidatas para setup)."""
    result = []
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell.professor is None and cell.level == 0:
                result.append(Position(row=r, col=c))
    return result
