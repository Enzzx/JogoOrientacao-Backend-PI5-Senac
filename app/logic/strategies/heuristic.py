"""Estratégia: heurística simples baseada em prioridades."""

from typing import List, Optional, Tuple
from app.schemas import Cell, PlayerTurnResponse, Position, SetupResponse, TeamID
from app.logic.config import WINNING_LEVEL
from app.logic.rules import (
    get_empty_low_level_cells,
    get_my_professors_positions,
    get_opponent_professors,
    get_valid_mentor_positions,
    get_valid_moves_for_professor,
    is_winning_move,
)


def heuristic_setup(board: List[List[Cell]]) -> Optional[SetupResponse]:
    """Posiciona o professor o mais próximo do centro possível.

    Heurística: posições centrais dão mais mobilidade no jogo.
    """
    candidates = get_empty_low_level_cells(board)
    if not candidates:
        return None

    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    center = (rows / 2, cols / 2)

    def distance_to_center(pos: Position) -> float:
        return abs(pos.row - center[0]) + abs(pos.col - center[1])

    candidates.sort(key=distance_to_center)
    best = candidates[0]
    return SetupResponse(row=best.row, col=best.col)


def heuristic_turn(
    board: List[List[Cell]], team_id: TeamID
) -> Optional[PlayerTurnResponse]:
    """Escolhe a melhor jogada baseado em heurísticas:

    1. Se posso vencer agora → vence
    2. Senão escolhe jogada que maximiza score
    """
    my_profs = get_my_professors_positions(board, team_id)
    if not my_profs:
        return None

    # 1. Procura jogada de vitória
    for prof_name, prof_pos in my_profs:
        for move_to in get_valid_moves_for_professor(board, prof_pos):
            if is_winning_move(board, move_to):
                return PlayerTurnResponse(
                    professor=prof_name,
                    move_to=move_to,
                    mentor_at=None,
                )

    # 2. Gera todas as jogadas válidas e pontua
    all_moves: List[Tuple[float, PlayerTurnResponse]] = []

    for prof_name, prof_pos in my_profs:
        for move_to in get_valid_moves_for_professor(board, prof_pos):
            mentor_options = get_valid_mentor_positions(board, move_to)

            if not mentor_options:
                continue

            for mentor_at in mentor_options:
                jogada = PlayerTurnResponse(
                    professor=prof_name,
                    move_to=move_to,
                    mentor_at=mentor_at,
                )
                score = _score_move(board, jogada, team_id)
                all_moves.append((score, jogada))

    if not all_moves:
        return None

    all_moves.sort(key=lambda x: x[0], reverse=True)
    return all_moves[0][1]


def _score_move(
    board: List[List[Cell]],
    move: PlayerTurnResponse,
    team_id: TeamID,
) -> float:
    """Avalia o quão boa é uma jogada.

    Heurísticas:
    + Orientar uma célula de nível 3 (cria oportunidade de vitória adjacente)
    + Mover para perto de células com level alto
    - Deixar adversário com vitória fácil
    """
    score = 0.0

    target_cell = board[move.move_to.row][move.move_to.col]
    score += target_cell.level * 2

    if move.mentor_at:
        mentor_cell = board[move.mentor_at.row][move.mentor_at.col]
        new_level = mentor_cell.level + 1

        if new_level == WINNING_LEVEL:
            score += 5
        elif new_level == 3:
            score += 3
        else:
            score += new_level * 0.5

    if move.mentor_at:
        mentor_cell = board[move.mentor_at.row][move.mentor_at.col]
        new_level = mentor_cell.level + 1
        if new_level == WINNING_LEVEL:
            if _opponent_can_reach(board, move.mentor_at, team_id):
                score -= 10

    return score


def _opponent_can_reach(
    board: List[List[Cell]],
    target: Position,
    team_id: TeamID,
) -> bool:
    """Verifica se algum professor adversário consegue alcançar a posição alvo."""
    opponent_profs = get_opponent_professors(team_id)
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell.professor not in opponent_profs:
                continue
            if abs(r - target.row) <= 1 and abs(c - target.col) <= 1:
                cell_here = board[r][c]
                target_cell = board[target.row][target.col]
                if abs(cell_here.level - target_cell.level) <= 1:
                    return True
    return False
