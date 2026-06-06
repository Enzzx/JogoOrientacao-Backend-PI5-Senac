"""Estratégia: jogadas aleatórias mas sempre válidas."""

import random
from typing import List, Optional
from app.schemas import Cell, PlayerTurnResponse, Position, SetupResponse, TeamID
from app.logic.rules import (
    get_empty_low_level_cells,
    get_my_professors_positions,
    get_valid_mentor_positions,
    get_valid_moves_for_professor,
    is_winning_move,
)


def random_setup(board: List[List[Cell]]) -> Optional[SetupResponse]:
    """Escolhe uma posição aleatória vazia de nível 0."""
    candidates = get_empty_low_level_cells(board)
    if not candidates:
        return None
    pos = random.choice(candidates)
    return SetupResponse(row=pos.row, col=pos.col)


def random_turn(
    board: List[List[Cell]], team_id: TeamID
) -> Optional[PlayerTurnResponse]:
    """Escolhe um movimento aleatório válido para algum dos meus professores."""
    my_profs = get_my_professors_positions(board, team_id)
    if not my_profs:
        return None

    random.shuffle(my_profs)

    for prof_name, prof_pos in my_profs:
        valid_moves = get_valid_moves_for_professor(board, prof_pos)
        if not valid_moves:
            continue

        random.shuffle(valid_moves)
        for move_to in valid_moves:
            if is_winning_move(board, move_to):
                return PlayerTurnResponse(
                    professor=prof_name,
                    move_to=move_to,
                    mentor_at=None,
                )

            mentor_options = _mentor_options_after_move(board, prof_pos, move_to)
            if not mentor_options:
                continue

            mentor_pos = random.choice(mentor_options)
            return PlayerTurnResponse(
                professor=prof_name,
                move_to=move_to,
                mentor_at=mentor_pos,
            )

    return None


def _mentor_options_after_move(
    board: List[List[Cell]],
    from_pos: Position,
    to_pos: Position,
) -> List[Position]:
    """Calcula as posições válidas de orientação após o professor se mover."""
    return get_valid_mentor_positions(board, to_pos)
