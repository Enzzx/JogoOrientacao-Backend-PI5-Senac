"""Estratégia: heurística simples baseada em prioridades."""

from typing import List, Optional, Tuple
from app.schemas import Cell, PlayerTurnResponse, Position, SetupResponse, TeamID
from app.logic.config import WINNING_LEVEL
from app.logic.rules import (
    find_professor_position,
    get_adjacent_positions,
    get_empty_low_level_cells,
    get_my_professors_positions,
    get_opponent_professors,
    get_team_professors,
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


def evaluate_board(board: List[List[Cell]], team_id: TeamID) -> float:
    """Avalia quão bom é o tabuleiro para team_id.

    Versão ofensiva: prioriza construir vantagem própria (caminho de
    níveis crescentes perto dos meus professores) em vez de só reagir.
    """
    opp_team = TeamID.LOVELACE if team_id == TeamID.TURING else TeamID.TURING
    my_profs = get_team_professors(team_id)
    opp_profs = get_team_professors(opp_team)

    score = 0.0
    my_best_prof_level = 0
    opp_best_prof_level = 0

    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell.professor in my_profs:
                score += cell.level * 3.0
                score += _professor_threat_value(board, r, c, cell.level)
                score += _progression_potential(board, r, c, cell.level)
                my_best_prof_level = max(my_best_prof_level, cell.level)
            elif cell.professor in opp_profs:
                score -= cell.level * 2.0
                score -= _professor_threat_value(board, r, c, cell.level)
                opp_best_prof_level = max(opp_best_prof_level, cell.level)

    # Corrida: incentiva tomar a iniciativa em vez de ficar em 0.0
    score += (my_best_prof_level - opp_best_prof_level) * 4.0

    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell.professor is not None:
                continue
            if cell.level == WINNING_LEVEL:
                if _team_can_reach_cell(board, team_id, r, c):
                    score += 6.0
                if _team_can_reach_cell(board, opp_team, r, c):
                    score -= 5.0
            elif cell.level == 3:
                if _team_can_reach_cell(board, team_id, r, c):
                    score += 2.5
                if _team_can_reach_cell(board, opp_team, r, c):
                    score -= 2.0

    return score


def _progression_potential(
    board: List[List[Cell]], row: int, col: int, prof_level: int
) -> float:
    """Recompensa ter células adjacentes que formam um caminho de subida.

    Incentiva construir uma 'escada' de níveis em vez de ficar parado.
    """
    value = 0.0
    for adj in get_adjacent_positions(board, Position(row=row, col=col)):
        adj_cell = board[adj.row][adj.col]
        if adj_cell.professor is not None:
            continue
        diff = adj_cell.level - prof_level
        if diff == 1:
            value += 1.5  # posso subir para cá
        elif diff == 0 and adj_cell.level > 0:
            value += 0.5
    return value


def _professor_threat_value(
    board: List[List[Cell]], row: int, col: int, level: int
) -> float:
    """Quão ameaçador é um professor baseado na proximidade de células altas."""
    value = 0.0
    for adj in get_adjacent_positions(board, Position(row=row, col=col)):
        adj_cell = board[adj.row][adj.col]
        if adj_cell.professor is None and abs(level - adj_cell.level) <= 1:
            if adj_cell.level == WINNING_LEVEL:
                value += 10.0
            elif adj_cell.level == 3:
                value += 2.0
    return value


def _team_can_reach_cell(
    board: List[List[Cell]], team_id: TeamID, row: int, col: int
) -> bool:
    """Verifica se algum professor do time consegue mover para (row, col)."""
    target_cell = board[row][col]
    for prof in get_team_professors(team_id):
        pos = find_professor_position(board, prof)
        if pos is None:
            continue
        if pos.row == row and pos.col == col:
            continue
        if abs(pos.row - row) <= 1 and abs(pos.col - col) <= 1:
            from_cell = board[pos.row][pos.col]
            if abs(from_cell.level - target_cell.level) <= 1:
                return True
    return False
