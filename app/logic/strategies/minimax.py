"""Estrategia: Minimax com poda alfa-beta, move ordering e top-K filtering.

Simula jogadas a frente alternando entre maximizar (meu turno) e
minimizar (turno do adversario). Usa evaluate_board nas folhas.
"""

import time
from typing import List, Optional

from app.schemas import Cell, PlayerTurnResponse, SetupResponse, TeamID
from app.logic.config import MINIMAX_DEPTH, MINIMAX_TIME_LIMIT_SECONDS
from app.logic.rules import (
    check_winner,
    generate_all_moves,
    get_opponent_team,
    simulate_turn,
)
from app.logic.strategies.heuristic import (
    _score_move,
    evaluate_board,
    heuristic_setup,
    heuristic_turn,
)


TOP_K_MOVES = 15

WIN_SCORE = 100_000.0
LOSS_SCORE = -100_000.0


def minimax_setup(board: List[List[Cell]]) -> Optional[SetupResponse]:
    """Setup usa a heurística (posiciona perto do centro)."""
    return heuristic_setup(board)


def minimax_turn(
    board: List[List[Cell]], team_id: TeamID
) -> Optional[PlayerTurnResponse]:
    """Escolhe a melhor jogada usando Minimax com poda alfa-beta."""
    start_time = time.time()

    try:
        moves = generate_all_moves(board, team_id)
        if not moves:
            return None

        # Vitória imediata: joga direto
        for move in moves:
            if move.mentor_at is None:
                return move

        # Move ordering: melhores jogadas primeiro
        scored = [(_score_move(board, m, team_id), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_moves = [m for _, m in scored[:TOP_K_MOVES]]

        best_move = top_moves[0]
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in top_moves:
            if time.time() - start_time > MINIMAX_TIME_LIMIT_SECONDS:
                print(f"[minimax] Limite de tempo atingido, usando melhor ate agora")
                break

            new_board = simulate_turn(board, move)
            value = _minimax(
                new_board,
                depth=MINIMAX_DEPTH - 1,
                alpha=alpha,
                beta=beta,
                maximizing=False,
                my_team=team_id,
                current_team=get_opponent_team(team_id),
                start_time=start_time,
            )

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

        elapsed = time.time() - start_time
        print(f"[minimax] Jogada em {elapsed:.2f}s (valor={best_value:.1f})")
        return best_move

    except Exception as e:
        import traceback
        print(f"[minimax] Erro: {e}")
        traceback.print_exc()
        print("[minimax] Fallback heuristica")
        return heuristic_turn(board, team_id)


def _minimax(
    board: List[List[Cell]],
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    my_team: TeamID,
    current_team: TeamID,
    start_time: float,
) -> float:
    """Recursão do minimax com poda alfa-beta."""
    winner = check_winner(board)
    if winner is not None:
        if winner == my_team:
            return WIN_SCORE + depth   # vitória mais rápida = melhor
        return LOSS_SCORE - depth      # derrota mais lenta = menos ruim

    if depth <= 0 or (time.time() - start_time > MINIMAX_TIME_LIMIT_SECONDS):
        return evaluate_board(board, my_team)

    moves = generate_all_moves(board, current_team)
    if not moves:
        if current_team == my_team:
            return LOSS_SCORE - depth
        return WIN_SCORE + depth

    scored = [(_score_move(board, m, current_team), m) for m in moves]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [m for _, m in scored[:TOP_K_MOVES]]

    next_team = get_opponent_team(current_team)

    if maximizing:
        value = float('-inf')
        for move in top:
            new_board = simulate_turn(board, move)
            value = max(value, _minimax(
                new_board, depth - 1, alpha, beta,
                False, my_team, next_team, start_time,
            ))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = float('inf')
        for move in top:
            new_board = simulate_turn(board, move)
            value = min(value, _minimax(
                new_board, depth - 1, alpha, beta,
                True, my_team, next_team, start_time,
            ))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
