"""Estratégia: Minimax com poda alfa-beta.

TODO: implementação completa. Por enquanto delega para a heurística.
"""

from typing import List, Optional
from app.schemas import Cell, PlayerTurnResponse, SetupResponse, TeamID
from app.logic.strategies.heuristic import heuristic_setup, heuristic_turn


def minimax_setup(board: List[List[Cell]]) -> Optional[SetupResponse]:
    """Placeholder: usa heurística por enquanto."""
    return heuristic_setup(board)


def minimax_turn(
    board: List[List[Cell]], team_id: TeamID
) -> Optional[PlayerTurnResponse]:
    """Placeholder: usa heurística por enquanto.

    TODO: implementar minimax real com:
    - Árvore de decisão
    - Poda alfa-beta
    - Função de avaliação do estado completo
    - Limite de tempo (MINIMAX_TIME_LIMIT_SECONDS)
    """
    return heuristic_turn(board, team_id)
