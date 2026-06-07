"""Lógica de decisão do jogador inteligente.

Expõe as funções choose_setup e choose_turn, que despacham para a
estratégia ativa configurada em config.py.
"""

from typing import List, Optional
from app.schemas import Cell, PlayerTurnResponse, SetupResponse, TeamID
from app.logic.config import ACTIVE_STRATEGY
from app.logic.strategies.random_bot import random_setup, random_turn
from app.logic.strategies.heuristic import heuristic_setup, heuristic_turn
from app.logic.strategies.minimax import minimax_setup, minimax_turn
from app.logic.strategies.rl import rl_setup, rl_turn


_STRATEGIES = {
    "random": (random_setup, random_turn),
    "heuristic": (heuristic_setup, heuristic_turn),
    "minimax": (minimax_setup, minimax_turn),
    "rl": (rl_setup, rl_turn),
}


def choose_setup(board: List[List[Cell]]) -> Optional[SetupResponse]:
    """Escolhe a posição de setup usando a estratégia ativa."""
    setup_fn, _ = _STRATEGIES.get(ACTIVE_STRATEGY, _STRATEGIES["random"])
    return setup_fn(board)


def choose_turn(
    board: List[List[Cell]], team_id: TeamID
) -> Optional[PlayerTurnResponse]:
    """Escolhe a jogada do turno usando a estratégia ativa."""
    _, turn_fn = _STRATEGIES.get(ACTIVE_STRATEGY, _STRATEGIES["random"])
    return turn_fn(board, team_id)
