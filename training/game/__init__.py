"""Simulador local do Jogo da Orientacao para treinamento de RL."""

from training.game.engine import GameEngine
from training.game.state import GameState
from training.game.actions import SetupAction, TurnAction
from training.game.constants import (
    BOARD_SIZE,
    WINNING_LEVEL,
    TEAM_TURING,
    TEAM_LOVELACE,
    PROFESSORS_BY_TEAM,
    Phase,
    GameStatus,
)

__all__ = [
    "GameEngine",
    "GameState",
    "SetupAction",
    "TurnAction",
    "BOARD_SIZE",
    "WINNING_LEVEL",
    "TEAM_TURING",
    "TEAM_LOVELACE",
    "PROFESSORS_BY_TEAM",
    "Phase",
    "GameStatus",
]
