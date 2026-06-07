"""Constantes do Jogo da Orientacao."""

from enum import Enum

BOARD_SIZE = 5
WINNING_LEVEL = 4
MAX_TURNS = 100  # limite para evitar partidas infinitas

TEAM_TURING = 1
TEAM_LOVELACE = 2

PROFESSORS_BY_TEAM = {
    TEAM_TURING: ["CLARO", "REY"],
    TEAM_LOVELACE: ["KARIN", "BEATRIZ"],
}

ALL_PROFESSORS = ["CLARO", "REY", "KARIN", "BEATRIZ"]


class Phase(str, Enum):
    SETUP_PLACEMENT = "setup_placement"
    PLAYER_TURN = "player_turn"


class GameStatus(str, Enum):
    PLAYING = "PLAYING"
    FINISHED = "FINISHED"


def get_opponent(team):
    return TEAM_LOVELACE if team == TEAM_TURING else TEAM_TURING


def get_professor_team(professor):
    for team, profs in PROFESSORS_BY_TEAM.items():
        if professor in profs:
            return team
    return None
