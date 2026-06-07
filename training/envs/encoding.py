"""Codificacao do estado e acoes para o ambiente Gymnasium."""

from typing import List, Optional, Tuple

import numpy as np

from training.game import (
    BOARD_SIZE,
    GameEngine,
    GameState,
    Phase,
    PROFESSORS_BY_TEAM,
    SetupAction,
    TurnAction,
    WINNING_LEVEL,
)
from training.game.constants import ALL_PROFESSORS

# ===== Observacao (estado -> tensor) =====
#
# Observacao tem formato 5x5xN canais:
# - Canal 0: nivel da celula (normalizado 0..1)
# - Canal 1..4: one-hot do professor (CLARO, REY, KARIN, BEATRIZ)
# - Canal 5: 1 se a celula esta vazia
# - Canal 6: 1 se eh meu time, 0 se eh adversario (no canal de professor)
# - Canal 7: fase do jogo (0 = setup, 1 = turn)
# - Canal 8: meu time (0 = Turing, 1 = Lovelace)
#
# Plus features globais como turn_number normalizado.

NUM_CHANNELS = 9


def encode_state(state: GameState, my_team: int) -> np.ndarray:
    """Codifica o estado do jogo como tensor (BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS).

    Args:
        state: estado atual do jogo
        my_team: meu time (1 = Turing, 2 = Lovelace)

    Returns:
        Tensor numpy float32 de shape (5, 5, 9)
    """
    obs = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)

    my_profs = set(PROFESSORS_BY_TEAM[my_team])

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = state.board[r][c]

            # Canal 0: nivel normalizado
            obs[r, c, 0] = cell.level / WINNING_LEVEL

            # Canais 1-4: one-hot do professor
            if cell.professor is not None:
                idx = ALL_PROFESSORS.index(cell.professor)
                obs[r, c, 1 + idx] = 1.0
            else:
                # Canal 5: celula vazia
                obs[r, c, 5] = 1.0

            # Canal 6: pertence ao meu time
            if cell.professor in my_profs:
                obs[r, c, 6] = 1.0

            # Canal 7: fase
            if state.phase == Phase.PLAYER_TURN:
                obs[r, c, 7] = 1.0

            # Canal 8: meu time (0 = Turing, 1 = Lovelace)
            obs[r, c, 8] = 0.0 if my_team == 1 else 1.0

    return obs


# ===== Acao (indice -> acao do jogo) =====
#
# O espaco de acao precisa ser FIXO (rede neural tem tamanho fixo).
# Solucao:
#   - SETUP: 25 acoes (uma para cada celula do tabuleiro 5x5)
#   - TURN: a acao e composta de:
#       * Professor (4 opcoes: CLARO, REY, KARIN, BEATRIZ)
#       * Move to (25 celulas)
#       * Mentor at (26 opcoes: 25 celulas + 1 "sem mentor" para jogada de vitoria)
#     Total = 4 * 25 * 26 = 2600 acoes possiveis
#
# Vamos usar um espaco de acao unificado:
#   - Indices 0..24 = setup (row, col)
#   - Indices 25..2624 = turn (encoded)
#
# Acoes invalidas sao mascaradas pelo agente (action masking).

SETUP_ACTIONS_COUNT = BOARD_SIZE * BOARD_SIZE  # 25
TURN_ACTIONS_COUNT = 4 * 25 * 26  # 2600
TOTAL_ACTIONS = SETUP_ACTIONS_COUNT + TURN_ACTIONS_COUNT  # 2625


def setup_action_to_index(action: SetupAction) -> int:
    """Codifica SetupAction como indice."""
    return action.row * BOARD_SIZE + action.col


def index_to_setup_action(index: int) -> SetupAction:
    """Decodifica indice como SetupAction."""
    row = index // BOARD_SIZE
    col = index % BOARD_SIZE
    return SetupAction(row=row, col=col)


def turn_action_to_index(action: TurnAction) -> int:
    """Codifica TurnAction como indice (offset SETUP_ACTIONS_COUNT)."""
    prof_idx = ALL_PROFESSORS.index(action.professor)
    move_idx = action.move_to_row * BOARD_SIZE + action.move_to_col

    if action.mentor_at_row is None:
        mentor_idx = 25  # sem mentor (jogada de vitoria)
    else:
        mentor_idx = action.mentor_at_row * BOARD_SIZE + action.mentor_at_col

    raw = prof_idx * (25 * 26) + move_idx * 26 + mentor_idx
    return SETUP_ACTIONS_COUNT + raw


def index_to_turn_action(index: int) -> TurnAction:
    """Decodifica indice como TurnAction."""
    raw = index - SETUP_ACTIONS_COUNT

    prof_idx = raw // (25 * 26)
    rest = raw % (25 * 26)
    move_idx = rest // 26
    mentor_idx = rest % 26

    professor = ALL_PROFESSORS[prof_idx]
    move_row = move_idx // BOARD_SIZE
    move_col = move_idx % BOARD_SIZE

    if mentor_idx == 25:
        return TurnAction(
            professor=professor,
            move_to_row=move_row,
            move_to_col=move_col,
            mentor_at_row=None,
            mentor_at_col=None,
        )

    mentor_row = mentor_idx // BOARD_SIZE
    mentor_col = mentor_idx % BOARD_SIZE

    return TurnAction(
        professor=professor,
        move_to_row=move_row,
        move_to_col=move_col,
        mentor_at_row=mentor_row,
        mentor_at_col=mentor_col,
    )


def get_valid_action_mask(state: GameState) -> np.ndarray:
    """Retorna mascara booleana de quais indices de acao sao validos.

    True = valido, False = invalido.
    Shape: (TOTAL_ACTIONS,)
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)

    if state.phase == Phase.SETUP_PLACEMENT:
        for action in GameEngine.valid_setup_actions(state):
            idx = setup_action_to_index(action)
            mask[idx] = True
    else:
        for action in GameEngine.valid_turn_actions(state):
            idx = turn_action_to_index(action)
            mask[idx] = True

    return mask


def decode_action(index: int, state: GameState):
    """Decodifica indice como acao do jogo apropriada para a fase atual."""
    if index < SETUP_ACTIONS_COUNT:
        return index_to_setup_action(index)
    return index_to_turn_action(index)
