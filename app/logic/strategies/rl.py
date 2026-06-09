"""Estrategia: agente treinado por Reinforcement Learning.

Carrega um modelo MaskablePPO treinado e usa para escolher jogadas.
O carregamento eh feito na inicializacao (singleton) para evitar
carregar a cada requisicao.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.schemas import (
    PlayerTurnResponse,
    Position,
    SetupResponse,
    TeamID,
)
from app.logic.strategies.heuristic import heuristic_setup, heuristic_turn


# ===== Configuracao =====

MODEL_PATH = os.getenv(
    "RL_MODEL_PATH",
    "training/models/ppo_vs_random_final.zip",
)

DETERMINISTIC = True


# ===== Constantes do encoding (devem casar com training/envs/encoding.py) =====

BOARD_SIZE = 5
WINNING_LEVEL = 4
NUM_CHANNELS = 9

ALL_PROFESSORS = ["CLARO", "REY", "KARIN", "BEATRIZ"]
PROFESSORS_BY_TEAM = {
    1: ["CLARO", "REY"],
    2: ["KARIN", "BEATRIZ"],
}

SETUP_ACTIONS_COUNT = BOARD_SIZE * BOARD_SIZE  # 25
TURN_ACTIONS_COUNT = 4 * 25 * 26  # 2600
TOTAL_ACTIONS = SETUP_ACTIONS_COUNT + TURN_ACTIONS_COUNT  # 2625


# ===== Singleton do modelo =====

_model = None
_load_attempted = False
_load_error = None


def _load_model():
    """Carrega o modelo na primeira chamada. Singleton."""
    global _model, _load_attempted, _load_error

    if _load_attempted:
        return _model

    _load_attempted = True

    if not Path(MODEL_PATH).exists():
        _load_error = f"Modelo nao encontrado em {MODEL_PATH}"
        print(f"[RL] AVISO: {_load_error}")
        print("[RL] Caindo para heuristica como fallback")
        return None

    try:
        from sb3_contrib import MaskablePPO

        print(f"[RL] Carregando modelo de {MODEL_PATH}...")
        _model = MaskablePPO.load(MODEL_PATH)
        print("[RL] Modelo carregado com sucesso")
        return _model
    except Exception as e:
        _load_error = f"Erro ao carregar modelo: {e}"
        print(f"[RL] ERRO: {_load_error}")
        print("[RL] Caindo para heuristica como fallback")
        return None


# ===== Encoding (espelha training/envs/encoding.py) =====

def _encode_state(board, my_team: int, phase: str) -> np.ndarray:
    """Codifica o estado como tensor 5x5x9 (igual ao treino)."""
    obs = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)

    my_profs = set(PROFESSORS_BY_TEAM[my_team])

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = board[r][c]

            obs[r, c, 0] = cell.level / WINNING_LEVEL

            if cell.professor is not None:
                idx = ALL_PROFESSORS.index(cell.professor)
                obs[r, c, 1 + idx] = 1.0
            else:
                obs[r, c, 5] = 1.0

            if cell.professor in my_profs:
                obs[r, c, 6] = 1.0

            if phase == "player_turn":
                obs[r, c, 7] = 1.0

            obs[r, c, 8] = 0.0 if my_team == 1 else 1.0

    return obs


def _setup_action_to_index(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def _index_to_setup_action(index: int) -> tuple:
    return (index // BOARD_SIZE, index % BOARD_SIZE)


def _turn_action_to_index(prof: str, move_r: int, move_c: int, mr, mc) -> int:
    prof_idx = ALL_PROFESSORS.index(prof)
    move_idx = move_r * BOARD_SIZE + move_c
    mentor_idx = 25 if mr is None else mr * BOARD_SIZE + mc
    raw = prof_idx * (25 * 26) + move_idx * 26 + mentor_idx
    return SETUP_ACTIONS_COUNT + raw


def _index_to_turn_action(index: int) -> dict:
    raw = index - SETUP_ACTIONS_COUNT
    prof_idx = raw // (25 * 26)
    rest = raw % (25 * 26)
    move_idx = rest // 26
    mentor_idx = rest % 26

    professor = ALL_PROFESSORS[prof_idx]
    move_row = move_idx // BOARD_SIZE
    move_col = move_idx % BOARD_SIZE

    if mentor_idx == 25:
        return {"professor": professor, "move_to": (move_row, move_col), "mentor_at": None}

    return {
        "professor": professor,
        "move_to": (move_row, move_col),
        "mentor_at": (mentor_idx // BOARD_SIZE, mentor_idx % BOARD_SIZE),
    }


# ===== Calculo da mascara de acoes validas =====

def _is_within(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _valid_setup_indices(board) -> list:
    return [
        _setup_action_to_index(r, c)
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if board[r][c].professor is None and board[r][c].level == 0
    ]


def _valid_turn_indices(board, team_id: int) -> list:
    result = []
    for prof in PROFESSORS_BY_TEAM[team_id]:
        prof_pos = None
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c].professor == prof:
                    prof_pos = (r, c)
                    break
            if prof_pos:
                break
        if not prof_pos:
            continue

        pr, pc = prof_pos
        current_level = board[pr][pc].level

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = pr + dr, pc + dc
                if not _is_within(nr, nc):
                    continue
                dest = board[nr][nc]
                if dest.professor is not None:
                    continue
                if abs(current_level - dest.level) > 1:
                    continue

                if dest.level == WINNING_LEVEL:
                    result.append(_turn_action_to_index(prof, nr, nc, None, None))
                    continue

                for mr_d in (-1, 0, 1):
                    for mc_d in (-1, 0, 1):
                        if mr_d == 0 and mc_d == 0:
                            continue
                        mr, mc = nr + mr_d, nc + mc_d
                        if not _is_within(mr, mc):
                            continue
                        mentor = board[mr][mc]
                        if mentor.professor is not None:
                            continue
                        if mentor.level >= WINNING_LEVEL:
                            continue
                        result.append(_turn_action_to_index(prof, nr, nc, mr, mc))

    return result


def _build_action_mask(board, my_team: int, phase: str) -> np.ndarray:
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
    indices = (
        _valid_setup_indices(board)
        if phase == "setup_placement"
        else _valid_turn_indices(board, my_team)
    )
    for idx in indices:
        mask[idx] = True
    return mask


# ===== Funcoes publicas =====

def rl_setup(board) -> Optional[SetupResponse]:
    """Escolhe posicao de setup usando o modelo RL."""
    model = _load_model()
    if model is None:
        print("[RL SETUP] Modelo nao carregado, usando heuristica")
        return heuristic_setup(board)

    try:
        print("[RL SETUP DEBUG] === Fase de setup ===")
        obs = _encode_state(board, my_team=1, phase="setup_placement")
        mask = _build_action_mask(board, my_team=1, phase="setup_placement")

        n_valid = int(mask.sum())
        print(f"[RL SETUP DEBUG] Posicoes validas para setup: {n_valid}")

        if not mask.any():
            print("[RL SETUP DEBUG] SEM POSICOES VALIDAS - retornando None")
            return None

        action, _ = model.predict(obs, action_masks=mask, deterministic=DETERMINISTIC)
        action = int(action)

        if action >= SETUP_ACTIONS_COUNT:
            print(f"[RL SETUP DEBUG] Acao invalida no setup: {action}, fallback")
            return heuristic_setup(board)

        row, col = _index_to_setup_action(action)
        print(f"[RL SETUP DEBUG] Posicionando em ({row},{col}) - nivel atual={board[row][col].level}")
        return SetupResponse(row=row, col=col)
    except Exception as e:
        import traceback
        print(f"[RL SETUP] Erro no setup: {e}")
        traceback.print_exc()
        print("[RL SETUP] Fallback heuristica")
        return heuristic_setup(board)


def rl_turn(board, team_id: TeamID) -> Optional[PlayerTurnResponse]:
    """Escolhe jogada usando o modelo RL."""
    model = _load_model()
    if model is None:
        print("[RL] Modelo nao carregado, usando heuristica")
        return heuristic_turn(board, team_id)

    try:
        team_int = int(team_id.value) if hasattr(team_id, "value") else int(team_id)

        # LOG: estado recebido
        print(f"\n[RL DEBUG] === Turno do time {team_int} ===")
        print("[RL DEBUG] Board recebido:")
        for r, row in enumerate(board):
            row_str = ""
            for c, cell in enumerate(row):
                prof = cell.professor[:3] if cell.professor else "..."
                row_str += f"[{cell.level}|{prof:>3}] "
            print(f"  {row_str}")

        # Posicoes dos meus professores
        my_profs = PROFESSORS_BY_TEAM[team_int]
        print(f"[RL DEBUG] Meus professores: {my_profs}")
        for prof in my_profs:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if board[r][c].professor == prof:
                        print(f"  {prof} em ({r},{c}) - nivel {board[r][c].level}")

        obs = _encode_state(board, my_team=team_int, phase="player_turn")
        mask = _build_action_mask(board, my_team=team_int, phase="player_turn")

        n_valid = int(mask.sum())
        print(f"[RL DEBUG] Acoes validas: {n_valid}")

        if not mask.any():
            print("[RL DEBUG] SEM ACOES VALIDAS - retornando None")
            return None

        action, _ = model.predict(obs, action_masks=mask, deterministic=DETERMINISTIC)
        action = int(action)

        if action < SETUP_ACTIONS_COUNT:
            print(f"[RL DEBUG] ERRO: acao de setup na fase de turno: {action}")
            return heuristic_turn(board, team_id)

        decoded = _index_to_turn_action(action)

        move_r, move_c = decoded["move_to"]
        response = PlayerTurnResponse(
            professor=decoded["professor"],
            move_to=Position(row=move_r, col=move_c),
            mentor_at=None,
        )

        if decoded["mentor_at"] is not None:
            mr, mc = decoded["mentor_at"]
            response.mentor_at = Position(row=mr, col=mc)

        # LOG: jogada escolhida
        print(f"[RL DEBUG] JOGADA: prof={response.professor}")
        print(f"  move_to=({move_r},{move_c}) nivel={board[move_r][move_c].level}")
        if response.mentor_at:
            mr, mc = response.mentor_at.row, response.mentor_at.col
            print(f"  mentor_at=({mr},{mc}) nivel atual={board[mr][mc].level}")
        else:
            print(f"  (jogada de vitoria - sem mentor_at)")

        return response
    except Exception as e:
        import traceback
        print(f"[RL] Erro no turn: {e}")
        traceback.print_exc()
        print("[RL] Fallback heuristica")
        return heuristic_turn(board, team_id)
