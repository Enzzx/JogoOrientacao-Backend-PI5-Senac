"""Funcoes de recompensa para o ambiente Gymnasium.

Estrutura de recompensas:
- Recompensa terminal: +/- baseado em vitoria/derrota
- Recompensas intermediarias (shaping): sinais menores durante o jogo
  que aceleram o aprendizado sem alterar a politica otima.

Constantes podem ser ajustadas para experimentar.
"""

from dataclasses import dataclass
from typing import Optional

from training.game import GameEngine, GameState
from training.game.constants import (
    BOARD_SIZE,
    PROFESSORS_BY_TEAM,
    WINNING_LEVEL,
    get_opponent,
)


# ===== Pesos das recompensas =====

@dataclass
class RewardConfig:
    """Pesos das recompensas. Permite tunar facilmente."""

    # Terminal (fim de jogo)
    win: float = 10.0
    loss: float = -10.0
    draw: float = -1.0          # leve penalidade para empate (incentiva ganhar)

    # Acao invalida
    invalid_action: float = -10.0

    # Intermediarias (shaping)
    raised_level_self: float = 0.1            # subi nivel de um aluno
    raised_level_to_3: float = 0.3            # subi pra nivel 3 (perto da vitoria)
    raised_level_to_4_safe: float = 0.5       # criei nivel 4 que so meu time alcanca
    raised_level_to_4_dangerous: float = -0.8 # criei nivel 4 que adversario alcanca

    move_closer_to_high_cell: float = 0.05    # me aproximei de celula nivel alto
    blocked_opponent_winning_move: float = 0.4

    # Diversidade de professores
    used_different_professor: float = 0.5     # bonus generoso por trocar de professor
    used_same_professor_streak: float = -0.5  # penalidade pesada por insistir no mesmo
    professor_idle_penalty_per_turn: float = -0.2  # penalidade forte por professor parado

    # Penalidade por turno (incentiva ganhar rapido)
    per_turn: float = -0.01


DEFAULT_REWARD_CONFIG = RewardConfig()


# ===== Calculo da recompensa =====

def compute_terminal_reward(
    state: GameState,
    my_team: int,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Recompensa final do episodio.

    state.winner:
      - my_team  -> vitoria
      - opponent -> derrota
      - 0        -> empate (limite de turnos)
      - None     -> nao terminou (nao deveria ser chamado nesse caso)
    """
    if state.winner is None:
        return 0.0
    if state.winner == my_team:
        return config.win
    if state.winner == 0:
        return config.draw
    return config.loss


def compute_step_reward(
    state_before: GameState,
    state_after: GameState,
    my_team: int,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
    professor_used: Optional[str] = None,
    last_professor: Optional[str] = None,
    prof_last_used_turn: Optional[dict] = None,
) -> float:
    """Recompensa intermediaria apos uma acao do agente.

    Compara o estado antes e depois da jogada para emitir sinais relevantes.

    NAO inclui recompensa terminal — essa eh calculada separadamente.
    """
    reward = 0.0

    # Penalidade leve por turno (incentiva resolver rapido)
    reward += config.per_turn

    # Detecta orientacao (subir nivel de aluno)
    level_delta_info = _detect_level_raise(state_before, state_after)
    if level_delta_info is not None:
        row, col, new_level = level_delta_info
        reward += config.raised_level_self

        if new_level == 3:
            reward += config.raised_level_to_3
        elif new_level == WINNING_LEVEL:
            # Quem consegue alcancar esse nivel 4?
            my_reach = _team_can_reach(state_after, my_team, row, col)
            opp_reach = _team_can_reach(state_after, get_opponent(my_team), row, col)

            if my_reach and not opp_reach:
                reward += config.raised_level_to_4_safe
            elif opp_reach:
                reward += config.raised_level_to_4_dangerous

    # Aproximacao de celulas com nivel alto
    if _moved_closer_to_high_cell(state_before, state_after, my_team):
        reward += config.move_closer_to_high_cell

    # Bloqueou jogada de vitoria do adversario?
    if _blocked_opponent_winning_move(state_before, state_after, my_team):
        reward += config.blocked_opponent_winning_move

    # Diversidade de professores: bonus/penalidade por troca
    if professor_used is not None and last_professor is not None:
        if professor_used != last_professor:
            reward += config.used_different_professor
        else:
            reward += config.used_same_professor_streak

    # Penalidade por professor ocioso
    if prof_last_used_turn is not None:
        my_profs = PROFESSORS_BY_TEAM[my_team]
        current_turn = state_after.turn_number
        for prof in my_profs:
            if state_after.get_professor_position(prof) is None:
                continue
            last_used = prof_last_used_turn.get(prof, 0)
            turns_idle = current_turn - last_used
            if last_used > 0 and turns_idle > 2:
                reward += config.professor_idle_penalty_per_turn * (turns_idle - 2)
            elif last_used == 0 and current_turn > 4:
                reward += config.professor_idle_penalty_per_turn * (current_turn - 4)

    return reward


# ===== Helpers =====

def _detect_level_raise(
    state_before: GameState, state_after: GameState
) -> Optional[tuple]:
    """Detecta se alguma celula teve o nivel aumentado.

    Retorna (row, col, novo_nivel) se sim, None caso contrario.
    """
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            before = state_before.board[r][c].level
            after = state_after.board[r][c].level
            if after > before:
                return (r, c, after)
    return None


def _team_can_reach(state: GameState, team: int, row: int, col: int) -> bool:
    """Verifica se algum professor do time consegue alcancar a celula
    no proximo turno (movimento valido)."""
    target_cell = state.board[row][col]

    for prof in PROFESSORS_BY_TEAM[team]:
        pos = state.get_professor_position(prof)
        if pos is None:
            continue
        pr, pc = pos

        if not (abs(pr - row) <= 1 and abs(pc - col) <= 1):
            continue
        if pr == row and pc == col:
            continue

        from_cell = state.board[pr][pc]
        if abs(from_cell.level - target_cell.level) > 1:
            continue

        return True
    return False


def _moved_closer_to_high_cell(
    state_before: GameState, state_after: GameState, my_team: int
) -> bool:
    """Verifica se algum professor meu se aproximou de uma celula nivel >= 2."""
    before_distance = _avg_min_distance_to_high_cells(state_before, my_team)
    after_distance = _avg_min_distance_to_high_cells(state_after, my_team)

    if before_distance is None or after_distance is None:
        return False

    return after_distance < before_distance


def _avg_min_distance_to_high_cells(
    state: GameState, team: int, min_level: int = 2
) -> Optional[float]:
    """Distancia media dos meus professores ate a celula de nivel alto mais proxima."""
    high_cells = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if state.board[r][c].level >= min_level:
                high_cells.append((r, c))

    if not high_cells:
        return None

    distances = []
    for prof in PROFESSORS_BY_TEAM[team]:
        pos = state.get_professor_position(prof)
        if pos is None:
            continue
        pr, pc = pos
        min_dist = min(
            max(abs(pr - hr), abs(pc - hc))  # distancia de Chebyshev (king)
            for hr, hc in high_cells
        )
        distances.append(min_dist)

    if not distances:
        return None
    return sum(distances) / len(distances)


def _blocked_opponent_winning_move(
    state_before: GameState, state_after: GameState, my_team: int
) -> bool:
    """Detecta se a jogada bloqueou uma celula que era vitoria do adversario.

    Heuristica simples: se antes da jogada havia uma celula de nivel 4 que
    o adversario podia alcancar, e depois ela foi ocupada por mim ou ficou
    inalcancavel, considera bloqueio.
    """
    opp = get_opponent(my_team)

    threats_before = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = state_before.board[r][c]
            if cell.level == WINNING_LEVEL and cell.professor is None:
                if _team_can_reach(state_before, opp, r, c):
                    threats_before.append((r, c))

    if not threats_before:
        return False

    for r, c in threats_before:
        cell_after = state_after.board[r][c]
        # Celula ocupada por mim agora = bloqueado
        if cell_after.professor is not None and cell_after.professor in PROFESSORS_BY_TEAM[my_team]:
            return True
        # Adversario nao alcanca mais
        if not _team_can_reach(state_after, opp, r, c):
            return True

    return False
