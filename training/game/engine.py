"""Motor do Jogo da Orientacao.

Responsavel por:
- Listar acoes validas em cada estado
- Aplicar acoes e gerar novo estado
- Detectar fim de jogo
"""

from typing import List, Optional, Tuple
import random

from training.game.actions import SetupAction, TurnAction
from training.game.constants import (
    BOARD_SIZE,
    GameStatus,
    MAX_TURNS,
    Phase,
    PROFESSORS_BY_TEAM,
    WINNING_LEVEL,
    get_opponent,
    get_professor_team,
)
from training.game.state import Cell, GameState


class GameEngine:
    """Motor do jogo."""

    @staticmethod
    def is_within_board(row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    @staticmethod
    def is_adjacent(r1: int, c1: int, r2: int, c2: int) -> bool:
        """Adjacencia tipo king (8 direcoes)."""
        if r1 == r2 and c1 == c2:
            return False
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1

    # ===== Acoes validas =====

    @staticmethod
    def valid_setup_actions(state: GameState) -> List[SetupAction]:
        """Posicoes validas para posicionar o proximo professor."""
        if state.phase != Phase.SETUP_PLACEMENT:
            return []

        result = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = state.board[r][c]
                if cell.professor is None and cell.level == 0:
                    result.append(SetupAction(row=r, col=c))
        return result

    @staticmethod
    def valid_turn_actions(state: GameState) -> List[TurnAction]:
        """Todas as jogadas validas para o time da vez."""
        if state.phase != Phase.PLAYER_TURN:
            return []

        result = []
        team = state.current_team
        my_profs_on_board = state.get_team_professors_on_board(team)

        for prof, pr, pc in my_profs_on_board:
            current_cell = state.board[pr][pc]

            # Procura destinos validos (8 adjacentes)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = pr + dr, pc + dc
                    if not GameEngine.is_within_board(nr, nc):
                        continue

                    dest_cell = state.board[nr][nc]

                    # Nao pode ter outro professor
                    if dest_cell.professor is not None:
                        continue

                    # Diferenca de nivel <= 1
                    if abs(current_cell.level - dest_cell.level) > 1:
                        continue

                    # Se chegou em celula nivel 4 = vitoria (sem mentor)
                    if dest_cell.level == WINNING_LEVEL:
                        result.append(TurnAction(
                            professor=prof,
                            move_to_row=nr,
                            move_to_col=nc,
                            mentor_at_row=None,
                            mentor_at_col=None,
                        ))
                        continue

                    # Senao, precisa escolher mentor_at adjacente ao novo destino
                    for mr_d in (-1, 0, 1):
                        for mc_d in (-1, 0, 1):
                            if mr_d == 0 and mc_d == 0:
                                continue
                            mr, mc = nr + mr_d, nc + mc_d
                            if not GameEngine.is_within_board(mr, mc):
                                continue

                            mentor_cell = state.board[mr][mc]
                            if mentor_cell.professor is not None:
                                continue
                            if mentor_cell.level >= WINNING_LEVEL:
                                continue

                            result.append(TurnAction(
                                professor=prof,
                                move_to_row=nr,
                                move_to_col=nc,
                                mentor_at_row=mr,
                                mentor_at_col=mc,
                            ))

        return result

    # ===== Aplicar acao =====

    @staticmethod
    def apply_setup(state: GameState, action: SetupAction) -> GameState:
        """Aplica uma acao de setup e retorna novo estado."""
        if state.phase != Phase.SETUP_PLACEMENT:
            raise ValueError(f"Estado nao esta em fase de setup: {state.phase}")

        prof = state.professor_to_place()
        if prof is None:
            raise ValueError("Nao ha professor para posicionar")

        # Valida acao
        valid = GameEngine.valid_setup_actions(state)
        if action not in valid:
            raise ValueError(f"Acao de setup invalida: {action}")

        new_state = state.copy()
        new_state.board[action.row][action.col].professor = prof
        new_state.setup_queue.pop(0)

        # Proxima fase ou proximo setup
        if not new_state.setup_queue:
            new_state.phase = Phase.PLAYER_TURN
            new_state.current_team = state.current_team  # comeca quem fez ultimo setup
            # ou alterna - vamos alternar para nao dar vantagem
            new_state.current_team = get_opponent(state.current_team)
        else:
            # Proximo a posicionar pertence a qual time?
            next_prof = new_state.setup_queue[0]
            new_state.current_team = get_professor_team(next_prof)

        return new_state

    @staticmethod
    def apply_turn(state: GameState, action: TurnAction) -> GameState:
        """Aplica uma acao de turno e retorna novo estado."""
        if state.phase != Phase.PLAYER_TURN:
            raise ValueError(f"Estado nao esta em fase de turno: {state.phase}")

        valid = GameEngine.valid_turn_actions(state)
        if action not in valid:
            raise ValueError(f"Acao de turno invalida: {action}")

        new_state = state.copy()

        # Posicao atual do professor
        pos = new_state.get_professor_position(action.professor)
        if pos is None:
            raise ValueError(f"Professor nao encontrado: {action.professor}")
        pr, pc = pos

        # Move o professor
        new_state.board[pr][pc].professor = None
        new_state.board[action.move_to_row][action.move_to_col].professor = action.professor

        # Verifica vitoria (chegou em nivel 4)
        dest_level = new_state.board[action.move_to_row][action.move_to_col].level
        if dest_level == WINNING_LEVEL:
            new_state.winner = state.current_team
            return new_state

        # Orienta (aumenta nivel)
        if action.mentor_at_row is not None and action.mentor_at_col is not None:
            mr, mc = action.mentor_at_row, action.mentor_at_col
            new_state.board[mr][mc].level += 1

        # Avanca turno
        new_state.turn_number += 1
        new_state.current_team = get_opponent(state.current_team)

        # Limite de turnos = empate (nenhum vencedor)
        if new_state.turn_number > MAX_TURNS:
            new_state.winner = 0  # 0 = empate (diferente de None que e jogando)

        return new_state

    # ===== Helpers de simulacao =====

    @staticmethod
    def play_random_game(seed: Optional[int] = None) -> GameState:
        """Simula uma partida inteira com jogadas aleatorias.

        Util para validar que o engine funciona.
        """
        if seed is not None:
            random.seed(seed)

        state = GameState.initial()

        while not state.is_finished():
            if state.phase == Phase.SETUP_PLACEMENT:
                actions = GameEngine.valid_setup_actions(state)
                if not actions:
                    break
                action = random.choice(actions)
                state = GameEngine.apply_setup(state, action)
            else:
                actions = GameEngine.valid_turn_actions(state)
                if not actions:
                    # Time atual sem jogadas = passa a vez (ou perde, dependendo da regra)
                    state.current_team = get_opponent(state.current_team)
                    state.turn_number += 1
                    if state.turn_number > MAX_TURNS:
                        state.winner = 0
                    continue
                action = random.choice(actions)
                state = GameEngine.apply_turn(state, action)

        return state
