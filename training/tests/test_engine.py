"""Testes do motor do jogo.

Pode rodar com: python -m pytest training/tests/
Ou simplesmente: python -m training.tests.test_engine
"""

from training.game import (
    BOARD_SIZE,
    GameEngine,
    GameState,
    Phase,
    SetupAction,
    TEAM_TURING,
    TEAM_LOVELACE,
)


def test_initial_state():
    state = GameState.initial()
    assert state.phase == Phase.SETUP_PLACEMENT
    assert state.turn_number == 1
    assert state.winner is None
    assert len(state.setup_queue) == 4
    # Todas as celulas vazias
    for row in state.board:
        for cell in row:
            assert cell.level == 0
            assert cell.professor is None
    print("OK test_initial_state")


def test_valid_setup_actions():
    state = GameState.initial()
    actions = GameEngine.valid_setup_actions(state)
    # Tabuleiro 5x5 todo vazio = 25 posicoes validas
    assert len(actions) == BOARD_SIZE * BOARD_SIZE
    print("OK test_valid_setup_actions")


def test_setup_phase_transitions():
    state = GameState.initial()

    # Posiciona os 4 professores
    state = GameEngine.apply_setup(state, SetupAction(row=0, col=0))
    assert state.phase == Phase.SETUP_PLACEMENT

    state = GameEngine.apply_setup(state, SetupAction(row=0, col=1))
    assert state.phase == Phase.SETUP_PLACEMENT

    state = GameEngine.apply_setup(state, SetupAction(row=1, col=0))
    assert state.phase == Phase.SETUP_PLACEMENT

    state = GameEngine.apply_setup(state, SetupAction(row=1, col=1))
    # Apos o 4o setup deve passar para player_turn
    assert state.phase == Phase.PLAYER_TURN
    assert len(state.setup_queue) == 0
    print("OK test_setup_phase_transitions")


def test_random_game_completes():
    state = GameEngine.play_random_game(seed=42)
    assert state.is_finished()
    # Tem vencedor 1, 2 ou 0 (empate)
    assert state.winner in (0, 1, 2)
    print(f"OK test_random_game_completes (winner={state.winner}, turns={state.turn_number})")


def test_multiple_random_games():
    """Roda 100 partidas aleatorias para validar estabilidade."""
    wins = {0: 0, 1: 0, 2: 0}
    for i in range(100):
        state = GameEngine.play_random_game(seed=i)
        assert state.is_finished()
        wins[state.winner] += 1
    print(f"OK test_multiple_random_games — Vitorias: Empate={wins[0]} Turing={wins[1]} Lovelace={wins[2]}")


if __name__ == "__main__":
    test_initial_state()
    test_valid_setup_actions()
    test_setup_phase_transitions()
    test_random_game_completes()
    test_multiple_random_games()
    print("\nTodos os testes passaram!")
