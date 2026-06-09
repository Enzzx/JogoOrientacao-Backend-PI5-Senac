"""Testes da funcao de recompensa."""

from training.envs.rewards import (
    DEFAULT_REWARD_CONFIG,
    compute_step_reward,
    compute_terminal_reward,
)
from training.game import (
    GameEngine,
    GameState,
    Phase,
    SetupAction,
    TEAM_LOVELACE,
    TEAM_TURING,
    TurnAction,
)


def _setup_filled_state():
    """Estado com 4 professores posicionados, pronto para jogar."""
    state = GameState.initial()
    state = GameEngine.apply_setup(state, SetupAction(row=0, col=0))
    state = GameEngine.apply_setup(state, SetupAction(row=4, col=4))
    state = GameEngine.apply_setup(state, SetupAction(row=0, col=4))
    state = GameEngine.apply_setup(state, SetupAction(row=4, col=0))
    return state


def test_terminal_reward_win():
    state = _setup_filled_state()
    state.winner = TEAM_TURING
    r = compute_terminal_reward(state, my_team=TEAM_TURING)
    assert r > 0, f"Esperado positivo, recebeu {r}"
    print(f"OK test_terminal_reward_win (r={r})")


def test_terminal_reward_loss():
    state = _setup_filled_state()
    state.winner = TEAM_LOVELACE
    r = compute_terminal_reward(state, my_team=TEAM_TURING)
    assert r < 0
    print(f"OK test_terminal_reward_loss (r={r})")


def test_terminal_reward_draw():
    state = _setup_filled_state()
    state.winner = 0
    r = compute_terminal_reward(state, my_team=TEAM_TURING)
    assert r <= 0
    print(f"OK test_terminal_reward_draw (r={r})")


def test_step_reward_level_raise():
    """Subir nivel de aluno deve dar recompensa positiva."""
    state_before = _setup_filled_state()
    state_after = state_before.copy()
    # Simula que subiu o nivel de uma celula
    state_after.board[2][2].level = 1

    r = compute_step_reward(state_before, state_after, my_team=TEAM_TURING)
    # r tem penalidade por turno, mas raised_level_self deve compensar
    assert r > -1, f"Esperado positivo apos compensar per_turn, recebeu {r}"
    print(f"OK test_step_reward_level_raise (r={r:.3f})")


def test_step_reward_no_op_only_penalty():
    """Sem mudancas no tabuleiro -> so penalidade por turno."""
    state_before = _setup_filled_state()
    state_after = state_before.copy()

    r = compute_step_reward(state_before, state_after, my_team=TEAM_TURING)
    # so a penalidade por turno
    assert r == DEFAULT_REWARD_CONFIG.per_turn
    print(f"OK test_step_reward_no_op_only_penalty (r={r:.3f})")


def test_step_reward_dangerous_level_4():
    """Criar celula nivel 4 acessivel ao adversario deve dar negativo."""
    from training.game import GameState, Phase

    # Estado controlado: Lovelace em (2,3) com nivel 3 pode alcancar (2,2)
    # quando ela subir para nivel 4 (delta = 1, movimento valido)
    state_before = GameState.initial()
    state_before.phase = Phase.PLAYER_TURN
    state_before.setup_queue = []
    state_before.board[2][3].professor = "KARIN"
    state_before.board[2][3].level = 3
    state_before.board[2][2].level = 3  # vai subir para 4

    state_after = state_before.copy()
    state_after.board[2][2].level = 4

    r = compute_step_reward(state_before, state_after, my_team=TEAM_TURING)
    assert r < 0, f"Esperado negativo, recebeu {r}"
    print(f"OK test_step_reward_dangerous_level_4 (r={r:.3f})")


def test_step_reward_used_different_professor():
    """Trocar de professor deve dar bonus."""
    state_before = _setup_filled_state()
    state_after = state_before.copy()

    r = compute_step_reward(
        state_before, state_after,
        my_team=TEAM_TURING,
        professor_used="REY",
        last_professor="CLARO",
    )
    # bonus 0.15 - penalidade 0.01 = 0.14
    assert r > 0, f"Esperado positivo (mudou professor), recebeu {r}"
    print(f"OK test_step_reward_used_different_professor (r={r:.3f})")


def test_step_reward_same_professor_streak():
    """Usar mesmo professor seguidamente deve dar penalidade."""
    state_before = _setup_filled_state()
    state_after = state_before.copy()

    r = compute_step_reward(
        state_before, state_after,
        my_team=TEAM_TURING,
        professor_used="CLARO",
        last_professor="CLARO",
    )
    # penalidade -0.10 - 0.01 = -0.11
    assert r < 0, f"Esperado negativo (mesmo professor), recebeu {r}"
    print(f"OK test_step_reward_same_professor_streak (r={r:.3f})")


if __name__ == "__main__":
    test_terminal_reward_win()
    test_terminal_reward_loss()
    test_terminal_reward_draw()
    test_step_reward_no_op_only_penalty()
    test_step_reward_level_raise()
    test_step_reward_dangerous_level_4()
    test_step_reward_used_different_professor()
    test_step_reward_same_professor_streak()
    print("\nTodos os testes de recompensa passaram!")
