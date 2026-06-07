"""Testes do ambiente Gymnasium."""

import numpy as np

from training.envs import JogoOrientacaoEnv
from training.envs.encoding import (
    NUM_CHANNELS,
    SETUP_ACTIONS_COUNT,
    TOTAL_ACTIONS,
    get_valid_action_mask,
)


def test_env_reset():
    env = JogoOrientacaoEnv(opponent="random", my_team=1)
    obs, info = env.reset(seed=42)
    assert obs.shape == (5, 5, NUM_CHANNELS)
    assert obs.dtype == np.float32
    assert info["my_team"] == 1
    print("OK test_env_reset")


def test_observation_space():
    env = JogoOrientacaoEnv(my_team=1)
    obs, _ = env.reset(seed=0)
    # Tensor deve estar entre 0 e 1
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0
    print("OK test_observation_space")


def test_random_actions_complete_episode():
    """Roda episodios usando acoes aleatorias validas."""
    env = JogoOrientacaoEnv(opponent="random", my_team=1)
    results = {"win": 0, "loss": 0, "draw": 0, "stuck": 0}

    for ep in range(20):
        obs, info = env.reset(seed=ep)
        done = False
        last_reward = 0.0
        while not done:
            mask = info["valid_action_mask"]
            valid_indices = np.where(mask)[0]

            # Sem acoes validas: engine trata no proximo step com acao qualquer
            if len(valid_indices) == 0:
                obs, last_reward, terminated, truncated, info = env.step(0)
                results["stuck"] += 1
                done = True
                break

            action = int(np.random.choice(valid_indices))
            obs, last_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if not info.get("no_valid_actions"):
            if last_reward > 0:
                results["win"] += 1
            elif last_reward < 0:
                results["loss"] += 1
            else:
                results["draw"] += 1

    print(f"OK test_random_actions_complete_episode — {results}")


def test_invalid_action_penalized():
    """Acao invalida deve resultar em recompensa negativa e fim do episodio."""
    env = JogoOrientacaoEnv(opponent="random", my_team=1)
    obs, info = env.reset(seed=1)

    # Encontra uma acao invalida
    mask = info["valid_action_mask"]
    invalid_indices = np.where(~mask)[0]
    if len(invalid_indices) == 0:
        print("SKIP test_invalid_action_penalized (sem acoes invalidas)")
        return

    invalid_action = int(invalid_indices[0])
    obs, reward, terminated, truncated, info = env.step(invalid_action)
    assert terminated
    assert reward < 0
    assert info.get("invalid_action") is True
    print("OK test_invalid_action_penalized")


def test_action_encoding_roundtrip():
    """Codifica e decodifica acoes para garantir que e bijetivo."""
    from training.envs.encoding import (
        index_to_setup_action,
        index_to_turn_action,
        setup_action_to_index,
        turn_action_to_index,
    )

    # Setup actions
    for idx in range(SETUP_ACTIONS_COUNT):
        action = index_to_setup_action(idx)
        assert setup_action_to_index(action) == idx

    # Turn actions
    for idx in range(SETUP_ACTIONS_COUNT, TOTAL_ACTIONS):
        action = index_to_turn_action(idx)
        assert turn_action_to_index(action) == idx

    print("OK test_action_encoding_roundtrip")


if __name__ == "__main__":
    test_env_reset()
    test_observation_space()
    test_action_encoding_roundtrip()
    test_invalid_action_penalized()
    test_random_actions_complete_episode()
    print("\nTodos os testes do ambiente passaram!")
