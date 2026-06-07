"""Avalia o modelo treinado jogando contra oponente aleatorio.

Uso:
    python -m training.train.evaluate
    python -m training.train.evaluate --model training/models/ppo_vs_random_final.zip
    python -m training.train.evaluate --episodes 500
"""

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from training.envs import JogoOrientacaoEnv
from training.envs.encoding import get_valid_action_mask


def mask_fn(env):
    return get_valid_action_mask(env.state)


def evaluate(model_path: str, episodes: int = 100, verbose: bool = False):
    """Roda N episodios e mostra estatisticas."""
    print(f"\n=== Avaliando {model_path} em {episodes} episodios ===\n")

    model = MaskablePPO.load(model_path)

    results = {"win": 0, "loss": 0, "draw": 0, "invalid": 0}
    total_turns = 0
    total_reward = 0.0

    env = JogoOrientacaoEnv(opponent="random", my_team=None)
    env = ActionMasker(env, mask_fn)

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0.0
        done = False

        while not done:
            mask = info.get("valid_action_mask")
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated

        winner = info.get("winner")
        my_team = info.get("my_team")

        if info.get("invalid_action"):
            results["invalid"] += 1
        elif winner == my_team:
            results["win"] += 1
        elif winner == 0 or winner is None:
            results["draw"] += 1
        else:
            results["loss"] += 1

        total_turns += info.get("turn_number", 0)
        total_reward += episode_reward

        if verbose:
            outcome = "W" if winner == my_team else ("D" if winner in (0, None) else "L")
            print(
                f"Ep {ep+1:3d}: {outcome} | "
                f"turnos={info.get('turn_number')} | "
                f"reward={episode_reward:+.2f}"
            )

    # Estatisticas
    total = sum(results.values())
    win_rate = results["win"] / total if total > 0 else 0

    print("\n=== Resultados ===")
    print(f"Vitorias: {results['win']:>4} ({results['win']/total:.1%})")
    print(f"Derrotas: {results['loss']:>4} ({results['loss']/total:.1%})")
    print(f"Empates : {results['draw']:>4} ({results['draw']/total:.1%})")
    if results["invalid"] > 0:
        print(f"Invalidas: {results['invalid']:>3}")
    print(f"\nTurnos medios: {total_turns / total:.1f}")
    print(f"Reward medio:  {total_reward / total:+.3f}")
    print(f"\nWin rate: {win_rate:.2%}")

    return win_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str,
        default="training/models/ppo_vs_random_final.zip",
        help="Caminho do modelo a avaliar",
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Quantos episodios rodar",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Mostra resultado de cada episodio",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERRO: modelo nao encontrado em {args.model}")
        print("Treine primeiro com: python -m training.train.train_vs_random")
        return

    evaluate(args.model, args.episodes, args.verbose)


if __name__ == "__main__":
    main()
