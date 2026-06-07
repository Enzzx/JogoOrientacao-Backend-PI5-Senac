"""Treina o agente RL contra um oponente aleatorio.

Fase inicial do treinamento. O objetivo eh fazer o agente vencer
o aleatorio com taxa alta (>80%) antes de evoluir para self-play.

Uso:
    python -m training.train.train_vs_random

Argumentos opcionais:
    --timesteps N    quantidade de passos de treino (padrao: 200_000)
    --resume PATH    caminho de modelo pre-existente para continuar treinando
"""

import argparse
import os
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from training.envs import JogoOrientacaoEnv
from training.train.callbacks import WinRateCallback


# ===== Diretorios =====
MODELS_DIR = Path("training/models")
LOGS_DIR = Path("training/logs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ppo_vs_random"


def mask_fn(env: JogoOrientacaoEnv) -> np.ndarray:
    """Retorna a mascara de acoes validas do estado atual."""
    from training.envs.encoding import get_valid_action_mask
    return get_valid_action_mask(env.state)


def make_env():
    """Cria o ambiente com action masking."""
    env = JogoOrientacaoEnv(opponent="random", my_team=None)
    env = ActionMasker(env, mask_fn)
    return env


def train(timesteps: int = 200_000, resume_path: str = None):
    """Treina o agente."""
    print(f"=== Treino vs Random | {timesteps} timesteps ===\n")

    # Vetoriza (necessario mesmo para 1 ambiente)
    vec_env = DummyVecEnv([make_env])

    if resume_path and os.path.exists(resume_path):
        print(f"Carregando modelo existente: {resume_path}")
        model = MaskablePPO.load(resume_path, env=vec_env)
    else:
        print("Criando novo modelo PPO\n")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(LOGS_DIR),
            policy_kwargs={
                "net_arch": [256, 256],
            },
        )

    # Callbacks
    win_rate_cb = WinRateCallback(window_size=200, verbose=1)
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=MODEL_NAME,
    )

    # Treina
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[win_rate_cb, checkpoint_cb],
            tb_log_name=MODEL_NAME,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuario.")

    # Salva modelo final
    final_path = MODELS_DIR / f"{MODEL_NAME}_final.zip"
    model.save(str(final_path))
    print(f"\nModelo salvo em: {final_path}")
    print(f"Logs do TensorBoard em: {LOGS_DIR}")
    print(f"Para visualizar: tensorboard --logdir {LOGS_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Quantidade de timesteps de treino",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Caminho de modelo para continuar treino",
    )
    args = parser.parse_args()

    train(timesteps=args.timesteps, resume_path=args.resume)


if __name__ == "__main__":
    main()
