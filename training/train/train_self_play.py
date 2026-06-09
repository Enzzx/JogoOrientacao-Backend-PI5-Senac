"""Treina o agente por self-play contra geracoes anteriores.

Uso:
    # Primeira vez (vai usar o modelo atual como geracao 0)
    python -m training.train.train_self_play \
        --base training/models/ppo_vs_random_final.zip \
        --timesteps 500000

    # Continuar evoluindo
    python -m training.train.train_self_play \
        --resume training/models/self_play/latest.zip \
        --timesteps 500000

    # Forcar nova geracao apos N timesteps
    python -m training.train.train_self_play \
        --timesteps 1000000 \
        --promote-every 250000
"""

import argparse
import shutil
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from training.envs import JogoOrientacaoEnv
from training.envs.opponents import OpponentPool
from training.train.callbacks import WinRateCallback


MODELS_DIR = Path("training/models")
SELF_PLAY_DIR = MODELS_DIR / "self_play"
LOGS_DIR = Path("training/logs")
SELF_PLAY_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def mask_fn(env):
    from training.envs.encoding import get_valid_action_mask
    return get_valid_action_mask(env.state)


def next_generation_name() -> str:
    """Determina o nome do proximo arquivo de geracao."""
    existing = list(SELF_PLAY_DIR.glob("gen_*.zip"))
    return f"gen_{len(existing):03d}.zip"


def bootstrap_pool_from_base(base_path: str):
    """Copia o modelo base como geracao 0 do pool, se o pool estiver vazio."""
    pool_files = list(SELF_PLAY_DIR.glob("gen_*.zip"))
    if pool_files:
        print(f"[bootstrap] Pool ja contem {len(pool_files)} geracao(oes), nao copiando base.")
        return

    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Modelo base nao encontrado: {base_path}")

    dest = SELF_PLAY_DIR / "gen_000.zip"
    shutil.copy(base, dest)
    print(f"[bootstrap] Modelo base copiado para {dest.name}")


def train(
    timesteps: int,
    base_path: str = None,
    resume_path: str = None,
    promote_every: int = 0,
):
    """Treina o agente por self-play."""
    print(f"\n=== Self-Play | {timesteps} timesteps ===\n")

    if base_path:
        bootstrap_pool_from_base(base_path)

    pool = OpponentPool(pool_dir=str(SELF_PLAY_DIR))
    pool_size = len(pool.list_generations())
    print(f"[pool] {pool_size} geracao(oes) disponivel(eis)")

    def make_env():
        env = JogoOrientacaoEnv(
            opponent="self_play",
            opponent_pool=pool,
            my_team=None,
        )
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_env])

    if resume_path and Path(resume_path).exists():
        print(f"[model] Carregando {resume_path}")
        model = MaskablePPO.load(resume_path, env=vec_env)
    elif pool.latest_generation():
        latest = pool.latest_generation()
        print(f"[model] Carregando geracao mais recente: {latest.name}")
        model = MaskablePPO.load(str(latest), env=vec_env)
    else:
        raise RuntimeError(
            "Nenhum modelo disponivel. Forneca --base ou --resume."
        )

    win_rate_cb = WinRateCallback(window_size=200, verbose=1)
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=str(SELF_PLAY_DIR / "checkpoints"),
        name_prefix="self_play",
    )

    steps_done = 0
    try:
        if promote_every > 0:
            while steps_done < timesteps:
                chunk = min(promote_every, timesteps - steps_done)
                model.learn(
                    total_timesteps=chunk,
                    callback=[win_rate_cb, checkpoint_cb],
                    tb_log_name="self_play",
                    reset_num_timesteps=False,
                    progress_bar=True,
                )
                steps_done += chunk

                new_gen = SELF_PLAY_DIR / next_generation_name()
                model.save(str(new_gen))
                print(f"\n[promote] Nova geracao salva: {new_gen.name}")

                pool._loaded_models = {}
        else:
            model.learn(
                total_timesteps=timesteps,
                callback=[win_rate_cb, checkpoint_cb],
                tb_log_name="self_play",
                progress_bar=True,
            )
    except KeyboardInterrupt:
        print("\n[interrupt] Treino interrompido pelo usuario.")

    final_path = SELF_PLAY_DIR / next_generation_name()
    model.save(str(final_path))
    print(f"\n[save] Modelo final: {final_path.name}")

    shutil.copy(final_path, SELF_PLAY_DIR / "latest.zip")

    print(f"\nPool atual: {len(pool.list_generations())} geracao(oes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, help="Modelo base para inicializar geracao 0")
    parser.add_argument("--resume", type=str, help="Modelo para continuar treino")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument(
        "--promote-every", type=int, default=250_000,
        help="A cada N timesteps, promove o modelo atual como nova geracao",
    )
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        base_path=args.base,
        resume_path=args.resume,
        promote_every=args.promote_every,
    )


if __name__ == "__main__":
    main()
