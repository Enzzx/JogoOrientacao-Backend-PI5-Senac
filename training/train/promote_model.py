"""Utilitario para gerenciar o pool de geracoes do self-play.

Uso:
    # Listar todas as geracoes
    python -m training.train.promote_model list

    # Promover um modelo arbitrario como nova geracao
    python -m training.train.promote_model promote training/models/checkpoints/foo.zip

    # Remover uma geracao (cuidado!)
    python -m training.train.promote_model remove gen_002.zip

    # Avaliar duas geracoes uma contra a outra
    python -m training.train.promote_model duel gen_001.zip gen_005.zip --episodes 100
"""

import argparse
import shutil
from pathlib import Path


SELF_PLAY_DIR = Path("training/models/self_play")


def list_generations():
    if not SELF_PLAY_DIR.exists():
        print("Pool vazio.")
        return
    gens = sorted(SELF_PLAY_DIR.glob("gen_*.zip"))
    if not gens:
        print("Pool vazio.")
        return
    print(f"\n{len(gens)} geracao(oes) no pool:")
    for gen in gens:
        size_mb = gen.stat().st_size / 1024 / 1024
        print(f"  {gen.name}  ({size_mb:.1f} MB)")


def promote(source_path: str):
    SELF_PLAY_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(source_path)
    if not src.exists():
        print(f"Arquivo nao encontrado: {source_path}")
        return

    existing = list(SELF_PLAY_DIR.glob("gen_*.zip"))
    next_name = f"gen_{len(existing):03d}.zip"
    dest = SELF_PLAY_DIR / next_name

    shutil.copy(src, dest)
    print(f"Promovido: {src.name} -> {dest.name}")


def remove(name: str):
    target = SELF_PLAY_DIR / name
    if not target.exists():
        print(f"Geracao nao encontrada: {name}")
        return
    target.unlink()
    print(f"Removida: {name}")


def duel(gen_a: str, gen_b: str, episodes: int = 100):
    """Avalia duas geracoes uma contra a outra."""
    import random

    from sb3_contrib import MaskablePPO
    from training.envs.encoding import decode_action, encode_state, get_valid_action_mask
    from training.game import GameEngine, GameState, Phase
    from training.game.constants import TEAM_LOVELACE, TEAM_TURING, get_opponent

    path_a = SELF_PLAY_DIR / gen_a
    path_b = SELF_PLAY_DIR / gen_b

    if not path_a.exists() or not path_b.exists():
        print("Uma das geracoes nao foi encontrada.")
        return

    model_a = MaskablePPO.load(str(path_a))
    model_b = MaskablePPO.load(str(path_b))

    wins = {"a": 0, "b": 0, "draw": 0}

    for ep in range(episodes):
        team_a = TEAM_TURING if ep % 2 == 0 else TEAM_LOVELACE
        team_b = get_opponent(team_a)

        state = GameState.initial(first_team=random.choice([TEAM_TURING, TEAM_LOVELACE]))

        while not state.is_finished():
            current_team = state.current_team
            model = model_a if current_team == team_a else model_b

            obs = encode_state(state, my_team=current_team)
            mask = get_valid_action_mask(state)

            if not mask.any():
                state.winner = get_opponent(current_team)
                break

            action_idx, _ = model.predict(obs, action_masks=mask, deterministic=False)
            action_obj = decode_action(int(action_idx), state)

            if state.phase == Phase.SETUP_PLACEMENT:
                state = GameEngine.apply_setup(state, action_obj)
            else:
                state = GameEngine.apply_turn(state, action_obj)

        if state.winner == team_a:
            wins["a"] += 1
        elif state.winner == team_b:
            wins["b"] += 1
        else:
            wins["draw"] += 1

    print(f"\nDuelo {gen_a} vs {gen_b} ({episodes} episodios):")
    print(f"  {gen_a}: {wins['a']} ({wins['a']/episodes:.1%})")
    print(f"  {gen_b}: {wins['b']} ({wins['b']/episodes:.1%})")
    print(f"  Empates: {wins['draw']}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("source")

    p_remove = sub.add_parser("remove")
    p_remove.add_argument("name")

    p_duel = sub.add_parser("duel")
    p_duel.add_argument("a")
    p_duel.add_argument("b")
    p_duel.add_argument("--episodes", type=int, default=100)

    args = parser.parse_args()

    if args.cmd == "list":
        list_generations()
    elif args.cmd == "promote":
        promote(args.source)
    elif args.cmd == "remove":
        remove(args.name)
    elif args.cmd == "duel":
        duel(args.a, args.b, args.episodes)


if __name__ == "__main__":
    main()
