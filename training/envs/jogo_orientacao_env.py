"""Ambiente Gymnasium do Jogo da Orientacao.

Implementa a interface padrao do Gymnasium para uso com bibliotecas de RL
como Stable Baselines 3.

Modos de jogo:
- vs_random: agente joga contra oponente aleatorio
- self_play: agente joga contra ele mesmo (mesma politica)
"""

from typing import Any, Dict, Optional, Tuple
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from training.envs.encoding import (
    BOARD_SIZE,
    NUM_CHANNELS,
    TOTAL_ACTIONS,
    decode_action,
    encode_state,
    get_valid_action_mask,
)
from training.game import (
    GameEngine,
    GameState,
    Phase,
    TEAM_LOVELACE,
    TEAM_TURING,
)
from training.game.constants import get_opponent


class JogoOrientacaoEnv(gym.Env):
    """Ambiente Gymnasium do Jogo da Orientacao."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent: str = "random",
        my_team: Optional[int] = None,
        max_turns: int = 100,
    ):
        """
        Args:
            opponent: "random" para oponente aleatorio
            my_team: 1 (Turing), 2 (Lovelace) ou None (sorteado a cada episodio)
            max_turns: limite de turnos
        """
        super().__init__()

        self.opponent_type = opponent
        self.my_team_fixed = my_team
        self.max_turns = max_turns

        # Espaco de observacao: tensor 5x5x9
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS),
            dtype=np.float32,
        )

        # Espaco de acao: discreto com TOTAL_ACTIONS possibilidades
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Estado interno
        self.state: Optional[GameState] = None
        self.my_team: int = TEAM_TURING

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia o ambiente."""
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Define o time do agente
        if self.my_team_fixed is None:
            self.my_team = random.choice([TEAM_TURING, TEAM_LOVELACE])
        else:
            self.my_team = self.my_team_fixed

        # Sorteia quem comeca (primeiro setup)
        first_team = random.choice([TEAM_TURING, TEAM_LOVELACE])
        self.state = GameState.initial(first_team=first_team)

        # Se o oponente comeca, ele joga ate ser nossa vez
        self._play_opponent_until_my_turn()

        obs = encode_state(self.state, self.my_team)
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executa uma acao do agente."""
        from training.envs.rewards import (
            DEFAULT_REWARD_CONFIG,
            compute_step_reward,
            compute_terminal_reward,
        )

        if self.state is None:
            raise RuntimeError("Ambiente nao foi reset")

        if self.state.current_team != self.my_team:
            raise RuntimeError("Nao e o turno do agente")

        config = DEFAULT_REWARD_CONFIG

        # Verifica mascara
        mask = get_valid_action_mask(self.state)

        # Nenhuma acao valida: agente perde por nao ter jogada
        if not mask.any():
            self.state.winner = get_opponent(self.my_team)
            terminal = compute_terminal_reward(self.state, self.my_team, config)
            obs = encode_state(self.state, self.my_team)
            return obs, terminal, True, False, {
                **self._build_info(),
                "no_valid_actions": True,
            }

        if not mask[action]:
            obs = encode_state(self.state, self.my_team)
            return (
                obs,
                config.invalid_action,
                True,
                False,
                {**self._build_info(), "invalid_action": True},
            )

        # Guarda estado antes para calcular shaping
        state_before = self.state.copy()

        # Aplica acao do agente
        action_obj = decode_action(action, self.state)
        if self.state.phase == Phase.SETUP_PLACEMENT:
            self.state = GameEngine.apply_setup(self.state, action_obj)
        else:
            self.state = GameEngine.apply_turn(self.state, action_obj)

        # Recompensa intermediaria pela jogada do agente
        step_reward = compute_step_reward(state_before, self.state, self.my_team, config)

        # Fim por vitoria do agente?
        if self.state.is_finished():
            terminal = compute_terminal_reward(self.state, self.my_team, config)
            obs = encode_state(self.state, self.my_team)
            return obs, step_reward + terminal, True, False, self._build_info()

        # Oponente joga
        self._play_opponent_until_my_turn()

        # Verifica fim (oponente pode ter ganhado)
        if self.state.is_finished():
            terminal = compute_terminal_reward(self.state, self.my_team, config)
            obs = encode_state(self.state, self.my_team)
            return obs, step_reward + terminal, True, False, self._build_info()

        obs = encode_state(self.state, self.my_team)
        truncated = self.state.turn_number > self.max_turns
        return obs, step_reward, False, truncated, self._build_info()

    def _play_opponent_until_my_turn(self):
        """Faz o oponente jogar ate ser a vez do agente (ou fim de jogo)."""
        while (
            not self.state.is_finished()
            and self.state.current_team != self.my_team
        ):
            if self.opponent_type == "random":
                had_action = self._opponent_random_play()
                if not had_action:
                    self.state.winner = self.my_team
                    return
            else:
                raise NotImplementedError(
                    f"Tipo de oponente nao suportado: {self.opponent_type}"
                )

    def _opponent_random_play(self) -> bool:
        """Oponente joga aleatoriamente uma acao valida.

        Retorna True se conseguiu jogar, False se nao havia acao disponivel.
        """
        if self.state.phase == Phase.SETUP_PLACEMENT:
            actions = GameEngine.valid_setup_actions(self.state)
            if not actions:
                return False
            action = random.choice(actions)
            self.state = GameEngine.apply_setup(self.state, action)
            return True
        else:
            actions = GameEngine.valid_turn_actions(self.state)
            if not actions:
                return False
            action = random.choice(actions)
            self.state = GameEngine.apply_turn(self.state, action)
            return True

    def _build_info(self) -> Dict[str, Any]:
        return {
            "my_team": self.my_team,
            "phase": self.state.phase.value if self.state else None,
            "turn_number": self.state.turn_number if self.state else 0,
            "current_team": self.state.current_team if self.state else None,
            "winner": self.state.winner if self.state else None,
            "valid_action_mask": get_valid_action_mask(self.state) if self.state else None,
        }

    def render(self):
        """Renderiza o tabuleiro no terminal."""
        if self.state is None:
            print("(estado vazio)")
            return

        print(f"\n=== Turno {self.state.turn_number} | Fase {self.state.phase.value} | Time {self.state.current_team} (meu time: {self.my_team}) ===")
        for r in range(BOARD_SIZE):
            row_str = ""
            for c in range(BOARD_SIZE):
                cell = self.state.board[r][c]
                prof = cell.professor[:3] if cell.professor else "..."
                row_str += f"[{cell.level}|{prof:>3}] "
            print(row_str)
        if self.state.winner is not None:
            print(f">>> Fim de jogo. Vencedor: {self.state.winner}")
