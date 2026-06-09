"""Gerenciador de oponentes para self-play."""

import random
from pathlib import Path
from typing import List, Optional

import numpy as np


class OpponentPool:
    """Pool de oponentes para self-play.

    Sorteia adversarios entre:
    - Versoes recentes da politica
    - Versoes antigas (evita catastrophic forgetting)
    - Oponente aleatorio (mantem capacidade basica)
    """

    def __init__(
        self,
        pool_dir: str = "training/models/self_play",
        recent_prob: float = 0.6,
        old_prob: float = 0.3,
        random_prob: float = 0.1,
    ):
        """
        Args:
            pool_dir: diretorio com os modelos salvos
            recent_prob: chance de pegar a geracao mais recente
            old_prob: chance de pegar uma geracao antiga
            random_prob: chance de jogar contra aleatorio
        """
        self.pool_dir = Path(pool_dir)
        self.recent_prob = recent_prob
        self.old_prob = old_prob
        self.random_prob = random_prob

        self._loaded_models = {}

    def list_generations(self) -> List[Path]:
        """Lista todos os modelos no pool, ordenados por geracao."""
        if not self.pool_dir.exists():
            return []
        return sorted(self.pool_dir.glob("gen_*.zip"))

    def latest_generation(self) -> Optional[Path]:
        """Retorna o caminho da geracao mais recente."""
        gens = self.list_generations()
        return gens[-1] if gens else None

    def sample_opponent(self) -> Optional[Path]:
        """Sorteia um oponente conforme as probabilidades configuradas.

        Retorna:
            - Path do modelo se for self-play
            - None se for oponente aleatorio
        """
        gens = self.list_generations()

        if not gens:
            return None

        roll = random.random()

        if roll < self.random_prob:
            return None

        if roll < self.random_prob + self.old_prob and len(gens) > 1:
            return random.choice(gens[:-1])

        return gens[-1]

    def load_model(self, model_path: Path):
        """Carrega modelo com cache."""
        from sb3_contrib import MaskablePPO

        key = str(model_path)
        if key not in self._loaded_models:
            print(f"[OpponentPool] Carregando {model_path.name}")
            self._loaded_models[key] = MaskablePPO.load(str(model_path))
        return self._loaded_models[key]

    def predict(self, model_path: Path, obs: np.ndarray, mask: np.ndarray) -> int:
        """Faz predicao usando um modelo do pool."""
        model = self.load_model(model_path)
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(action)
