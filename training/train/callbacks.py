"""Callbacks customizados para acompanhar o treino."""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class WinRateCallback(BaseCallback):
    """Acompanha a taxa de vitoria do agente nos ultimos N episodios."""

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.recent_results = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        # Quando episodio termina, registra vitoria/derrota
        for info in self.locals.get("infos", []):
            if info.get("winner") is not None and info.get("my_team") is not None:
                if info["winner"] == info["my_team"]:
                    self.recent_results.append(1)
                elif info["winner"] == 0:
                    self.recent_results.append(0)
                else:
                    self.recent_results.append(-1)

        # Loga periodicamente
        if self.num_timesteps % 5000 == 0 and len(self.recent_results) > 0:
            wins = sum(1 for r in self.recent_results if r == 1)
            losses = sum(1 for r in self.recent_results if r == -1)
            draws = sum(1 for r in self.recent_results if r == 0)
            total = len(self.recent_results)
            win_rate = wins / total if total > 0 else 0

            self.logger.record("custom/win_rate", win_rate)
            self.logger.record("custom/wins", wins)
            self.logger.record("custom/losses", losses)
            self.logger.record("custom/draws", draws)

            if self.verbose >= 1:
                print(
                    f"[step {self.num_timesteps}] "
                    f"win_rate={win_rate:.2%} "
                    f"({wins}W / {losses}L / {draws}D em {total} jogos)"
                )

        return True
