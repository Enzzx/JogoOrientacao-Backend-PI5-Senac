"""Treina o agente jogando contra ele mesmo (self-play).

Esta etapa eh para depois que o agente ja vence consistentemente
o oponente aleatorio (>80% win rate).

NAO USE ainda. Implementaremos depois.

Uso futuro:
    python -m training.train.train_self_play --base training/models/ppo_vs_random_final.zip
"""

# TODO: implementar quando o agente estiver vencendo o aleatorio com >80% win rate
#
# Self-play envolve:
# 1. Carregar o modelo treinado
# 2. Criar ambiente onde o oponente eh uma copia do modelo
# 3. Manter um "pool" de versoes antigas para evitar overfitting
# 4. Treinar contra esse pool
# 5. Periodicamente: atualizar pool com a nova versao
