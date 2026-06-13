# Jogador Inteligente PI5 — Jogo da Orientação

## Estrutura da aplicação

O backend é uma API em **FastAPI** responsável por receber o estado do tabuleiro e devolver a jogada do jogador inteligente.

```
.
├── app/                        # Codigo de producao (API)
│   ├── main.py                 # Endpoints /health e /move
│   ├── schemas.py              # DTOs (Pydantic)
│   └── logic/
│       ├── config.py           # Estrategia ativa e parametros
│       ├── rules.py            # Regras do jogo + simulacao de jogadas
│       └── strategies/
│           ├── random_bot.py   # Bot aleatorio
│           ├── heuristic.py    # Heuristica + funcao de avaliacao
│           ├── minimax.py      # Minimax alfa-beta (EM USO)
│           └── rl.py           # Reinforcement Learning (exploratorio)
│
└── training/                   # Codigo de RL (exploratorio, nao em producao)
    ├── game/                   # Simulador local do jogo
    ├── envs/                   # Ambiente Gymnasium + recompensas
    └── train/                  # Scripts de treino PPO/self-play
```

### Endpoints

- **`GET /health`**: retorna status da API e timestamp, usado para verificar se o serviço está no ar.
- **`POST /move`**: recebe o estado do jogo (`AITurnRequest`, com tabuleiro, time, fase do turno etc.) e devolve a jogada:
  - na fase `setup_placement`, retorna `SetupResponse` (linha/coluna para posicionar o professor);
  - na fase `player_turn`, retorna `PlayerTurnResponse` (qual professor mover, para onde, e qual aluno orientar).

### Módulo de regras (`rules.py`)

Centraliza toda a lógica do jogo: encontrar posições de professores, calcular movimentos válidos (adjacência + diferença de nível ≤ 1), posições válidas de orientação, verificação de vitória (professor em célula nível 4), geração de todas as jogadas possíveis para um time e simulação de uma jogada aplicada ao tabuleiro (`simulate_turn`), usada pelo minimax para gerar estados futuros.

### Dispatcher de estratégias (`logic/__init__.py`)

A função `choose_setup`/`choose_turn` escolhe a estratégia configurada em `config.py` (`ACTIVE_STRATEGY`). Cada estratégia implementa as funções `*_setup` e `*_turn` com a mesma assinatura, o que permite trocar de algoritmo só mudando essa configuração.

---

## Jogador Inteligente - Estratégia

### Primeira tentativa: Reinforcement Learning

A primeira abordagem foi treinar um agente com **RL** (MaskablePPO, via `sb3_contrib`), usando uma codificação do tabuleiro como tensor 5x5x9 e uma máscara de ações válidas para garantir que o modelo só escolhesse jogadas legais.

Contra o bot aleatório o agente apresentou bons resultados, chegando a aproximadamente **96% de win rate**. No entanto, percebemos um problema recorrente: o modelo desenvolvia a tendência de **usar sempre o mesmo professor**, deixando o segundo praticamente parado o jogo todo. Isso limitava bastante a estratégia (perdia mobilidade e opções de jogada) e tornava o jogador menos consistente contra adversários mais competentes — ajustar isso via reward shaping/treino exigiria mais tempo de treinamento do que tínhamos disponível.

O código do RL (`rl.py`) foi mantido no projeto como estratégia alternativa (com fallback automático para a heurística caso o modelo não seja encontrado ou dê erro), mas a estratégia ativa em produção é o **minimax**, por ser mais robusta, previsível e não depender de treino. É possível reativar o RL trocando `ACTIVE_STRATEGY` para `"rl"` em `app/logic/config.py`.

### Estratégia escolhida: Heurística + Minimax com poda alfa-beta

Diante do problema do RL, optamos por uma abordagem mais clássica: **minimax com poda alfa-beta**, apoiado por uma função heurística de avaliação de tabuleiro. O algoritmo simula várias jogadas à frente, alternando entre maximizar o próprio ganho e minimizar o ganho do adversário, escolhendo sempre a jogada que leva ao melhor resultado garantido.

#### Por que Minimax?

O Jogo da Orientação tem características ideais para busca em árvore:

- **Informação completa:** todo o tabuleiro é visível
- **Determinístico:** não há aleatoriedade nas regras
- **Dois jogadores alternados:** turnos bem definidos
- **Tabuleiro pequeno (5×5):** permite buscar várias jogadas à frente

Para esse tipo de jogo, Minimax com uma boa função de avaliação é mais forte e mais previsível que abordagens de aprendizado, jogando de forma consistente sem precisar de treino prévio.

#### Heurística (`heuristic.py`)

Serve duas funções:
1. `_score_move`: pontua uma jogada individual (favorece mover para células de nível alto, orientar alunos para nível 3 ou 4, e penaliza criar uma vitória que o adversário consiga aproveitar). Usada para ordenar jogadas antes de explorá-las no minimax.
2. `evaluate_board`: avalia o tabuleiro como um todo na perspectiva de um time — soma os níveis dos próprios professores, considera "ameaças" (proximidade de células de nível alto), potencial de progressão (uma "escada" de níveis subindo) e quem consegue alcançar células de nível 3/4 primeiro. É essa função que avalia as folhas da árvore do minimax.

#### Como funciona o Minimax (`minimax.py`)

A cada turno, o algoritmo constrói uma árvore de possibilidades:

1. Gera todas as jogadas válidas para o time atual (`generate_all_moves`).
2. Se existe jogada de vitória imediata, joga direto (sem precisar buscar).
3. Caso contrário, ordena as jogadas pela heurística (`_score_move`) e mantém só as **top 15** (`TOP_K_MOVES`) — isso reduz bastante o fator de ramificação sem perder as jogadas mais promissoras.
4. Para cada jogada candidata, simula o tabuleiro resultante (`simulate_turn`) e chama a recursão `_minimax`, alternando entre maximizar (nosso turno) e minimizar (turno do adversário), com poda alfa-beta para cortar ramos que não vão influenciar a decisão final.
5. Continua alternando até a profundidade definida.
6. Nas folhas (profundidade 0, vitória/derrota detectada, ou tempo esgotado), retorna `evaluate_board`.
7. Escolhe a jogada que leva ao melhor resultado garantido.

#### Otimizações implementadas

Como o jogo tem muitas jogadas possíveis por turno (40–66 em média), um Minimax puro não caberia no limite de 5 segundos. Foram aplicadas quatro otimizações:

**1. Poda alfa-beta**
Corta ramos da árvore que não podem influenciar a decisão final, reduzindo drasticamente o número de estados avaliados.

**2. Move ordering**
As jogadas são ordenadas pela heurística antes de explorar a árvore. Avaliar primeiro as jogadas promissoras faz a poda alfa-beta cortar muito mais ramos.

**3. Top-K filtering**
Em cada nível da árvore, apenas as K melhores jogadas (segundo a heurística) são consideradas, controlando o fator de ramificação.

**4. Limite de tempo**
A busca verifica o tempo decorrido a cada chamada recursiva e em cada jogada candidata no nível raiz. Se o limite for atingido, a busca para automaticamente e retorna a **melhor jogada encontrada até o momento** (em vez de travar ou não responder), garantindo que nunca ultrapasse os 5 segundos da API.

#### Função de avaliação

Quando a árvore atinge a profundidade máxima (ou é interrompida), cada tabuleiro é avaliado por `evaluate_board`, uma função heurística que considera:

- Proximidade dos meus professores a células de nível alto
- Quão perto estou de uma vitória (célula nível 4 alcançável)
- Ameaças do adversário (células nível 4 que ele pode alcançar — peso defensivo maior)
- Nível das células ocupadas por cada time
- Potencial de células nível 3 livres adjacentes

Vitórias recebem pontuação máxima (priorizando ganhar rápido) e derrotas pontuação mínima (priorizando adiar a perda).

#### Parâmetros

Configuráveis em `app/logic/config.py`:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `MINIMAX_DEPTH` | 6 | Quantas jogadas à frente analisar |
| `TOP_K_MOVES` | 15 | Jogadas consideradas por nível |
| `MINIMAX_TIME_LIMIT_SECONDS` | 4.5 | Limite antes de parar a busca (margem de segurança) |

Esses valores foram ajustados manualmente: testamos diferentes profundidades e medimos o tempo médio de resposta em várias posições de tabuleiro, até achar um valor que aproveitasse bem o tempo disponível sem passar do limite de 5 segundos.

#### Fallback

Se qualquer exceção ocorrer durante o minimax (ex: erro inesperado de estado), o código cai para a estratégia heurística pura (`heuristic_turn`), garantindo que a API sempre devolva uma jogada válida dentro do tempo.

### Testes

- Testamos a estratégia rodando partidas completas contra o bot aleatório (`random_bot.py`) e contra a própria heurística, observando se o minimax consistentemente vencia e se conseguia identificar e executar jogadas de vitória quando disponíveis.
- Medimos o tempo de resposta em diferentes estados de tabuleiro (early/mid/late game) para validar que, mesmo no pior caso, o corte por tempo evita estourar os 5 segundos.
- Validamos os endpoints `/health` e `/move` manualmente, enviando os mesmos payloads usados pelo orquestrador (incluindo as duas fases: `setup_placement` e `player_turn`), conferindo se as respostas seguem o formato esperado (`SetupResponse`/`PlayerTurnResponse`).
