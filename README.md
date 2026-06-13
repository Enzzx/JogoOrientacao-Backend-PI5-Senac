# Jogador Inteligente — PI5 Senac

Backend do jogador inteligente para o **Jogo da Orientação** do Projeto Integrador 5 do Senac.

**API:** https://pi5-api-production.up.railway.app

---

## Como o jogador funciona

A IA usa **Minimax com poda alfa-beta** como estratégia principal. O algoritmo simula várias jogadas à frente, alternando entre maximizar o próprio ganho e minimizar o ganho do adversário, escolhendo sempre a jogada que leva ao melhor resultado garantido.

### Por que Minimax?

O Jogo da Orientação tem características ideais para busca em árvore:

- **Informação completa:** todo o tabuleiro é visível
- **Determinístico:** não há aleatoriedade nas regras
- **Dois jogadores alternados:** turnos bem definidos
- **Tabuleiro pequeno (5×5):** permite buscar várias jogadas à frente

Para esse tipo de jogo, Minimax com uma boa função de avaliação é mais forte e mais previsível que abordagens de aprendizado, jogando de forma consistente sem precisar de treino prévio.

---

## Endpoints da API

### `GET /health`

Verifica se a API está rodando.

```json
{ "status": "ok", "timestamp": "2026-06-10T12:00:00" }
```

### `POST /move`

Endpoint chamado pelo orquestrador de partidas. Recebe o estado do turno e retorna a jogada.

**Request — fase setup:**
```json
{
  "game_id": "...",
  "turn_number": 1,
  "turn_phase": "setup_placement",
  "your_team": 1,
  "professor_to_place": "CLARO",
  "board": [[]]
}
```

**Response:** `{ "row": 2, "col": 2 }`

**Request — fase turn:**
```json
{
  "game_id": "...",
  "turn_number": 5,
  "turn_phase": "player_turn",
  "your_team": 1,
  "board": [[]]
}
```

**Response:**
```json
{
  "professor": "CLARO",
  "move_to": { "row": 1, "col": 2 },
  "mentor_at": { "row": 2, "col": 3 }
}
```

---

## Como rodar localmente

**Requisitos:** Python 3.12+

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

**Rodar a API:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Swagger: http://localhost:8001/docs

---

## Trocar de estratégia

A estratégia ativa é controlada em `app/logic/config.py`:

```python
ACTIVE_STRATEGY = "minimax"  # opcoes: "random", "heuristic", "minimax", "rl"
```

Se a estratégia ativa falhar, a API cai automaticamente para a heurística como fallback.

---

## Stack

| Componente | Tecnologia |
|------------|-----------|
| Linguagem | Python 3.12 |
| Framework Web | FastAPI |
| Algoritmo principal | Minimax com poda alfa-beta |
| Deploy | Railway |

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Win rate vs Random Bot | ~100% |
| Tempo de resposta médio | 1–3 s (limite de 3.5 s configurado) |
| Usa todos os professores do time | Sim |
| Detecta vitórias garantidas | Sim |
