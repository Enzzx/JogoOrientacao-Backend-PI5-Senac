from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AITurnRequest, TurnPhase
from app.logic import choose_setup, choose_turn

app = FastAPI(
    title="Jogador Inteligente PI5",
    description="API do jogador inteligente para o Jogo da Orientação",
    version="0.1.0",
)

# CORS liberado para facilitar testes locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Endpoint de saúde — verifica se a API está rodando."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/move")
async def move(request: AITurnRequest):
    """Recebe o estado do jogo e retorna a jogada do jogador inteligente."""
    print(f"[move] game={request.game_id} turn={request.turn_number} phase={request.turn_phase}")

    if request.turn_phase == TurnPhase.SETUP_PLACEMENT:
        result = choose_setup(request.board)
        if result is None:
            raise HTTPException(status_code=422, detail="Sem posições disponíveis para setup")
        print(f"[setup] -> row={result.row} col={result.col}")
        return result

    if request.turn_phase == TurnPhase.PLAYER_TURN:
        result = choose_turn(request.board, request.your_team)
        if result is None:
            raise HTTPException(status_code=422, detail="Sem jogadas válidas disponíveis")
        print(f"[turn] -> professor={result.professor} move_to={result.move_to} mentor_at={result.mentor_at}")
        return result

    raise HTTPException(status_code=400, detail=f"Fase desconhecida: {request.turn_phase}")
