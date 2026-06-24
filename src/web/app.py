from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

from src.configuration.config import WEB_STATIC_DIR
from src.web.schemas import Answer, Question
from src.web.service import ChatService

_service: ChatService | None = None


def _get_service() -> ChatService:
    if _service is None:
        raise RuntimeError("ChatService not initialized — app startup may have failed.")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = ChatService()
    yield
    _service = None


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.post("/api/chat")
async def read_item(question: Question) -> Answer:
    result = await asyncio.to_thread(_get_service().chat, question.message)
    return Answer(message=result)


if __name__ == "__main__":
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000)
