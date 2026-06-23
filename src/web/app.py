import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

from src.configuration.config import WEB_STATIC_DIR
from src.web.schemas import Answer, Question
from src.web.service import ChatService

app = FastAPI()

app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")

service = ChatService()


@app.get("/")
async def root():
    # 重定向到index.html
    return RedirectResponse(url="/static/index.html")


@app.post("/api/chat")
async def read_item(question: Question) -> Answer:
    result = service.chat(question.message)
    return Answer(message=result)


if __name__ == "__main__":
    uvicorn.run('src.web.app:app', host="0.0.0.0", port=8000)
