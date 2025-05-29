from fastapi import FastAPI
from src.api import routes

app = FastAPI(
    title="Nha Trang Travel Chatbot API",
    version="0.1"
)

app.include_router(routes.router)
