import os

from fastapi import FastAPI

from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from trainer import train_model
from model_loader import load_model
from predictor import predict_category
from graph_intent import detect_graphs
from chat import chat
from config import MODEL_FILE, VECTORIZER_FILE


app = FastAPI()
app.mount("/graphs", StaticFiles(directory="graphs"), name="graphs")

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    train_model()

model, vectorizer = load_model()

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


class ProductRequest(BaseModel):

    prompt: str


class ChatMessage(BaseModel):

    role: str
    content: str


class ChatRequest(BaseModel):

    prompt: str | None = None
    messages: list[ChatMessage] = []


class TrainResponse(BaseModel):

    message: str
    samples: int
    categories: list[str]
    accuracy: float | None


@app.get("/")
def home():

    return {

        "message": "AI Product Assistant Running"

    }


@app.post("/predict")
def predict(request: ProductRequest):

    return predict_category(

        request.prompt,

        model,

        vectorizer
    )


@app.post("/chat")
def chat_api(request: ChatRequest):

    prompt = request.prompt
    if request.messages:
        prompt = next(
            (message.content for message in reversed(request.messages)
             if message.role == "user"),
            prompt
        )

    if not prompt:
        return {"reply": "Tell me about a product and I’ll predict its category."}

    return chat(

        prompt,

        model,

        vectorizer
    )

@app.post("/train", response_model=TrainResponse)
def train():

    global model, vectorizer

    stats = train_model()
    model, vectorizer = load_model()

    return {

        "message": "Model retrained successfully",

        "samples": stats["samples"],

        "categories": stats["categories"],

        "accuracy": stats["accuracy"]
    }


@app.get("/test-graph")
def test_graph():

    prompt = "count and pie chart"

    graphs = detect_graphs(prompt)

    print("Detected graphs:", graphs)

    return {
        "prompt": prompt,
        "graphs": graphs
    }