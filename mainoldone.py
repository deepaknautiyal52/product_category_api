from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

# -----------------------------
# STEP 1: TRAIN MODEL (Only once)
# -----------------------------

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

if not os.path.exists(MODEL_FILE):

    # Sample training data (replace later with DB data)
    data = {
        "product_name": [
            "iPhone 15",
            "Samsung Galaxy",
            "MacBook Pro",
            "Dell Laptop",
            "Nike Running Shoes",
            "Adidas Sneakers",
            "Office Chair",
            "Wooden Table",
            "Refrigerator",
            "Washing Machine"
        ],
        "category": [
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Footwear",
            "Footwear",
            "Furniture",
            "Furniture",
            "Home Appliances",
            "Home Appliances"
        ]
    }

    df = pd.DataFrame(data)
    # X = df["product_name"]   # Text feature
    # y = df["category"]       # Target label
     
    # X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.3, random_state=42
    # )
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["product_name"])
    y = df["category"]
    

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

# -----------------------------
# STEP 2: LOAD MODEL
# -----------------------------

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# -----------------------------
# STEP 3: FASTAPI APP
# -----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProductRequest(BaseModel):
    product_name: str

@app.post("/predict")
def predict_category(request: ProductRequest):
    vector = vectorizer.transform([request.product_name])
    prediction = model.predict(vector)

    return {
        "product": request.product_name,
        "predicted_category": prediction[0]
    }
