from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import re

# -----------------------------
# STEP 1: FILE PATHS
# -----------------------------

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# -----------------------------
# STEP 2: TEXT CLEANING FUNCTION
# -----------------------------

def clean_text(text):

    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -----------------------------
# STEP 3: TRAIN MODEL (ONLY ONCE)
# -----------------------------

if not os.path.exists(MODEL_FILE):

    # Sample training data
    # Later you can replace this with MySQL DB data

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

    # -----------------------------
    # CLEAN TRAINING DATA
    # -----------------------------

    df["product_name"] = df["product_name"].apply(clean_text)

    # -----------------------------
    # TF-IDF VECTORIZER
    # -----------------------------

    vectorizer = TfidfVectorizer(

        lowercase=True,

        # Better context understanding
        ngram_range=(1, 2),

        # Ignore extremely common words
        max_df=0.95,

        # Keep even low-frequency words
        min_df=1
    )

    # -----------------------------
    # CONVERT TEXT → NUMBERS
    # -----------------------------

    X = vectorizer.fit_transform(df["product_name"])

    y = df["category"]

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------

    model = LogisticRegression(

        max_iter=1000,

        # Helpful for unbalanced data
        class_weight="balanced"
    )

    model.fit(X, y)   

    # -----------------------------
    # SAVE MODEL & VECTORIZER
    # -----------------------------

    joblib.dump(model, MODEL_FILE) 

    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("✅ Model trained successfully")

# -----------------------------
# STEP 4: LOAD MODEL
# -----------------------------

model = joblib.load(MODEL_FILE)

vectorizer = joblib.load(VECTORIZER_FILE)

print("✅ Model loaded successfully")

# -----------------------------
# STEP 5: FASTAPI APP
# -----------------------------

app = FastAPI()

# -----------------------------
# HOME ROUTE
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "FastAPI Product Category API Running Successfully"
    }

# -----------------------------
# ENABLE CORS
# -----------------------------

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# -----------------------------
# REQUEST MODEL
# -----------------------------

class ProductRequest(BaseModel):

    product_name: str

# -----------------------------
# STEP 6: PREDICT CATEGORY API
# -----------------------------
@app.get("/predict")
def predict_get():

     return {
        "status": "API Running",
        "message": "Use POST request with product_name for prediction"
    }
    
@app.post("/predict")
def predict_category(request: ProductRequest):

    # -----------------------------
    # CLEAN USER INPUT
    # -----------------------------

    cleaned_product = clean_text(request.product_name)

    # -----------------------------
    # CONVERT INPUT → VECTOR
    # -----------------------------

    vector = vectorizer.transform([cleaned_product])

    # -----------------------------
    # GET PREDICTION
    # -----------------------------

    prediction = model.predict(vector)

    # -----------------------------
    # GET CONFIDENCE SCORE
    # -----------------------------

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities)

    # -----------------------------
    # LOW CONFIDENCE HANDLING
    # -----------------------------

    if confidence < 0.40:

        predicted_category = "Uncertain"

    else:

        predicted_category = prediction[0]

    # -----------------------------
    # RETURN RESPONSE
    # -----------------------------

    return {

        "product": request.product_name,

        "cleaned_product": cleaned_product,

        "predicted_category": predicted_category,

        "confidence": round(float(confidence), 2)
    }

# -----------------------------
# OPTIONAL CHAT ENDPOINT
# -----------------------------

@app.post("/chat")
def chat(request: ProductRequest):

    cleaned_product = clean_text(request.product_name)

    vector = vectorizer.transform([cleaned_product])

    prediction = model.predict(vector)

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities)

    if confidence < 0.40:

        reply = (
            f"I am not fully confident about "
            f"'{request.product_name}'. "
            f"It may belong to "
            f"'{prediction[0]}' category."
        )

    else:

        reply = (
            f"✅ '{request.product_name}' belongs to "
            f"'{prediction[0]}' category."
        )

    return {

        "reply": reply,

        "predicted_category": prediction[0],

        "confidence": round(float(confidence), 2)
    }