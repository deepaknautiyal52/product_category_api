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
# STEP 3: TRAIN MODEL
# -----------------------------

# FORCE RETRAIN
# Delete old model files automatically

if os.path.exists(MODEL_FILE):
    os.remove(MODEL_FILE)

if os.path.exists(VECTORIZER_FILE):
    os.remove(VECTORIZER_FILE)

# -----------------------------
# TRAIN NEW MODEL
# -----------------------------

if not os.path.exists(MODEL_FILE):

    data = {

        "product_name": [

            # ELECTRONICS
            "Apple smartphone with iOS",
            "Premium mobile phone with camera",
            "Samsung Android mobile device",
            "Gaming laptop for developers",
            "High performance Dell office laptop",
            "Wireless Bluetooth earbuds",
            "Apple MacBook for coding",
            "Smart LED television",
            "Portable electronic tablet",
            "Noise cancelling headphones",
            "Android mobile phone",
            "iPhone with powerful camera",
            "Laptop for office work",
            "Wireless gaming mouse",

            # FOOTWEAR
            "Nike running shoes for sports",
            "Adidas casual sneakers",
            "Comfortable gym training shoes",
            "Leather formal shoes for men",
            "Women high heel sandals",
            "Sports footwear for athletes",
            "Jogging sneakers",
            "Running sports shoes",
            "Gym footwear for men",
            "Casual shoes for walking",

            # FURNITURE
            "Wooden office chair",
            "Large dining table",
            "Comfortable sofa set",
            "Study desk for students",
            "Modern bedroom furniture",
            "Wooden cupboard storage",
            "Office furniture chair",
            "Computer desk table",
            "Dining room chair",
            "Wooden study table",

            # HOME APPLIANCES
            "Double door refrigerator",
            "Automatic washing machine",
            "Kitchen microwave oven",
            "Electric air conditioner",
            "Home water purifier",
            "Vacuum cleaner machine",
            "Kitchen appliance for cooking",
            "Cooling refrigerator machine",
            "Washing appliance",
            "Microwave for kitchen"
        ],

        "category": [

            # ELECTRONICS
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",
            "Electronics",

            # FOOTWEAR
            "Footwear",
            "Footwear",
            "Footwear",
            "Footwear",
            "Footwear",
            "Footwear",
            "Footwear",
            "Footwear",
            "Footwear",
            "Footwear",

            # FURNITURE
            "Furniture",
            "Furniture",
            "Furniture",
            "Furniture",
            "Furniture",
            "Furniture",
            "Furniture",
            "Furniture",
            "Furniture",
            "Furniture",

            # HOME APPLIANCES
            "Home Appliances",
            "Home Appliances",
            "Home Appliances",
            "Home Appliances",
            "Home Appliances",
            "Home Appliances",
            "Home Appliances",
            "Home Appliances",
            "Home Appliances",
            "Home Appliances"
        ]
    }

    # -----------------------------
    # DATAFRAME
    # -----------------------------

    df = pd.DataFrame(data)

    # -----------------------------
    # CLEAN TRAINING DATA
    # -----------------------------

    df["product_name"] = df["product_name"].apply(clean_text)

    # -----------------------------
    # TF-IDF VECTORIZER
    # -----------------------------

    vectorizer = TfidfVectorizer(lowercase=True,ngram_range=(1, 3),stop_words='english',max_df=0.95,min_df=1)

    # -----------------------------
    # CONVERT TEXT → VECTOR
    # -----------------------------

    X = vectorizer.fit_transform(df["product_name"]) # convert it into number or embedding format example - 1.31,2.31 etc
    y = df["category"]

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------

    model = LogisticRegression(max_iter=2000,class_weight="balanced") 
    model.fit(X, y)   # place it into x and y axis

    # -----------------------------
    # SAVE MODEL
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
# ENABLE CORS
# -----------------------------

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)

# -----------------------------
# HOME ROUTE
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "FastAPI Product Category API Running Successfully"
    }

# -----------------------------
# REQUEST MODEL
# -----------------------------

class ProductRequest(BaseModel):

    prompt: str

# -----------------------------
# GET PREDICT ROUTE
# -----------------------------

@app.get("/predict")
def predict_get():

    return {
        "status": "API Running",
        "message": "Use POST request with prompt"
    }

# -----------------------------
# POST PREDICTION API
# -----------------------------

@app.post("/predict")
def predict_category(request: ProductRequest):

    # -----------------------------
    # USER PROMPT
    # -----------------------------

    user_prompt = request.prompt

    # -----------------------------
    # CLEAN PROMPT
    # -----------------------------

    cleaned_prompt = clean_text(user_prompt)

    # -----------------------------
    # VECTORIZE INPUT
    # -----------------------------

    vector = vectorizer.transform([cleaned_prompt])

    # -----------------------------
    # PREDICTION
    # -----------------------------

    prediction = model.predict(vector)

    # -----------------------------
    # CONFIDENCE
    # -----------------------------

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities)

    # -----------------------------
    # DEBUG LOGS
    # -----------------------------

    print("Prompt:", user_prompt)

    print("Cleaned Prompt:", cleaned_prompt)

    print("Prediction:", prediction)

    print("Probabilities:", probabilities)

    print("Classes:", model.classes_)

    # -----------------------------
    # FINAL RESPONSE
    # -----------------------------

    predicted_category = prediction[0]

    ai_message = (
        f"Based on your prompt, "
        f"this product belongs to "
        f"'{predicted_category}' category."
    )

    return {

        "prompt": user_prompt,

        "cleaned_prompt": cleaned_prompt,

        "predicted_category": predicted_category,

        "confidence": round(float(confidence), 2),

        "message": ai_message
    }

# -----------------------------
# CHAT API
# -----------------------------

@app.post("/chat")
def chat(request: ProductRequest):

    user_prompt = request.prompt.strip().lower()

    # --------------------------------
    # GREETINGS
    # --------------------------------

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening"
    ]

    if user_prompt in greetings:

        return {
            "reply": (
                "Hello 👋\n\n"
                "I am your Product Assistant.\n\n"
                "You can ask me things like:\n"
                "- Apple smartphone with camera\n"
                "- Nike running shoes\n"
                "- Wooden office chair\n"
                "- Microwave oven for kitchen\n"
            )
        }

    # --------------------------------
    # HELP
    # --------------------------------

    if "help" in user_prompt:

        return {
            "reply": (
                "I can help you with:\n\n"
                "1. Product category prediction\n"
                "2. Product search\n"
                "3. Product recommendations\n\n"
                "Try typing a product description."
            )
        }

    # --------------------------------
    # THANK YOU
    # --------------------------------

    if "thank" in user_prompt:

        return {
            "reply": "You're welcome 😊"
        }

    # --------------------------------
    # GOODBYE
    # --------------------------------

    if user_prompt in ["bye", "goodbye", "see you"]:

        return {
            "reply": "Goodbye 👋 Have a great day!"
        }

    # --------------------------------
    # PRODUCT CATEGORY PREDICTION
    # --------------------------------

    cleaned_prompt = clean_text(user_prompt) #cleaned prompt

    vector = vectorizer.transform([cleaned_prompt]) #vector or embedding format example - 1.31,2.31 etc 

    prediction = model.predict(vector)   #predicted category

    probabilities = model.predict_proba(vector)[0] #probabilities for each category

    confidence = max(probabilities) #confidence level

    predicted_category = prediction[0] #predicted category

    return {

        "reply": (
            f"Based on your description, "
            f"this product belongs to "
            f"'{predicted_category}' category."
        ),

        "predicted_category": predicted_category,

        "confidence": round(float(confidence), 2)
    }