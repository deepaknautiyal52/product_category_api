import os

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import clean_text
#from dataset import get_training_data
from database import get_training_data
from config import MODEL_FILE
from config import VECTORIZER_FILE


def train_model():

    # -------------------------------------
    # Delete old model files (if available)
    # -------------------------------------

    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
        print("Old model.pkl deleted")

    if os.path.exists(VECTORIZER_FILE):
        os.remove(VECTORIZER_FILE)
        print("Old vectorizer.pkl deleted")

    # -----------------------------
    # TRAIN NEW MODEL
    # -----------------------------

    df = get_training_data()
    #df = pd.DataFrame(df)  # Ensure df is a DataFrame

    df["product_name"] = df["product_name"].apply(clean_text)

    vectorizer = TfidfVectorizer(

        lowercase=True,

        ngram_range=(1,3),

        stop_words="english"

    )

    X = vectorizer.fit_transform(df["product_name"]) # convert product names to TF-IDF features

    y = df["category_name"]

    model = LogisticRegression(

        max_iter=2000,

        class_weight="balanced"

    ) # in this case, we are using Logistic Regression as the model for multi-class classification

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        # Refit on the full dataset so the saved model uses all available data
        model.fit(X, y)
    except ValueError:
        # Not enough samples per category for a stratified holdout split
        model.fit(X, y)
        accuracy = None

    joblib.dump(model,MODEL_FILE) #in x and y format

    joblib.dump(vectorizer,VECTORIZER_FILE) #in embedding format

    accuracy_label = f"{accuracy:.2%}" if accuracy is not None else "n/a (too few samples to hold out)"
    print(f"Model trained successfully (holdout accuracy: {accuracy_label}, samples: {len(df)})")

    return {
        "samples": len(df),
        "categories": sorted(y.unique().tolist()),
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None
    }


if __name__=="__main__":

    train_model()
