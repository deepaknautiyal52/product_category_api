import joblib

from config import MODEL_FILE
from config import VECTORIZER_FILE


def load_model():

    model = joblib.load(MODEL_FILE)   # Load the trained model from the specified file

    vectorizer = joblib.load(VECTORIZER_FILE) # Load the trained vectorizer from the specified file

    return model, vectorizer