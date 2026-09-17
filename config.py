import os

# -----------------------------
# BASE DIRECTORY
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# MODEL FILES
# -----------------------------

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")

# -----------------------------
# TRAINING SETTINGS
# -----------------------------

MAX_ITER = 2000

CONFIDENCE_THRESHOLD = 0.40

# ==========================
# MYSQL CONFIGURATION
# ==========================

DB_HOST = os.getenv("DB_HOST", "localhost")

DB_USER = os.getenv("DB_USER", "root")

DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DB_NAME = os.getenv("DB_NAME", "ai_product_db")

DB_PORT = int(os.getenv("DB_PORT", "3306"))