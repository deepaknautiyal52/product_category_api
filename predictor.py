from preprocess import clean_text
from config import CONFIDENCE_THRESHOLD


def predict_category(prompt, model, vectorizer):

    cleaned_prompt = clean_text(prompt)

    vector = vectorizer.transform([cleaned_prompt])  # convert it into vector or embadding format

    prediction = model.predict(vector) #do prediction using the trained model

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities)
    ranked_categories = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True
    )
    top_categories = [
        {
            "category": str(category),
            "confidence": round(float(probability), 2)
        }
        for category, probability in ranked_categories[:3]
    ]

    return {
        "prompt": prompt,
        "cleaned_prompt": cleaned_prompt,
        "predicted_category": str(prediction[0]),
        "confidence": round(float(confidence), 2),
        "top_categories": top_categories,
        "needs_clarification": bool(confidence < CONFIDENCE_THRESHOLD)
    }