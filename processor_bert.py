from sentence_transformers import SentenceTransformer
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "log_classifier_model.joblib")

transformer_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
classifier_model = joblib.load(MODEL_PATH)

def classify_logs_with_bert(log_messages):
    embeddings = transformer_model.encode(
        log_messages,
        show_progress_bar=False
    )

    probabilities = classifier_model.predict_proba(embeddings)
    predictions = []

    for probs, emb in zip(probabilities, embeddings):
        if max(probs) < 0.5:
            predictions.append("Unclassified")
        else:
            predictions.append(
                classifier_model.predict([emb])[0]
            )

    return predictions
