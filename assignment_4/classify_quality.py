import fasttext
import os

def classify_quality(text: str) -> tuple[str, float]:
    MODEL_PATH = "/data/c-mrohatgi/quality_classifier.bin"
    model = fasttext.load_model(MODEL_PATH)
    sanitized = text.replace("\n", " ")
    labels, probs = model.predict(sanitized, k=1)
    top_label, top_prob = labels[0], probs[0]

    if top_label == "__label__positive":
        return "wiki", top_prob
    else:
        return "cc", top_prob
