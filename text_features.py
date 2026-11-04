"""Simple writing-style (stylometric) features used in the notebook and the web app."""
import re

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "word_count",
    "avg_word_length",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "punctuation_per_100_words",
    "paragraph_count",
]


def text_features(text: str) -> dict:
    """Return a dictionary with simple style features of one text."""
    words = re.findall(r"[A-Za-z']+", text)
    n_words = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences] or [0]
    paragraphs = [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    punctuation = len(re.findall(r"[,;:!?\"()\-]", text))
    return {
        "word_count": n_words,
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "sentence_count": len(sentences),
        "avg_sentence_length": float(np.mean(sentence_lengths)),
        "sentence_length_std": float(np.std(sentence_lengths)),
        "type_token_ratio": len({w.lower() for w in words}) / n_words if n_words else 0.0,
        "punctuation_per_100_words": 100 * punctuation / n_words if n_words else 0.0,
        "paragraph_count": len(paragraphs),
    }


def features_frame(texts) -> pd.DataFrame:
    """Apply text_features to many texts and return a DataFrame."""
    return pd.DataFrame([text_features(t) for t in texts], columns=FEATURE_NAMES)
