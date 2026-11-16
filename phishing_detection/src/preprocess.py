"""Phase 2 - Data Cleaning & NLP Preprocessing.

Uses NLTK for tokenisation, stopword removal, and lemmatisation (PorterStemmer).
Falls back gracefully if NLTK data is unavailable so the pipeline never breaks.
"""
import re
import html
import string

# --- NLTK setup (downloaded once; graceful fallback) ---
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    _STOPWORDS = set(stopwords.words("english"))
    _STEMMER = PorterStemmer()
    _NLTK_OK = True
except LookupError:
    # If NLTK data not found, fall back to sklearn stopwords
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _STOPWORDS = set(ENGLISH_STOP_WORDS)
    _STEMMER = None
    _NLTK_OK = False

# --- Compiled regexes ---
HTML_RE = re.compile(r"<[^>]+>")
URL_RE  = re.compile(r"(https?://\S+|www\.\S+)", re.I)
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Full NLP preprocessing pipeline:

    1. HTML entity decode and tag stripping
    2. Replace URLs with a placeholder token (preserves URL *presence* as a feature)
    3. Lowercase
    4. Punctuation removal
    5. NLTK tokenisation
    6. Stopword removal (NLTK English stopwords)
    7. Porter Stemming (reduces vocabulary size, improves generalisation)
    """
    # Step 1 – HTML
    text = html.unescape(str(text))
    text = HTML_RE.sub(" ", text)

    # Step 2 – URLs → token
    text = URL_RE.sub(" urltoken ", text)

    # Step 3 – lowercase
    text = text.lower()

    # Step 4 – punctuation
    text = PUNCT_RE.sub(" ", text)

    # Step 5 – tokenise (NLTK word_tokenize if available, else split)
    if _NLTK_OK:
        tokens = word_tokenize(text)
    else:
        tokens = SPACE_RE.split(text.strip())

    # Step 6 – stopwords + length filter
    tokens = [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]

    # Step 7 – stemming
    if _STEMMER:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


def build_combined(df, subject_col: str = "subject", body_col: str = "body"):
    """Concatenate subject and body so the model sees the full email text."""
    import pandas as pd
    subj = df[subject_col].fillna("") if subject_col in df.columns else pd.Series([""] * len(df))
    body = df[body_col].fillna("")
    return (subj.astype(str) + " . " + body.astype(str))
