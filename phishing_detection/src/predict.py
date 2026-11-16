"""Live classification helper used by the web app / CLI."""
import joblib
import pandas as pd
from .preprocess import clean_text, build_combined
from .features import metadata_frame, extract_metadata

_MODEL = None

def load_model(path="outputs/best_model.joblib"):
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(path)
    return _MODEL


def classify(subject: str, sender_domain: str, body: str, model_path="outputs/best_model.joblib"):
    df = pd.DataFrame([{"subject": subject, "sender_domain": sender_domain, "body": body}])
    df["clean_text"] = build_combined(df).map(clean_text)
    X = pd.concat([df[["clean_text"]], metadata_frame(df)], axis=1)
    prob = float(load_model(model_path).predict_proba(X)[0, 1])
    meta = extract_metadata(subject, sender_domain, body)
    flags = []
    if meta["has_ip_url"]: flags.append("URL uses a raw IP address")
    if meta["has_shortener"]: flags.append("URL shortener used")
    if meta["suspicious_url_tld"]: flags.append("Link points to a suspicious domain extension")
    if meta["url_has_at"]: flags.append("URL contains '@' (hides real destination)")
    if meta["sender_bad_tld"]: flags.append("Sender domain has a suspicious extension")
    if meta["urgency_hits"] >= 3: flags.append("Multiple urgency / credential-related words")
    if meta["n_exclaim"] >= 2: flags.append("Excessive exclamation marks")
    return {"label": "PHISHING" if prob >= 0.5 else "LEGITIMATE", "phishing_probability": prob, "red_flags": flags}
