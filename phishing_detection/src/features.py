"""Phase 3 - Feature Engineering.

Two complementary feature sets are combined:

1. **TF-IDF** (Term Frequency-Inverse Document Frequency)
   - Uni-grams and bi-grams extracted from the NLTK-cleaned text.
   - Up to 5,000 features; min_df=2 removes noise; sublinear_tf dampens
     the effect of very frequent terms.

2. **Structural / Metadata features** (14 hand-crafted signals)
   - URL analysis  : count, IP-based URLs, shortener services, '@' in URL,
                     suspicious TLDs (.xyz, .top, .ru, etc.), average URL length.
   - Sender domain : bad TLD, hyphen presence (typo-squatting pattern).
   - Content tells  : HTML markup, exclamation count, uppercase ratio,
                      digit ratio, body length, urgency-word hits.

The ColumnTransformer `make_preprocessor()` glues both sets together so
that any scikit-learn Pipeline can consume a single DataFrame.
"""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------ constants
URL_RE = re.compile(r"(https?://[^\s<>'\"]+|www\.[^\s<>'\"]+)", re.I)
IP_RE  = re.compile(r"https?://\d{1,3}(\.\d{1,3}){3}")
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
              "is.gd", "buff.ly", "rb.gy", "cutt.ly")
BAD_TLDS = (".xyz", ".top", ".info", ".click", ".ru", ".tk", ".biz",
            ".zip", ".cc", ".gq", ".ml", ".cf")

# Phishing-associated urgency / credential vocabulary
URGENT_WORDS = [
    "urgent", "immediately", "suspend", "verify", "confirm", "expire",
    "final notice", "action required", "password", "winner", "won",
    "prize", "claim", "refund", "locked", "unusual activity",
    "security alert", "bank", "billing", "login", "account",
    "dear customer", "click here", "limited time", "free gift",
    "congratulations", "you have been selected",
]

# Columns produced by extract_metadata — keep in sync with ColumnTransformer
META_COLS = [
    "n_urls", "has_ip_url", "has_shortener", "url_has_at",
    "suspicious_url_tld", "avg_url_len", "sender_bad_tld",
    "sender_hyphen", "has_html", "n_exclaim",
    "upper_ratio", "digit_ratio", "body_len", "urgency_hits",
]


# -------------------------------------------------------------- feature funcs
def extract_metadata(subject: str, sender_domain: str, body: str) -> dict:
    """Compute 14 structural / linguistic features on the *raw* (uncleaned) email.

    Working on the raw text is intentional: HTML tags, URLs, and capitalisation
    are cleaned away in the NLP branch but carry valuable signal here.
    """
    body   = str(body)
    subject = str(subject)
    sender  = str(sender_domain).lower()

    urls    = URL_RE.findall(body)
    low     = (subject + " " + body).lower()
    letters = [c for c in body if c.isalpha()]

    return {
        "n_urls":           len(urls),
        "has_ip_url":       int(bool(IP_RE.search(body))),
        "has_shortener":    int(any(s in u.lower() for u in urls for s in SHORTENERS)),
        "url_has_at":       int(any("@" in u for u in urls)),
        "suspicious_url_tld": int(
            any(re.search(r"\.(xyz|top|info|click|ru|tk|biz|zip|cc|gq|ml|cf)(/|$|\?)", u.lower())
                for u in urls)
        ),
        "avg_url_len":      float(np.mean([len(u) for u in urls])) if urls else 0.0,
        "sender_bad_tld":   int(sender.endswith(BAD_TLDS)),
        "sender_hyphen":    int("-" in sender.split(".")[0]),   # hyphen in domain label
        "has_html":         int("<html" in body.lower() or "<a " in body.lower()),
        "n_exclaim":        body.count("!"),
        "upper_ratio":      (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "digit_ratio":      (sum(c.isdigit() for c in body) / len(body)) if body else 0.0,
        "body_len":         len(body),
        "urgency_hits":     sum(w in low for w in URGENT_WORDS),
    }


def metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply extract_metadata row-wise and return a tidy DataFrame."""
    rows = [
        extract_metadata(s, d, b)
        for s, d, b in zip(df["subject"], df["sender_domain"], df["body"])
    ]
    return pd.DataFrame(rows, columns=META_COLS, index=df.index)


def make_preprocessor(use_text: bool = True, use_meta: bool = True,
                       max_features: int = 5000) -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer combining TF-IDF and metadata.

    Parameters
    ----------
    use_text      : include the TF-IDF text branch
    use_meta      : include the MinMax-scaled metadata branch
    max_features  : vocabulary cap for TF-IDF
    """
    parts = []
    if use_text:
        parts.append((
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=max_features,
                min_df=2,
                sublinear_tf=True,       # log(1+tf) — reduces dominance of very common terms
            ),
            "clean_text",
        ))
    if use_meta:
        parts.append((
            "meta",
            MinMaxScaler(),              # scales to [0,1] so metadata works with Naive Bayes
            META_COLS,
        ))
    return ColumnTransformer(parts)
