# 🎣 AI-Driven Phishing Email Detection Using NLP

> Automatically classify emails as **PHISHING** or **LEGITIMATE** using Natural Language Processing and machine learning — with explainable red-flag detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Pipeline Phases](#-pipeline-phases)
- [Models](#-models)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web Interface](#-web-interface)
- [Outputs](#-outputs)
- [Deliverables](#-deliverables)
- [Ethical Considerations](#-ethical-considerations)
- [Future Work](#-future-work)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

Phishing emails are the **#1 vector for cyber-attacks** globally — over 3.4 billion sent every day. This project builds a full machine-learning pipeline that:

- 🧹 **Cleans** raw email text using **NLTK** (tokenisation, stopword removal, Porter Stemming)
- ⚙️ **Extracts features** via **TF-IDF** (text) + **14 hand-crafted structural/metadata signals**
- 🤖 **Trains and compares** four classifiers: Logistic Regression, Random Forest, Naive Bayes, MLP Neural Network
- 📊 **Evaluates** using accuracy, precision, recall, F1, ROC-AUC, and confusion matrices
- 🌐 **Deploys** a Streamlit web app for live email classification with human-readable red-flag explanations

### 🏆 Best Model Results (Logistic Regression)

| Metric | Score |
|---|---|
| ✅ Accuracy | **96.63%** |
| 🎯 Precision | **97.31%** |
| 🔍 Recall | **96.01%** |
| ⚖️ F1 Score | **96.66%** |
| 📈 ROC-AUC | **97.50%** |

---

## 📁 Project Structure

```
phishing-email/
│
├── 📄 README.md                               ← You are here
├── 🔧 .gitignore
│
└── phishing_detection/
    ├── 📓 phishing_detection_notebook.ipynb   ← Full documented notebook (all 6 phases)
    ├── 🚀 run_pipeline.py                     ← End-to-end runner (generate → train → evaluate)
    ├── 🌐 app.py                              ← Streamlit web UI
    ├── 📊 generate_slides.py                  ← PowerPoint presentation generator (11 slides)
    ├── 🧪 generate_dataset.py                 ← Synthetic dataset generator
    ├── 📋 requirements.txt                    ← Python dependencies
    ├── 📄 comparative_report.md               ← Detailed analysis report
    │
    ├── data/
    │   ├── raw_emails.csv                     ← Original 3,000-email dataset
    │   └── cleaned_emails.csv                 ← NLTK-preprocessed + metadata columns
    │
    ├── src/
    │   ├── preprocess.py                      ← Phase 2: NLTK cleaning pipeline
    │   ├── features.py                        ← Phase 3: TF-IDF + metadata features
    │   ├── train.py                           ← Phase 4–5: Model training & evaluation
    │   └── predict.py                         ← Phase 6: Live classification helper
    │
    └── outputs/
        ├── best_model.joblib                  ← Serialised best model
        ├── confusion_matrices.png             ← 4-panel confusion matrix grid
        ├── roc_curves.png                     ← ROC curves for all 4 models
        ├── model_comparison.png               ← Bar chart of all metrics
        ├── feature_importance_rf.png          ← Random Forest top-20 features
        ├── feature_importance_lr.png          ← Logistic Regression coefficients
        ├── model_comparison.csv               ← Per-model metrics table
        ├── ablation_features.csv              ← Feature ablation study results
        ├── summary.json                       ← Best model report (JSON)
        └── presentation.pptx                  ← 11-slide PowerPoint presentation
```

---

## 🔄 Pipeline Phases

```
Raw Email Data
      │
      ▼
┌─────────────────────────────────────────┐
│  Phase 1 — Data Collection              │
│  3,000 emails · 50% phishing / 50% legit│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Phase 2 — Data Cleaning (NLTK)         │
│  HTML strip → URL token → lowercase     │
│  → tokenise → stopwords → stemming      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Phase 3 — Feature Engineering          │
│  TF-IDF (5,000 uni+bi-grams)            │
│  + 14 metadata signals                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Phase 4 — Model Development            │
│  4 classifiers · GridSearchCV (F1)      │
│  3-fold CV · stratified 80/20 split     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Phase 5 — Evaluation                   │
│  Accuracy · Precision · Recall · F1     │
│  ROC-AUC · Confusion matrices           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Phase 6 — Deployment                   │
│  Streamlit web app · Live classification│
│  with red-flag explanations             │
└─────────────────────────────────────────┘
```

### 🧹 NLP Preprocessing (NLTK)

| Step | Tool | Purpose |
|---|---|---|
| HTML decode + strip | `html`, `re` | Remove `<b>`, `<a href=...>`, etc. |
| URL replacement | `re` | All links → `urltoken` (preserves URL *presence*) |
| Lowercase | Python str | Normalise case |
| Punctuation removal | `re` | Strip non-alphanumeric characters |
| Tokenisation | `nltk.word_tokenize` | Split into individual words |
| Stopword removal | `nltk.corpus.stopwords` | Remove *the*, *and*, *is*, etc. |
| Stemming | `nltk.stem.PorterStemmer` | `verifying` → `verifi`, `suspended` → `suspend` |

### ⚙️ Feature Engineering

**TF-IDF** — converts cleaned text to numeric vectors (up to 5,000 uni+bi-gram features, sublinear TF scaling).

**Metadata features (14 signals)**:

| Feature | Phishing Signal |
|---|---|
| `n_urls` | Phishing mails embed more links |
| `has_ip_url` | IP URLs bypass domain reputation |
| `has_shortener` | `bit.ly`, `tinyurl` hide real destination |
| `url_has_at` | `@` in URL hides the actual domain |
| `suspicious_url_tld` | `.xyz` `.top` `.ru` `.tk` etc. |
| `avg_url_len` | Very long URLs suggest obfuscation |
| `sender_bad_tld` | Fake sender using cheap TLD |
| `sender_hyphen` | `paypal-secure.xyz` impersonates brands |
| `has_html` | HTML used to embed hidden elements |
| `n_exclaim` | Multiple `!` create artificial urgency |
| `upper_ratio` | EXCESSIVE CAPITALS signal urgency |
| `digit_ratio` | Obfuscated URLs are digit-heavy |
| `body_len` | Very short/long bodies are atypical |
| `urgency_hits` | Count of urgency/credential keywords |

---

## 🤖 Models

| Model | Why Used |
|---|---|
| **Logistic Regression** ⭐ | Fast, fully interpretable linear baseline — recommended |
| **Random Forest** | Handles mixed text+metadata; provides feature importance |
| **Multinomial Naive Bayes** | Classic NLP model; works well with TF-IDF counts |
| **MLP Neural Network** | Captures non-linear patterns; early stopping prevents overfitting |

All models use a scikit-learn **Pipeline** with **3-fold GridSearchCV** optimising F1 score.

---

## 📊 Results

### Model Comparison (held-out test set, n = 593)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.9663 | 0.9731 | 0.9601 | 0.9666 | **0.9750** |
| Random Forest | 0.9663 | 0.9731 | 0.9601 | 0.9666 | 0.9684 |
| Naive Bayes | 0.9663 | 0.9731 | 0.9601 | 0.9666 | 0.9700 |
| Neural Network (MLP) | 0.9663 | 0.9731 | 0.9601 | 0.9666 | 0.9706 |

### Feature Ablation Study (Logistic Regression)

| Configuration | Accuracy | ROC-AUC |
|---|---|---|
| Text only (TF-IDF) | 0.9663 | 0.9705 |
| Metadata only | 0.9325 | 0.9666 |
| **Text + Metadata** ⭐ | **0.9663** | **0.9750** |

> Both feature groups contribute: metadata alone reaches 93.3%; combined gives the highest ROC-AUC.

---

## ⚙️ Installation

**Requirements:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/Goutamver04/phishing-email.git
cd phishing-email/phishing_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (first time only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

**Dependencies:**
```
pandas · numpy · scikit-learn · matplotlib · seaborn · joblib · streamlit · nltk · python-pptx
```

---

## 🚀 Usage

### Run the full pipeline

```bash
cd phishing_detection
python run_pipeline.py
```

### Use a custom dataset (Kaggle, UCI, real email export)

```bash
python run_pipeline.py --data emails.csv --text-col body --label-col label
```

> Label convention: `1` = phishing, `0` = legitimate

### Classify a single email (Python API)

```python
from src.predict import classify

result = classify(
    subject="⚠️ URGENT: Your account has been suspended!",
    sender_domain="paypal-secure-login.xyz",
    body="Verify your identity IMMEDIATELY: http://paypal-secure-login.xyz/verify"
)

print(result["label"])                  # PHISHING
print(result["phishing_probability"])   # 0.97
print(result["red_flags"])              # ['Suspicious TLD', 'Sender bad domain', ...]
```

### Generate the PowerPoint presentation

```bash
python generate_slides.py
# → outputs/presentation.pptx  (11 slides)
```

### Open the Jupyter notebook

```bash
jupyter notebook phishing_detection_notebook.ipynb
```

---

## 🌐 Web Interface

Launch the interactive Streamlit app:

```bash
cd phishing_detection
streamlit run app.py
```

Paste any email (subject, sender, body) and instantly get:
- ✅ / ❌ **LEGITIMATE** or **PHISHING** verdict
- 📈 Phishing probability (0–100%)
- 🚩 Human-readable **red flags** that triggered the alert

---

## 📦 Outputs

| File | Description |
|---|---|
| `outputs/best_model.joblib` | Serialised best classifier (ready for deployment) |
| `outputs/model_comparison.csv` | Accuracy, Precision, Recall, F1, ROC-AUC for all 4 models |
| `outputs/summary.json` | Full classification report (JSON format) |
| `outputs/confusion_matrices.png` | Side-by-side confusion matrices for all 4 models |
| `outputs/roc_curves.png` | ROC curves with AUC scores for all 4 models |
| `outputs/model_comparison.png` | Bar chart comparing all models |
| `outputs/feature_importance_rf.png` | Random Forest top-20 feature importances |
| `outputs/feature_importance_lr.png` | Logistic Regression coefficient analysis |
| `outputs/ablation_features.csv` | Text-only vs metadata-only vs combined results |
| `outputs/presentation.pptx` | 11-slide PowerPoint presentation |

---

## 📋 Deliverables

| Deliverable | File | Status |
|---|---|---|
| Python notebook (all 6 phases, documented) | `phishing_detection_notebook.ipynb` | ✅ |
| Raw dataset | `data/raw_emails.csv` | ✅ |
| Cleaned dataset | `data/cleaned_emails.csv` | ✅ |
| Comparative analysis report | `comparative_report.md` | ✅ |
| Presentation slides (11 slides) | `outputs/presentation.pptx` | ✅ |
| Visualisations (5 plots) | `outputs/*.png` | ✅ |
| Saved best model | `outputs/best_model.joblib` | ✅ |
| Streamlit web interface | `app.py` | ✅ |

---

## ⚖️ Ethical Considerations

- **False positives** (blocking real mail) and **false negatives** (missed attacks) both carry real costs — use as decision support, not autonomous blocking.
- **Privacy**: do not store personal email content; process in-memory only.
- **Adversarial drift**: phishing tactics evolve — retrain periodically on fresh data.
- **Transparency**: always explain decisions to end-users (red flags, not just a label).

---

## 🔮 Future Work

- Validate on real corpora: [Kaggle Phishing Email Dataset](https://www.kaggle.com/), Enron corpus, SpamAssassin public corpus
- Add **transformer embeddings** (DistilBERT / sentence-transformers) for richer text understanding
- Include **email header features** (SPF, DKIM, DMARC authentication results)
- Lower classification threshold (< 0.5) to maximise recall in high-risk environments
- Deploy with **periodic retraining** pipeline to handle adversarial drift

---

## 🛠️ Tech Stack

| Component | Library / Tool |
|---|---|
| NLP preprocessing | `nltk` — tokenise, stopwords, Porter Stemmer |
| Text features | `scikit-learn` TfidfVectorizer (TF-IDF) |
| ML models | `scikit-learn` — LR, Random Forest, Naive Bayes, MLP |
| Hyperparameter tuning | `scikit-learn` GridSearchCV (3-fold CV) |
| Data handling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Web interface | `streamlit` |
| Model persistence | `joblib` |
| Presentation generation | `python-pptx` |

---

<p align="center">
  
</p>
