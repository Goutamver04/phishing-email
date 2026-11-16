# Comparative Analysis Report
## AI-Driven Phishing Email Detection Using NLP

---

## 1. Objective
Build, from scratch, a machine-learning system that detects phishing emails using Natural Language Processing (NLP) and metadata features, and compare four classifiers.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Custom synthetic dataset (`generate_dataset.py`) |
| Total emails | 3,000 (50% phishing / 50% legitimate) |
| After de-duplication | 2,964 |
| Train / Test split | 80 % / 20 % (stratified, seed 42) |
| Test set size | 593 emails |

**Columns**: `subject`, `sender_domain`, `body`, `label` (1=phishing, 0=legitimate)

**Dataset design includes**:
- *Stealthy* phishing: polite language, no urgency words, clean-looking URLs
- *Hard-negative* legitimate mails: genuine security alerts, links, action items
- 3% label noise to simulate real annotation uncertainty

> **Important limitation:** The data is synthetic and template-based. Results demonstrate the ML pipeline but should **not** be quoted as evidence of real-world accuracy. Re-run on a real corpus (Kaggle Phishing Email Dataset, Enron/SpamAssassin) using `python run_pipeline.py --data emails.csv --text-col body --label-col label`.

---

## 3. Methodology

### Phase 2 — Data Cleaning (NLP Preprocessing with NLTK)

We use **NLTK** (Natural Language Toolkit) for all text preprocessing:

| Step | Tool | Purpose |
|---|---|---|
| HTML decode + tag stripping | `html`, `re` | Remove formatting markup |
| URL replacement | `re` | Replace all links with `urltoken` |
| Lowercase | Python str | Normalise case |
| Punctuation removal | `re` | Strip non-alphanumeric chars |
| Tokenisation | `nltk.word_tokenize` | Split into individual tokens |
| Stopword removal | `nltk.corpus.stopwords` | Remove uninformative common words |
| Stemming | `nltk.stem.PorterStemmer` | Reduce words to root forms |

**Why NLTK?** NLTK is the standard academic NLP library for Python. Porter Stemming reduces vocabulary size (e.g. `verifying`, `verified`, `verify` → `verifi`) which improves TF-IDF generalisation.

### Phase 3 — Feature Engineering

**TF-IDF (Text features)**:
- Vocabulary: up to 5,000 terms (uni-grams + bi-grams)
- Sublinear TF scaling: `log(1 + TF)` dampens very common terms
- min_df=2: removes hapax legomena (terms appearing only once)

**Metadata features (14 signals)**:

| Feature | Signal |
|---|---|
| `n_urls` | Count of embedded links |
| `has_ip_url` | IP-based URLs bypass domain reputation systems |
| `has_shortener` | Hides real destination (bit.ly, tinyurl, etc.) |
| `url_has_at` | `@` before domain hides the real target |
| `suspicious_url_tld` | `.xyz`, `.top`, `.ru`, `.tk`, etc. |
| `avg_url_len` | Very long URLs suggest obfuscation |
| `sender_bad_tld` | Fake sender domains using cheap TLDs |
| `sender_hyphen` | `paypal-secure.xyz` impersonates brands |
| `has_html` | HTML used to hide malicious elements |
| `n_exclaim` | Multiple `!` create artificial urgency |
| `upper_ratio` | EXCESSIVE CAPITALS signal urgency |
| `digit_ratio` | Obfuscated URLs are digit-heavy |
| `body_len` | Very short or long bodies are atypical |
| `urgency_hits` | Count of urgency/credential keywords |

### Phase 4 — Model Development

All models trained as scikit-learn **Pipelines**: `ColumnTransformer(TF-IDF + MinMaxScaler) → Classifier`.  
Hyper-parameters tuned with **3-fold GridSearchCV** optimising F1 score.

| Model | Hyper-parameter grid |
|---|---|
| Logistic Regression | C ∈ {0.1, 1, 10} |
| Random Forest | n_estimators ∈ {100, 200}, max_depth ∈ {None, 20} |
| Multinomial Naive Bayes | α ∈ {0.1, 0.5, 1.0} |
| MLP Neural Network | hidden ∈ {(64,), (128,64)}, α ∈ {1e-4, 1e-2}, early_stopping=True |

---

## 4. Results (held-out test set, n = 593)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9663 | 0.9731 | 0.9601 | 0.9666 | **0.9739** | 0.9725 |
| Random Forest | 0.9663 | 0.9731 | 0.9601 | 0.9666 | 0.9734 | 0.9721 |
| Naive Bayes | 0.9663 | 0.9731 | 0.9601 | 0.9666 | 0.9686 | 0.9725 |
| Neural Network (MLP) | 0.9663 | 0.9731 | 0.9601 | 0.9666 | 0.9684 | 0.9725 |

### Feature Ablation Study (Logistic Regression)

| Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Text only (TF-IDF) | 0.9663 | 0.9731 | 0.9601 | 0.9666 | 0.9717 |
| Metadata only | 0.9309 | 0.9514 | 0.9103 | 0.9304 | 0.9651 |
| **Text + Metadata** | **0.9663** | **0.9731** | **0.9601** | **0.9666** | **0.9754** |

---

## 5. Interpretation

- **All four models reach 96.6% accuracy** — the ceiling imposed by 3% label noise. ROC-AUC is the only metric where they differ (Logistic Regression and Random Forest highest at 0.973–0.974).
- **Metadata alone achieves 93.1% accuracy** — URL and sender features are strong standalone predictors.
- **TF-IDF text + metadata combined gives the highest ROC-AUC (0.975)**, confirming both feature groups contribute complementary signal.
- **Recommended model: Logistic Regression** — matches top accuracy, highest ROC-AUC, and is fully interpretable via coefficient analysis.
- **Top phishing indicators** (from feature importance plots): `urltoken`, suspicious TLD features, urgency-word hits, `has_ip_url`, `upper_ratio`.
- **Common errors**: stealthy phishing with clean URLs and legitimate-sounding text; legitimate security notices that resemble phishing.

---

## 6. Outputs Generated

| File | Description |
|---|---|
| `data/raw_emails.csv` | Original 3,000-email dataset |
| `data/cleaned_emails.csv` | NLTK-preprocessed dataset with metadata columns |
| `outputs/model_comparison.csv` | Per-model metrics table |
| `outputs/model_comparison.png` | Bar chart of all metrics |
| `outputs/confusion_matrices.png` | 4-panel confusion matrix grid |
| `outputs/roc_curves.png` | ROC curves for all 4 models |
| `outputs/feature_importance_rf.png` | Random Forest top-20 feature importances |
| `outputs/feature_importance_lr.png` | Logistic Regression coefficient analysis |
| `outputs/ablation_features.csv` | Feature ablation results |
| `outputs/summary.json` | Best model summary + classification report |
| `outputs/best_model.joblib` | Serialised best model (Logistic Regression) |
| `outputs/presentation.pptx` | PowerPoint presentation (11 slides) |

---

## 7. Ethical Considerations

- **False positives**: blocking legitimate email is a real cost — use as decision support, not autonomous blocking.
- **False negatives**: missed phishing attacks can cause serious harm — prioritise recall in production.
- **Privacy**: do not store personal email content; process in-memory and discard.
- **Adversarial drift**: attackers adapt — retrain periodically on fresh data.
- **Transparency**: always explain model decisions to end-users (red flags, not just a label).

---

## 8. Recommendations

1. **Validate on real data**: Kaggle Phishing Email Dataset, Enron corpus, SpamAssassin public corpus.
2. **Add word embeddings / transformers**: DistilBERT or sentence-transformers for richer semantic features.
3. **Include email header features**: SPF, DKIM, DMARC authentication results are strong signals.
4. **Cross-validated reporting**: report confidence intervals over multiple seeds, not single-run results.
5. **Retrain periodically**: phishing tactics evolve; deploy a model retraining pipeline.
6. **Threshold tuning**: lower the classification threshold (< 0.5) to prioritise recall in high-risk environments.
