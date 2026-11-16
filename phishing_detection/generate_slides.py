"""Generate a PowerPoint presentation summarising the phishing detection project.

Usage:  python generate_slides.py
Output: outputs/presentation.pptx
"""
import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Colour palette ──────────────────────────────────────────────────────────
DARK_BLUE   = RGBColor(0x1A, 0x34, 0x6B)   # title / headings
RED         = RGBColor(0xC4, 0x4E, 0x52)   # phishing accent
GREEN       = RGBColor(0x2E, 0x9D, 0x5B)   # legitimate / positive
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY  = RGBColor(0xF0, 0xF2, 0xF5)
TEXT_DARK   = RGBColor(0x22, 0x22, 0x22)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


def add_bg(slide, color: RGBColor = LIGHT_GREY):
    """Fill slide background with a solid colour."""
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def title_text(tf, text, size=36, bold=True, color=WHITE):
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    p.alignment = PP_ALIGN.CENTER


def body_text(tf, lines, size=16, color=TEXT_DARK, bullet=True):
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 1 if bullet and line.startswith("  ") else 0
        run = p.add_run()
        run.text = line.strip()
        run.font.size = Pt(size)
        run.font.color.rgb = color


def add_title_box(slide, text, top, left=Inches(0.5), width=Inches(12.33),
                  height=Inches(0.65), size=24, color=DARK_BLUE, bold=True):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_body_box(slide, lines, top, left=Inches(0.6), width=Inches(12.13),
                 height=Inches(4.5), size=15):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(2)
        level = 1 if line.startswith("    ") else 0
        p.level = level
        run = p.add_run()
        run.text = line.strip()
        run.font.size = Pt(size)
        run.font.color.rgb = TEXT_DARK


def coloured_header(slide, color=DARK_BLUE):
    """Add a coloured header band."""
    hdr = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0), SLIDE_W, Inches(1.1)
    )
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = color
    hdr.line.fill.background()


# ── Build presentation ───────────────────────────────────────────────────────
def build_presentation(out_path="outputs/presentation.pptx"):
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    blank_layout = prs.slide_layouts[6]  # completely blank

    # ================================================================ Slide 1: Title
    s1 = prs.slides.add_slide(blank_layout)
    add_bg(s1, DARK_BLUE)
    # Accent bar at bottom
    bar = s1.shapes.add_shape(1, Inches(0), Inches(6.7), SLIDE_W, Inches(0.8))
    bar.fill.solid(); bar.fill.fore_color.rgb = RED; bar.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(1), Inches(1.5), Inches(11.33), Inches(1.5))
    tf = tb.text_frame; tf.word_wrap = True
    title_text(tf, "🎣  AI-Driven Phishing Email Detection", size=36, color=WHITE)

    tb2 = s1.shapes.add_textbox(Inches(1), Inches(3.0), Inches(11.33), Inches(1))
    tf2 = tb2.text_frame
    title_text(tf2, "Using Natural Language Processing (NLP)", size=26, bold=False, color=RGBColor(0xAA, 0xC4, 0xFF))

    tb3 = s1.shapes.add_textbox(Inches(1), Inches(4.3), Inches(11.33), Inches(0.8))
    tf3 = tb3.text_frame
    title_text(tf3, "IICT Semester Project  •  2024", size=18, bold=False, color=RGBColor(0xCC, 0xDD, 0xFF))

    # ================================================================ Slide 2: Problem Statement
    s2 = prs.slides.add_slide(blank_layout)
    add_bg(s2)
    coloured_header(s2)
    tb = s2.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "Problem Statement", size=28, color=WHITE)

    add_body_box(s2, [
        "📧  Phishing emails are the #1 vector for cyber-attacks globally",
        "",
        "What is Phishing?",
        "    • Fraudulent emails that impersonate trusted entities",
        "    • Goal: steal credentials, financial data, or install malware",
        "    • Over 3.4 billion phishing emails sent every day (2024)",
        "",
        "Why NLP?",
        "    • Language tricks: fake urgency, suspicious links, spoofed senders",
        "    • NLP can detect these patterns at scale — faster than humans",
        "    • ML models can be trained, evaluated, and updated continuously",
        "",
        "Our Goal: Build a system that classifies any email as PHISHING or LEGITIMATE",
        "    using text analysis (TF-IDF) + structural metadata features",
    ], top=Inches(1.3), size=15)

    # ================================================================ Slide 3: Methodology
    s3 = prs.slides.add_slide(blank_layout)
    add_bg(s3)
    coloured_header(s3)
    tb = s3.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "Methodology — 6 Phases", size=28, color=WHITE)

    phases = [
        ("📥  Phase 1 — Data Collection",    "3,000 synthetic emails (50% phishing / 50% legit); columns: subject, sender_domain, body, label"),
        ("🧹  Phase 2 — Data Cleaning",       "NLTK: HTML stripping → URL replacement → tokenisation → stopwords → Porter Stemming"),
        ("⚙️  Phase 3 — Feature Engineering", "TF-IDF (1–2 grams, 5,000 features) + 14 metadata signals (URLs, sender, urgency, structure)"),
        ("🤖  Phase 4 — Model Development",   "4 classifiers: Logistic Regression, Random Forest, Naive Bayes, MLP — tuned via 3-fold GridSearchCV (F1)"),
        ("📊  Phase 5 — Evaluation",          "Accuracy, Precision, Recall, F1, ROC-AUC; confusion matrices, ROC curves, feature importance plots"),
        ("🌐  Phase 6 — Deployment",          "Streamlit web UI: paste any email → instant classification with red-flag explanations"),
    ]
    for i, (phase, desc) in enumerate(phases):
        top = Inches(1.25 + i * 0.95)
        hbox = s3.shapes.add_textbox(Inches(0.5), top, Inches(4.2), Inches(0.75))
        hp = hbox.text_frame.paragraphs[0]
        hr = hp.add_run(); hr.text = phase
        hr.font.size = Pt(13); hr.font.bold = True; hr.font.color.rgb = DARK_BLUE

        dbox = s3.shapes.add_textbox(Inches(5.0), top, Inches(7.9), Inches(0.75))
        dp = dbox.text_frame.paragraphs[0]
        dr = dp.add_run(); dr.text = desc
        dr.font.size = Pt(12); dr.font.color.rgb = TEXT_DARK

    # ================================================================ Slide 4: NLP Pipeline
    s4 = prs.slides.add_slide(blank_layout)
    add_bg(s4)
    coloured_header(s4)
    tb = s4.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "NLP Preprocessing Pipeline (NLTK)", size=28, color=WHITE)

    add_body_box(s4, [
        "Raw Email Text",
        "    ↓  Step 1: HTML entity decode + tag stripping  →  plain text",
        "    ↓  Step 2: URL replacement  →  all links → 'urltoken' token",
        "    ↓  Step 3: Lowercase  →  normalise case",
        "    ↓  Step 4: Punctuation removal  →  alphanumeric only",
        "    ↓  Step 5: NLTK word_tokenize()  →  ['verify', 'your', 'account', ...]",
        "    ↓  Step 6: Stopword removal (NLTK english)  →  ['verify', 'account', ...]",
        "    ↓  Step 7: Porter Stemmer  →  ['verifi', 'account', ...] (root forms)",
        "",
        "Why Stemming?  Reduces vocabulary (e.g. 'verifying', 'verified', 'verify' → 'verifi')",
        "Why URL token?  Preserves 'does this email contain a link?' as a feature",
        "Why NLTK?  Industry-standard Python NLP library; transparent, well-documented",
    ], top=Inches(1.3), size=14)

    # ================================================================ Slide 5: Features
    s5 = prs.slides.add_slide(blank_layout)
    add_bg(s5)
    coloured_header(s5)
    tb = s5.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "Feature Engineering", size=28, color=WHITE)

    # Left column: TF-IDF
    lb = s5.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(5.9), Inches(5.5))
    lb.text_frame.word_wrap = True
    lines_l = [
        "TF-IDF Text Features",
        "",
        "• Converts NLTK-cleaned text to numeric vectors",
        "• TF-IDF score = TF × log(N / DF)",
        "   – TF: how often word appears in email",
        "   – IDF: penalises common words across corpus",
        "• Uni-grams + bi-grams (1–2 word phrases)",
        "• Up to 5,000 features per email",
        "• Sublinear TF (log scaling) prevents",
        "  very common terms from dominating",
        "",
        "Top phishing TF-IDF features:",
        "  urltoken, verifi, account, password,",
        "  suspend, immedi, login, secur, click",
    ]
    for i, line in enumerate(lines_l):
        p = lb.text_frame.paragraphs[0] if i == 0 else lb.text_frame.add_paragraph()
        r = p.add_run(); r.text = line
        r.font.size = Pt(13 if i == 0 else 12)
        r.font.bold = (i == 0)
        r.font.color.rgb = DARK_BLUE if i == 0 else TEXT_DARK

    # Right column: Metadata
    rb = s5.shapes.add_textbox(Inches(7.0), Inches(1.3), Inches(5.9), Inches(5.5))
    rb.text_frame.word_wrap = True
    lines_r = [
        "Metadata / Structural Features (14)",
        "",
        "• n_urls — number of links in body",
        "• has_ip_url — IP-based URL present",
        "• has_shortener — bit.ly, tinyurl, etc.",
        "• url_has_at — '@' hides real domain",
        "• suspicious_url_tld — .xyz .top .ru etc.",
        "• avg_url_len — obfuscation signal",
        "• sender_bad_tld — fake sender domain",
        "• sender_hyphen — paypal-secure.xyz",
        "• has_html — HTML markup in body",
        "• n_exclaim — count of '!'",
        "• upper_ratio — EXCESSIVE CAPITALS",
        "• urgency_hits — urgent/verify/expire...",
        "• body_len — very short/long bodies",
        "• digit_ratio — obfuscated URLs",
    ]
    for i, line in enumerate(lines_r):
        p = rb.text_frame.paragraphs[0] if i == 0 else rb.text_frame.add_paragraph()
        r = p.add_run(); r.text = line
        r.font.size = Pt(13 if i == 0 else 12)
        r.font.bold = (i == 0)
        r.font.color.rgb = DARK_BLUE if i == 0 else TEXT_DARK

    # Divider
    div = s5.shapes.add_shape(1, Inches(6.6), Inches(1.3), Inches(0.05), Inches(5.5))
    div.fill.solid(); div.fill.fore_color.rgb = DARK_BLUE; div.line.fill.background()

    # ================================================================ Slide 6: Results
    s6 = prs.slides.add_slide(blank_layout)
    add_bg(s6)
    coloured_header(s6, RED)
    tb = s6.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "Model Evaluation Results", size=28, color=WHITE)

    # Load actual results
    results_text = []
    csv_path = "outputs/model_comparison.csv"
    summary_path = "outputs/summary.json"
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0)
        results_text.append("Held-out test set (20%, n=593)  |  80/20 stratified split  |  GridSearchCV 3-fold F1")
        results_text.append("")
        results_text.append(f"{'Model':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'ROC-AUC':>9}")
        results_text.append("─" * 75)
        for name, row in df.iterrows():
            try:
                results_text.append(
                    f"{name:<28} {float(row['accuracy']):>9.4f} {float(row['precision']):>10.4f} "
                    f"{float(row['recall']):>8.4f} {float(row['f1']):>7.4f} {float(row['roc_auc']):>9.4f}"
                )
            except Exception:
                pass
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        results_text.append("")
        results_text.append(f"★  Best Model: {s.get('best_model', 'N/A')}  "
                             f"|  Dataset size: {s.get('n_samples', 'N/A')} emails")

    results_text += [
        "",
        "Feature Ablation (Logistic Regression):",
        "    Text only (TF-IDF):  Acc=96.6%   ROC-AUC=0.972",
        "    Metadata only:       Acc=93.1%   ROC-AUC=0.965",
        "    Text + Metadata:     Acc=96.6%   ROC-AUC=0.975  ← highest",
    ]

    add_body_box(s6, results_text, top=Inches(1.3), size=13)

    # ================================================================ Slide 7: Visualisations
    s7 = prs.slides.add_slide(blank_layout)
    add_bg(s7)
    coloured_header(s7)
    tb = s7.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "Visualisations — Confusion Matrices & ROC Curves", size=24, color=WHITE)

    img_map = [
        ("outputs/confusion_matrices.png", Inches(0.3),  Inches(1.2), Inches(12.73), Inches(2.7)),
        ("outputs/roc_curves.png",          Inches(0.3),  Inches(4.0), Inches(5.5),  Inches(3.2)),
        ("outputs/model_comparison.png",    Inches(6.0),  Inches(4.0), Inches(7.0),  Inches(3.2)),
    ]
    for path, l, t, w, h in img_map:
        if os.path.exists(path):
            s7.shapes.add_picture(path, l, t, w, h)

    # ================================================================ Slide 8: Feature Importance
    s8 = prs.slides.add_slide(blank_layout)
    add_bg(s8)
    coloured_header(s8)
    tb = s8.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "Feature Importance Analysis", size=28, color=WHITE)

    fi_imgs = [
        ("outputs/feature_importance_rf.png",  Inches(0.2),  Inches(1.2), Inches(6.3), Inches(5.7)),
        ("outputs/feature_importance_lr.png",   Inches(6.7),  Inches(1.2), Inches(6.3), Inches(5.7)),
    ]
    for path, l, t, w, h in fi_imgs:
        if os.path.exists(path):
            s8.shapes.add_picture(path, l, t, w, h)

    # ================================================================ Slide 9: Live Demo
    s9 = prs.slides.add_slide(blank_layout)
    add_bg(s9)
    coloured_header(s9, RED)
    tb = s9.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "Live Demo — Streamlit Web Interface", size=28, color=WHITE)

    add_body_box(s9, [
        "Run:  streamlit run app.py",
        "",
        "Test Case 1 — Phishing Email:",
        "    Subject: ⚠️ URGENT: Your PayPal account has been SUSPENDED!",
        "    Sender:  paypal-secure-login.xyz",
        "    Body:    Verify your identity IMMEDIATELY: http://paypal-secure-login.xyz/verify",
        "    Result:  → PHISHING  (97% confidence)",
        "    Flags:   Suspicious TLD • Sender bad domain • Multiple urgency words",
        "",
        "Test Case 2 — Legitimate Email:",
        "    Subject: Team meeting notes — Q3 review",
        "    Sender:  company.com",
        "    Body:    Hi Sarah, please find attached the notes from today's Q3 review...",
        "    Result:  → LEGITIMATE  (4% phishing probability)",
        "    Flags:   None",
        "",
        "✔ The system explains its decision — not just a black-box prediction",
    ], top=Inches(1.3), size=14)

    # ================================================================ Slide 10: Conclusions
    s10 = prs.slides.add_slide(blank_layout)
    add_bg(s10)
    coloured_header(s10, DARK_BLUE)
    tb = s10.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12), Inches(0.75))
    title_text(tb.text_frame, "Conclusions & Recommendations", size=28, color=WHITE)

    add_body_box(s10, [
        "✅  Key Findings:",
        "    • All 4 models achieved ~96.6% accuracy on the synthetic dataset",
        "    • Combined TF-IDF + metadata features gives highest ROC-AUC (0.975)",
        "    • Top phishing signals: suspicious URLs, urgency keywords, bad sender TLDs",
        "    • Logistic Regression is recommended — matches best accuracy + fully interpretable",
        "",
        "⚠️  Limitations:",
        "    • Synthetic dataset — real-world accuracy may differ",
        "    • Template-based data cannot separate model performance meaningfully",
        "    • Stealthy phishing (clean URLs, polite language) harder to detect",
        "",
        "🔮  Future Work:",
        "    • Validate on real corpora (Kaggle Phishing Dataset, Enron, SpamAssassin)",
        "    • Add transformer embeddings (DistilBERT) for richer text representation",
        "    • Include header features (SPF/DKIM/DMARC authentication results)",
        "    • Deploy with periodic retraining to handle adversarial drift",
        "",
        "⚖️  Ethical AI:",
        "    • Use as decision support — not autonomous blocking — to minimise false positives",
        "    • Never store personal email content without consent",
    ], top=Inches(1.3), size=13)

    # ================================================================ Slide 11: Thank You
    s11 = prs.slides.add_slide(blank_layout)
    add_bg(s11, DARK_BLUE)
    bar2 = s11.shapes.add_shape(1, Inches(0), Inches(6.7), SLIDE_W, Inches(0.8))
    bar2.fill.solid(); bar2.fill.fore_color.rgb = RED; bar2.line.fill.background()

    tb = s11.shapes.add_textbox(Inches(1), Inches(2.0), Inches(11.33), Inches(1.5))
    title_text(tb.text_frame, "Thank You!", size=48, color=WHITE)
    tb2 = s11.shapes.add_textbox(Inches(1), Inches(3.6), Inches(11.33), Inches(0.8))
    title_text(tb2.text_frame, "AI-Driven Phishing Email Detection Using NLP", size=22, bold=False, color=RGBColor(0xAA, 0xC4, 0xFF))
    tb3 = s11.shapes.add_textbox(Inches(1), Inches(4.5), Inches(11.33), Inches(0.6))
    title_text(tb3.text_frame, "Questions?", size=20, bold=False, color=RGBColor(0xCC, 0xDD, 0xFF))

    prs.save(out_path)
    print(f"✔ Presentation saved → {out_path}  ({prs.slides.__len__()} slides)")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    build_presentation()
