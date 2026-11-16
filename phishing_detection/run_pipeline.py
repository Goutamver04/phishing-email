"""End-to-end runner:  python run_pipeline.py [--data file.csv --text-col body --label-col label]"""
import argparse, os
from src.train import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw_emails.csv")
    ap.add_argument("--text-col", default="body")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--subject-col", default="subject")
    ap.add_argument("--sender-col", default="sender_domain")
    a = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(a.data) and a.data == "data/raw_emails.csv":
        import generate_dataset; generate_dataset.main()
    res, abl, best = run(a.data, text_col=a.text_col, label_col=a.label_col, subject_col=a.subject_col, sender_col=a.sender_col)
    print("\n=== Model comparison ===\n", res.drop(columns="best_params").astype(float).round(4))
    print("\n=== Feature ablation (Logistic Regression) ===\n", abl and __import__("pandas").DataFrame(abl).T.round(4))
