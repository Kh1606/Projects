# evaluate_detailed.py

import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt

# ── USER CONFIG ────────────────────────────────────────────────────────
MODEL_DIR  = "D:/text/models/manual/best_model"   # your trained HF folder
TEST_CSV   = "D:/text/test.csv"                   # CSV with columns "text","label"
RESULT_DIR = "D:/text/results_detailed"           # will be created if needed
BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1) Load test set
    ds = load_dataset("csv", data_files={"test": TEST_CSV})["test"]
    texts = ds["text"]
    labels = np.array(ds["label"])

    # 2) Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE).eval()

    # 3) Inference → logits, probabilities, predictions
    all_logits, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            enc   = tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_tensors="pt"
            )
            enc = {k:v.to(DEVICE) for k,v in enc.items()}
            out = model(**enc).logits
            probs = torch.softmax(out, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_logits.append(out.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_probs  = np.vstack(all_probs)
    all_preds  = np.concatenate(all_preds)

    # 4) Compute basic metrics
    acc       = accuracy_score(labels, all_preds)
    prec_w    = precision_score(labels, all_preds, average="weighted")
    rec_w     = recall_score(labels, all_preds, average="weighted")
    f1_w      = f1_score(labels, all_preds, average="weighted")

    # 5) Classification report (dict + str)
    report_dict = classification_report(
        labels, all_preds, output_dict=True, digits=4
    )
    report_str  = classification_report(
        labels, all_preds, digits=4
    )

    # 6) Save overall & per-class to TXT
    with open(os.path.join(RESULT_DIR, "overall_metrics.txt"), "w") as f:
        f.write(
            f"Accuracy : {acc:.4f}\n"
            f"Precision: {prec_w:.4f}\n"
            f"Recall   : {rec_w:.4f}\n"
            f"F1-score : {f1_w:.4f}\n"
        )
    with open(os.path.join(RESULT_DIR, "class_metrics.txt"), "w") as f:
        f.write(report_str)

    # 7) Save classification report as CSV
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(RESULT_DIR, "classification_report.csv"))

    # 8) Confusion matrix heatmap
    cm = confusion_matrix(labels, all_preds)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
    plt.close()

    # 9) ROC curves (one-vs-rest)
    from sklearn.preprocessing import label_binarize
    classes = sorted(report_dict.keys(), key=lambda x: x if x.isdigit() else "z")
    classes = [int(c) for c in classes if c.isdigit()]
    y_true_bin = label_binarize(labels, classes=classes)
    plt.figure()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        auc_score   = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_score:.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"))
    plt.close()

    # 10) Precision–Recall curves
    plt.figure()
    for i, cls in enumerate(classes):
        p, r, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        auc_pr  = auc(r, p)
        plt.plot(r, p, label=f"Class {cls} (AUC={auc_pr:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "pr_curve.png"))
    plt.close()

    # 11) Threshold analysis per class
    thresholds = np.linspace(0, 1, 101)
    for i, cls in enumerate(classes):
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            pred_bin = (all_probs[:, i] >= t).astype(int)
            precisions.append(precision_score(y_true_bin[:,i], pred_bin, zero_division=0))
            recalls.append(recall_score(y_true_bin[:,i], pred_bin, zero_division=0))
            f1s.append(f1_score(y_true_bin[:,i], pred_bin, zero_division=0))
        plt.figure()
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls,    label="Recall")
        plt.plot(thresholds, f1s,        label="F1-score")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"Threshold Analysis (Class {cls})")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(
            RESULT_DIR, f"threshold_analysis_class_{cls}.png"
        ))
        plt.close()

    # 12) Prediction confidence histogram
    max_probs = all_probs.max(axis=1)
    plt.figure()
    plt.hist(max_probs, bins=50)
    plt.xlabel("Max Predicted Probability")
    plt.ylabel("Count")
    plt.title("Prediction Confidence Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(
        RESULT_DIR, "confidence_distribution.png"
    ))
    plt.close()

    # 13) Minimal terminal output
    print("\nOverall metrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec_w:.4f}")
    print(f"  Recall   : {rec_w:.4f}")
    print(f"  F1-score : {f1_w:.4f}")
    print("\nClass-by-class metrics:")
    for cls in classes:
        m = report_dict[str(cls)]
        print(f"  Class {cls} → P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1-score']:.4f}")

if __name__ == "__main__":
    main()
