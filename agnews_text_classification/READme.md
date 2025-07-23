# 📰 AG‑News Text‑Classification Pipeline

Fine‑tuning **BERT** on the AG News dataset with a simple, reproducible workflow:

| stage | script | goal |
|-------|--------|------|
| 1️⃣ Prepare data | **`prepare_data.py`** | Download AG News, create `train.csv`, `val.csv`, `test.csv` |
| 2️⃣ Train | **`fine_tune.py`** | Fine‑tune `bert‑base‑uncased` and save the best checkpoint |
| 3️⃣ Evaluate | **`eval2.py`** | Compute accuracy/F1, draw plots, threshold curves, etc. |

---

## 📂 Project structure
```
text-classification/
├─ data/ # CSVs (created in step 1)
│ ├─ train.csv
│ ├─ val.csv
│ └─ test.csv
├─ models/ # checkpoints (created in step 2)
│ └─ best_model/
├─ results/ # plots & reports (created in step 3)
│ ├─ overall_metrics.txt
│ ├─ class_metrics.txt
│ └─ confusion_matrix.png
├─ preprocessing.py
├─ train.py
├─ evaluation.py
├─ requirements.txt
└─ README.md
```
```
Overall metrics
Accuracy  : 0.9430
Precision : 0.9431
Recall    : 0.9430
F1‑score  : 0.9430
```
