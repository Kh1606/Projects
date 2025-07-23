# fine_tune.py
import os
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── CONFIG ────────────────────────────────────────────────────────────────
MODEL_NAME    = "bert-base-uncased"
OUTPUT_DIR    = "D:/text/models/manual"    # where best_model/ will be saved
TRAIN_CSV     = "D:/text/train.csv"
VAL_CSV       = "D:/text/val.csv"
EPOCHS        = 3
MAX_LENGTH    = 128
TRAIN_BS      = 16
EVAL_BS       = 32
LOGGING_STEPS = 50
# ──────────────────────────────────────────────────────────────────────────

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted")
    }

def main():
    # 1) Load & tokenize
    ds = load_dataset("csv", data_files={"train":TRAIN_CSV, "validation":VAL_CSV})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tok(batch):
        return tokenizer(batch["text"],
                        truncation=True,
                        padding="max_length",
                        max_length=MAX_LENGTH)
    ds = ds.map(tok, batched=True)
    ds.set_format("torch", columns=["input_ids","attention_mask","label"])
    train_ds, val_ds = ds["train"], ds["validation"]

    # 2) Prepare model
    num_labels = len(set(train_ds["label"]))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # 3) Epoch-by-epoch train + eval
    best_f1, best_epoch = 0.0, 0
    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, f"epoch_{epoch}"),
            num_train_epochs=1,
            per_device_train_batch_size=TRAIN_BS,
            per_device_eval_batch_size=EVAL_BS,
            logging_steps=LOGGING_STEPS,
            report_to=[],           # disable WandB/other loggers
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        # train one epoch
        trainer.train()

        # evaluate
        metrics = trainer.evaluate()
        f1  = metrics.get("eval_f1")
        acc = metrics.get("eval_accuracy")
        print(f"-- Val Accuracy: {acc:.4f} | Val F1: {f1:.4f}")

        # save best checkpoint
        if f1 is not None and f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_path = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_path, exist_ok=True)
            trainer.save_model(best_path)
            print(f">>> New best model (epoch {epoch}, F1={f1:.4f}) saved to {best_path}")

    print(f"\nDone. Best F1={best_f1:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
