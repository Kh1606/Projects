import pandas as pd
from datasets import load_dataset

def main():
    ds = load_dataset("ag_news")
    # HuggingFace default splits: train (120k) / test (7.6k)
    train_valid = ds["train"].train_test_split(test_size=0.1, seed=42)
    train = train_valid["train"]
    val   = train_valid["test"]
    test  = ds["test"]

    # Convert and save
    for split, d in zip(["train","val","test"], [train, val, test]):
        df = pd.DataFrame({
            "text": d["text"],
            "label": d["label"]
        })
        df.to_csv(f"../data/{split}.csv", index=False)

if __name__ == "__main__":
    main()
