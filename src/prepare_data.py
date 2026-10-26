from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from utils import DATA_DIR, SEED, ensure_dirs, save_json, set_seed


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    ds = load_dataset("lansinuote/ChnSentiCorp")
    meta = {"source": "lansinuote/ChnSentiCorp", "seed": SEED, "splits": {}}
    for split in ("train", "validation", "test"):
        df = pd.DataFrame(ds[split])
        df = df.dropna(subset=["text"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0]
        out = DATA_DIR / f"{split}.csv"
        df.to_csv(out, index=False)
        counts = df["label"].value_counts().to_dict()
        meta["splits"][split] = {
            "n": int(len(df)),
            "label_counts": {str(k): int(v) for k, v in counts.items()},
            "path": f"data/{split}.csv",
        }
        print(split, len(df), counts)
    save_json(meta, DATA_DIR / "meta.json")
    print("Saved to", DATA_DIR)


if __name__ == "__main__":
    main()
