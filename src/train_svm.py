from __future__ import annotations

import time

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from utils import (
    DATA_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    SEED,
    compute_metrics,
    ensure_dirs,
    save_json,
    set_seed,
)


def tokenize(text: str) -> str:
    return " ".join(jieba.lcut(str(text)))


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    t0 = time.time()
    print("Tokenizing with jieba...")
    train_text = train["text"].map(tokenize)
    val_text = val["text"].map(tokenize)
    test_text = test["text"].map(tokenize)

    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    clf = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
    clf.fit(X_train, train["label"].values)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    val_m = compute_metrics(val["label"].values, val_pred)
    test_m = compute_metrics(test["label"].values, test_pred)
    elapsed = time.time() - t0

    result = {
        "method": "TF-IDF + LinearSVM",
        "val": val_m,
        "test": test_m,
        "elapsed_sec": elapsed,
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "max_features": 30000,
        "ngram_range": [1, 2],
    }
    save_json(result, RESULTS_DIR / "svm.json")
    pd.DataFrame({"y_true": test["label"], "y_pred": test_pred}).to_csv(
        RESULTS_DIR / "svm_test_preds.csv", index=False
    )
    print("VAL", val_m["accuracy"], val_m["macro_f1"])
    print("TEST", test_m["accuracy"], test_m["macro_f1"])
    print("Saved", RESULTS_DIR / "svm.json")


if __name__ == "__main__":
    main()
