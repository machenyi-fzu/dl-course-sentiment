from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from utils import FIGURES_DIR, LABEL_NAMES, RESULTS_DIR, ensure_dirs

matplotlib.rcParams["font.sans-serif"] = [
    "Noto Serif CJK SC",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_json(name):
    path = RESULTS_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ensure_dirs()
    candidates = [
        ("TF-IDF+SVM", "svm.json", "svm_test_preds.csv"),
        ("BiLSTM-随机词向量", "bilstm_random.json", "bilstm_random_test_preds.csv"),
        ("BiLSTM-冻结词向量", "bilstm_frozen_emb.json", "bilstm_frozen_emb_test_preds.csv"),
        ("BiLSTM-预训练尝试", "bilstm_pretrained.json", "bilstm_pretrained_test_preds.csv"),
        ("BERT-冻结编码器", "bert_freeze.json", "bert_freeze_test_preds.csv"),
        ("BERT-微调末两层", "bert_finetune_last2.json", "bert_finetune_last2_test_preds.csv"),
    ]
    rows = []
    pred_files = []
    for label, jf, pf in candidates:
        jp = RESULTS_DIR / jf
        if not jp.exists():
            continue
        data = load_json(jf)
        rows.append({
            "method": label,
            "val_acc": data["val"]["accuracy"],
            "val_macro_f1": data["val"]["macro_f1"],
            "test_acc": data["test"]["accuracy"],
            "test_macro_f1": data["test"]["macro_f1"],
            "elapsed_sec": data.get("elapsed_sec"),
            "raw_file": jf,
        })
        pp = RESULTS_DIR / pf
        if pp.exists():
            pred_files.append((label, pp))

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    summary.to_json(RESULTS_DIR / "summary.json", force_ascii=False, indent=2, orient="records")
    print(summary)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(summary))
        w = 0.35
        ax.bar(x - w / 2, summary["test_acc"], w, label="Accuracy")
        ax.bar(x + w / 2, summary["test_macro_f1"], w, label="Macro-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["method"], rotation=20, ha="right")
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("分数")
        ax.set_title("测试集性能对比")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "metrics_compare.png", dpi=160)
        plt.close(fig)

    main_preds = [p for p in pred_files if p[0] in (
        "TF-IDF+SVM", "BiLSTM-随机词向量", "BERT-冻结编码器", "BERT-微调末两层"
    )]
    if not main_preds:
        main_preds = pred_files[:4]
    n = len(main_preds)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6))
        if n == 1:
            axes = [axes]
        for ax, (label, path) in zip(axes, main_preds):
            df = pd.read_csv(path)
            cm = pd.crosstab(df["y_true"], df["y_pred"])
            cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
            ax.set_title(label)
            ax.set_xlabel("预测")
            ax.set_ylabel("真实")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=160)
        plt.close(fig)

    curve_sources = []
    for label, jf, _ in candidates:
        jp = RESULTS_DIR / jf
        if not jp.exists():
            continue
        data = load_json(jf)
        hist = data.get("history")
        if hist:
            curve_sources.append((label, hist))
    if curve_sources:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, hist in curve_sources:
            xs = [h["epoch"] for h in hist]
            ys = [h.get("val_macro_f1") or h.get("val_f1") for h in hist]
            if ys[0] is None:
                ys = [h.get("val_macro_f1", h.get("val_accuracy")) for h in hist]
            if any(v is None for v in ys):
                ys = []
                for h in hist:
                    if "val_macro_f1" in h:
                        ys.append(h["val_macro_f1"])
                    elif "val_accuracy" in h:
                        ys.append(h["val_accuracy"])
                    else:
                        ys.append(None)
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("验证集 Macro-F1 / Acc")
        ax.set_title("训练过程验证曲线")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "training_curves.png", dpi=160)
        plt.close(fig)

    print("Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
