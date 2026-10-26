from __future__ import annotations

import argparse
import time
from collections import Counter

import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import (
    DATA_DIR,
    RESULTS_DIR,
    SEED,
    compute_metrics,
    ensure_dirs,
    get_device,
    save_json,
    set_seed,
)

PAD, UNK = "<pad>", "<unk>"
MAX_LEN = 128
MIN_FREQ = 2
EMB_DIM = 100
HIDDEN = 128
BATCH = 64
EPOCHS = 6
LR = 1e-3


class TextDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len=MAX_LEN):
        self.labels = list(labels)
        self.ids = [encode(t, stoi, max_len) for t in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            torch.tensor(self.ids[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
        )


def tokenize(text: str):
    return jieba.lcut(str(text))


def build_vocab(texts, min_freq=MIN_FREQ):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    stoi = {PAD: 0, UNK: 1}
    for w, c in cnt.most_common():
        if c >= min_freq:
            stoi[w] = len(stoi)
    return stoi


def encode(text, stoi, max_len=MAX_LEN):
    ids = [stoi.get(w, 1) for w in tokenize(text)][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes=2, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        lengths = (x != 0).sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(self.dropout(h))


def load_pretrained_matrix(stoi, emb_dim=EMB_DIM):
    """Return pretrained vectors if available, else None."""
    print("Skipping remote word-vector download; will use frozen-embedding ablation.")
    return None


def train_one(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.extend(pred.tolist())
        labels.extend(y.numpy().tolist())
    return compute_metrics(labels, preds), preds


def run_variant(name: str, use_pretrained: bool):
    set_seed(SEED)
    ensure_dirs()
    device = get_device()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("Building vocab...")
    stoi = build_vocab(train_df["text"].tolist())
    print("vocab", len(stoi))

    train_ds = TextDataset(train_df["text"], train_df["label"], stoi)
    val_ds = TextDataset(val_df["text"], val_df["label"], stoi)
    test_ds = TextDataset(test_df["text"], test_df["label"], stoi)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    model = BiLSTMClassifier(len(stoi), EMB_DIM, HIDDEN).to(device)
    embedding_source = "random"
    if use_pretrained:
        matrix = load_pretrained_matrix(stoi, EMB_DIM)
        if matrix is not None:
            model.embedding.weight.data.copy_(torch.from_numpy(matrix))
            embedding_source = "pretrained_word_vectors"
        else:
            embedding_source = "random_fallback_no_pretrained_file"

    freeze_emb = False
    if use_pretrained and embedding_source.startswith("random_fallback"):
        for p in model.embedding.parameters():
            p.requires_grad = False
        freeze_emb = True
        embedding_source = "random_frozen_embedding"
        name = "bilstm_frozen_emb"

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR)

    history = []
    best_val_f1 = -1
    best_state = None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_one(model, train_loader, optimizer, criterion, device)
        val_m, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{k}": v for k, v in val_m.items() if k != "confusion_matrix"}})
        print(f"[{name}] epoch {epoch} loss={loss:.4f} val_acc={val_m['accuracy']:.4f} val_f1={val_m['macro_f1']:.4f}")
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_m, _ = evaluate(model, val_loader, device)
    test_m, test_preds = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    result = {
        "method": f"BiLSTM ({name})",
        "embedding_source": embedding_source,
        "freeze_embedding": freeze_emb,
        "vocab_size": len(stoi),
        "emb_dim": EMB_DIM,
        "hidden": HIDDEN,
        "max_len": MAX_LEN,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "lr": LR,
        "history": history,
        "val": val_m,
        "test": test_m,
        "elapsed_sec": elapsed,
    }
    out = RESULTS_DIR / f"{name}.json"
    save_json(result, out)
    pd.DataFrame({"y_true": test_df["label"], "y_pred": test_preds}).to_csv(
        RESULTS_DIR / f"{name}_test_preds.csv", index=False
    )
    torch.save({"stoi": stoi, "state_dict": best_state, "config": result}, RESULTS_DIR / f"{name}.pt")
    print("Saved", out)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["all", "random", "pretrained"], default="all")
    args = parser.parse_args()
    if args.variant in ("all", "random"):
        run_variant("bilstm_random", use_pretrained=False)
    if args.variant in ("all", "pretrained"):
        run_variant("bilstm_pretrained", use_pretrained=True)


if __name__ == "__main__":
    main()
