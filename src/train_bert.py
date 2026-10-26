from __future__ import annotations

import argparse
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
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

MODEL_CANDIDATES = [
    "hfl/rbt3",
    "hfl/rbt6",
    "hfl/chinese-bert-wwm-ext",
]
MAX_LEN = 128
BATCH = 16
EPOCHS_FREEZE = 3
EPOCHS_FINETUNE = 2
LR_HEAD = 2e-3
LR_FT = 2e-5


class SentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.enc = tokenizer(
            list(map(str, texts)),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.labels[i],
        }


class BertClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int = 2, freeze: bool = True, unfreeze_last_n: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_classes)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0:
                layers = None
                if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                    layers = self.encoder.encoder.layer
                elif hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
                    layers = self.encoder.transformer.layer
                if layers is not None:
                    for layer in layers[-unfreeze_last_n:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
                    for p in self.encoder.pooler.parameters():
                        p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(out, "pooler_output", None) is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


def resolve_model():
    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            print("Trying model", name)
            tok = AutoTokenizer.from_pretrained(name)
            _ = AutoModel.from_pretrained(name)
            print("Using", name)
            return name, tok
        except Exception as e:
            last_err = e
            print("Fail", name, type(e).__name__, str(e)[:160])
    raise RuntimeError(f"No pretrained model available: {last_err}")


def run_epoch(model, loader, optimizer, scheduler, criterion, device, train=True):
    model.train(train)
    total_loss = 0.0
    preds, labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            if train:
                optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * len(y)
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    metrics = compute_metrics(labels, preds)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics, preds


def run_variant(tag: str, freeze: bool, unfreeze_last_n: int, epochs: int, lr: float, model_name: str, tokenizer):
    set_seed(SEED)
    ensure_dirs()
    device = get_device()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_ds = SentDataset(train_df["text"], train_df["label"], tokenizer)
    val_ds = SentDataset(val_df["text"], val_df["label"], tokenizer)
    test_ds = SentDataset(test_df["text"], test_df["label"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    model = BertClassifier(model_name, freeze=freeze, unfreeze_last_n=unfreeze_last_n).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] trainable {n_trainable}/{n_total}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    history = []
    best_f1 = -1
    best_state = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scheduler, criterion, device, train=True)[0]
        va = run_epoch(model, val_loader, None, None, criterion, device, train=False)[0]
        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["accuracy"],
            "val_loss": va["loss"],
            "val_accuracy": va["accuracy"],
            "val_macro_f1": va["macro_f1"],
        })
        print(f"[{tag}] epoch {epoch} train_loss={tr['loss']:.4f} val_acc={va['accuracy']:.4f} val_f1={va['macro_f1']:.4f}")
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_m = run_epoch(model, val_loader, None, None, criterion, device, train=False)[0]
    test_m, test_preds = run_epoch(model, test_loader, None, None, criterion, device, train=False)
    for m in (val_m, test_m):
        m.pop("loss", None)
    elapsed = time.time() - t0

    result = {
        "method": f"BERT ({tag})",
        "model_name": model_name,
        "freeze_encoder": freeze,
        "unfreeze_last_n": unfreeze_last_n,
        "max_len": MAX_LEN,
        "batch_size": BATCH,
        "epochs": epochs,
        "lr": lr,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "history": history,
        "val": val_m,
        "test": test_m,
        "elapsed_sec": elapsed,
    }
    save_json(result, RESULTS_DIR / f"{tag}.json")
    pd.DataFrame({"y_true": test_df["label"], "y_pred": test_preds}).to_csv(
        RESULTS_DIR / f"{tag}_test_preds.csv", index=False
    )
    print("Saved", RESULTS_DIR / f"{tag}.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["all", "freeze", "finetune"], default="all")
    args = parser.parse_args()
    model_name, tokenizer = resolve_model()
    if args.variant in ("all", "freeze"):
        run_variant(
            "bert_freeze",
            freeze=True,
            unfreeze_last_n=0,
            epochs=EPOCHS_FREEZE,
            lr=LR_HEAD,
            model_name=model_name,
            tokenizer=tokenizer,
        )
    if args.variant in ("all", "finetune"):
        run_variant(
            "bert_finetune_last2",
            freeze=True,
            unfreeze_last_n=2,
            epochs=EPOCHS_FINETUNE,
            lr=LR_FT,
            model_name=model_name,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
