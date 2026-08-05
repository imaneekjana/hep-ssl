import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from colliderml.core import load_tables, collect_tables
from sklearn.metrics import roc_auc_score
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import CaloEvent
from src.data.dataset import ColliderMLHits, projected_hits
from src.models.gnn import GravNetEncoder

class LabeledGraphDataset(IterableDataset):

    def __init__(self, events, labels, seed):
        super().__init__()

        self.events = ColliderMLHits(events, split=None, shuffle_files=False, log=False, seed=seed)

        self.labels = labels

    def __iter__(self):
        for event_dict, label in zip(self.events, self.labels):
            event = event_dict["calo_hit_features"]

            if isinstance(event, CaloEvent):
                hits = event.hits
            else: hits = np.asarray(event)

            eta_phi_hits = projected_hits(hits, grid_size=32, typ="hits")["eta-phi"]

            yield Data(x=torch.tensor(eta_phi_hits, dtype=torch.float32), y=torch.tensor(label, dtype=torch.long))

def load_channel(channel, data_dir, max_events):
    config = {"dataset_id": "CERN/ColliderML-Release-1",
              "channels": channel,
              "pileup": "pu0",
              "objects": ["calo_hits"],
              "split": "train",
              "lazy": False,
              "max_events": max_events,
              "data_dir": str(data_dir)}

    tables = load_tables(config)
    frames = collect_tables(tables)
    return frames["calo_hits"]

def extract_embeddings(encoder, events, labels, batch_size, seed, device):
    dataset = LabeledGraphDataset(events, labels, seed)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    embeddings = []
    saved_labels = []
    encoder.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings.append(encoder(batch).cpu().numpy())

            saved_labels.append(batch.y.cpu().numpy())
    return(np.concatenate(embeddings), np.concatenate(saved_labels))

def evaluate(classifier, features, labels):
    classifier.eval()

    with torch.no_grad():
        logits = classifier(features)
        loss = F.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        accuracy = (logits.argmax(dim=1) == labels).float().mean()

    return (loss.item(), accuracy.item(), probabilities.cpu().numpy())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True, type=Path)

    parser.add_argument("--data-dir", required=True, type=Path)

    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--events-per-class", default=1500, type=int)

    parser.add_argument("--encoder-batch-size", default=64, type=int)

    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--lr", default=1e-3, type=float)

    parser.add_argument("--weight-decay", default=1e-4, type=float)

    parser.add_argument("--seed", default=53, type=int)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint["args"]
    split_seed = int(saved_args["seed"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ttbar = load_channel("ttbar", args.data_dir, args.events_per_class)
    ggf = load_channel("ggf", args.data_dir, args.events_per_class)

    ttbar = ttbar.with_columns(pl.lit(0).alias("_label"))
    ggf = ggf.with_columns(pl.lit(1).alias("_label"))

    combined = pl.concat([ttbar, ggf]).sample(fraction=1.0, shuffle=True, seed=split_seed)
    labels = combined["_label"].to_numpy()
    events = combined.drop("_label")

    n_total = len(events)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_events = events[: n_train]
    val_events = events[n_train: n_train + n_val]
    test_events = events[n_train + n_val: ]

    train_labels = labels[: n_train]
    val_labels = labels[n_train: n_train + n_val]
    test_labels = labels[n_train + n_val: ]

    encoder = GravNetEncoder(in_features=3, 
                             hidden_dim=saved_args["hidden_dim"],
                             latent_dim=saved_args["latent_dim"],
                             proj_dim=saved_args["proj_dim"],
                             k=saved_args["gravnet_k"],
                             space_dim=saved_args["space_dim"],
                             propagate_dim=saved_args["propagate_dim"])

    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.projection_head = nn.Identity()
    encoder = encoder.to(device)

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    train_x, train_y = extract_embeddings(encoder, train_events, train_labels, args.encoder_batch_size, split_seed, device)

    val_x, val_y = extract_embeddings(encoder, val_events, val_labels, args.encoder_batch_size, split_seed, device)

    test_x, test_y = extract_embeddings(encoder, test_events, test_labels, args.encoder_batch_size, split_seed, device)

    np.savez_compressed(args.output_dir / "latents.npz", train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y)

    feature_mean = train_x.mean(axis=0, keepdims=True)
    feature_std = train_x.std(axis=0, keepdims=True) + 1e-8

    train_x = (train_x - feature_mean) / feature_std
    val_x = (val_x - feature_mean) / feature_std
    test_x = (test_x - feature_mean) / feature_std

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.long)
    test_y = torch.tensor(test_y, dtype=torch.long)

    classifier = nn.Linear(train_x.shape[1], 2)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        optimizer.zero_grad()

        train_logits = classifier(train_x)
        train_loss = F.cross_entropy(train_logits, train_y)

        train_loss.backward()
        optimizer.step()

        train_accuracy = (train_logits.argmax(dim=1) == train_y).float().mean().item()

        val_loss, val_accuracy, _ = evaluate(classifier, val_x, val_y)

        history.append({"epoch": epoch,
                        "train_loss": train_loss.item(),
                        "val_loss": val_loss,
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_accuracy})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(classifier.state_dict())

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss.item():.6f} | "
              f"val_loss={val_loss:.6f} | "
              f"train_acc={train_accuracy:.4f} | "
              f"val_acc={val_accuracy:.4f}")

    classifier.load_state_dict(best_state)

    val_loss, val_accuracy, _ = evaluate(classifier, val_x, val_y)
    test_loss, test_accuracy, test_probabilities = evaluate(classifier, test_x, test_y)
    test_auc = roc_auc_score(test_y.numpy(), test_probabilities)

    np.savez_compressed(args.output_dir / "test_predictions.npz", labels=test_y.numpy(), probabilities=test_probabilities,)

    metrics = {"best_epoch": best_epoch,
               "best_val_loss": best_val_loss,
               "val_accuracy": val_accuracy,
               "test_loss": test_loss,
               "test_accuracy": test_accuracy,
               "test_roc_auc": test_auc,
               "labels": {"ttbar": 0, "ggf": 1}}

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    torch.save({"model_state_dict": best_state, "feature_mean": feature_mean, "feature_std": feature_std}, args.output_dir / "linear_probe.pt")

    print("\nClassifier results")
    print(f"Best epoch: {best_epoch}")
    print(f"Test accuracy: "
          f"{test_accuracy:.4f}")
    print(f"Test ROC-AUC: "
          f"{test_auc:.4f}")
                
if __name__ == "__main__":
    main()