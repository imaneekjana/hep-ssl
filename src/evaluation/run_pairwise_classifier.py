import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from colliderml.core import collect_tables, load_tables
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from src.data.dataset import ColliderMLHits, projected_hits
from src.evaluation.classification import evaluate_two_embedding_sets
from src.models.gnn import GravNetEncoder


VALID_CLASSIFIERS = (
    "knn",
    "logistic_regression",
    "gradient_boosting",
    "random_forest",
    "mlp",
)

class CleanPlanarGraphDataset(IterableDataset):
    """
    Convert clean ColliderML events into normalized eta-phi graphs"
    No augmentation is applied
    """

    def __init__(self, events, feature_mean, feature_std, grid_size=32, seed=42):
        super().__init__()
        self.events = ColliderMLHits(events, split=None, shuffle_files=False, log=False, seed=seed)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_std = np.asarray(feature_std, dtype=np.float64)
        self.grid_size = grid_size

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        for event_dict in self.events:
            raw_hits = event_dict["calo_hit_features"]
            nodes = projected_hits(raw_hits, grid_size=self.grid_size, typ="hits")["eta-phi"]

            if nodes.ndim !=2 or nodes.shape[1] !=3:
                raise RuntimeError(
                    "Expected eta-phi graph nodes with shape "
                    f"(n_nodes, 3), received {nodes.shape}."
                )

            if len(nodes) == 0:
                raise RuntimeError("empty eta-phi graph")

            if not np.isfinite(nodes).all():
                raise RuntimeError(
                    "NaN or Inf found before normalization."
                )

            normalized_nodes = (nodes - self.feature_mean) / (self.feature_std)

            if not np.isfinite(normalized_nodes).all():
                raise RuntimeError(
                    "NaN or Inf found after normalization."
                )
            yield Data(x=torch.tensor(normalized_nodes, dtype= torch.float32))

def load_channel(channel, events_per_channel, data_dir):
    config = {"dataset_id": "CERN/ColliderML-Release-1",
              "channels": channel,
              "pileup": "pu0",
              "objects": ["calo_hits"],
              "split": "train",
              "lazy": False,
              "max_events": events_per_channel,
              "data_dir": str(data_dir)}
    tables = load_tables(config)
    frames = collect_tables(tables)
    events = frames["calo_hits"]

    if len(events) != events_per_channel:
        raise RuntimeError(
            f"Requested {events_per_channel} events "
            f"from {channel}, but loaded {len(events)}."
        )

    print(f"Loaded {len(events)} events "
          f"from channel {channel}.")

    return events

def load_projection_stats(stats_path):
    saved = np.load(stats_path, allow_pickle=True).item()

    if "means" not in saved or "stds" not in saved:
        raise KeyError(f"{stats_path} does not contain "
                       "'means' and 'stds'.")

    means = saved["means"]
    stds = saved["stds"]

    if len(means) != 1 or len(stds) !=1:
        raise ValueError("This classifier expects exactly one "
                         "projection: eta-phi.")

    feature_mean = np.asarray(means[0], dtype = np.float64)
    feature_std = np.asarray(stds[0], dtype = np.float64)

    if feature_mean.shape != (3,):
        raise ValueError("Expected eta-phi mean with shape (3,), "
                         f"received {feature_mean.shape}.")

    if feature_std.shape != (3,):
        raise ValueError("Expected eta-phi std with shape (3,), "
                         f"received {feature_std.shape}.")

    if np.any(feature_std <= 0):
        raise ValueError("All projection standard deviations "
                         "must be positive.")
    return feature_mean, feature_std

def build_frozen_encoder(checkpoint, device):
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain "
                       "'model_state_dict'.")

    if "args" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'args'.")

    saved_args = checkpoint["args"]
    encoder = GravNetEncoder(in_features=3, hidden_dim=int(saved_args["hidden_dim"]), latent_dim=int(saved_args["latent_dim"]), proj_dim=int(saved_args["proj_dim"]), k = int(saved_args["gravnet_k"]), space_dim = int(saved_args["space_dim"]), propagate_dim = int(saved_args["propagate_dim"]))

    encoder.load_state_dict(checkpoint["model_state_dict"])

    encoder.projection_head = nn.Identity()

    encoder = encoder.to(device)
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    return encoder, saved_args

def extract_embeddings(encoder, events, feature_mean, feature_std, batch_size, seed, device):
    dataset = CleanPlanarGraphDataset(events = events, feature_mean=feature_mean, feature_std=feature_std, grid_size=32, seed=seed)
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = 0, drop_last = False)

    embedding_batches = []
    encoder.eval()

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            embeddings = encoder(batch)
            embedding_batches.append(embeddings.cpu().numpy())

    if not embedding_batches:
        raise RuntimeError("Embedding DataLoader produced no batches.")

    result = np.concatenate(embedding_batches, axis=0)

    if result.shape[0] != len(events):
        raise RuntimeError(f"Expected {len(events)} embeddings, "
                           f"but extracted {result.shape[0]}.")

    if not np.isfinite(result).all():
        raise RuntimeError("Extracted embeddings contain NaN or Inf.")

    return result

def parse_args():
    parser = argparse.ArgumentParser(description=("Extract frozen GravNet embeddings and run Aneek's downstream classifiers."))

    parser.add_argument(
        "--pretraining-dir",
        required=True,
        type=Path,
        help=(
            "Extracted pretraining experiment directory "
            "containing checkpoints/best.pt and stats.npy."
        ),
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--encoder-batch-size",
        default=64,
        type=int,
    )

    parser.add_argument(
        "--train-fraction",
        default=0.8,
        type=float,
    )

    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=VALID_CLASSIFIERS,
        default=[
            "logistic_regression",
            "knn",
            "gradient_boosting",
        ],
    )

    parser.add_argument(
        "--classifier-random-state",
        default=42,
        type=int,
    )

    return parser.parse_args()

def main():
    args = parse_args()

    pretraining_dir = (
        args.pretraining_dir
        .expanduser()
        .resolve()
    )

    data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    checkpoint_path = (
        pretraining_dir
        / "checkpoints"
        / "best.pt"
    )

    stats_path = pretraining_dir / "stats.npy"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Missing statistics file: {stats_path}"
        )

    if output_dir.exists() and any(
        output_dir.iterdir()
    ):
        raise FileExistsError(
            f"Output directory is not empty: "
            f"{output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    encoder, saved_args = build_frozen_encoder(
        checkpoint=checkpoint,
        device=device,
    )

    channel_a = str(saved_args["channel_a"])
    channel_b = str(saved_args["channel_b"])
    events_per_channel = int(
        saved_args["events_per_channel"]
    )
    pretraining_seed = int(saved_args["seed"])

    random.seed(pretraining_seed)
    np.random.seed(pretraining_seed)
    torch.manual_seed(pretraining_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            pretraining_seed
        )

    feature_mean, feature_std = (
        load_projection_stats(stats_path)
    )

    print(
        f"Pretraining experiment: "
        f"{pretraining_dir.name}"
    )
    print(
        f"Classification task: "
        f"{channel_a} vs {channel_b}"
    )
    print(
        f"Events per class: "
        f"{events_per_channel}"
    )
    print(f"Device: {device}")
    print(f"Feature mean: {feature_mean}")
    print(f"Feature std:  {feature_std}")

    events_class0 = load_channel(
        channel=channel_a,
        events_per_channel=events_per_channel,
        data_dir=data_dir,
    )

    events_class1 = load_channel(
        channel=channel_b,
        events_per_channel=events_per_channel,
        data_dir=data_dir,
    )

    embeddings_class0 = extract_embeddings(
        encoder=encoder,
        events=events_class0,
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=args.encoder_batch_size,
        seed=pretraining_seed,
        device=device,
    )

    embeddings_class1 = extract_embeddings(
        encoder=encoder,
        events=events_class1,
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=args.encoder_batch_size,
        seed=pretraining_seed,
        device=device,
    )

    print(
        f"{channel_a} embeddings: "
        f"{embeddings_class0.shape}"
    )
    print(
        f"{channel_b} embeddings: "
        f"{embeddings_class1.shape}"
    )

    np.savez_compressed(
        output_dir / "embeddings.npz",
        embeddings_class0=embeddings_class0,
        embeddings_class1=embeddings_class1,
        class0_name=np.asarray(channel_a),
        class1_name=np.asarray(channel_b),
    )

    summary = {
        "experiment": pretraining_dir.name,
        "channel_a": channel_a,
        "channel_b": channel_b,
        "events_per_channel": events_per_channel,
        "embedding_dimension": int(
            embeddings_class0.shape[1]
        ),
        "train_fraction": args.train_fraction,
        "classifier_random_state": (
            args.classifier_random_state
        ),
        "classifiers": {},
    }

    for classifier_name in args.classifiers:
        print()
        print("=" * 60)
        print(
            f"Running classifier: "
            f"{classifier_name}"
        )
        print("=" * 60)

        classifier_output = (
            output_dir / classifier_name
        )

        result = evaluate_two_embedding_sets(
            embeddings_class0=embeddings_class0,
            embeddings_class1=embeddings_class1,
            classifier=classifier_name,
            train_fraction=args.train_fraction,
            random_state=(
                args.classifier_random_state
            ),
            scale_embeddings=True,
            shuffle_within_class=True,
            shuffle_combined_sets=True,
            class_names=(
                channel_a,
                channel_b,
            ),
            output_dir=classifier_output,
            verbose=True,
        )

        summary["classifiers"][
            classifier_name
        ] = result.metrics

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("All classifiers completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
