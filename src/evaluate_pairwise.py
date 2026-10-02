"""Frozen clean evaluation using the same preparation and model as pretraining."""

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--prepared", type=Path, help="Relocated prepared directory; must match checkpoint metadata.")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to RUN_DIR/checkpoints/best.pt.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--encoder-mode", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--probe-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Downstream estimator seed; does not resplit events.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    if args.classifier_c <= 0 or args.probe_alpha < 0:
        parser.error("classifier-c must be positive and probe-alpha nonnegative")
    return args


def build_frozen_encoder(checkpoint, *, encoder_mode="pretrained", random_seed=42, device="cpu"):
    """One constructor for both sources; random initialization never loads weights."""
    import torch
    from src.models.multispace import MultiSpaceEncoder

    if encoder_mode not in {"pretrained", "random"}:
        raise ValueError(f"Unknown encoder mode: {encoder_mode}")
    config = checkpoint["config"]
    # Construction occurs on CPU and does not perturb the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(random_seed)
        model = MultiSpaceEncoder(mode=config["mode"], **config["model"])
    if encoder_mode == "pretrained":
        model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def default_output_dir(run_dir, suffix):
    run_dir = Path(run_dir)
    if run_dir.parent.name == "pretraining":
        return run_dir.parent.parent / "classifier" / run_dir.name / suffix
    return run_dir / "evaluation" / suffix


def main(argv=None):
    args = parse_args(argv)
    import torch
    from torch.utils.data import DataLoader
    from src.data.events import load_prepared
    from src.data.views import CleanDataset, collate_clean
    from src.evaluation.representations import evaluate_representations, export_representations
    from src.training.checkpoint import assert_prepared_matches, load_checkpoint

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint or run_dir / "checkpoints" / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    configured_prepared = checkpoint["config"].get("prepared_dir")
    if args.prepared is None and configured_prepared is None:
        raise ValueError("Supply --prepared: checkpoint does not locate a prepared directory.")
    prepared_path = (args.prepared or Path(configured_prepared)).expanduser().resolve()
    prepared = load_prepared(prepared_path)
    assert_prepared_matches(checkpoint, prepared.metadata)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    suffix = "pretrained" if args.encoder_mode == "pretrained" else f"random_{args.random_seed}"
    output_dir = args.output_dir or default_output_dir(run_dir, suffix)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {output_dir}")
    model = build_frozen_encoder(checkpoint, encoder_mode=args.encoder_mode,
                                 random_seed=args.random_seed, device=device)
    dataset = CleanDataset(prepared, split=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False, collate_fn=collate_clean)
    metadata = {
        "schema_version": 1, "encoder_mode": args.encoder_mode,
        "random_encoder_seed": args.random_seed if args.encoder_mode == "random" else None,
        "model_mode": checkpoint["config"]["mode"],
        "network_readouts_trained": (args.encoder_mode == "pretrained"
                                     and checkpoint["config"]["mode"] == "five_anisotropic_physics"
                                     and checkpoint["config"]["objective"]["gamma"] > 0),
        "model_config": checkpoint["config"]["model"],
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
        "prepared_fingerprint": checkpoint["prepared_fingerprint"],
        "preprocessing": prepared.metadata,
        "clean_inputs": True,
        "representation_source": "h_before_contrastive_projector",
    }
    arrays = export_representations(model, loader, output_dir / "representations.npz",
                                    device=device, metadata=metadata)
    result = evaluate_representations(arrays, output_dir, classifier_c=args.classifier_c,
                                      probe_alpha=args.probe_alpha, seed=args.seed,
                                      target_stats=prepared.metadata["target_stats"])
    (output_dir / "evaluation_config.json").write_text(json.dumps({
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_device": str(device), "prepared_fingerprint": checkpoint["prepared_fingerprint"],
    }, indent=2), encoding="utf-8")
    print(f"Exported {len(arrays['event_key'])} clean events to {output_dir}")
    for name, roles in result["metrics"]["classification"].items():
        print(f"{name}: test accuracy={roles['test']['accuracy']:.4f}, ROC-AUC={roles['test']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
