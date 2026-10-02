"""Two-view trainer shared by all four trainable experiment modes."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from src.config import load_config, save_config, validate_config
from src.data.events import load_prepared
from src.data.views import PairDataset, collate_pairs
from src.models.multispace import MultiSpaceEncoder
from src.losses.multitask import MultiTaskObjective
from src.training.checkpoint import (
    SCHEMA_VERSION, assert_prepared_matches, capture_rng_state, load_checkpoint,
    metadata_fingerprint, restore_rng_state, save_checkpoint, source_state,
)


class EventBatchSampler(Sampler):
    """Training drops tails; evaluation merges a final singleton into its predecessor."""
    def __init__(self, size, batch_size, *, shuffle=False, drop_last=False, seed=42):
        if batch_size < 2 or size < 2 or (drop_last and size < batch_size):
            raise ValueError("Need at least two events and a full training batch")
        self.size, self.batch_size = size, batch_size
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = np.arange(self.size)
        if self.shuffle:
            indices = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(indices)
        batches = [indices[i:i + self.batch_size].tolist() for i in range(0, self.size, self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches.pop()
        elif len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        if self.drop_last:
            return self.size // self.batch_size
        count = (self.size + self.batch_size - 1) // self.batch_size
        return count - int(self.size % self.batch_size == 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda; GravNet CPU/CUDA are supported")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(value)


def build_optimizer(model, objective, config):
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    metric_parameters = [p for p in objective.parameters() if p.requires_grad]
    all_parameters = model_parameters + metric_parameters
    if len(all_parameters) != len({id(p) for p in all_parameters}):
        raise ValueError("Optimizer parameter groups contain duplicates")
    groups = [{"params": model_parameters, "weight_decay": config["weight_decay"]}]
    if metric_parameters:
        groups.append({"params": metric_parameters, "weight_decay": 0.0})
    return torch.optim.Adam(groups, lr=config["lr"])


def _loader(dataset, sampler, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      collate_fn=collate_pairs, generator=generator, persistent_workers=False)


def run_epoch(model, objective, loader, device, *, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    old_model, old_objective = model.training, objective.training
    model.train(training)
    objective.train(training)
    totals, count, steps = {}, 0, 0
    try:
        with torch.set_grad_enabled(training):
            for batch in loader:
                keys = batch["event_key"]
                size = len(keys)
                if size < 2 or len(set(keys)) != size:
                    raise ValueError("Contrastive batches require distinct source events")
                graphs = [batch[name].to(device) for name in ("view1", "view2")]
                targets = [{k: v.to(device) for k, v in batch[name].items()}
                           for name in ("targets1", "targets2")]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    outputs = [model(graph) for graph in graphs]
                    values = objective(*outputs, *targets)
                loss = values["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite objective")
                if training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    for group in optimizer.param_groups:
                        for parameter in group["params"]:
                            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                                raise FloatingPointError("Nonfinite gradient")
                    scaler.step(optimizer)
                    scaler.update()
                for name, value in values.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
                count += size
                steps += 1
    finally:
        model.train(old_model)
        objective.train(old_objective)
    if count == 0:
        raise ValueError("Empty epoch")
    return {key: value / count for key, value in totals.items()}, steps


def train(config, prepared_dir, run_dir, *, resume=None, stop_after_epoch=None):
    """Train through an epoch boundary; optional stop leaves scheduler horizon unchanged."""
    checkpoint = load_checkpoint(resume) if resume is not None else None
    if checkpoint is not None:
        saved = copy.deepcopy(checkpoint["config"])
        if config is not None:
            supplied = copy.deepcopy(config)
            supplied["prepared_dir"] = saved["prepared_dir"]
            if supplied != saved:
                raise ValueError("Resume config differs; optimizer/schedule/method overrides are disabled")
        config = saved
    supplied_config = config is not None
    config = validate_config(copy.deepcopy(config) if supplied_config else load_config())
    if prepared_dir is None and config["prepared_dir"] is None:
        raise ValueError("A prepared directory is required")
    prepared_dir = Path(prepared_dir or config["prepared_dir"]).expanduser().resolve()
    prepared = load_prepared(prepared_dir)
    if checkpoint is not None:
        assert_prepared_matches(checkpoint, prepared.metadata)
    else:
        preparation = prepared.metadata["preparation_config"]
        for section in ("data", "grid", "targets"):
            if supplied_config:
                for key, value in config[section].items():
                    # None explicitly means resolve from the frozen preparation.
                    if value is not None and value != preparation[section][key]:
                        raise ValueError(f"{section}.{key} disagrees with prepared data; reuse its configuration")
            config[section] = copy.deepcopy(preparation[section])
        config["prepared_dir"] = str(prepared_dir)
        print("Using frozen data, grid and target definitions from prepared artifacts", flush=True)
    run_dir = Path(run_dir)
    if checkpoint is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
        save_config(config, run_dir / "config.json")
        (run_dir / "preprocessing.json").write_text(json.dumps(prepared.metadata, indent=2, allow_nan=False) + "\n")
    else:
        if not (run_dir / "config.json").is_file():
            raise ValueError("Resume requires the original run directory (it can be relocated)")
        disk_config = json.loads((run_dir / "config.json").read_text())
        if disk_config != checkpoint["config"]:
            raise ValueError("Run configuration differs from checkpoint")
    tc = config["training"]
    device = resolve_device(tc["device"])
    if tc["amp"] and device.type != "cuda":
        raise ValueError("AMP training currently requires CUDA")
    seed_everything(tc["seed"])
    model = MultiSpaceEncoder(config["mode"], **config["model"]).to(device)
    objective = MultiTaskObjective(config["mode"], proj_dim=config["model"]["proj_dim"], **config["objective"]).to(device)
    optimizer = build_optimizer(model, objective, tc)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"], eta_min=0)
    scaler = torch.amp.GradScaler("cuda", enabled=tc["amp"])
    training = PairDataset(prepared, "train", config["augmentation"], seed=tc["augmentation_seed"])
    validation = PairDataset(prepared, "val", config["augmentation"], seed=tc["validation_seed"])
    train_sampler = EventBatchSampler(len(training), tc["batch_size"], shuffle=True, drop_last=True, seed=tc["seed"])
    val_sampler = EventBatchSampler(len(validation), tc["batch_size"])
    train_loader = _loader(training, train_sampler, tc["num_workers"], tc["seed"] + 1)
    val_loader = _loader(validation, val_sampler, tc["num_workers"], tc["seed"] + 2)
    start, global_step, best, history = 0, 0, float("inf"), []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        objective.load_state_dict(checkpoint["objective_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best, history = checkpoint["best_validation"], checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"])
    end = tc["epochs"] if stop_after_epoch is None else min(stop_after_epoch, tc["epochs"])
    if end <= start:
        raise ValueError("No epochs remain within the requested stopping boundary")
    provenance = source_state()
    for epoch in range(start, end):
        training.set_epoch(epoch)
        validation.set_epoch(0)
        train_sampler.set_epoch(epoch)
        train_metrics, steps = run_epoch(model, objective, train_loader, device,
                                        optimizer=optimizer, scaler=scaler, amp=tc["amp"])
        val_metrics, _ = run_epoch(model, objective, val_loader, device, amp=tc["amp"])
        global_step += steps
        improved = val_metrics["loss"] < best
        best = min(best, val_metrics["loss"])
        metric_diagonals = {name: values.detach().cpu().tolist()
                            for name, values in objective.metric_diagonals().items()}
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        "train": train_metrics, "val": val_metrics, "lambda": metric_diagonals})
        scheduler.step()
        payload = {
            "schema_version": SCHEMA_VERSION, **provenance, "config": config,
            "model_state": model.state_dict(), "objective_state": objective.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
            "best_validation": best, "best_criterion": "validation_total", "history": history,
            "rng_state": capture_rng_state(), "preprocessing": prepared.metadata,
            "prepared_fingerprint": metadata_fingerprint(prepared.metadata),
        }
        save_checkpoint(payload, run_dir / "checkpoints" / "last.pt")
        if improved:
            save_checkpoint(payload, run_dir / "checkpoints" / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(f"epoch={epoch + 1}/{tc['epochs']} train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f}", flush=True)
    return run_dir / "checkpoints" / "last.pt"
