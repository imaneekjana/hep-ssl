import uproot
import awkward as ak
import random
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import os
import csv
import numpy as np

import torch
from torch.utils.data import IterableDataset

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool

import h5py
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from livelossplot import PlotLosses


"""

Contrastive Learning training protocol

"""


class Contrastive_Learning(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        self.scaler = GradScaler("cuda", enabled=(self.args.fp16_precision and self.arge.device.type == "cuda"))
        self.history = {"epoch": [], "train_loss": [], "val_loss": []}
        

    def info_nce_loss(self, features):
        current_batch_size = (features.shape[0] // self.args.n_views)
        labels = torch.arange(current_batch_size, device=features.device).repeat(self.args.n_views)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def forward(self, view):

        latent_vec = self.model(view)

        return latent_vec
        
    def test(self, loader, desc="", seed=341):
        was_training = self.model.training

        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        amp_enabled = (self.args.fp16_precision and self.args.device.type =="cuda")

        try:
            with torch.no_grad():
                for view1, view2 in tqdm(loader, desc=desc):
                    view1 = view1.to(self.args.device)
                    view2 = view2.to(self.args.device)

                    with autocast(device_type=self.args.device.type, enabled=amp_enabled):
                        feat1 = self.forward(view1)
                        feat2 = self.forward(view2)

                        features = torch.cat((feat1, feat2), dim=0)

                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                    total_loss += loss.item()
                    num_batches += 1
            if num_batches == 0:
                raise RuntimeError("The evaluation DataLoader produced no batches.")

            average_loss = total_loss / num_batches

        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)

            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            self.model.train(was_training)

        return average_loss


    def save_history(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "loss_history.csv")

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerows(zip(self.history["epoch"], self.history["train_loss"], self.history["val_loss"]))

        if len(self.history["epoch"]) == 0:
            return

        figure_path = os.path.join(output_dir, "loss_curve.png")

        plt.figure(figsize=(8, 5))

        plt.plot(self.history["epoch"], self.history["train_loss"], label="Train Loss")

        plt.plot(self.history["epoch"], self.history["val_loss"], label="Validation Loss")

        plt.xalbel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path, dip=150)
        plt.close()

    def save_checkpoint(self, path, epoch, train_loss, val_loss, best_val_loss):
        checkpoint_directory=os.path.dirname(path)

        if checkpoint_directory:
            os.makedirs(checkpoint_directory, exist_ok=True)

        checkpoint = {"epoch": epoch,
                      "model_state_dict": self.model.state_dict(),
                      "optimizer_state_dict": self.optimizer.state_dict(),
                      "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler is not None else None),
                      "scaler_state_dict": self.scaler.state_dict(),
                      "train_loss": train_loss,
                      "val_loss": val_loss,
                      "best_val_loss": best_val_loss,
                      "history": self.history,
                      "args": vars(self.args)}
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.args.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer:
            optimizer_state = checkpoint.get("optimizer_state_dict")

            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

            scheduler_state = checkpoint.get("scheduler_state_dict")

            if (self.scheduler is not None and scheduler_state is not None):
                self.scheduler.load_state_dict(scheduler_state)

            scaler_state = checkpoint.get("scaler_state_dict")

            if scaler_state:
                self.scaler.load_state_dict(scaler_state)

        saved_history = checkpoint.get("history")

        if saved_history is not None:
            self.history = saved_history

        start_epoch = checkpoint["epoch"] + 1

        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint: {path}")
        print(f"Resume from epoch: {start_epoch}")

        return start_epoch, best_val_loss



    def train(self, train_loader, val_loader, checkpoint_dir, output_dir, start_epoch=0, best_val_loss=float("inf")):
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir,exist_ok=True)
        last_checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
        best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")

        amp_enabled = (self.args.fp16_precision and self.args.device.type =="cuda")

        for epoch_counter in tqdm(range(start_epoch, self.args.epochs), desc="epoch"):
            self.model.train()

            total_train_loss = 0.0
            num_train_batches = 0

            for view1, view2 in tqdm(train_loader, desc = f"train epoch {epoch_counter + 1}", leave = False):
                view1 = view1.to(self.args.device)
                view2 = view2.to(self.args.device)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=self.args.device.type, enabled=amp_enabled):
                    feat1 = self.forward(view1)
                    feat2 = self.forward(view2)

                    features = torch.cat((feat1, feat2), dim = 0)

                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_train_loss += loss.item()
                num_train_batches += 1

            if num_train_batches == 0:
                raise RuntimeError("The training DataLoader produced no batches")

            train_loss = (total_train_loss / num_train_batches)
            val_loss = self.test(val_loader, desc="validation", seed = self.args.seed + 200)

            epoch_number = epoch_counter + 1
            self.history["epoch"].append(epoch_number)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            learning_rate_used = (self.optimizer.param_groups[0]["lr"])

            if self.scheduler is not None:
                self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                self.save_checkpoint(path=best_checkpoint_path, epoch=epoch_counter, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val_loss)

                print(f"Saved new best checkpoint: "
                      f"{best_checkpoint_path}")

            self.save_checkpoint(path=last_checkpoint_path, epoch=epoch_counter, train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val_loss)
            self.save_history(output_dir)
            print(f"Epoch {epoch_number}/{self.args.epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | "
                  f"best_val_loss={best_val_loss:.6f} | "
                  f"lr={learning_rate_used:.8f}")

        return best_val_loss
            