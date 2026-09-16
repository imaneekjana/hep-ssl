"""
Imports

"""

import random
import uproot
import awkward as ak
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import os
import sys
import json
import h5py


import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from livelossplot import PlotLosses
from torch_geometric.loader import DataLoader
import torch_cluster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import *
from src.data.dataset import ColliderMLHits, ContrastiveLearningDatasetPlanar, ContrastiveLearningGraphDatasetPlanar, EventGraphBuilder
from src.models.gnn import *
from src.models.contrastive_learning import *


from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.backends.cudnn as cudnn
from torchvision import models



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='ColliderML Training')


#parser.add_argument('-data', metavar='DIR', default='./datasets',help='path to dataset')

parser.add_argument('-dataset-name', default='colliderml',
                    help='dataset name')

parser.add_argument('-a', '--arch', metavar='ARCH', default='gnn',
                    help='model architecture: ')




parser.add_argument('--trainevents', default=1500, type=int,
                    help='number of events used for training ')

parser.add_argument('--split', default='ttbar_ggf_total3000', help='Name for the training output directory')

parser.add_argument('--rotation', default=np.pi/8, type=float, help='scale for random global rotation')

parser.add_argument('--energy-noise', default=0.05, type=float, help='scale for (log)-energy noise')

parser.add_argument('--resume', default=None, type=str, help='Path to a checkpoint file used to resume training')





parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument("--data-dir", default=None, type=str, help="Directory used by ColliderML to story or read dataset files")

parser.add_argument("--output-dir", default=None, type=str, help="Parent directory for checkpoints and training outputs")

parser.add_argument('--epochs', default=18, type=int, metavar='N',
                    help='number of total epochs to run')


parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=32, type=int,
                    help='latent space dimension where constrastive loss is applied')

parser.add_argument('--feat_dim', default=4, type=int,
                    help='feature dimension')

parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')

parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

parser.add_argument('--experiment-name', default='baseline', type=str, help="Unique name for this experiment")

parser.add_argument('--xyz-noise', default=5.0, type=float, help="Std for Gaussian noise added to xyz")

parser.add_argument('--shift-std', default=0.0, type=float, help="XY translation standard deviation; 0 disables shifting")

parser.add_argument("--crop-fraction", default=0.0, type=float, help="Spatial crop fraction; 0 disables cropping")

parser.add_argument("--rotation-mode", choices=["uniform", "gaussian"], default="uniform", help="Distribution used to sample the rotation angle")

parser.add_argument("--hidden-dim", default=16, type=int)

parser.add_argument("--latent-dim", default=64, type=int)

parser.add_argument("--proj-dim", default=32, type=int)

parser.add_argument("--gravnet-k", default=8, type=int)

parser.add_argument("--space-dim", default=4, type=int)

parser.add_argument("--propagate-dim", default=16, type=int)

parser.add_argument("--train-augmentation-order", default="rotate+energy_noise+xyz_noise", type=str, help=(
    "Ordered augmentation pipeline. "
    "Join transform names with '+' "
    "Available transforms: rotate, energy_noise, xyz_noise, shift, crop. "
    "Use 'none' to disable all augmentations."
))

args = parser.parse_args()
print("Experiment configuration:")
for key, value in sorted(vars(args).items()):
    print(f"  {key}: {value}")

## save the args---done Later

assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True    
else:
    args.device = torch.device('cpu')


    
# Fixing seed after parsing args
if args.seed is not None:
    SEED = args.seed

else:

    SEED = 42  # or any fixed integer


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)




"""
Loading ColliderML data

"""

from colliderml.core import load_tables, collect_tables
import polars as pl

if args.data_dir is not None:
    data_dir = Path(args.data_dir).expanduser().resolve()
elif os.environ.get("COLLIDERML_DATA_DIR"):
    data_dir = Path(os.environ["COLLIDERML_DATA_DIR"]).expanduser().resolve()
else:
    data_dir = PROJECT_ROOT / "colliderml-data"

data_dir.mkdir(parents=True, exist_ok=True)
print(f"ColliderML data directory: {data_dir}")



## ttbar data
cfg1 = {
    "dataset_id": "CERN/ColliderML-Release-1",
    "channels": "ttbar",
    "pileup": "pu0",
    "objects": ["calo_hits"],
    "split": "train",
    "lazy": False,
    "max_events": args.trainevents,
    "data_dir": str(data_dir)
}
tables1 = load_tables(cfg1)
frames1 = collect_tables(tables1)

calo_hits1 = frames1["calo_hits"]


## dihiggs data
cfg2 = {
    "dataset_id": "CERN/ColliderML-Release-1",
    "channels": "dihiggs",
    "pileup": "pu0",
    "objects": ["calo_hits"],
    "split": "train",
    "lazy": False,
    "max_events": args.trainevents,
    "data_dir": str(data_dir)
}
tables2 = load_tables(cfg2)
frames2 = collect_tables(tables2)

calo_hits2 = frames2["calo_hits"]

## ggf data
cfg3 = {
    "dataset_id": "CERN/ColliderML-Release-1",
    "channels": "ggf",
    "pileup": "pu0",
    "objects": ["calo_hits"],
    "split": "train",
    "lazy": False,
    "max_events": args.trainevents,
    "data_dir": str(data_dir)
}
tables3 = load_tables(cfg3)
frames3 = collect_tables(tables3)

calo_hits3 = frames3["calo_hits"]

# Concatenate
#combined = pl.concat([calo_hits1, calo_hits2, calo_hits3])
combined = pl.concat([calo_hits1, calo_hits3])
#combined = pl.concat([calo_hits2, calo_hits3])

#combined = calo_hits2 #Only dihiggs

# Shuffle rows
combined = combined.sample(fraction=1.0, with_replacement=False, shuffle=True, seed=SEED)

#print(combined.shape)
#print(combined.head())

calo_hits = combined



"""

Preparing dataset

"""


def build_train_augmentation(args):
    transform_factories = {
        "rotate": lambda: RandomRotateXY(
            angle_range = (-args.rotation, args.rotation), gaussian=(args.rotation_mode == "gaussian")
        ),
        "energy_noise": lambda: EnergyWhiteNoise(sigma=args.energy_noise),
        "xyz_noise": lambda: NoiseXYZ(sigma=args.xyz_noise),
        "shift": lambda: RandomShift(shift_std=(args.shift_std, args.shift_std, 0.0)),
        "crop": lambda: RandomSpatialCrop(crop_fraction=args.crop_fraction)
    }

    order_spec = args.train_augmentation_order.strip().lower()

    if order_spec in {"", "none"}:
        transform_names = []
    else:
        transform_names = [name.strip() for name in order_spec.split("+") if name.strip()]

    unknown_names = [name for name in transform_names if name not in transform_factories]

    if unknown_names:
        valid_names = ", ".join(transform_factories.keys())
        raise ValueError(
            f"Unknown augmentation names: {unknown_names}. "
            f"Valid names are: {valid_names}"
        )

    if len(transform_names) != len(set(transform_names)):
        raise ValueError(
            "Duplicate transforms are not allowed in "
            f"--train-augmentation-order: {transform_names}"
        )

    transforms = [transform_factories[name]() for name in transform_names]

    print("Training augmentation pipeline: " + (" -> ".join(transform_names) if transform_names else "none"))

    return Compose(transforms)

train_augment = build_train_augmentation(args)

eval_augment = Compose([
    RandomRotateXY(angle_range=(-np.pi / 8, np.pi / 8), gaussian = False),
    EnergyWhiteNoise(sigma=0.05),
    NoiseXYZ(sigma=5.0)
])

print(
    "Evaluation augmentation pipeline: "
    "rotate -> energy_noise -> xyz_noise"
)

n_total = len(calo_hits)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

train_hits = calo_hits[:n_train]
val_hits = calo_hits[n_train:n_train + n_val]
test_hits = calo_hits[n_train + n_val:]
dataset_train_base = ColliderMLHits(train_hits, split=None, shuffle_files=True, log=False, seed=SEED)
dataset_val_base = ColliderMLHits(val_hits, split=None, shuffle_files=False, log=False, seed=SEED + 100)
dataset_test_base = ColliderMLHits(test_hits, split=None, shuffle_files=False, log=False, seed=SEED + 200)

print(
    f"Dataset split: "
    f"train={len(dataset_train_base)}, "
    f"val={len(dataset_val_base)}, "
    f"test={len(dataset_test_base)}"
)


r = 0 # r does not matter for GravNetEncoder, but for GraphConvEncoder
#r = 1
#r = 3



MEANS = None
STDS = None



"""
projs: subset of ['eta-phi', 'x-y', 'z-rho', 'z-phi']

typ: can either be 'image' or 'hits'
"""



dataset_train = ContrastiveLearningGraphDatasetPlanar(ContrastiveLearningDatasetPlanar(dataset_train_base, MEANS, STDS,  train_augment, projs=['eta-phi'], grid_size=32, typ='hits'), builder=EventGraphBuilder(radius=r))

dataset_val = ContrastiveLearningGraphDatasetPlanar(ContrastiveLearningDatasetPlanar(dataset_val_base, MEANS, STDS, eval_augment, projs=['eta-phi'], grid_size=32, typ='hits'), builder=EventGraphBuilder(radius=r))

dataset_test = ContrastiveLearningGraphDatasetPlanar(ContrastiveLearningDatasetPlanar(dataset_test_base, MEANS, STDS, eval_augment, projs=["eta-phi"], grid_size=32, typ="hits"), builder=EventGraphBuilder(radius=r))



train_loader = DataLoader(dataset_train, batch_size=args.batch_size, drop_last=True, num_workers=0)
val_loader   = DataLoader(dataset_val,   batch_size=args.batch_size, drop_last=False, num_workers=0)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size, drop_last=False, num_workers=0)



print("Data preprocessing working fine.")










"""
TRAIN THE MODEL

"""




"""
Specify the model
"""

#model = PointNetEncoder()

model = GravNetEncoder(in_features=3, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, proj_dim=args.proj_dim, k=args.gravnet_k, space_dim=args.space_dim, propagate_dim=args.propagate_dim) # adjust the in_features properly, 2+1=3 for planar projections of hits




def count_trainable_parameters(mymodel):
    return sum(p.numel() for p in mymodel.parameters() if p.requires_grad)

print("Trainable parameters:", count_trainable_parameters(model))




"""
Specify the folder
"""


if args.output_dir is not None:
    output_root = Path(args.output_dir).expanduser().resolve()

elif os.environ.get("COLLIDERML_OUT_DIR"):
    output_root = Path(os.environ["COLLIDERML_OUT_DIR"]).expanduser().resolve()

else:
    output_root = PROJECT_ROOT / "outputs"

run_name = args.experiment_name

folder = output_root / run_name
checkpoint_dir = folder /"checkpoints"

if folder.exists() and args.resume is None:
    raise FileExistsError(
        f"Experiment output directory already exists: {folder}. "
        "Use a new --experiment-name, remove the old directory, "
        "or provide --resume to continue an existing run."
    )

folder.mkdir(parents = True, exist_ok = True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"Training output directory: {folder}")

np.save(folder / "stats.npy",
        {"means": MEANS,
         "stds": STDS,
         "events": len(calo_hits),
         "train_events": len(dataset_train_base),
         "val_events": len(dataset_val_base),
         "test_events": len(dataset_test_base)})

def serializable_args(args):
    basic_types = (int, float, str, bool, type(None))
    return {
        key: value if isinstance(value, basic_types) else str(value)
        for key, value in vars(args).items()
    }

with open(folder / "args.json", "w", encoding="utf-8") as args_file:
    json.dump(serializable_args(args), args_file, indent=4)

"""
Prepare the model and optimizer

"""

model = model.to(args.device)

print(f"Training device: {args.device}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

"""
Train the model

"""
simclr = Contrastive_Learning(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
start_epoch = 0
best_val_loss = float("inf")

if args.resume is not None:
    resume_path = Path(args.resume).expanduser().resolve()

    if not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resume_path}")

    start_epoch, best_val_loss = (simclr.load_checkpoint(path=resume_path, load_optimizer=True))

best_val_loss = simclr.train(train_loader=train_loader, val_loader=val_loader, checkpoint_dir=str(checkpoint_dir), output_dir=str(folder), start_epoch=start_epoch, best_val_loss=best_val_loss)



best_checkpoint_path = (checkpoint_dir / "best.pt")

if not best_checkpoint_path.is_file():
    raise FileNotFoundError(
        "Training finished without producing best.pt: "
        f"{best_checkpoint_path}"
    )

simclr.load_checkpoint(path=best_checkpoint_path, load_optimizer=False)

test_loss = simclr.test(loader=test_loader, desc="test", seed=args.seed + 671)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Final test loss: "
    f"{test_loss:.6f}"
)

with open(folder / "test_metrics.json", "w", encoding="utf-8") as metrics_file:
    json.dump({
        "best_validation_loss":best_val_loss,
        "test_loss": test_loss
    }, metrics_file, indent=4)