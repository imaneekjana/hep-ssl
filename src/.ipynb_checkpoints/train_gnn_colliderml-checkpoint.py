import uproot
import awkward as ak
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import os
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
from torch.cuda.amp import GradScaler, autocast
from livelossplot import PlotLosses

from torch.utils.data import Dataset, DataLoader, random_split

from torch_geometric.loader import DataLoader

from src.data.augmentation import *
from src.data.dataset import *
from src.models.gnn import *
from src.models.contrastive_learning import *


import sys

#sys.path.append('/global/cfs/cdirs/m4474/aneek/particlemind/src/datasets')


#os.chdir('/global/cfs/cdirs/m4474/aneek/particlemind')

#from CLDHits import CLDHits

"""
Loading ColliderML data

"""

from colliderml.core import load_tables, collect_tables


## ttbar data
cfg1 = {
    "dataset_id": "CERN/ColliderML-Release-1",
    "channels": "ttbar",
    "pileup": "pu0",
    "objects": ["calo_hits"],
    "split": "train",
    "lazy": False,
    "max_events": 3200,
    "data_dir":"/pscratch/sd/a/aneekj02/colliderml-data"
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
    "max_events": 3200,
    "data_dir":"/pscratch/sd/a/aneekj02/colliderml-data"
}
tables2 = load_tables(cfg1)
frames2 = collect_tables(tables1)

calo_hits2 = frames2["calo_hits"]

import polars as pl

# Concatenate
combined = pl.concat([calo_hits1, calo_hits2])

# Shuffle rows
combined = combined.sample(fraction=1.0, with_replacement=False, seed=42)

#print(combined.shape)
#print(combined.head())

calo_hits = combined






"""
Data Preprocessor

"""

class ColliderMLHits(IterableDataset):
    def __init__(
        self, calo_hits, split, shuffle_files=False, train_fraction=0.8):
        """
        Initialize the dataset.

        Args:
            calo_hits : calo_hit data for events.
            shuffle_files (bool): Whether to shuffle the order of parquet files.
        """
        
        self.calo_hits = calo_hits
        self.shuffle_files = shuffle_files
        

        self.split = split
        if self.split is not None:
            split_index = int(len(self.calo_hits) * train_fraction)
            if self.split == "train":
                self.calo_hits = self.calo_hits[:split_index]
            elif self.split == "val":
                self.calo_hits = self.calo_hits[split_index:]

        if self.shuffle_files:
            self.shuffle_shards()

    def __len__(self):
        """
        Return the number of events in the dataset.
        """
        return len(self.calo_hits) 

    def shuffle_shards(self):
        """
        Shuffle the events in calo_hits
        """
        random.shuffle(self.calo_hits)

    def __iter__(self):
        logger = logging.getLogger(__name__)
        self.sample_counter = 0  # Reset sample counter for each iteration or each epoch
        #worker_info = torch.utils.data.get_worker_info()
        

        
        data = self.calo_hits
        
        for event_i in range(len(self.calo_hits)):

            data_i = data[event_i]

            calo_hit_features = np.column_stack(
                (
                    data_i['x'].to_numpy()[0],
                    data_i['y'].to_numpy()[0],
                    data_i['z'].to_numpy()[0],
                    data_i['total_energy'].to_numpy()[0]

                )
            )

            f_i = len(data_i['x'].to_numpy()[0]) if len(data_i['x'].to_numpy()[0])<8000 else 8000

            yield {

                "calo_hit_features": calo_hit_features[0:f_i]

            }
         


"""

Preparing dataset

"""

augment = Compose([
    RandomRotateXY((0, 2*np.pi))
    ,
    RandomShift((20.0, 20.0, 0.0)),
    RandomSpatialCrop(0.7),
    EnergyWhiteNoise(0.2)
])



dataset_train = ColliderMLHits(calo_hits, "train")
dataset_val = ColliderMLHits(calo_hits, "val")

dataset_train = ContrastiveLearningGraphDataset(ContrastiveLearningDataset(dataset_train, augment))
dataset_val = ContrastiveLearningGraphDataset(ContrastiveLearningDataset(dataset_val, augment))



train_loader = DataLoader(dataset_train, batch_size=32, drop_last=True)
val_loader   = DataLoader(dataset_val,   batch_size=32, drop_last=True)




model = GNNEncoder()

def count_trainable_parameters(mymodel):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Trainable parameters:", count_trainable_parameters(model))

'''
for x1, x2 in train_loader:

    f1 = model(x1)
    f2 = model(x2)

    print(f1.shape)

    print('Its working. Vamos!!')

    #print(x1)

    
    break
'''
print("Vamos!!")



"""
TRAIN THE MODEL

"""


from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

#parser.add_argument('-data', metavar='DIR', default='./datasets',help='path to dataset')
'''
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')'''

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--epochs', default=5, type=int, metavar='N',
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

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=16, type=int,
                    help='latent space dimension where constrastive loss is applied')

parser.add_argument('--feat_dim', default=32, type=int,
                    help='feature dimension')

parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')

parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


args, unknown = parser.parse_known_args()

assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True    
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1


model = GNNEncoder()


# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

#  It’s a no-op if the 'gpu_index' argument is a negative integer or None.



with torch.cuda.device(args.gpu_index):
    
    simclr = Contrastive_Learning(model=model, optimizer=optimizer, scheduler=scheduler, args=args)

    print("Lesss gooo")
    
    simclr.train(train_loader, val_loader, save_model=True, folder = '/global/cfs/cdirs/m4474/aneek/particlemind_aneek/saved_models_colliderml/', wandb_=True, key='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', name='Test-run-CaloHitsGNN-ColliderML-mixed-ttbar-dihiggs')




