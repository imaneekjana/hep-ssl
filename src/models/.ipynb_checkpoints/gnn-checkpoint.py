import uproot
import awkward as ak
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.nn import radius_graph

import torch
from torch.utils.data import IterableDataset


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool


'''
Designing a GNN type encoder

'''

class GNNEncoder(nn.Module):


    """

    A GNN type encoder with 3 convolutions and a projection

    feature dimension progression: 4 -> hidden_dim -> 2*hidden_dim -> 4*hidden_dim -> latent_dim

    """
    def __init__(self, hidden_dim=64, latent_dim=128, proj_dim=32):
        super().__init__()

        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        self.conv1 = EdgeConv(mlp(2*4, hidden_dim))
        self.conv2 = EdgeConv(mlp(2*hidden_dim, hidden_dim))
        self.conv3 = EdgeConv(mlp(2*hidden_dim, hidden_dim))

        self.project = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, latent_dim)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )

    def forward(self, data):

        """
        data is a graph object

        """

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        # Global pooling → event-level vector
        x = global_mean_pool(x, batch)

        z = self.project(x)

        zp = self.projection_head(z)

        return zp

