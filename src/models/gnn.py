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
import torch_cluster


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import EdgeConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax


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




"""
POINT-NET ENCODER
"""


class PointNetEncoder(nn.Module):
    def __init__(self, in_features = 4, hidden_dim = 16, latent_dim=64, proj_dim=32):
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim

        self.mlp1 = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.hidden_dim, 2*self.hidden_dim),
            nn.LayerNorm(2*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2*self.hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.ReLU()
        )

        self.project = nn.Sequential(
            nn.Linear(
            3*self.latent_dim, 
            self.latent_dim,
            ),
            nn.LayerNorm(self.latent_dim)
        )
        
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, 8*self.hidden_dim),
            nn.LayerNorm(8*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(8*self.hidden_dim, self.proj_dim)
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        h = self.mlp1(x)

        h = self.mlp2(h)

        log_e = data.x[:, 3]                          # raw energy per hit [N]
        w = softmax(log_e, batch)                     # normalize within each graph [N]
        
        h_max = global_max_pool(h, batch)
        h_mean = global_mean_pool(h, batch)
        h_w = global_add_pool(h * w.unsqueeze(-1), batch)

        h = torch.cat([h_max,h_mean, h_w], dim=-1)

        z = self.project(h)


        zp = self.head(z)

        return zp




"""
GRAPHCONV BASED ENCODER
"""

class GraphConvEncoder(nn.Module):
    """
    GraphConv-based encoder for event-level data (e.g. calorimeter hits).

    Each event:
        nodes = hits [x, y, z, E]
        graph = kNN in spatial coordinates
        output = event embedding
    """

    def __init__(self, in_features=4, hidden_dim=16, latent_dim=64, proj_dim=32):
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim

        # input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # GraphConv layers
        self.conv1 = GraphConv(self.hidden_dim, 2*self.hidden_dim)
        self.conv2 = GraphConv(2*self.hidden_dim, 4*self.hidden_dim)
        self.conv3 = GraphConv(4*self.hidden_dim, 8*self.hidden_dim)

        # normalization (very important for stability)
        self.norm1 = nn.LayerNorm(2*self.hidden_dim)
        self.norm2 = nn.LayerNorm(4*self.hidden_dim)
        self.norm3 = nn.LayerNorm(8*self.hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(8*self.hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.ReLU()
        )

        # event-level projection
        self.event_mlp = nn.Sequential(
            nn.Linear(3 * self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        # projection head (contrastive / SSL ready)
        self.projection_head = nn.Sequential(
            nn.Linear(self.latent_dim, 8*self.hidden_dim),
            nn.LayerNorm(8*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(8*self.hidden_dim, self.proj_dim)
        )

    def forward(self, data):
        x, batch, edge_index = data.x, data.batch, data.edge_index

        log_e = data.x[:, -1]                          # raw energy per hit [N]
        w = softmax(log_e, batch)                     # normalize within each graph [N]
        

        # 1. embed hits
        x = self.input_proj(x)

        # 2. build graph from geometry (kNN in (x,y,z))
        #edge_index = knn_graph(x[:, :3], k=self.k, batch=batch)

        # 3. message passing stack (GraphConv)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)

        x = self.proj(x)

        # 4. multi-scale pooling (important for physics structure)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_w = global_add_pool(x * w.unsqueeze(-1), batch)

        x = torch.cat([x_mean, x_max, x_w], dim=-1)

        # 5. event embedding
        z = self.event_mlp(x)

        # 6. projection head
        zp = self.projection_head(z)

        return zp




"""
GRAVNET BASED ENCODER
"""

from torch_geometric.nn import GravNetConv

class GravNetEncoder(nn.Module):
    """
    GravNet-based encoder for event-level data (e.g. calorimeter hits).
    Each event:
        nodes = hits [x, y, z, E]
        graph = learned dynamically by GravNet (no pre-built kNN needed)
        output = event embedding

    GravNet learns its own latent space for graph construction,
    making it well-suited for irregular calorimeter geometry.
    """
    def __init__(self, in_features=4, hidden_dim=16, latent_dim=64, proj_dim=32,
                 k=8, space_dim=4, propagate_dim=16):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim
        self.k = k

        # input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # GravNet layers
        # GravNetConv(in, out, space_dimensions, propagate_dimensions, k)
        # space_dim     : dimensionality of learned graph-construction space
        # propagate_dim : dimensionality of messages passed along edges
        self.conv1 = GravNetConv(self.hidden_dim,   2*self.hidden_dim, space_dimensions=space_dim, propagate_dimensions=propagate_dim, k=k)
        self.conv2 = GravNetConv(2*self.hidden_dim, 4*self.hidden_dim, space_dimensions=space_dim, propagate_dimensions=propagate_dim, k=k)
        self.conv3 = GravNetConv(4*self.hidden_dim, 8*self.hidden_dim, space_dimensions=space_dim, propagate_dimensions=propagate_dim, k=k)

        # normalization
        self.norm1 = nn.LayerNorm(2*self.hidden_dim)
        self.norm2 = nn.LayerNorm(4*self.hidden_dim)
        self.norm3 = nn.LayerNorm(8*self.hidden_dim)

        # post-conv projection to fixed latent dim
        self.proj = nn.Sequential(
            nn.Linear(8*self.hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.ReLU(),
        )

        # event-level aggregation MLP
        self.event_mlp = nn.Sequential(
            nn.Linear(3 * self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # projection head (contrastive / SSL ready)
        self.projection_head = nn.Sequential(
            nn.Linear(self.latent_dim, 8*self.hidden_dim),
            nn.LayerNorm(8*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(8*self.hidden_dim, self.proj_dim),
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        # note: no edge_index needed — GravNet builds its own graph

        log_e = data.x[:, -1]                          # raw energy per hit [N]
        w = softmax(log_e, batch)  

        # 1. embed hits
        x = self.input_proj(x)

        # 2. message passing stack (GravNet builds graph dynamically per layer)
        x = self.conv1(x, batch)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x, batch)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x, batch)
        x = self.norm3(x)
        x = self.proj(x)

        # 3. multi-scale pooling (important for physics structure)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x_w = global_add_pool(x * w.unsqueeze(-1), batch)
        x = torch.cat([x_mean, x_max, x_w], dim=-1)  # [B, 2*latent_dim]

        # 4. event embedding
        z = self.event_mlp(x)   # [B, latent_dim]

        # 5. projection head
        zp = self.projection_head(z)  # [B, proj_dim]

        return zp
