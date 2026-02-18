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

from .augmentation import *




'''
Creating graph for each event out of hits

'''

class EventGraphBuilder:
    def __init__(self, radius=3, max_neighbors=32):
        self.r = radius
        self.neighbors = max_neighbors

    def __call__(self, hits):
        """
        hits: numpy array or tensor [N, 4]
        """

        if not torch.is_tensor(hits):
            hits = torch.tensor(hits, dtype=torch.float)

        pos = hits[:, :3]      # spatial coordinates
        features = hits        # (x, y, z, E)

        #edge_index = knn_graph(pos, k=self.k, loop=False)

        pos = hits[:, :3]
        N = pos.shape[0]

        # Compute pairwise squared distances
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 3]
        dist2 = (diff ** 2).sum(-1)

        # Select edges within radius (exclude self)
        row, col = torch.where((dist2 <= self.r ** 2) & (dist2 > 0))
        edge_index = torch.stack([row, col], dim=0)

        return Data(x=features, edge_index=edge_index)





###-------------------------------xxxxxxxxx-------------------------------######




class ContrastiveLearningDataset(IterableDataset):
    """
    Wraps an iterable dataset of calorimeter events contained in the output of CLDHits.

    Yields:
        original, view1, view2
    """

    def __init__(self, base_dataset, transform=None):

        """
        base_dataset is an iterable dataset with each item being a dictionary with the key "calo_hit_features".

        transform must be a Transform object, see augmentation.py
        
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):

        return len(self.base_dataset)

    def __iter__(self):
        for event_dict in self.base_dataset:

            event = event_dict["calo_hit_features"]

            if not isinstance(event, CaloEvent):
                event = CaloEvent(event)

            if self.transform is None:
                view1 = event.copy()
                view2 = event.copy()
            else:
                view1 = self.transform(event)
                view2 = self.transform(event)

            
                
            event_dict["calo_hit_features_1"] = view1.hits
            event_dict["calo_hit_features_2"] = view2.hits
            event_dict["calo_hit_features"] = event.hits

            yield event_dict
            


class ContrastiveLearningGraphDataset(IterableDataset):
    """
    Wraps an iterable dataset of calorimeter events.

    Yields:
         view1_graph, view2_graph
    """

    def __init__(self, base_dataset, builder=EventGraphBuilder):

        """
        base_dataset must be an iterable with dictionaries having the keys, "calo_hit_features_1" and "calo_hit_features_2".

        base_dataset can be an output of ContrastiveLearningDataset

        """
        super().__init__()
        self.base_dataset = base_dataset
        self.builder = builder

    def __len__(self):

        return len(self.base_dataset)

    def __iter__(self):
        for event_dict in self.base_dataset:

            view1 = event_dict["calo_hit_features_1"]
            view2 = event_dict["calo_hit_features_2"]

            view1_graph = self.builder()(view1)
            view2_graph = self.builder()(view2)

            yield view1_graph, view2_graph
            

