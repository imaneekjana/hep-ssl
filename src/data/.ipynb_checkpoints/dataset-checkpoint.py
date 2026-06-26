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
        features = hits        # (x, y, z, log_E)
        

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


import numpy as np

def to_cylindrical(hits):
    """
    hits: ndarray of shape (N,4)
          columns = [x, y, z, logE]

    returns:
          [r, phi, eta, logE]
    """

    x = hits[:, 0]
    y = hits[:, 1]
    z = hits[:, 2]
    logE = hits[:, 3]

    r = np.sqrt(x**2 + y**2)

    phi = np.arctan2(y, x)

    theta = np.arctan2(r, z)

    eta = -np.log(np.tan(theta / 2))

    hits_c = np.stack([r, phi, eta, logE], axis=1)

    return hits_c


class ContrastiveLearningDataset(IterableDataset):
    """
    Wraps an iterable dataset of calorimeter events contained in the output of CLDHits.

    Yields:
        original, view1, view2
    """

    def __init__(self, base_dataset, mean, std, transform=None, cylindrical=False):

        """
        base_dataset is an iterable dataset with each item being a dictionary with the key "calo_hit_features".

        transform must be a Transform object, see augmentation.py
        
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.mean = mean
        self.std = std
        self.cylindrical = cylindrical

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

            out_dict = {}


            if self.cylindrical == False:

                out_dict["calo_hit_features_1"] = (view1.hits-self.mean)/(self.std + 1e-8)
                out_dict["calo_hit_features_2"] = (view2.hits-self.mean)/(self.std + 1e-8)
                out_dict["calo_hit_features"] = (event.hits-self.mean)/(self.std + 1e-8)

            else:

                out_dict["calo_hit_features_1"] = (to_cylindrical(view1.hits)-self.mean)/(self.std + 1e-8)
                out_dict["calo_hit_features_2"] = (to_cylindrical(view2.hits)-self.mean)/(self.std + 1e-8)
                out_dict["calo_hit_features"] = (to_cylindrical(event.hits)-self.mean)/(self.std + 1e-8)

                

            

            '''
            out_dict["calo_hit_features_1"] = view1.hits
            out_dict["calo_hit_features_2"] = view2.hits
            out_dict["calo_hit_features"] = event.hits
            '''

            yield out_dict
            


class ContrastiveLearningGraphDataset(IterableDataset):
    """
    Wraps an iterable dataset of calorimeter events.

    Yields:
         view1_graph, view2_graph
    """

    def __init__(self, base_dataset, builder=EventGraphBuilder()):

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

            view1_graph = self.builder(view1)
            view2_graph = self.builder(view2)

            yield view1_graph, view2_graph
            




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

            x   = data_i['x'].to_numpy()[0]
            y   = data_i['y'].to_numpy()[0]
            z   = data_i['z'].to_numpy()[0]
            e   = data_i['total_energy'].to_numpy()[0]

            # Log-transform energy
            e_log = np.log(e + 1e-6)

            calo_hit_features = np.column_stack((x, y, z, e_log)).astype(np.float32)

           

            #f_i = len(data_i['x'].to_numpy()[0]) if len(data_i['x'].to_numpy()[0])<8000 else 8000

            yield {

                "calo_hit_features": calo_hit_features

            }
         
