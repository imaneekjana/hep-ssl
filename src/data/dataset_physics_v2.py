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

from .augmentation_physics_v2 import *




'''
Creating graph for each event out of hits

'''

class EventGraphBuilder:
    def __init__(self, radius=3, max_neighbors=32):
        self.r = radius
        self.neighbors = max_neighbors

    def __call__(self, hits):
        """
        hits: numpy array or tensor [N, d+1], d spatial coordinates + 1 energy
        """

        if not torch.is_tensor(hits):
            hits = torch.tensor(hits, dtype=torch.float)

        pos = hits[:, :-1]      # spatial coordinates
        features = hits        # (x, y, z, log_E)
        

        #edge_index = knn_graph(pos, k=self.k, loop=False)

        pos = hits[:, :-1]
        N = pos.shape[0]

        # Compute pairwise squared distances
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  
        dist2 = (diff ** 2).sum(-1)

        # Select edges within radius (exclude self)
        row, col = torch.where((dist2 <= self.r ** 2) & (dist2 > 0))
        edge_index = torch.stack([row, col], dim=0)

        return Data(x=features, edge_index=edge_index)





###-------------------------------Helpers-------------------------------######


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

    s_phi = np.sin(phi)

    theta = np.arctan2(r, z)

    eta = -np.log(np.tan(theta / 2))

    hits_c = np.stack([r, s_phi, eta, logE], axis=1)

    return hits_c


def bin_points_to_grid(
    x, y, v,
    x_min, x_max, y_min, y_max,
    nx, ny,
    cutoff=0.0,
    typ='image',
    grid_transform=None,
):
    
    """
    Bin points (x, y) with values v onto a 2D grid by summing values
    of points falling into each cell.

    Parameters
    ----------
    x, y, v : array-like, shape (N,)
        Coordinates and values of the N points.
    x_min, x_max : float
        Range of the grid in x.
    y_min, y_max : float
        Range of the grid in y.
    nx, ny : int
        Number of grid cells along x and y.

    Returns
    -------
    grid : ndarray, shape (ny, nx)
        Summed values per cell. grid[i, j] corresponds to the cell
        spanning y_edges[i]:y_edges[i+1] and x_edges[j]:x_edges[j+1].
    x_edges : ndarray, shape (nx+1,)
        Bin edges along x.
    y_edges : ndarray, shape (ny+1,)
        Bin edges along y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)

    if not (len(x) == len(y) == len(v)):
        raise ValueError("x, y, v must have the same length")

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    # weights=v makes histogram2d SUM v instead of counting points
    grid_T, xe, ye = np.histogram2d(
        x, y, bins=[x_edges, y_edges], weights=v
    )

    # histogram2d returns shape (nx, ny) indexed [x_bin, y_bin];
    # transpose to the conventional image/grid layout (ny, nx) = [row=y, col=x]

    grid = grid_T.T

    # Physics-aware cell augmentations act on linear energy, before masking
    # empty cells and before taking log(E_cell).
    if grid_transform is not None:
        grid = grid_transform(grid)

    x_centers = (xe[:-1] + xe[1:]) / 2
    y_centers = (ye[:-1] + ye[1:]) / 2
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers)  # shape (ny, nx) each

    mask = grid > cutoff

    # .ravel() / boolean masking already gives 1D; np.asarray + dtype makes it explicit
    cell_centers_x = np.asarray(X_centers[mask], dtype=float).ravel()
    cell_centers_y = np.asarray(Y_centers[mask], dtype=float).ravel()
    cell_values    = np.asarray(grid[mask], dtype=float).ravel()

    assert len(x_centers) == nx
    assert len(y_centers) == ny


    if typ == 'image':

        image = np.empty((3, ny, nx), dtype=np.float32)

        image[2] = grid
        image[0] = x_centers[None, :]
        image[1] = y_centers[:, None]

        return image


    elif typ == 'hits':

        return np.column_stack((cell_centers_x, cell_centers_y, np.log(cell_values+ 1e-6))).astype(np.float64)



def projected_hits(
    hits,
    grid_size=32,
    typ='image',
    grid_transform=None,
):

    """
    hits: ndarray of shape (N,4)
          columns = [x, y, z, E]

    returns:
          dict object with keys 'eta-phi', 'x-y', 'z-rho', 'z-phi'
    """

    x = hits[:, 0]
    y = hits[:, 1]
    z = hits[:, 2]
    energy = hits[:, 3]

    rho = np.sqrt(x**2 + y**2)

    phi = np.arctan2(y, x)

    #s_phi = np.sin(phi)

    theta = np.arctan2(rho, z)

    eta = -np.log(np.tan(theta / 2))

    type_dict = {}

    type_dict['eta-phi'] = bin_points_to_grid(
        eta,
        phi,
        energy,
        min(eta),
        max(eta),
        -np.pi,
        np.pi,
        grid_size,
        grid_size,
        0.0,
        typ,
        grid_transform=grid_transform,
    )

    type_dict['x-y'] = bin_points_to_grid(x, y, energy, min(x), max(x), min(y), max(y), grid_size, grid_size, 0.0, typ)

    type_dict['z-rho'] = bin_points_to_grid(z, rho, energy, min(z), max(z), min(rho), max(rho), grid_size, grid_size, 0.0, typ)

    type_dict['z-phi'] = bin_points_to_grid(z, phi, energy, min(z), max(z), -np.pi, np.pi, grid_size, grid_size, 0.0, typ)


    return type_dict

    

   

    



##------------------------------------------------------------------------------


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



"""
Contrastive learning dataset in various 2-dimensional projections
"""


class ContrastiveLearningDatasetPlanar(IterableDataset):
    """
    Wraps an iterable dataset of calorimeter events contained in the output of ColliderMLHits.

    Yields: a dictionary with keys "calo_hit_features_1", "calo_hit_features_2", and "calo_hit_features" representing planar projections for view1, view2, and original  respectively
    
    """

    def __init__(
        self,
        base_dataset,
        mean=None,
        std=None,
        transform=None,
        grid_transform=None,
        projs=['eta-phi'],
        grid_size=32,
        typ='hits',
    ):

        """
        base_dataset is an iterable dataset with each item being a dictionary with the key "calo_hit_features".

        transform must be a Transform object, see augmentation.py

        projs: subset of ['eta-phi', 'x-y', 'z-rho', 'z-phi']

        typ: can either be 'image' or 'hits'
        
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.grid_transform = grid_transform
        self.mean = mean
        self.std = std
        self.projs = projs
        self.grid_size = grid_size
        self.typ = typ
        
        

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

            if self.mean==None and self.std==None:

                projected_hits1= projected_hits(
                    view1.hits,
                    self.grid_size,
                    self.typ,
                    grid_transform=self.grid_transform,
                )
                projected_hits2= projected_hits(
                    view2.hits,
                    self.grid_size,
                    self.typ,
                    grid_transform=self.grid_transform,
                )
                projected_hits0= projected_hits(event.hits, self.grid_size, self.typ)

                out_dict["calo_hit_features_1"] = [projected_hits1[key] for key in self.projs]

                out_dict["calo_hit_features_2"] = [projected_hits2[key] for key in self.projs]

                out_dict["calo_hit_features"] = [projected_hits0[key] for key in self.projs]

            else:

                assert len(self.mean) == len(self.projs)
                assert len(self.std) == len(self.projs)


                projected_hits1= projected_hits(
                    view1.hits,
                    self.grid_size,
                    self.typ,
                    grid_transform=self.grid_transform,
                )
                projected_hits2= projected_hits(
                    view2.hits,
                    self.grid_size,
                    self.typ,
                    grid_transform=self.grid_transform,
                )
                projected_hits0= projected_hits(event.hits, self.grid_size, self.typ)

                out_dict["calo_hit_features_1"] = [(projected_hits1[self.projs[i]]-self.mean[i])/self.std[i] for i in range(len(self.projs))]

                out_dict["calo_hit_features_2"] = [(projected_hits2[self.projs[i]]-self.mean[i])/self.std[i] for i in range(len(self.projs))]

                out_dict["calo_hit_features"] = [(projected_hits0[self.projs[i]]-self.mean[i])/self.std[i] for i in range(len(self.projs))]

                

            yield out_dict
            

            





### Conversion from "hits" to "graphs"
            


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
            



class ContrastiveLearningGraphDatasetPlanar(IterableDataset):
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

            view1_graphs = [self.builder(view1_) for view1_ in view1]
            view2_graphs = [self.builder(view2_) for view2_ in view2]


            if len(view1_graphs) == 1:
                
                yield view1_graphs[0], view2_graphs[0]
                
            else:

                yield view1_graphs, view2_graphs

            
            




class ColliderMLHits(IterableDataset):
    def __init__(
        self, calo_hits, split=None, shuffle_files=False, train_fraction=0.8, log=False, seed=42):
        """
        Initialize the dataset.

        Args:
            calo_hits : calo_hit data for events.
            shuffle_files (bool): Whether to shuffle the order of parquet files.
        """
        super().__init__()
        
        self.calo_hits = calo_hits
        self.shuffle_files = shuffle_files
        self.log = log
        self.seed = seed
        self.epoch = 0
        
        self.split = split
        if self.split is not None:
            split_index = int(len(self.calo_hits) * train_fraction)
            if self.split == "train":
                self.calo_hits = self.calo_hits[:split_index]
            elif self.split == "val":
                self.calo_hits = self.calo_hits[split_index:]

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

        if self.shuffle_files:
            data = self.calo_hits.sample(
                fraction = 1.0,
                with_replacement = False,
                shuffle = True,
                seed = self.seed + self.epoch
            )
            self.epoch += 1
        else:
            data = self.calo_hits
        
        for event_i in range(len(data)):

            data_i = data.slice(event_i, 1)

            x   = data_i['x'].to_numpy()[0]
            y   = data_i['y'].to_numpy()[0]
            z   = data_i['z'].to_numpy()[0]
            e   = data_i['total_energy'].to_numpy()[0]

            # Log-transform energy
            e_log = np.log(e + 1e-6)

            if self.log:
                calo_hit_features = np.column_stack((x, y, z, e_log)).astype(np.float32)
            else:
                calo_hit_features = np.column_stack((x, y, z, e)).astype(np.float32)
                

            

           

            #f_i = len(data_i['x'].to_numpy()[0]) if len(data_i['x'].to_numpy()[0])<8000 else 8000

            yield {

                "calo_hit_features": calo_hit_features

            }
