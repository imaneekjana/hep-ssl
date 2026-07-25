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

from colliderml.physics import assign_primary_ancestor, CALO_DETECTOR_CODES

from .augmentation import *
from .coarsening import voxelize_hits
from .graph_builder import EventGraphBuilder



"""
Extracting info from ColliderML dataset
"""

class ColliderMLHits(IterableDataset):
    def __init__(
        self, calo_hits, split, shuffle_files=False, train_fraction=0.8, log=False):
        """
        Initialize the dataset.

        Args:
            calo_hits : calo_hit data for events.
            shuffle_files (bool): Whether to shuffle the order of parquet files.
        """
        
        self.calo_hits = calo_hits
        self.shuffle_files = shuffle_files
        self.log = log
        
        

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
            
            # preparing ECAL and HCAL masks
            
            detector_id = data_i['detector'].to_numpy()[0]
            
            _ecal_ids = [
                CALO_DETECTOR_CODES["ecal_neg_endcap"],
                CALO_DETECTOR_CODES["ecal_barrel"],
                CALO_DETECTOR_CODES["ecal_pos_endcap"],
            ]
            _hcal_ids = [
                CALO_DETECTOR_CODES["hcal_neg_endcap"],
                CALO_DETECTOR_CODES["hcal_barrel"],
                CALO_DETECTOR_CODES["hcal_pos_endcap"],
            ]
            mask_ecal = np.isin(detector_id, _ecal_ids)
            mask_hcal = np.isin(detector_id, _hcal_ids)
            
            
            

            # Log-transform energy if flagged
            
            e_log = np.log(e + 1e-6)

            if self.log==True:
                calo_hit_features = np.column_stack((x, y, z, e_log)).astype(np.float32)
            else:
                calo_hit_features = np.column_stack((x, y, z, e)).astype(np.float32)
                

            

           

            #f_i = len(data_i['x'].to_numpy()[0]) if len(data_i['x'].to_numpy()[0])<8000 else 8000

            yield {

                "calo_hit_features": calo_hit_features,
                "ecal_mask": mask_ecal,
                "hcal_mask": mask_hcal,

            }
         




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


def bin_points_to_grid(x, y, v, x_min, x_max, y_min, y_max, nx, ny, cutoff=0.0, typ='image'):
    
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



def projected_hits(hits, grid_size=32, typ='image'):

    """
    hits: ndarray of shape (N,4)
          columns = [x, y, z, E]

    returns:
          dict object with keys 'eta-phi', 'x-y', 'z-rho', 'z-phi'
    """

    x = hits[:, 0]
    y = hits[:, 1]
    z = hits[:, 2]
    E = hits[:, 3]

    rho = np.sqrt(x**2 + y**2)

    phi = np.arctan2(y, x)

    #s_phi = np.sin(phi)

    theta = np.arctan2(rho, z)

    eta = -np.log(np.tan(theta / 2))

    type_dict = {}

    type_dict['eta-phi'] = bin_points_to_grid(eta, phi, E, min(eta), max(eta), min(phi), max(phi), grid_size, grid_size, 0.0, typ)

    type_dict['x-y'] = bin_points_to_grid(x, y, E, min(x), max(x), min(y), max(y), grid_size, grid_size, 0.0, typ)

    type_dict['z-rho'] = bin_points_to_grid(z, rho, E, min(z), max(z), min(rho), max(rho), grid_size, grid_size, 0.0, typ)

    type_dict['z-phi'] = bin_points_to_grid(z, phi, E, min(z), max(z), min(phi), max(phi), grid_size, grid_size, 0.0, typ)


    return type_dict

    

   

    



##=============================== Contrastive Learning Dataset preparations ====================#########


class ContrastiveLearningDataset(IterableDataset):
    """
    Wraps an iterable dataset of calorimeter events contained in the output of CLDHits.

    Yields:
        original, view1, view2
    """

    def __init__(self, base_dataset, mean=None, std=None, transform=None, voxel_size=None, cylindrical=False):

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
        self.voxel_size = voxel_size
        self.feature_axis = 1

    def __len__(self):

        return len(self.base_dataset)

    def compute_mean_std(self, max_samples=50):
        
        array = []

        for i, x_dict in enumerate(self):
            
            if i >= max_samples:
                break

            xi = x_dict["calo_hit_features_0"]

            xi = np.moveaxis(np.asarray(xi), self.feature_axis, 0) 
            xi = xi.reshape(xi.shape[0], -1) 

            array.append(xi)

        X = np.concatenate(array, axis=1) 

        mean = X.mean(axis=1) 
        std = X.std(axis=1)

        self.mean = mean
        self.std = std

        return mean, std

        

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

            if self.voxel_size is not None:

                view1.hits = voxelize_hits(view1.hits, voxel_size = self.voxel_size, origin=None, return_coarse_hits=True)['coarse_hits']
                view2.hits = voxelize_hits(view2.hits, voxel_size = self.voxel_size, origin=None, return_coarse_hits=True)['coarse_hits']
                event.hits = voxelize_hits(event.hits, voxel_size = self.voxel_size, origin=None, return_coarse_hits=True)['coarse_hits']


            if self.mean is None and self.std is None:

                if self.cylindrical == False:
                    

                    out_dict["calo_hit_features_1"] = view1.hits
                    out_dict["calo_hit_features_2"] = view2.hits
                    out_dict["calo_hit_features_0"] = event.hits
                else:

                    out_dict["calo_hit_features_1"] = to_cylindrical(view1.hits)
                    out_dict["calo_hit_features_2"] = to_cylindrical(view2.hits)
                    out_dict["calo_hit_features_0"] = to_cylindrical(event.hits)

            else:
                if self.cylindrical == False:

                    out_dict["calo_hit_features_1"] = (view1.hits-self.mean)/(self.std + 1e-8)
                    out_dict["calo_hit_features_2"] = (view2.hits-self.mean)/(self.std + 1e-8)
                    out_dict["calo_hit_features_0"] = (event.hits-self.mean)/(self.std + 1e-8)
                else:

                    out_dict["calo_hit_features_1"] = (to_cylindrical(view1.hits)-self.mean)/(self.std + 1e-8)
                    out_dict["calo_hit_features_2"] = (to_cylindrical(view2.hits)-self.mean)/(self.std + 1e-8)
                    out_dict["calo_hit_features_0"] = (to_cylindrical(event.hits)-self.mean)/(self.std + 1e-8)


                
    

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

    def __init__(self, base_dataset, mean=None, std=None, transform=None, projs=['eta-phi'], grid_size=32, typ='hits'):

        """
        base_dataset is an iterable dataset with each item being a dictionary with the key "calo_hit_features".

        transform must be a Transform object, see augmentation.py

        projs: subset of ['eta-phi', 'x-y', 'z-rho', 'z-phi']

        typ: can either be 'image' or 'hits'
        
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.mean = mean
        self.std = std
        self.projs = projs
        self.grid_size = grid_size
        self.typ = typ

        if self.typ == 'image':
            self.feature_axis = 0
        elif self.typ == 'hits':
            self.feature_axis = 1
                    
        
        

    def __len__(self):

        return len(self.base_dataset)

    def compute_mean_std(self, max_samples=50):
        
        arrays = [[] for _ in range(len(self.projs))]

        for i, x_dict in enumerate(self):
            
            if i >= max_samples:
                break

            x = x_dict["calo_hit_features_0"]

            x_ = [np.moveaxis(np.asarray(xi), self.feature_axis, 0) for xi in x]
            x_ = [xi.reshape(xi.shape[0], -1) for xi in x_]

            for i, arr in enumerate(arrays):

                arr.append(x_[i])


        X_ = [np.concatenate(arr, axis=1) for arr in arrays]

        mean = [X.mean(axis=1) for X in X_]
        std = [X.std(axis=1) for X in X_]

        self.mean = mean
        self.std = std

        return mean, std


    def normalize(self, x, mean_, std_):
        shape = [1] * x.ndim
        shape[self.feature_axis] = len(mean_)

        mean_ = mean_.reshape(shape)
        std_ = std_.reshape(shape)

        return (x - mean_) / (std_ + 1e-8)

    

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

                projected_hits1= projected_hits(view1.hits, self.grid_size, self.typ)
                projected_hits2= projected_hits(view2.hits, self.grid_size, self.typ)
                projected_hits0= projected_hits(event.hits, self.grid_size, self.typ)

                out_dict["calo_hit_features_1"] = [projected_hits1[key] for key in self.projs]

                out_dict["calo_hit_features_2"] = [projected_hits2[key] for key in self.projs]

                out_dict["calo_hit_features_0"] = [projected_hits0[key] for key in self.projs]

                

            else:

                assert len(self.mean) == len(self.projs)
                assert len(self.std) == len(self.projs)


                projected_hits1= projected_hits(view1.hits, self.grid_size, self.typ)
                projected_hits2= projected_hits(view2.hits, self.grid_size, self.typ)
                projected_hits0= projected_hits(event.hits, self.grid_size, self.typ)


                '''

                out_dict["calo_hit_features_1"] = [(projected_hits1[self.projs[i]]-self.mean[i])/self.std[i] for i in len(self.projs)]

                out_dict["calo_hit_features_2"] = [(projected_hits2[self.projs[i]]-self.mean[i])/self.std[i] for i in len(self.projs)]

                out_dict["calo_hit_features_0"] = [(projected_hits1[self.projs[i]]-self.mean[i])/self.std[i] for i in len(self.projs)]

                '''

                out_dict["calo_hit_features_1"] = [self.normalize(x=projected_hits1[self.projs[i]], mean_=self.mean[i], std_=self.std[i]) for i in range(len(self.projs))]

                out_dict["calo_hit_features_2"] = [self.normalize(x=projected_hits2[self.projs[i]], mean_=self.mean[i], std_=self.std[i]) for i in range(len(self.projs))]

                out_dict["calo_hit_features_0"] = [self.normalize(x=projected_hits0[self.projs[i]], mean_=self.mean[i], std_=self.std[i]) for i in range(len(self.projs))]

                

            yield out_dict
            

            





### Conversion from "hits" to "graphs"
            


class ContrastiveLearningGraphDataset(IterableDataset):
    """
    Wraps an iterable dataset of calorimeter events.

    Yields:
         view1_graph, view2_graph
    """

    def __init__(self, base_dataset, builder=EventGraphBuilder(method='radius', radius=0.0, from_assignment=False, max_neighbors=14, knn_neighbor=0), cluster_voxel_size=None):

        """
        base_dataset must be an iterable with dictionaries having the keys, "calo_hit_features_1" and "calo_hit_features_2".

        base_dataset can be an output of ContrastiveLearningDataset

        """
        super().__init__()
        self.base_dataset = base_dataset
        self.builder = builder
        self.cluster_voxel_size=cluster_voxel_size

    def __len__(self):

        return len(self.base_dataset)

    def __iter__(self):
        for event_dict in self.base_dataset:

            view1 = event_dict["calo_hit_features_1"]
            view2 = event_dict["calo_hit_features_2"]

            if self.cluster_voxel_size == None:

                view1_graph = self.builder(view1, assignment=None)
                view2_graph = self.builder(view2, assignment=None)

            else:

                assignment1 = voxelize_hits(hits=view1, voxel_size=self.cluster_voxel_size, return_coarse_hits=False)['assignment']
                assignment2 = voxelize_hits(hits=view2, voxel_size=self.cluster_voxel_size, return_coarse_hits=False)['assignment']

                view1_graph = self.builder(view1, assignment=assignment1)
                
                view2_graph = self.builder(view2, assignment=assignment2)

            
            

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




class ContrastiveLearningImageDatasetPlanar(IterableDataset):
    """
    Wraps an iterable dataset of calorimeter events.

    Yields:
         view1_image, view2_image
    """

    def __init__(self, base_dataset, only_energy=True):

        """
        base_dataset must be an iterable with dictionaries having the keys, "calo_hit_features_1" and "calo_hit_features_2".

        base_dataset can be an output of ContrastiveLearningDataset

        """
        super().__init__()
        self.base_dataset = base_dataset
        self.only_energy = only_energy
        

    def __len__(self):

        return len(self.base_dataset)

    def __iter__(self):
        for event_dict in self.base_dataset:
            

            view1 = event_dict["calo_hit_features_1"]
            view2 = event_dict["calo_hit_features_2"]

           
            if len(view1) == 1:

                if self.only_energy==True:
                    
                    yield view1[0][[-1]], view2[0][[-1]]

                else:
                    yield view1[0], view2[0]
                 
                
            else:

                yield view1, view2


            
            




