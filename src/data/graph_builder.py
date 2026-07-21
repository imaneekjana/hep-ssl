import uproot
import awkward as ak
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np




from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from torch_geometric.nn import radius_graph

from typing import Optional
from torch import Tensor
import torch
from torch.utils.data import IterableDataset

from .augmentation import *
from .coarsening import *

"""
Edge index creating helpers
"""
def build_radius_edge_index_from_assignment_1(
    hits: Tensor,
    assignment: Tensor,
    radius: float,
    max_neighbors: Optional[int] = 14,
    directed: bool = False,
) -> Tensor:
    """
    Build radius-based edges between hits belonging to the same cluster.

    The clusters can be voxel assignments returned by `voxelize_hits`,
    but this function works with any non-overlapping cluster assignment.

    Parameters
    ----------
    hits:
        Tensor of shape [N, 4], with columns [x, y, z, energy].

    assignment:
        Long tensor of shape [N]. `assignment[i]` is the cluster or
        voxel index containing hit i.

    radius:
        Two hits are connected when their Euclidean distance is less
        than `radius`.

    max_neighbors:
        Optional maximum number of outgoing neighbors retained for
        each source hit. The closest neighbors are kept.

    directed:
        If True, include both i -> j and j -> i.
        If False, include each pair only once, with i < j.

    Returns
    -------
    edge_index:
        Long tensor of shape [2, E].
    """
    if not torch.is_tensor(hits):
        hits = torch.as_tensor(hits, dtype=torch.float32)

    if not torch.is_tensor(assignment):
        assignment = torch.as_tensor(
            assignment,
            dtype=torch.long,
            device=hits.device,
        )
    else:
        assignment = assignment.to(
            device=hits.device,
            dtype=torch.long,
        )

    if hits.ndim != 2 or hits.shape[1] != 4:
        raise ValueError(
            f"`hits` must have shape [N, 4], received {tuple(hits.shape)}."
        )

    if assignment.ndim != 1 or assignment.shape[0] != hits.shape[0]:
        raise ValueError(
            "`assignment` must have shape [N], matching the number of hits."
        )

    if radius <= 0:
        raise ValueError("`radius` must be positive.")

    if max_neighbors is not None and max_neighbors <= 0:
        raise ValueError("`max_neighbors` must be positive or None.")

    positions = hits[:, :3]
    num_hits = hits.shape[0]
    device = hits.device

    if num_hits == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # Group hits with the same assignment into contiguous blocks.
    sorted_assignment, permutation = torch.sort(assignment)
    sorted_positions = positions[permutation]

    _, counts = torch.unique_consecutive(
        sorted_assignment,
        return_counts=True,
    )

    radius_squared = radius**2
    source_parts = []
    target_parts = []

    start = 0

    for count_tensor in counts:
        count = int(count_tensor.item())
        stop = start + count

        if count <= 1:
            start = stop
            continue

        cluster_positions = sorted_positions[start:stop]
        original_indices = permutation[start:stop]

        # Pairwise squared distances only inside this cluster.
        differences = (
            cluster_positions[:, None, :]
            - cluster_positions[None, :, :]
        )
        distance_squared = differences.square().sum(dim=-1)

        valid = (
            (distance_squared < radius_squared)
            & (distance_squared > 0)
        )

        if max_neighbors is None:
            local_source, local_target = torch.where(valid)

        else:
            local_source_parts = []
            local_target_parts = []

            for local_i in range(count):
                candidate_j = torch.where(valid[local_i])[0]

                if candidate_j.numel() == 0:
                    continue

                if candidate_j.numel() > max_neighbors:
                    candidate_distances = distance_squared[
                        local_i, candidate_j
                    ]

                    nearest = torch.topk(
                        candidate_distances,
                        k=max_neighbors,
                        largest=False,
                    ).indices

                    candidate_j = candidate_j[nearest]

                local_source_parts.append(
                    torch.full(
                        (candidate_j.numel(),),
                        local_i,
                        dtype=torch.long,
                        device=device,
                    )
                )
                local_target_parts.append(candidate_j)

            if not local_source_parts:
                start = stop
                continue

            local_source = torch.cat(local_source_parts)
            local_target = torch.cat(local_target_parts)

        source = original_indices[local_source]
        target = original_indices[local_target]

        if not directed:
            keep = source < target
            source = source[keep]
            target = target[keep]

        source_parts.append(source)
        target_parts.append(target)

        start = stop

    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return torch.stack(
        [
            torch.cat(source_parts),
            torch.cat(target_parts),
        ],
        dim=0,
    )


def build_radius_edge_index_from_assignment_2(
    hits: Tensor,
    assignment: Tensor,
    radius: float = 0.0,
):
    """
    Build edges between spatially nearby hits belonging to the same cluster.

    Parameters
    ----------
    hits:
        Tensor or array of shape [N, D + 1].
        The final column is assumed to be energy.

    assignment:
        Tensor or array of shape [N].
        assignment[i] gives the cluster containing hit i.

    radius:
        Maximum spatial distance between connected hits.

    Returns
    -------
    edge_index:
        Long tensor of shape [2, E].
    """
    hits = torch.as_tensor(
        hits,
        dtype=torch.float32,
    )

    assignment = torch.as_tensor(
        assignment,
        dtype=torch.long,
        device=hits.device,
    )

    if hits.ndim != 2:
        raise ValueError("`hits` must have shape [N, D + 1].")

    if assignment.ndim != 1 or assignment.shape[0] != hits.shape[0]:
        raise ValueError(
            "`assignment` must have shape [N] and match `hits`."
        )

    if radius <= 0:
        raise ValueError("`radius` must be positive.")

    pos = hits[:, :-1]

    # dist2[i, j] = squared spatial distance between nodes i and j.
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist2 = diff.square().sum(dim=-1)

    # same_cluster[i, j] is True when i and j have equal assignments.
    same_cluster = (
        assignment.unsqueeze(1)
        == assignment.unsqueeze(0)
    )

    valid_edges = (
        (dist2 <= radius**2)
        & (dist2 > 0)
        & same_cluster
    )

    row, col = torch.where(valid_edges)

    return torch.stack([row, col], dim=0)




def build_radius_edge_index_(hits, radius=0.0):
    

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
    row, col = torch.where((dist2 <= radius ** 2) & (dist2 > 0))
    edge_index = torch.stack([row, col], dim=0)

    return edge_index




"""
Creating graph for each event out of hits

"""

class EventGraphBuilder:
    
    def __init__(self, method='radius', radius=0.0, from_assignment=False, max_neighbors=14, knn_neighbor=0):
        
        self.method = method
        self.r = radius
        self.neighbors = max_neighbors
        self.knn_neighbor = knn_neighbor ## NOT IMPLEMENTED YET
        self.from_assignment = from_assignment
        
        

    def __call__(self, hits, assignment=None):
        """
        hits: numpy array or tensor [N, d+1], d spatial coordinates + 1 energy
        """
        

        features = hits

        if not self.from_assignment:

            if self.method == 'radius':
                
                edge_index = build_radius_edge_index_(hits, self.r)
            
        
        else:
            
            if assignment is None or hits.shape[0] != assignment.shape[0]:
                raise ValueError('If from_assignment==True, assigment of shape (N,) must be provided')

            if self.method == 'radius':
                
                edge_index = build_radius_edge_index_from_assignment_1(hits, assignment, self.r, self.neighbors)

        
        if assignment is None:
            
            assigment = torch.zeros(hits.shape[0], dtype=float)
        



        return Data(x=features, edge_index=edge_index, assignment=assignment)



