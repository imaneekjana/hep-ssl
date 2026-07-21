"""

In this python file I am building my models, trying out different things

"""

"""
IMPORTS
"""

#from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, GravNetConv, global_mean_pool



######=========================== Helpers for pooling =============================##################


"""
The following function pools node-embeddings according to predefined assignments in a graph
~~~ a kind of 'coarsening' for creating some 'super-node' level embeddings of each event ~~~~~
"""

def pool_nodes_by_assignment(
    node_embeddings: Tensor,
    assignment: Tensor,
    batch: Tensor,
    reduction: Literal[
        "sum",
        "mean",
        "energy_weighted",
    ] = "mean",
    energies: Optional[Tensor] = None,
):
    """
    Pool nodes using the pair `(batch ID, local cluster ID)`.
    """
    assignment = assignment.to(
        device=node_embeddings.device,
        dtype=torch.long,
    )

    batch = batch.to(
        device=node_embeddings.device,
        dtype=torch.long,
    )

    cluster_keys = torch.stack(
        [batch, assignment],
        dim=1,
    )

    unique_keys, pooled_index, counts = torch.unique(
        cluster_keys,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    num_clusters = unique_keys.shape[0]
    embedding_dim = node_embeddings.shape[1]

    cluster_batch = unique_keys[:, 0]
    local_cluster_id = unique_keys[:, 1]

    cluster_embeddings = node_embeddings.new_zeros(
        (num_clusters, embedding_dim)
    )

    if reduction == "energy_weighted":
        if energies is None:
            raise ValueError(
                "Energy-weighted pooling requires `energies`."
            )

        energies = energies.to(
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )

        cluster_embeddings.index_add_(
            0,
            pooled_index,
            node_embeddings * energies.unsqueeze(-1),
        )

        cluster_energy = energies.new_zeros(num_clusters)
        cluster_energy.index_add_(
            0,
            pooled_index,
            energies,
        )

        cluster_embeddings = (
            cluster_embeddings
            / cluster_energy.clamp_min(1e-12).unsqueeze(-1)
        )

    else:
        cluster_embeddings.index_add_(
            0,
            pooled_index,
            node_embeddings,
        )

        if reduction == "mean":
            cluster_embeddings = (
                cluster_embeddings
                / counts.to(node_embeddings.dtype).unsqueeze(-1)
            )

        elif reduction != "sum":
            raise ValueError(
                f"Unknown pooling reduction: {reduction}"
            )

    return {
        "cluster_embeddings": cluster_embeddings,
        "cluster_batch": cluster_batch,
        "local_cluster_id": local_cluster_id,
        "cluster_counts": counts,
    }




####============================== Neural Networks ================================################


class My_Model_01(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        cluster_dim: int = 64,
        latent_dim: int = 64,
        proj_dim: int = 32,
        num_gnn_layers: int = 3,
        model_type: Literal["GraphConv", "GCNConv", "GravNetConv"] = "GravNetConv",
        model_args: dict = {'k': 5, 'space_dim': 8, 'propagate_dim': 8},
        cluster_pooling: str = "mean",
    ):
        super().__init__()

        self.cluster_pooling = cluster_pooling

        self.model_type = model_type

        dimensions = (
            [input_dim]
            + [hidden_dim] * (num_gnn_layers - 1)
            + [cluster_dim]
        )

        if model_type=="GravNetConv":

            self.convolutions = nn.ModuleList(
                [
                    GravNetConv(
                        dimensions[i],
                        dimensions[i + 1],
                        space_dimensions=model_args['space_dim'],
                        propagate_dimensions=model_args['propagate_dim'],
                        k=model_args['k']
                    )
                    for i in range(num_gnn_layers)
                ]
            )
        elif model_type=="GraphConv":

            self.convolutions = nn.ModuleList(
                [
                    GraphConv(
                        dimensions[i],
                        dimensions[i + 1],
                    )
                    for i in range(num_gnn_layers)
                ]
            )
        elif model_type=="GCNConv":

            self.convolutions = nn.ModuleList(
                [
                    GCNConv(
                        dimensions[i],
                        dimensions[i + 1],
                    )
                    for i in range(num_gnn_layers)
                ]
            )
        
            

        

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dimensions[i + 1])
                for i in range(num_gnn_layers)
            ]
        )

        self.cluster_mlp = nn.Sequential(
            nn.Linear(cluster_dim, 2*cluster_dim),
            nn.ReLU(),
            nn.Linear(2*cluster_dim, cluster_dim),
        )

        self.event_mlp = nn.Sequential(
            nn.Linear(cluster_dim, 2*cluster_dim),
            nn.ReLU(),
            nn.Linear(2*cluster_dim, latent_dim),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(
        self,
        data,
    ):

        x, edge_index, batch, assignment = data.x, data.edge_index, data.batch, data.assignment
        
        h = x

        if self.model_type=="GravNetConv":

            for conv, norm in zip(
                self.convolutions,
                self.norms,
            ):
                h = conv(h, batch)
                h = norm(h)
                h = torch.relu(h)

        else:

            for conv, norm in zip(
                self.convolutions,
                self.norms,
            ):
                h = conv(h, edge_index)
                h = norm(h)
                h = torch.relu(h)
            

        
        fine_node_embeddings = h

        pooled = pool_nodes_by_assignment(
            node_embeddings=fine_node_embeddings,
            assignment=assignment,
            batch=batch,
            reduction=self.cluster_pooling,
            energies=x[:, -1],
        )

        cluster_embeddings = self.cluster_mlp(
            pooled["cluster_embeddings"]
        )

        cluster_batch = pooled["cluster_batch"]

        # One embedding per event.
        event_embeddings = global_mean_pool(
            cluster_embeddings,
            cluster_batch,
        )

        event_embeddings = self.event_mlp(event_embeddings)

        event_embeddings = self.projection_head(event_embeddings)

        return {
            "fine_node_embeddings": fine_node_embeddings,
            "cluster_embeddings": cluster_embeddings,
            "cluster_batch": cluster_batch,
            "cluster_counts": pooled["cluster_counts"],
            "local_cluster_id": pooled["local_cluster_id"],
            "event_embeddings": event_embeddings,
        }




