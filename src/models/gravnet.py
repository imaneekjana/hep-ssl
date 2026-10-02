"""Shared GravNet node encoder and pooling with physical energy weights.

The node operations match the former ``gnn.GravNetEncoder`` through its
post-convolution projection. Fixed-grid preprocessing, raw-energy pooling,
and the two observed summaries are deliberate changes to the old baseline.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GravNetConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


def graph_batch_indices(graph):
    """Allow both a PyG Batch and a single, unbatched Data object."""
    batch = getattr(graph, "batch", None)
    if batch is None:
        batch = torch.zeros(graph.x.shape[0], device=graph.x.device, dtype=torch.long)
    if batch.ndim != 1 or batch.shape[0] != graph.x.shape[0]:
        raise ValueError("graph.batch must have one graph index per node.")
    if batch.dtype != torch.long or batch.numel() == 0:
        raise ValueError("Expected nonempty graphs and int64 batch indices.")
    if batch[0] != 0 or torch.any(batch[1:] < batch[:-1]):
        raise ValueError("GravNet requires sorted, contiguous graph batch indices.")
    counts = torch.bincount(batch)
    if torch.any(counts == 0):
        raise ValueError("Empty graphs are not supported.")
    return batch, counts


def pool_nodes(node_h, batch, energy, summary):
    """Concatenate mean, max, E/sum(E) pooling and observed summaries.

    ``energy`` is linear observed GeV, independently retained from graph.x.
    ``summary`` contains two standardized *observed* log-energy quantities.
    FP16/BF16 aggregation uses FP32; FP64 reference calculations stay FP64.
    """
    if energy.ndim != 1 or energy.shape[0] != node_h.shape[0]:
        raise ValueError("graph.energy must be a vector aligned with nodes.")
    if not torch.isfinite(energy).all() or torch.any(energy < 0):
        raise ValueError("Pooling requires finite, nonnegative linear energies.")
    n_graphs = int(batch.max().item()) + 1
    if summary.ndim == 1 and n_graphs == 1:
        summary = summary.unsqueeze(0)
    if tuple(summary.shape) != (n_graphs, 2):
        raise ValueError("graph.summary must have shape [number_of_graphs, 2].")
    if not torch.isfinite(summary).all():
        raise ValueError("Observed energy summaries must be finite.")
    dtype = torch.float64 if node_h.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=node_h.device.type, enabled=False):
        h = node_h.to(dtype)
        energy = energy.to(device=h.device, dtype=dtype)
        total = global_add_pool(energy[:, None], batch, size=n_graphs)[:, 0]
        if not torch.isfinite(total).all() or torch.any(total <= 0):
            raise ValueError("Every graph must have finite positive observed energy.")
        weights = energy / total[batch]
        mean = global_mean_pool(h, batch, size=n_graphs)
        maximum = global_max_pool(h, batch, size=n_graphs)
        weighted = global_add_pool(h * weights[:, None], batch, size=n_graphs)
        return torch.cat((mean, maximum, weighted, summary.to(h)), dim=-1)


class GravNetBackbone(nn.Module):
    """Original three-layer GravNet, ending at latent node features.

    PyG GravNetConv uses torch-cluster's kNN including the node itself. The
    tested torch-cluster 1.6.3 CPU backend uses min(k, N) neighbors separately
    for each graph. k is never reduced according to other graphs in a batch.
    No external edge_index is constructed and no nodes are copied.
    """

    def __init__(self, hidden_dim=16, latent_dim=64, k=8,
                 space_dim=4, propagate_dim=16):
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("latent_dim", latent_dim),
                            ("k", k), ("space_dim", space_dim),
                            ("propagate_dim", propagate_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.k = k
        self.latent_dim = latent_dim
        self.input_proj = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        try:
            self.conv1 = GravNetConv(hidden_dim, 2 * hidden_dim, space_dim, propagate_dim, k)
            self.conv2 = GravNetConv(2 * hidden_dim, 4 * hidden_dim, space_dim, propagate_dim, k)
            self.conv3 = GravNetConv(4 * hidden_dim, 8 * hidden_dim, space_dim, propagate_dim, k)
        except ImportError as exc:
            raise ImportError(
                "The real PyG GravNet encoder requires a torch-cluster build "
                "matching the installed PyTorch version. No substitute encoder is used."
            ) from exc
        self.norm1 = nn.LayerNorm(2 * hidden_dim)
        self.norm2 = nn.LayerNorm(4 * hidden_dim)
        self.norm3 = nn.LayerNorm(8 * hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(8 * hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU(),
        )

    def forward(self, graph):
        if graph.x.ndim != 2 or graph.x.shape[1] != 3:
            raise ValueError("Expected standardized [eta, phi, logE] node features.")
        batch, _ = graph_batch_indices(graph)
        x = self.input_proj(graph.x)
        x = F.relu(self.norm1(self.conv1(x, batch)))
        x = F.relu(self.norm2(self.conv2(x, batch)))
        return self.proj(self.norm3(self.conv3(x, batch)))

    def pool(self, graph):
        node_h = self(graph)
        batch, _ = graph_batch_indices(graph)
        return pool_nodes(node_h, batch, graph.energy, graph.summary)
