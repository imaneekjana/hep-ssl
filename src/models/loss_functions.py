###~~~~~~~~~~~~~~~~~~~~~~~~~~ Python file for loss functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~############

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor




#####======================= InfoNCE loss (SSL based) =====================================############

class InfoNCE_Loss(nn.Module):

    """
    InfoNCE loss to be applied on embeddings of augmentations of a batch of events
    """

    def __init__(self, batch_size: int = 32, n_views: int = 2, temperature: float = 0.1):

        super().__init__()

        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z1, z2):

        features = torch.cat((z1, z2), dim=0)

        device = features.device
        
    
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)
    
        features = F.normalize(features, dim=1)
    
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape
    
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
    
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    
        logits = logits / self.temperature
    
        criterion = (self.criterion).to(features.device)
    
        this_loss = criterion(logits, labels)
        
        return this_loss

        



#####====================== OT based Sinkhorn loss for for a pair of node-embeddings ==============######


from geomloss import SamplesLoss


class EventWiseGeomOT_Loss(nn.Module):
    """
    Event-wise Sinkhorn OT for batched node/cluster embeddings.

    Parameters
    ----------
    loss_type:
        "sinkhorn" returns the Sinkhorn divergence when debias=True.
        Set debias=False for regularized entropic OT.

    p:
        Ground-cost exponent.

        p=1 corresponds roughly to ||x-y||.
        p=2 corresponds roughly to ||x-y||^2.

    blur:
        Sinkhorn scale parameter.

        For p=2, GeomLoss internally relates the entropic scale to
        blur**p. Therefore, blur is not exactly identical to epsilon
        in a hand-written Sinkhorn implementation.

    reach:
        Enables unbalanced OT when not None.

        Larger reach penalizes mass destruction more strongly.
        reach=None gives balanced OT.

    backend:
        "tensorized":
            Explicit cost matrix. Good for small point sets.

        "online":
            Uses KeOps and avoids storing the full cost matrix.

        "multiscale":
            Intended mainly for low-dimensional geometric point clouds.

        "auto":
            Lets GeomLoss choose.

    normalize_embeddings:
        L2-normalize embeddings before computing OT.

        With normalized embeddings and p=2:

            ||z_i - z_j||^2 = 2(1 - cosine_similarity).

        Thus squared Euclidean cost becomes proportional to cosine distance.
    """

    def __init__(
        self,
        p: int = 2,
        blur: float = 0.05,
        reach: Optional[float] = None,
        debias: bool = True,
        backend: Literal[
            "auto",
            "tensorized",
            "online",
            "multiscale",
        ] = "auto",
        normalize_embeddings: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
        scaling: float = 0.9,
    ) -> None:
        super().__init__()

        if p not in {1, 2}:
            raise ValueError("p must be 1 or 2.")

        if blur <= 0:
            raise ValueError("blur must be positive.")

        if reach is not None and reach <= 0:
            raise ValueError("reach must be positive or None.")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be 'mean', 'sum', or 'none'."
            )

        self.normalize_embeddings = normalize_embeddings
        self.reduction = reduction

        self.loss = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            reach=reach,
            debias=debias,
            backend=backend,
            scaling=scaling,
        )

    @staticmethod
    def _prepare_weights(
        weights: Optional[Tensor],
        n: int,
        reference: Tensor,
    ) -> Tensor:
        if weights is None:
            weights = torch.ones(
                n,
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            weights = weights.to(
                device=reference.device,
                dtype=reference.dtype,
            )

        if weights.ndim != 1 or weights.shape[0] != n:
            raise ValueError(f"Weights must have shape [{n}].")

        if torch.any(weights < 0):
            raise ValueError("OT weights must be non-negative.")

        if weights.sum().detach().item() <= 0:
            raise ValueError("Each event must have positive total mass.")

        return weights / weights.sum().clamp_min(1e-12)

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        batch1: Tensor,
        batch2: Tensor,
        mass1: Optional[Tensor] = None,
        mass2: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        z1, z2:
            Batched cluster embeddings with shapes [N1, D] and [N2, D].

        batch1, batch2:
            Event IDs for each cluster, shapes [N1] and [N2].

        mass1, mass2:
            Optional cluster masses, usually cluster energies.
        """
        if z1.ndim != 2 or z2.ndim != 2:
            raise ValueError("z1 and z2 must both be two-dimensional.")

        if z1.shape[1] != z2.shape[1]:
            raise ValueError("Embedding dimensions must match.")

        if batch1.shape != (z1.shape[0],):
            raise ValueError("batch1 must have shape [N1].")

        if batch2.shape != (z2.shape[0],):
            raise ValueError("batch2 must have shape [N2].")

        if self.normalize_embeddings:
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)

        event_ids1 = torch.unique(batch1, sorted=True)
        event_ids2 = torch.unique(batch2, sorted=True)

        if not torch.equal(event_ids1, event_ids2):
            raise ValueError(
                "Both views must contain the same event IDs."
            )

        event_losses = []

        for event_id in event_ids1:
            mask1 = batch1 == event_id
            mask2 = batch2 == event_id

            event_z1 = z1[mask1]
            event_z2 = z2[mask2]

            event_mass1 = self._prepare_weights(
                None if mass1 is None else mass1[mask1],
                n=event_z1.shape[0],
                reference=event_z1,
            )

            event_mass2 = self._prepare_weights(
                None if mass2 is None else mass2[mask2],
                n=event_z2.shape[0],
                reference=event_z2,
            )

            # GeomLoss weighted-point-cloud interface:
            # loss(alpha, x, beta, y)
            event_loss = self.loss(
                event_mass1,
                event_z1,
                event_mass2,
                event_z2,
            )

            event_losses.append(event_loss)

        event_losses = torch.stack(event_losses)

        if self.reduction == "mean":
            return event_losses.mean()

        if self.reduction == "sum":
            return event_losses.sum()

        return event_losses