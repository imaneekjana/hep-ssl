from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor




####----------------------------------VOXELIZATION FOR COARSE GRAINING-------------------------########

def voxelize_hits(
    hits: Tensor,
    voxel_size: Union[float, Sequence[float], Tensor],
    origin: Optional[Union[Sequence[float], Tensor]] = None,
    return_coarse_hits = True,
    return_assignment: bool = True,
    return_counts: bool = False,
    eps: float = 1e-12,
    log: bool = False,
):
    """
    Coarsen calorimeter hits by grouping them into d-dimensional spatial voxels.

    Each input hit is assumed to have the form

        [x1, ..., xd, energy]

    Hits occupying the same voxel are replaced by one coarse hit.
    The coarse-hit energy is the sum of constituent energies, while
    its position is the energy-weighted centroid:

        E_voxel = sum_i E_i

        r_voxel = sum_i E_i * r_i / sum_i E_i

    If the total energy in a voxel is numerically zero, the ordinary
    mean position of the hits in that voxel is used instead.

    Parameters
    ----------
    hits:
        Tensor of shape [N, d+1], with columns [x1, ..., xd, energy].

    voxel_size:
        Either a scalar, giving the same voxel size along all three
        coordinate axes, or a sequence/tensor of shape [d]:

            voxel_size = (dx1, ..., dxd)

    origin:
        Coordinate origin used to define voxel boundaries. If None,
        the minimum coordinate of the current event is used.

        For consistent detector-wide voxel boundaries across events,
        pass a fixed origin instead of leaving this as None.

    return_coarse_hits:
        If True, returns the coarse hits.

    return_assignment:
        If True, also return a tensor of shape [N]. Entry i gives the
        coarse-voxel index assigned to original hit i.

    return_counts:
        If True, also return the number of original hits in each voxel.

    eps:
        Small value used when checking whether the summed energy in a
        voxel is zero.
    log: 
        If True, the last feature/weight is logged in the final output.

    Returns
    -------
    coarse_hits:
        Tensor of shape [M, d+1], where M <= N. The columns are

            [x_energy_weighted, ...,
             xd_energy_weighted, summed_energy]

    assignment:
        Optional tensor of shape [N].

    counts:
        Optional tensor of shape [M].

    Examples
    --------
    >>> hits = torch.tensor([
    ...     [0.1, 0.2, 0.1, 2.0],
    ...     [0.4, 0.3, 0.2, 1.0],
    ...     [1.2, 0.1, 0.1, 3.0],
    ... ])
    >>> coarse_hits, assignment = voxelize_hits(
    ...     hits,
    ...     voxel_size=1.0,
    ...     origin=(0.0, 0.0, 0.0),
    ...     return_assignment=True,
    ... )
    """
    if not torch.is_tensor(hits):
        hits = torch.as_tensor(hits, dtype=torch.float32)

    if hits.ndim != 2:
        raise ValueError(
            f"`hits` must have shape [N, d+1], but received {tuple(hits.shape)}."
        )

    d = (hits.shape[1]-1) # Number of spatial dimension

    if not hits.is_floating_point():
        hits = hits.float()

    if hits.shape[0] == 0:
        coarse_hits = hits.clone()
        outputs = [coarse_hits]

        if return_assignment:
            outputs.append(
                torch.empty(0, dtype=torch.long, device=hits.device)
            )

        if return_counts:
            outputs.append(
                torch.empty(0, dtype=torch.long, device=hits.device)
            )

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    if not torch.isfinite(hits).all():
        raise ValueError("`hits` contains NaN or infinite values.")

    positions = hits[:, :-1]
    energies = hits[:, -1]

    voxel_size = torch.as_tensor(
        voxel_size,
        dtype=positions.dtype,
        device=positions.device,
    )

    if voxel_size.ndim == 0:
        voxel_size = voxel_size.repeat(d)

    if voxel_size.shape != (d,):
        raise ValueError(
            f"`voxel_size` must be a scalar or contain {d} values "
            "(dx1, ..., dxd)."
        )

    if torch.any(voxel_size <= 0):
        raise ValueError("All voxel sizes must be positive.")

    if origin is None:
        origin_tensor = positions.min(dim=0).values
    else:
        origin_tensor = torch.as_tensor(
            origin,
            dtype=positions.dtype,
            device=positions.device,
        )

        if origin_tensor.shape != (d,):
            raise ValueError(f"`origin` must contain {d} coordinates.")

    # Integer voxel coordinate for every hit: [N, d].
    voxel_coordinates = torch.floor(
        (positions - origin_tensor) / voxel_size
    ).to(torch.long)

    # `assignment[i]` gives the unique voxel containing hit i.
    unique_voxels, assignment, counts = torch.unique(
        voxel_coordinates,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    num_voxels = unique_voxels.shape[0]

    output = {}

    if return_assignment:
        output['assignment'] = assignment # store the assignment


    if return_coarse_hits: # Calculate coarse hits and return 
        
        # Sum energy in every voxel.
        coarse_energy = torch.zeros(
            num_voxels,
            dtype=energies.dtype,
            device=energies.device,
        )
        coarse_energy.index_add_(0, assignment, energies)

        # Sum E_i * r_i in every voxel.
        weighted_position_sum = torch.zeros(
            (num_voxels, d),
            dtype=positions.dtype,
            device=positions.device,
        )
        weighted_position_sum.index_add_(
            0,
            assignment,
            positions * energies.unsqueeze(-1),
        )

        energy_weighted_position = (
            weighted_position_sum
            / coarse_energy.unsqueeze(-1).clamp_min(eps)
        )

        # Compute ordinary mean positions as a fallback for voxels whose
        # total energy is zero.
        position_sum = torch.zeros(
            (num_voxels, d),
            dtype=positions.dtype,
            device=positions.device,
        )
        position_sum.index_add_(0, assignment, positions)

        mean_position = position_sum / counts.to(
            positions.dtype
        ).unsqueeze(-1)

        zero_energy_mask = coarse_energy.abs() <= eps

        coarse_position = torch.where(
            zero_energy_mask.unsqueeze(-1),
            mean_position,
            energy_weighted_position,
        )

        coarse_hits = torch.cat(
            [coarse_position, coarse_energy.unsqueeze(-1)],
            dim=-1,
        )

        if log==True:
        
            wt = coarse_hits[:,-1]
            coarse_hits[:,-1] = torch.log(wt+1e-15)

        output['coarse_hits'] = coarse_hits

    if return_counts:
        
        output['counts'] = counts

    return output






    