"""One shared backbone and configurable single/five-space representation heads."""

from torch import nn

from .gravnet import GravNetBackbone


SPACE_ORDER = ("general", "energy", "eta", "phi", "local")
TASK_DIMS = {"energy": 2, "eta": 8, "phi": 4, "local": 4}
MODES = (
    "single_cosine", "single_anisotropic", "five_anisotropic_no_aux",
    "five_anisotropic_physics",
)


class MultiSpaceEncoder(nn.Module):
    """Branch before the former event MLP, at the 194-D pooled input.

    ``encode`` returns unnormalized h and needs one graph view, no labels.
    Projectors and optional linear physical readouts are only used by forward.
    A random baseline is the same selected architecture without learned weights.
    """

    def __init__(self, mode="five_anisotropic_physics", hidden_dim=16,
                 latent_dim=64, proj_dim=32, k=8, space_dim=4, propagate_dim=16):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown model mode {mode!r}; choose one of {MODES}.")
        if isinstance(proj_dim, bool) or not isinstance(proj_dim, int) or proj_dim < 1:
            raise ValueError("proj_dim must be a positive integer.")
        self.mode = mode
        self.space_order = ("general",) if mode.startswith("single_") else SPACE_ORDER
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim
        self.pooled_dim = 3 * latent_dim + 2
        self.backbone = GravNetBackbone(hidden_dim, latent_dim, k, space_dim, propagate_dim)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.pooled_dim, latent_dim), nn.LayerNorm(latent_dim),
                nn.ReLU(), nn.Linear(latent_dim, latent_dim),
            ) for name in self.space_order
        })
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(latent_dim, 8 * hidden_dim), nn.LayerNorm(8 * hidden_dim),
                nn.ReLU(), nn.Linear(8 * hidden_dim, proj_dim),
            ) for name in self.space_order
        })
        self.readouts = nn.ModuleDict({
            name: nn.Linear(latent_dim, size) for name, size in TASK_DIMS.items()
        } if mode == "five_anisotropic_physics" else {})

    def encode(self, graph_batch):
        pooled = self.backbone.pool(graph_batch)
        return {name: self.heads[name](pooled) for name in self.space_order}

    def forward(self, graph_batch):
        h = self.encode(graph_batch)
        return {
            "h": h,
            "z": {name: self.projectors[name](h[name]) for name in self.space_order},
            "pred": {name: readout(h[name]) for name, readout in self.readouts.items()},
        }
