import ast
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GravNetConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

from src.losses.multitask import MultiTaskObjective
from src.models.gravnet import pool_nodes
from src.models.multispace import MultiSpaceEncoder, SPACE_ORDER, TASK_DIMS


def sample_graph(nodes=12):
    return Data(x=torch.randn(nodes, 3), energy=torch.rand(nodes) + 0.5,
                summary=torch.randn(1, 2))


class PhysicalPoolingTests(unittest.TestCase):
    def test_physical_pooling_manual_values_and_independent_feature_scaling(self):
        node_h = torch.tensor([[1., 5.], [3., 1.], [2., 9.]])
        energy = torch.tensor([1., 3., 4.])
        batch = torch.tensor([0, 0, 1])
        summary = torch.tensor([[10., 11.], [12., 13.]])
        pooled = pool_nodes(node_h, batch, energy, summary)
        expected = torch.tensor([[2., 3., 3., 5., 2.5, 2., 10., 11.],
                                 [2., 9., 2., 9., 2., 9., 12., 13.]])
        torch.testing.assert_close(pooled, expected)
        # Demonstrate the method change: standardized logE softmax is different.
        standardized_loge = torch.log(energy[:2]) / 2
        wrong = (torch.softmax(standardized_loge, dim=0)[:, None] * node_h[:2]).sum(0)
        self.assertFalse(torch.allclose(wrong, pooled[0, 4:6]))

    def test_zero_total_energy_is_rejected(self):
        with self.assertRaises(ValueError):
            pool_nodes(torch.ones(2, 64), torch.zeros(2, dtype=torch.long),
                       torch.zeros(2), torch.ones(1, 2))


class RealGravNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(111)
        try:
            self.model = MultiSpaceEncoder()
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_shapes_real_backbone_and_all_module_gradients(self):
        first = Batch.from_data_list([sample_graph(), sample_graph(15), sample_graph(10)])
        second = Batch.from_data_list([sample_graph(), sample_graph(15), sample_graph(10)])
        out1, out2 = self.model(first), self.model(second)
        self.assertEqual(self.model.backbone.pool(first).shape, (3, 194))
        self.assertEqual(tuple(out1["h"]), SPACE_ORDER)
        self.assertEqual(torch.cat(list(out1["h"].values()), dim=-1).shape, (3, 320))
        for name in SPACE_ORDER:
            self.assertEqual(out1["h"][name].shape, (3, 64))
            self.assertEqual(out1["z"][name].shape, (3, 32))
        for name, size in TASK_DIMS.items():
            self.assertEqual(out1["pred"][name].shape, (3, size))
        targets = {name: torch.randn(3, size) for name, size in TASK_DIMS.items()}
        targets["eta"] = torch.softmax(targets["eta"], dim=-1)
        objective = MultiTaskObjective()
        parameters = list(self.model.parameters()) + list(objective.parameters())
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "weight_decay": 1e-4},
            {"params": objective.parameters(), "weight_decay": 0},
        ], lr=0.001)
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        loss = objective(out1, out2, targets, targets)["loss"]
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in parameters))
        for module in [self.model.backbone, *self.model.heads.values(),
                       *self.model.projectors.values(), *self.model.readouts.values()]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        optimizer.step()
        self.assertTrue(all(float(p.abs().sum()) > 0 for p in objective.parameters()))

    def test_one_view_encode_and_no_cross_graph_messages(self):
        self.model.eval()
        one, two = sample_graph(), sample_graph(17)
        alone = self.model.encode(one)
        batched = self.model.encode(Batch.from_data_list([one, two]))
        with torch.no_grad():
            two.x += 100
            two.energy *= 5
            two.summary += 200
        changed = self.model.encode(Batch.from_data_list([one, two]))
        for name in SPACE_ORDER:
            torch.testing.assert_close(alone[name], batched[name][:1], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(batched[name][:1], changed[name][:1], atol=2e-6, rtol=2e-5)
        self.assertEqual(alone["general"].shape, (1, 64))

    def test_target_mutation_cannot_change_encoder_output(self):
        graph = sample_graph()
        targets = {name: torch.randn(1, size) for name, size in TASK_DIMS.items()}
        before = self.model(graph)
        for target in targets.values():
            target.add_(1000)
        after = self.model(graph)
        for name in SPACE_ORDER:
            torch.testing.assert_close(before["h"][name], after["h"][name], rtol=0, atol=0)

    def test_small_graph_native_neighbors_do_not_change_other_graphs_k(self):
        from torch_cluster import knn

        self.model.eval()
        large = sample_graph(20)
        large_alone = self.model.encode(large)
        for nodes in (1, 2, 7, 8):
            small = sample_graph(nodes)
            small_alone = self.model.encode(small)
            batch = Batch.from_data_list([small, large])
            together = self.model.encode(batch)
            edges = knn(batch.x, batch.x, 8, batch.batch, batch.batch)
            counts = torch.bincount(edges[0])
            self.assertEqual(counts[:nodes].tolist(), [min(nodes, 8)] * nodes)
            self.assertEqual(counts[nodes:].tolist(), [8] * 20)
            for name in SPACE_ORDER:
                torch.testing.assert_close(small_alone[name], together[name][:1],
                                           atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(large_alone[name], together[name][1:],
                                           atol=2e-6, rtol=2e-5)
        for layer in (self.model.backbone.conv1, self.model.backbone.conv2, self.model.backbone.conv3):
            self.assertEqual(layer.k, 8)

    def test_original_node_encoder_operations_are_preserved(self):
        source = Path(__file__).resolve().parents[1] / "src/models/gnn.py"
        parsed = ast.parse(source.read_text())
        original = next(node for node in parsed.body
                        if isinstance(node, ast.ClassDef) and node.name == "GravNetEncoder")
        namespace = {"nn": nn, "F": F, "torch": torch, "GravNetConv": GravNetConv,
                     "softmax": softmax, "global_mean_pool": global_mean_pool,
                     "global_max_pool": global_max_pool, "global_add_pool": global_add_pool}
        exec(compile(ast.Module(body=[original], type_ignores=[]), str(source), "exec"), namespace)
        old = namespace["GravNetEncoder"](in_features=3)
        backbone = self.model.backbone
        compatible = {name: tensor for name, tensor in old.state_dict().items()
                      if name in backbone.state_dict()}
        backbone.load_state_dict(compatible, strict=True)
        captured = []
        hook = old.proj.register_forward_hook(lambda module, args, output: captured.append(output))
        graph = Batch.from_data_list([sample_graph(), sample_graph(15)])
        old(graph)
        hook.remove()
        torch.testing.assert_close(backbone(graph), captured[0], rtol=0, atol=0)

    def test_single_modes_share_194_input_and_no_aux_modes_have_no_readouts(self):
        for mode in ("single_cosine", "single_anisotropic", "five_anisotropic_no_aux"):
            model = MultiSpaceEncoder(mode=mode)
            out = model(sample_graph())
            self.assertEqual(model.pooled_dim, 194)
            self.assertEqual(out["pred"], {})
            self.assertEqual(len(out["h"]), 1 if mode.startswith("single_") else 5)


if __name__ == "__main__":
    unittest.main()
