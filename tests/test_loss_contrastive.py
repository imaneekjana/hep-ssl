import io
import unittest

import torch
from torch.nn import functional as F

from src.losses.contrastive import AnInfoNCE, CosineInfoNCE, positive_indices
from src.losses.multitask import MultiTaskObjective, SPACE_ORDER


class ContrastiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)

    def test_identity_matches_cosine_loss_and_projection_gradients(self):
        for batch_size in (2, 5):
            z1 = torch.randn(batch_size, 32, dtype=torch.float64, requires_grad=True)
            z2 = torch.randn(batch_size, 32, dtype=torch.float64, requires_grad=True)
            ani = AnInfoNCE(32, tau=0.17).double()
            loss_an = ani(z1, z2)
            grad_an = torch.autograd.grad(loss_an, (z1, z2), retain_graph=True)
            loss_cos = CosineInfoNCE(tau=0.17)(z1, z2)
            grad_cos = torch.autograd.grad(loss_cos, (z1, z2))
            self.assertEqual(loss_an.dtype, torch.float64)
            torch.testing.assert_close(loss_an, loss_cos, atol=1e-12, rtol=1e-12)
            for one, two in zip(grad_an, grad_cos):
                torch.testing.assert_close(one, two, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(ani.diagonal(), torch.ones(32, dtype=torch.float64))

    def test_nonuniform_full_distance_matches_direct_reference(self):
        metric = AnInfoNCE(32, tau=0.11).double()
        with torch.no_grad():
            metric.raw_lambda.copy_(torch.linspace(-2, 2, 32))
        z = torch.randn(8, 32, dtype=torch.float64, requires_grad=True)
        u = F.normalize(z, dim=-1)
        lam = metric.diagonal()
        reference = ((u[:, None] - u[None, :]).square() * lam).sum(-1)
        distances = metric.pairwise_distances(u)
        torch.testing.assert_close(distances, reference, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(distances, distances.T, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(distances.diagonal(), torch.zeros(8, dtype=torch.float64),
                                   atol=1e-14, rtol=0)
        scores = -reference / (2 * 0.11)
        scores = scores.masked_fill(torch.eye(8, dtype=torch.bool), -torch.inf)
        ref_loss = F.cross_entropy(scores, torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))
        torch.testing.assert_close(metric(z[:4], z[4:]), ref_loss, atol=1e-12, rtol=1e-12)
        # Weighted inner product misses the candidate-dependent norm term.
        wrong = ((u * lam) @ u.T / 0.11).masked_fill(torch.eye(8, dtype=torch.bool), -torch.inf)
        self.assertGreater(abs(float(F.cross_entropy(wrong, positive_indices(4)) - ref_loss)), 1e-4)

    def test_metric_has_gradient_updates_and_constant_trace(self):
        metric = AnInfoNCE(32, tau=0.07)
        optimizer = torch.optim.Adam(metric.parameters(), lr=0.01, weight_decay=0)
        one, two = torch.randn(4, 32), torch.randn(4, 32)
        metric(one, two).backward()
        self.assertGreater(float(metric.raw_lambda.grad.abs().sum()), 0)
        optimizer.step()
        self.assertGreater(float(metric.raw_lambda.abs().sum()), 0)
        self.assertAlmostEqual(float(metric.diagonal().sum()), 32.0, places=5)
        self.assertTrue(torch.all(metric.diagonal() > 0))

    def test_cpu_autocast_and_low_precision_inputs_use_fp32_core(self):
        metric = AnInfoNCE(32, tau=0.07)
        for dtype in (torch.float16, torch.bfloat16):
            one = torch.randn(3, 32).to(dtype).requires_grad_()
            two = torch.randn(3, 32).to(dtype).requires_grad_()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss = metric(one, two)
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            torch.testing.assert_close(loss, metric(one.float(), two.float()))
            loss.backward()
            self.assertTrue(torch.isfinite(one.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_uses_fp32(self):
        metric = AnInfoNCE(32, tau=0.07).cuda()
        one = torch.randn(4, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        two = torch.randn(4, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = metric(one, two)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(one.grad).all())

    def test_pair_indices_and_invalid_batches(self):
        self.assertEqual(positive_indices(3).tolist(), [3, 4, 5, 0, 1, 2])
        with self.assertRaises(ValueError):
            AnInfoNCE()(torch.randn(1, 32), torch.randn(1, 32))
        with self.assertRaises(ValueError):
            AnInfoNCE()(torch.randn(3, 16), torch.randn(3, 16))
        with self.assertRaises(ValueError):
            CosineInfoNCE(0)

    def test_metric_underflow_is_reported(self):
        metric = AnInfoNCE()
        with torch.no_grad():
            metric.raw_lambda[0] = 1e6
        with self.assertRaises(FloatingPointError):
            metric.diagonal()

    def test_anisotropic_checkpoint_roundtrip_preserves_learned_metric(self):
        config = {"proj_dim": 32, "tau": 0.19}
        metric = AnInfoNCE(**config).double()
        optimizer = torch.optim.Adam(metric.parameters(), lr=0.05, weight_decay=0)
        one, two = torch.randn(4, 32, dtype=torch.float64), torch.randn(4, 32, dtype=torch.float64)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            metric(one, two).backward()
            optimizer.step()
        self.assertGreater(float(metric.raw_lambda.detach().abs().sum()), 0)
        buffer = io.BytesIO()
        torch.save({"config": config, "objective_state": metric.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
        restored = AnInfoNCE(**checkpoint["config"]).double()
        restored.load_state_dict(checkpoint["objective_state"], strict=True)
        self.assertEqual(restored.tau, config["tau"])
        torch.testing.assert_close(restored.raw_lambda, metric.raw_lambda, rtol=0, atol=0)
        torch.testing.assert_close(restored.diagonal(), metric.diagonal(), rtol=0, atol=0)
        torch.testing.assert_close(restored(one, two), metric(one, two), rtol=0, atol=0)
        expected_grad = torch.autograd.grad(metric(one, two), metric.raw_lambda)[0]
        actual_grad = torch.autograd.grad(restored(one, two), restored.raw_lambda)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


class MultiTaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.targets = {
            "energy": torch.randn(3, 2), "eta": torch.softmax(torch.randn(3, 8), -1),
            "phi": torch.randn(3, 4), "local": torch.randn(3, 4),
        }

    def output(self):
        return {"z": {name: torch.randn(3, 32, requires_grad=True) for name in SPACE_ORDER},
                "pred": {name: torch.randn_like(target, requires_grad=True)
                         for name, target in self.targets.items()}}

    def test_loss_means_views_tasks_and_spaces_and_targets_detach(self):
        objective = MultiTaskObjective(gamma=2)
        out1, out2 = self.output(), self.output()
        targets2 = {name: target.clone().requires_grad_() for name, target in self.targets.items()}
        result = objective(out1, out2, self.targets, targets2)
        expected_cl = torch.stack([result[f"cl/{s}"] for s in SPACE_ORDER]).mean()
        expected_phys = torch.stack([result[f"physics/{s}"] for s in self.targets]).mean()
        torch.testing.assert_close(result["loss"], expected_cl + 2 * expected_phys)
        result["loss"].backward()
        self.assertTrue(all(t.grad is None for t in targets2.values()))
        self.assertTrue(all(float(p.grad.abs().sum()) > 0 for p in objective.parameters()))
        self.assertEqual(len({id(p) for p in objective.parameters()}), 5)

    def test_changing_one_readout_changes_only_corresponding_auxiliary(self):
        objective = MultiTaskObjective()
        out1, out2 = self.output(), self.output()
        first = objective(out1, out2, self.targets, self.targets)
        out1["pred"]["energy"] = out1["pred"]["energy"] + 5
        second = objective(out1, out2, self.targets, self.targets)
        self.assertNotEqual(float(first["physics/energy"]), float(second["physics/energy"]))
        for name in ("eta", "phi", "local"):
            torch.testing.assert_close(first[f"physics/{name}"], second[f"physics/{name}"])
        torch.testing.assert_close(first["contrastive"], second["contrastive"])

    def test_gamma_zero_exactly_recovers_five_head_contrastive(self):
        out1, out2 = self.output(), self.output()
        zero = MultiTaskObjective(gamma=0)(out1, out2)
        no_aux = MultiTaskObjective(mode="five_anisotropic_no_aux")(out1, out2)
        torch.testing.assert_close(zero["loss"], no_aux["loss"])
        self.assertEqual(float(zero["physics"]), 0.0)
        zero["loss"].backward()
        self.assertTrue(all(p.grad is None for p in out1["pred"].values()))

    def test_single_modes_and_eta_zero_probability_bins(self):
        one, two = self.output(), self.output()
        cosine = MultiTaskObjective(mode="single_cosine")
        anisotropic = MultiTaskObjective(mode="single_anisotropic")
        self.assertEqual(list(cosine.parameters()), [])
        self.assertEqual(len(list(anisotropic.parameters())), 1)
        torch.testing.assert_close(cosine(one, two)["loss"], anisotropic(one, two)["loss"])
        self.targets["eta"][:] = 0
        self.targets["eta"][:, 0] = 1
        result = MultiTaskObjective()(one, two, self.targets, self.targets)
        self.assertTrue(torch.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
