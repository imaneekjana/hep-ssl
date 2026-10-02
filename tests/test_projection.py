import json
import unittest

import numpy as np

from src.data.projection import GridSpec, energy_summary, project_hits


def make_hits(eta, phi, energy):
    eta, phi, energy = np.broadcast_arrays(eta, phi, energy)
    radius = np.full(eta.size, 1000.0)
    return np.column_stack((
        radius * np.cos(phi.ravel()),
        radius * np.sin(phi.ravel()),
        radius * np.sinh(eta.ravel()),
        energy.ravel(),
    ))


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.grid = GridSpec.uniform(-2.0, 2.0, n_eta=4, n_phi=8)

    def test_grid_json_round_trip(self):
        payload = json.loads(json.dumps(self.grid.to_dict()))
        self.assertEqual(GridSpec.from_dict(payload), self.grid)
        self.assertEqual(self.grid.shape, (8, 4))
        self.assertEqual(payload["axis_order"], ["phi", "eta"])

    def test_sum_then_log_and_permutation(self):
        hits = make_hits([-0.8, -0.7], [0.1, 0.2], [2.0, 3.0])
        out = project_hits(hits, self.grid)
        self.assertEqual(out.node_features.shape, (1, 3))
        np.testing.assert_allclose(out.node_energy, [5.0])
        np.testing.assert_allclose(out.node_features[:, 2], [np.log(5.0 + 1e-6)])
        np.testing.assert_allclose(
            project_hits(hits[::-1], self.grid).energy_grid, out.energy_grid
        )

    def test_axis_order_and_fixed_centers(self):
        out = project_hits(make_hits(-0.2, 0.2, 7.0), self.grid)
        self.assertEqual(out.energy_grid.shape, (8, 4))
        self.assertEqual(out.energy_grid[4, 1], 7.0)
        np.testing.assert_allclose(out.node_features[0, :2], [-0.5, np.pi / 8])
        other = project_hits(make_hits(-0.9, 0.1, 2.0), self.grid)
        np.testing.assert_allclose(
            out.node_features[:, :2], other.node_features[:, :2]
        )

    def test_phi_endpoint_wrap(self):
        hits = np.array([[-1.0, 0.0, 0.0, 2.0], [-1.0, -0.0, 0.0, 3.0]])
        out = project_hits(hits, self.grid)
        self.assertEqual(out.energy_grid[0, 2], 5.0)
        self.assertEqual(np.count_nonzero(out.energy_grid), 1)

    def test_integer_bin_rotation(self):
        phi = self.grid.phi_centers
        hits = make_hits(-0.5, phi, np.arange(1.0, 9.0))
        shifted = make_hits(-0.5, phi + 2 * np.pi / 8, np.arange(1.0, 9.0))
        np.testing.assert_allclose(
            project_hits(shifted, self.grid).energy_grid,
            np.roll(project_hits(hits, self.grid).energy_grid, 1, axis=0),
        )

    def test_dropped_energy_accounting(self):
        good_and_outside = make_hits([0.2, 3.0], [0.2, 0.2], [2.0, 3.0])
        bad_geometry = np.array([[0, 0, 1, 5], [np.nan, 1, 1, 7]])
        out = project_hits(np.vstack((good_and_outside, bad_geometry)), self.grid)
        d = out.diagnostics
        self.assertEqual(d["n_invalid_geometry_hits"], 2)
        self.assertEqual(d["n_outside_eta_hits"], 1)
        self.assertEqual(d["retained_energy_gev"], 2.0)
        self.assertEqual(d["invalid_geometry_energy_gev"], 12.0)
        self.assertEqual(d["outside_eta_energy_gev"], 3.0)
        self.assertAlmostEqual(d["dropped_energy_fraction"], 15.0 / 17.0)

    def test_cutoff_after_sum_and_independent_paths(self):
        grid = GridSpec.uniform(-2, 2, n_eta=4, n_phi=8, cell_cutoff_gev=1.0)
        reference = make_hits([-0.8, -0.7, 0.2], [0.1, 0.2, 0.1], [0.6, 0.6, 0.5])
        observed = reference.copy()
        observed[:2, 3] = 0.0
        ref = project_hits(reference, grid)
        obs = project_hits(observed, grid)
        np.testing.assert_allclose(ref.node_energy, [1.2])
        self.assertEqual(ref.diagnostics["below_cutoff_energy_gev"], 0.5)
        self.assertTrue(obs.is_empty)
        self.assertFalse(ref.is_empty)

    def test_summary_scaling_and_no_energy_alias(self):
        out = project_hits(make_hits([-0.5, 0.5], [0.2, 1.0], [2.0, 3.0]), self.grid)
        expected = np.log([5.0, 5.0 / np.cosh(0.5)])
        np.testing.assert_allclose(energy_summary(out.energy_grid, self.grid), expected)
        np.testing.assert_allclose(
            energy_summary(3 * out.energy_grid, self.grid),
            expected + np.log(3.0),
        )
        out.node_features[:, 2] = 999.0
        np.testing.assert_allclose(out.node_energy, [2.0, 3.0])
        self.assertEqual(out.energy_grid.sum(), 5.0)

    def test_empty_event_without_fake_nodes(self):
        out = project_hits(np.empty((0, 4)), self.grid)
        self.assertTrue(out.is_empty)
        self.assertEqual(out.node_features.shape, (0, 3))
        self.assertIsNone(out.diagnostics["dropped_energy_fraction"])
        with self.assertRaises(ValueError):
            energy_summary(out.energy_grid, self.grid)

    def test_invalid_inputs_rejected(self):
        for energy in (-1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                project_hits(make_hits(0.2, 0.2, energy), self.grid)
        with self.assertRaises(ValueError):
            GridSpec.uniform(2.0, -2.0)
        with self.assertRaises(ValueError):
            GridSpec.uniform(-2, 2, n_eta=0)


if __name__ == "__main__":
    unittest.main()
