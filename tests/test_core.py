import unittest

import numpy as np

from skyrmion_transport.bloch import (
    bloch_hamiltonian, fhs_chern_subspace,
    hermiticity_error,
    slowest_strip_evanescent_mode,
    slowest_strip_mode_boundary_green,
    strip_bloch_multipliers,
    strip_bloch_multipliers_boundary_green,
    uniform_folded_energies,
)
from skyrmion_transport.multiterminal import (
    four_terminal_hall_observables, four_terminal_observables,
    landauer_buttiker_matrix, solve_voltage_probes,
    scattering_matrix_from_green, standard_four_contacts,
    terminal_currents_channel_reference, terminal_currents_from_scattering,
    transmission_matrix,
    transmission_matrix_sparse,
)
from skyrmion_transport.leads import uniform_lead_channel_summary
from skyrmion_transport.hall import contact_spectral_maps
from skyrmion_transport.statistics import finite_temperature_average, fermi_window_mass
from skyrmion_transport.textures import (
    TextureKind,
    lattice_topological_charge,
    make_array_texture,
    make_cellwise_disordered_array_texture,
    make_skyrmionium_wall,
    make_texture,
    max_norm_error,
    topological_charge_profile_x,
    windowed_topological_charge,
)
from skyrmion_transport.transport import (
    clean_lead_modes,
    direct_inverse_transmission,
    fit_exponential_length,
    paired_prefix_transmission,
    two_terminal_transmission,
)


class TextureTests(unittest.TestCase):
    def test_norm_and_topology(self):
        for kind in TextureKind:
            m = make_texture(kind, 30, 30, 8)
            self.assertLess(max_norm_error(m), 1e-12)
        q0, _ = lattice_topological_charge(make_texture("skyrmionium_q_zero", 30, 30, 8))
        q0_quintic, _ = lattice_topological_charge(
            make_texture("skyrmionium_q_zero_quintic", 30, 30, 8)
        )
        qdw, _ = lattice_topological_charge(
            make_texture("nonwinding_double_wall_q_zero", 30, 30, 8)
        )
        sky = make_texture("skyrmionium_q_zero", 30, 30, 8)
        double_wall = make_texture("nonwinding_double_wall_q_zero", 30, 30, 8)
        self.assertLess(np.max(np.abs(sky[..., 2] - double_wall[..., 2])), 1e-14)
        qp, _ = lattice_topological_charge(make_texture("skyrmion_q_plus", 30, 30, 8))
        qm, _ = lattice_topological_charge(make_texture("skyrmion_q_minus", 30, 30, 8))
        self.assertLess(abs(q0), 1e-10)
        self.assertLess(abs(q0_quintic), 1e-10)
        self.assertLess(abs(qdw), 1e-10)
        self.assertAlmostEqual(qp, 1.0, places=10)
        self.assertAlmostEqual(qm, -1.0, places=10)

    def test_padding_profile_and_wall_counterfactuals(self):
        base = make_array_texture("skyrmionium_q_zero", 18, 1, 1, 8)
        self.assertTrue(np.array_equal(
            base,
            make_array_texture("skyrmionium_q_zero", 18, 1, 1, 8, padding=(0, 0)),
        ))
        padded = make_array_texture(
            "skyrmionium_q_zero", 18, 1, 1, 8, padding=(2, 3)
        )
        self.assertEqual(padded.shape, (22, 24, 3))
        self.assertLess(max_norm_error(padded), 1e-12)
        q, _ = lattice_topological_charge(padded)
        density, profile = topological_charge_profile_x(padded)
        self.assertAlmostEqual(q, 0.0, places=10)
        self.assertEqual(profile.shape, (padded.shape[0] - 1,))
        self.assertAlmostEqual(np.sum(density), np.sum(profile), places=14)
        self.assertAlmostEqual(
            windowed_topological_charge(padded, 0, padded.shape[0]), q, places=14
        )
        q_inner, _ = lattice_topological_charge(
            make_skyrmionium_wall("inner", 30, 30, 8)
        )
        q_outer, _ = lattice_topological_charge(
            make_skyrmionium_wall("outer", 30, 30, 8)
        )
        self.assertAlmostEqual(abs(q_inner), 1.0, places=10)
        self.assertAlmostEqual(abs(q_outer), 1.0, places=10)
        self.assertAlmostEqual(q_inner + q_outer, 0.0, places=10)

    def test_cellwise_texture_disorder_interface(self):
        perfect = make_array_texture("skyrmionium_q_zero", 10, 2, 2, 4.0)
        rebuilt = make_cellwise_disordered_array_texture(
            "skyrmionium_q_zero", 10, 2, 2, 4.0
        )
        self.assertTrue(np.array_equal(perfect, rebuilt))
        radii = np.array([[0.2, -0.2], [0.1, -0.1]])
        offsets = np.zeros((2, 2, 2))
        offsets[0, 0] = (0.2, -0.1)
        vacancies = np.array([[False, True], [False, False]])
        disordered = make_cellwise_disordered_array_texture(
            "skyrmionium_q_zero",
            10,
            2,
            2,
            4.0,
            radius_offsets=radii,
            center_offsets=offsets,
            vacancies=vacancies,
        )
        self.assertLess(max_norm_error(disordered), 1e-12)
        self.assertTrue(np.all(disordered[:10, 10:20, 2] == 1.0))


class BlochTests(unittest.TestCase):
    def test_uniform_folded_dispersion(self):
        A, J, t = 3, 1.5, 1.0
        cell = make_texture("uniform", A, A, 1)
        kx, ky = 0.173, -0.221
        H = bloch_hamiltonian(cell, kx, ky, J, t)
        self.assertLess(hermiticity_error(H), 1e-12)
        expected = uniform_folded_energies(A, kx, ky, J, t)
        self.assertLess(np.max(np.abs(np.linalg.eigvalsh(H) - expected)), 1e-11)

    def test_uniform_chern_is_zero(self):
        cell = make_texture("uniform", 3, 3, 1)
        chern, flux = fhs_chern_subspace(cell, n_occ=9, nk=5, J=5.0, t=1.0)
        self.assertLess(abs(chern), 1e-12)
        self.assertEqual(flux.shape, (5, 5))

    def test_uniform_chain_complex_band(self):
        texture = make_texture("uniform", 1, 1, 1)
        multipliers = strip_bloch_multipliers(texture, 3.0, J=0.0, t=1.0)
        expected_modulus = (3.0 - np.sqrt(5.0)) / 2.0
        inside = np.abs(multipliers)[np.abs(multipliers) < 1.0]
        self.assertEqual(inside.size, 2)  # two degenerate spin channels
        self.assertLess(np.max(np.abs(np.abs(inside) - expected_modulus)), 1e-12)
        mode = slowest_strip_evanescent_mode(texture, 3.0, J=0.0, t=1.0)
        expected_kappa = -np.log(expected_modulus)
        self.assertFalse(mode["has_propagating_mode"])
        self.assertAlmostEqual(mode["kappa_per_a"], expected_kappa, places=12)
        self.assertAlmostEqual(
            mode["xi_transmission_a"], 1.0 / (2.0 * expected_kappa), places=12
        )

    def test_boundary_green_complex_band(self):
        texture = make_texture("uniform", 2, 1, 1)
        values, residuals = strip_bloch_multipliers_boundary_green(
            texture, 3.0, J=0.0, t=1.0
        )
        expected_atomic_modulus = (3.0 - np.sqrt(5.0)) / 2.0
        expected_cell_modulus = expected_atomic_modulus**2
        inside = np.abs(values)[np.abs(values) < 1.0]
        self.assertGreaterEqual(inside.size, 2)
        self.assertLess(np.min(np.abs(inside - expected_cell_modulus)), 1e-11)
        self.assertLess(np.max(residuals), 1e-8)
        mode = slowest_strip_mode_boundary_green(texture, 3.0, J=0.0, t=1.0)
        expected_kappa = -np.log(expected_atomic_modulus)
        self.assertFalse(mode["has_propagating_mode"])
        self.assertAlmostEqual(mode["kappa_per_a"], expected_kappa, places=11)
        self.assertLess(mode["reciprocal_pair_error"], 1e-10)


class TransportTests(unittest.TestCase):
    def test_rgf_matches_full_inverse(self):
        texture = make_texture("skyrmionium_q_zero", 6, 4, 1.5)
        E, J, t, eta = 0.17, 1.5, 1.0, 1e-7
        rgf = two_terminal_transmission(texture, np.array([E]), J, t, eta=eta)[0]
        full = direct_inverse_transmission(texture, E, J, t, eta=eta)
        self.assertLess(abs(rgf - full), 1e-10)

    def test_clean_channel_count(self):
        texture = make_texture("uniform", 8, 6, 2)
        energies = np.array([-0.8, 0.0, 0.9])
        T = two_terminal_transmission(texture, energies, 1.5, 1.0, eta=1e-9)
        N = clean_lead_modes(energies, 6, 1.5, 1.0)
        self.assertLess(np.max(np.abs(T - N)), 2e-5)
        for energy, expected in zip(energies, N):
            summary = uniform_lead_channel_summary(energy, 6, 1.5, 1.0)
            self.assertEqual(summary["n_total"], expected)
            self.assertEqual(
                summary["polarization"],
                (summary["n_up"] - summary["n_down"]) / expected,
            )

    def test_paired_prefix_matches_separate_rgf(self):
        texture = make_array_texture("skyrmionium_q_zero", 6, 2, 1, 2.0)
        disorder = np.random.default_rng(123).uniform(-0.1, 0.1, texture.shape[:2])
        energies = np.array([-0.4, 0.17, 0.6])
        paired = paired_prefix_transmission(
            texture, (6, 12), energies, 1.5, 1.0, eta=1e-8,
            onsite_disorder=disorder,
        )
        for length in (6, 12):
            separate = two_terminal_transmission(
                texture[:length], energies, 1.5, 1.0, eta=1e-8,
                onsite_disorder=disorder[:length],
            )
            self.assertLess(np.max(np.abs(paired[length] - separate)), 1e-12)

    def test_exponential_fit_keeps_tiny_transmissions(self):
        lengths = np.array([18, 36, 72, 144])
        transmission = np.exp(-0.2 * lengths - 1.3)
        fit = fit_exponential_length(lengths, transmission)
        self.assertGreater(fit["r2"], 1 - 1e-12)
        self.assertAlmostEqual(fit["slope"], -0.2, places=12)


class LandauerButtikerTests(unittest.TestCase):
    def test_gauge_and_conservation(self):
        # A symmetric matrix is a simple physical, port-balanced test case.
        # Generic random T is invalid because a unitary scattering matrix also
        # constrains each lead's total incoming and outgoing transmission.
        T = np.array([[0, 1.1, 0.3, 0.4], [1.1, 0, 0.5, 0.2],
                      [0.3, 0.5, 0, 0.8], [0.4, 0.2, 0.8, 0]])
        L = landauer_buttiker_matrix(T)
        self.assertLess(np.max(np.abs(np.sum(L, axis=0))), 1e-14)
        V, I = solve_voltage_probes(T, {0: 0.5, 1: -0.5})
        self.assertLess(abs(I.sum()), 1e-13)
        self.assertLess(max(abs(I[2]), abs(I[3])), 1e-13)
        self.assertLess(np.max(np.abs(L @ (V + 3.7) - I)), 1e-13)

    def test_physical_four_terminal_symmetries(self):
        L, W, R = 10, 8, 2.5
        contacts = standard_four_contacts(L, W, 3)
        values = {}
        for kind in ("uniform", "skyrmion_q_plus", "skyrmion_q_minus"):
            m = make_texture(kind, L, W, R)
            T, diag = transmission_matrix(m, 0.2, 1.5, 1.0, contacts, eta=1e-9)
            obs = four_terminal_observables(T)
            self.assertLess(obs["current_conservation_error"], 1e-12)
            self.assertLess(obs["probe_current_error"], 1e-12)
            self.assertLess(obs["gauge_invariance_error"], 2e-7)
            values[kind] = obs["Rxy_h_over_e2"]
            blocks, spins, unitarity = scattering_matrix_from_green(diag)
            charge_s, spin_s = terminal_currents_from_scattering(blocks, spins, obs["voltages"])
            charge_ref, spin_ref = terminal_currents_channel_reference(
                blocks, spins, obs["voltages"]
            )
            self.assertLess(np.max(np.abs(charge_ref - charge_s)), 1e-10)
            self.assertLess(np.max(np.abs(spin_ref - spin_s)), 1e-10)
            self.assertLess(unitarity, 2e-7)
            self.assertLess(np.max(np.abs(charge_s - obs["currents"])), 2e-7)
            eq_charge, eq_spin = terminal_currents_from_scattering(blocks, spins, np.ones(4))
            self.assertLess(np.max(np.abs(eq_charge)), 2e-7)
            self.assertLess(np.max(np.abs(eq_spin)), 2e-7)
        self.assertLess(abs(values["uniform"]), 1e-10)
        self.assertLess(abs(values["skyrmion_q_plus"] + values["skyrmion_q_minus"]), 1e-10)

    def test_publication_hall_angle_definition(self):
        voltages = np.array([0.5, -0.5, -0.02, 0.03])
        charge = np.array([2.0, -2.0, 0.0, 0.0])
        spin = np.array([0.1, -0.1, -0.08, 0.12])
        result = four_terminal_hall_observables(
            voltages, charge, spin, source_channels=10
        )
        self.assertAlmostEqual(result["charge_hall_angle"], 0.05)
        self.assertAlmostEqual(result["spin_hall_angle"], 0.05)
        self.assertTrue(result["valid_hall_point"])

    def test_sparse_matches_dense(self):
        L, W = 8, 6
        m = make_texture("skyrmionium_q_zero", L, W, 2.0)
        contacts = standard_four_contacts(L, W, 2)
        dense, _ = transmission_matrix(m, -0.31, 1.5, 1.0, contacts)
        sparse, _ = transmission_matrix_sparse(m, -0.31, 1.5, 1.0, contacts)
        self.assertLess(np.max(np.abs(dense - sparse)), 1e-10)

    def test_probe_interface_coupling_scales_self_energy(self):
        L, W = 8, 6
        texture = make_texture("uniform", L, W, 2.0)
        full = standard_four_contacts(L, W, 2, probe_J=0.0)
        half = standard_four_contacts(
            L, W, 2, probe_J=0.0, probe_coupling=0.5
        )
        _, full_diag = transmission_matrix(
            texture, -0.31, 1.5, 1.0, full
        )
        _, half_diag = transmission_matrix(
            texture, -0.31, 1.5, 1.0, half
        )
        for probe in (2, 3):
            self.assertLess(np.max(np.abs(
                half_diag["sigmas"][probe] - 0.25 * full_diag["sigmas"][probe]
            )), 1e-13)
            self.assertLess(np.max(np.abs(
                half_diag["gammas"][probe] - 0.25 * full_diag["gammas"][probe]
            )), 1e-13)
        for longitudinal in (0, 1):
            self.assertLess(np.max(np.abs(
                half_diag["sigmas"][longitudinal]
                - full_diag["sigmas"][longitudinal]
            )), 1e-13)

    def test_zero_onsite_disorder_preserves_sparse_result(self):
        L, W = 8, 6
        m = make_texture("skyrmionium_q_zero", L, W, 2.0)
        contacts = standard_four_contacts(L, W, 2)
        clean, _ = transmission_matrix_sparse(m, -0.31, 1.5, 1.0, contacts)
        zero_disorder, _ = transmission_matrix_sparse(
            m, -0.31, 1.5, 1.0, contacts, onsite_disorder=np.zeros((L, W))
        )
        self.assertLess(np.max(np.abs(clean - zero_disorder)), 1e-13)

    def test_finite_temperature_matrix_convolution(self):
        energy = np.linspace(-0.2, 0.2, 4001)
        matrix = np.broadcast_to(np.eye(3), (energy.size, 3, 3))
        averaged = finite_temperature_average(energy, matrix, 0.0, 0.01)
        mass = fermi_window_mass(energy, 0.0, 0.01)
        self.assertGreater(mass, 1 - 1e-8)
        self.assertLess(np.max(np.abs(averaged - np.eye(3))), 1e-8)

    def test_sparse_contact_spectral_maps_are_finite_and_symmetric(self):
        L, W = 8, 6
        texture = make_texture("uniform", L, W, 2.0)
        contacts = standard_four_contacts(L, W, 2, probe_J=0.0)
        _, diagnostics = transmission_matrix_sparse(
            texture, 0.2, 1.5, 1.0, contacts, eta=1e-9
        )
        maps = contact_spectral_maps(diagnostics, (L, W))
        self.assertEqual(maps["ldos"].shape, (L, W))
        self.assertEqual(maps["source_injectivity"].shape, (L, W))
        self.assertEqual(maps["bond_current_x"].shape, (L - 1, W))
        self.assertEqual(maps["bond_current_y"].shape, (L, W - 1))
        for value in maps.values():
            self.assertTrue(np.all(np.isfinite(value)))
        self.assertLess(np.max(abs(maps["ldos"] - maps["ldos"][:, ::-1])), 1e-10)


if __name__ == "__main__":
    unittest.main()
