"""Disorder ensembles and finite-temperature convolution."""

from __future__ import annotations

import numpy as np


def anderson_disorder(shape: tuple[int, int], strength: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-0.5 * strength, 0.5 * strength, size=shape)


def ensemble_summary(samples: np.ndarray) -> dict[str, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0]
    if n < 2:
        raise ValueError("At least two disorder samples are required")
    return {
        "mean": np.mean(samples, axis=0),
        "std": np.std(samples, axis=0, ddof=1),
        "sem": np.std(samples, axis=0, ddof=1) / np.sqrt(n),
        "median": np.median(samples, axis=0),
        "q05": np.quantile(samples, 0.05, axis=0),
        "q25": np.quantile(samples, 0.25, axis=0),
        "q75": np.quantile(samples, 0.75, axis=0),
        "q95": np.quantile(samples, 0.95, axis=0),
    }


def fermi_derivative(energies: np.ndarray, ef: float, kbt: float) -> np.ndarray:
    if kbt <= 0:
        raise ValueError("kBT must be positive")
    x = np.clip((np.asarray(energies) - ef) / (2.0 * kbt), -350.0, 350.0)
    return 0.25 / kbt / np.cosh(x) ** 2


def finite_temperature_conductance(
    energies: np.ndarray,
    transmission: np.ndarray,
    ef_values: np.ndarray,
    kbt: float,
) -> np.ndarray:
    """Return dimensionless G/(e^2/h) by numerical Fermi-window convolution."""
    energies = np.asarray(energies, dtype=float)
    transmission = np.asarray(transmission, dtype=float)
    return np.asarray([
        np.trapezoid(transmission * fermi_derivative(energies, ef, kbt), energies)
        for ef in np.atleast_1d(ef_values)
    ])


def finite_temperature_average(
    energies: np.ndarray,
    values: np.ndarray,
    ef: float,
    kbt: float,
) -> np.ndarray:
    """Convolve an energy-resolved scalar, vector, or matrix with -df/dE.

    The first axis of ``values`` must correspond to ``energies``.  No artificial
    normalization is applied: callers can use :func:`fermi_window_mass` to
    verify that the sampled energy interval captures the Fermi window.
    """
    energies = np.asarray(energies, dtype=float)
    values = np.asarray(values)
    if values.shape[0] != energies.size:
        raise ValueError("The first values axis must match energies")
    weights = fermi_derivative(energies, ef, kbt)
    reshape = (energies.size,) + (1,) * (values.ndim - 1)
    return np.trapezoid(values * weights.reshape(reshape), energies, axis=0)


def fermi_window_mass(energies: np.ndarray, ef: float, kbt: float) -> float:
    """Return the captured integral of -df/dE over a finite energy grid."""
    energies = np.asarray(energies, dtype=float)
    return float(np.trapezoid(fermi_derivative(energies, ef, kbt), energies))
