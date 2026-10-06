"""Parte 1: verificacao pelo problema de Kepler."""

from __future__ import annotations

import argparse
import math

import numpy as np

from verificacao_kepler import INTEGRATORS, KeplerModel, circular_error, observed_order


def elliptic_initial_state(semi_major_axis: float, eccentricity: float) -> np.ndarray:
    """Estado no periastro para uma orbita de Kepler com mu=1."""
    radius = semi_major_axis * (1.0 - eccentricity)
    speed = math.sqrt((1.0 + eccentricity) / (semi_major_axis * (1.0 - eccentricity)))
    return np.array([radius, 0.0, 0.0, 0.0, speed, 0.0], dtype=np.float64)


def orbital_invariants(state: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    position = state[:3]
    velocity = state[3:]
    energy = 0.5 * np.dot(velocity, velocity) - 1.0 / np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    semi_major_axis = -1.0 / (2.0 * energy)
    eccentricity = math.sqrt(1.0 + 2.0 * energy * np.dot(angular_momentum, angular_momentum))
    return energy, angular_momentum, semi_major_axis, eccentricity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=1.0)
    arguments = parser.parse_args()

    print("Tabela de ordem para a orbita circular")
    print("metodo       erro radial       p_obs")
    for name, step_function in INTEGRATORS.items():
        print(f"{name:<12} {circular_error(step_function, arguments.radius, arguments.dt):.6e}   {observed_order(step_function, arguments.radius, arguments.dt):.4f}")

    print("\nInvariantes analiticos no periastro")
    for eccentricity in (0.0, 0.5, 0.9):
        state = elliptic_initial_state(1.0, eccentricity)
        energy, angular_momentum, semi_major_axis, measured_eccentricity = orbital_invariants(state)
        print(f"e={eccentricity:.1f}: E={energy:.6f}, |L|={np.linalg.norm(angular_momentum):.6f}, a={semi_major_axis:.6f}, e={measured_eccentricity:.6f}")


if __name__ == "__main__":
    main()
