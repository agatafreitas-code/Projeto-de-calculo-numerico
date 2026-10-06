"""Parte 6: campo vetorizado e medicao de desempenho."""

from __future__ import annotations

import argparse
import time

import numpy as np

from projeto import MASS, POSITIONS


def vectorized_acceleration(positions: np.ndarray, eps: float = 0.18) -> np.ndarray:
    """Calcula aceleracao para P particulas simultaneamente, forma (P, 3)."""
    difference = positions[:, None, :] - POSITIONS[None, :, :]
    distance_squared = np.sum(difference * difference, axis=2) + eps**2
    contributions = -MASS[None, :, None] * difference / distance_squared[:, :, None] ** 1.5
    return np.sum(contributions, axis=1)


def active_particle_demo(particles: int, duration: float = 5.0, dt: float = 0.05, eps: float = 0.18, gamma: float = 0.15, rcap: float = 0.22, rescape: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Avanca particulas simultaneamente e remove as que terminaram."""
    rng = np.random.default_rng(1234)
    states = np.zeros((particles, 6), dtype=np.float64)
    states[:, :2] = rng.uniform(-3.0, 3.0, (particles, 2))
    active_fraction = []
    elapsed = 0.0
    while elapsed < duration and len(states):
        difference = states[:, None, :3] - POSITIONS[None, :, :]
        distance_squared = np.sum(difference * difference, axis=2) + eps**2
        acceleration = np.sum(-MASS[None, :, None] * difference / distance_squared[:, :, None] ** 1.5, axis=1) - gamma * states[:, 3:]
        states[:, :3] += dt * states[:, 3:]
        states[:, 3:] += dt * acceleration
        distances = np.linalg.norm(states[:, :3], axis=1)
        captured = np.any(np.linalg.norm(states[:, None, :3] - POSITIONS[None, :, :], axis=2) < rcap, axis=1)
        energy = 0.5 * np.sum(states[:, 3:] ** 2, axis=1) - np.sum(MASS[None, :] / np.sqrt(np.sum((states[:, None, :3] - POSITIONS[None, :, :]) ** 2, axis=2) + eps**2), axis=1)
        escaped = (distances > rescape) & (np.sum(states[:, :3] * states[:, 3:], axis=1) > 0.0) & (energy > 0.0)
        states = states[~(captured | escaped)]
        elapsed += dt
        active_fraction.append(len(states) / particles)
    return np.arange(len(active_fraction)) * dt, np.asarray(active_fraction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=20)
    arguments = parser.parse_args()

    positions = np.zeros((arguments.particles, 3), dtype=np.float64)
    positions[:, :2] = np.random.default_rng(1234).uniform(-3.0, 3.0, (arguments.particles, 2))
    started = time.perf_counter()
    for _ in range(arguments.repetitions):
        vectorized_acceleration(positions)
    elapsed = time.perf_counter() - started
    evaluations = arguments.particles * arguments.repetitions
    print(f"particulas={arguments.particles}")
    print(f"avaliacoes={evaluations}")
    print(f"tempo_total={elapsed:.6f}s")
    print(f"avaliacoes_por_segundo={evaluations / elapsed:.3e}")
    times, active_fraction = active_particle_demo(arguments.particles, arguments.repetitions * 0.05)
    print(f"fracao_ativa_final={active_fraction[-1] if len(active_fraction) else 0.0:.6e}")


if __name__ == "__main__":
    main()
