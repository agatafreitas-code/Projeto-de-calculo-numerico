"""Parte 4: estimativa do expoente de Lyapunov por Benettin."""

from __future__ import annotations

import argparse
import math
import numpy as np

from projeto import Parameters, right_hand_side


def benettin(initial_state: np.ndarray, parameters: Parameters, delta0: float = 1e-10, tau: float = 0.5, intervals: int = 100) -> float:
    """Estima lambda com RK4 e reescalonamento periodico da perturbacao."""
    state = initial_state.astype(np.float64, copy=True)
    rng = np.random.default_rng(42)
    direction = rng.standard_normal(len(state))
    direction = direction / np.linalg.norm(direction) * delta0
    perturbed = state + direction
    accumulated = 0.0
    steps = max(1, int(round(tau / parameters.max_step)))
    dt = tau / steps

    def advance(current: np.ndarray) -> np.ndarray:
        for _ in range(steps):
            slope_1 = right_hand_side(0.0, current, parameters)
            slope_2 = right_hand_side(0.0, current + 0.5 * dt * slope_1, parameters)
            slope_3 = right_hand_side(0.0, current + 0.5 * dt * slope_2, parameters)
            slope_4 = right_hand_side(0.0, current + dt * slope_3, parameters)
            current = current + dt * (slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4) / 6.0
        return current

    for _ in range(intervals):
        state = advance(state)
        perturbed = advance(perturbed)
        separation_vector = perturbed - state
        separation = np.linalg.norm(separation_vector)
        if separation == 0.0:
            return float("-inf")
        accumulated += np.log(separation / delta0)
        perturbed = state + delta0 * separation_vector / separation
    return accumulated / (intervals * tau)


def predictability_horizon(lambda_value: float, delta0: float = 2e-16, tolerance: float = 0.1) -> float:
    if lambda_value <= 0.0:
        return float("inf")
    return math.log(tolerance / delta0) / lambda_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--intervals", type=int, default=100)
    parser.add_argument("--boundary-x", type=float, default=0.2)
    parser.add_argument("--boundary-y", type=float, default=0.2)
    arguments = parser.parse_args()

    parameters = Parameters(gamma=0.0)
    initial_state = np.array([arguments.x, arguments.y, 0.0, 0.0, 0.0, 0.0])
    boundary_state = np.array([arguments.boundary_x, arguments.boundary_y, 0.0, 0.0, 0.0, 0.0])
    interior_lambda = benettin(initial_state, parameters, intervals=arguments.intervals)
    boundary_lambda = benettin(boundary_state, parameters, intervals=arguments.intervals)
    print(f"lambda_interior={interior_lambda:.6e}")
    print(f"lambda_fronteira={boundary_lambda:.6e}")
    print(f"tprev_interior={predictability_horizon(interior_lambda):.6e}")
    print(f"tprev_fronteira={predictability_horizon(boundary_lambda):.6e}")


if __name__ == "__main__":
    main()
