"""Verificacao dos integradores no problema de Kepler.

Executar com:
    python verificacao_kepler.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KeplerModel:
    """Problema de uma massa unitaria na origem, sem arrasto."""

    eps: float = 0.0

    def acceleration(self, position: np.ndarray) -> np.ndarray:
        distance_squared = np.dot(position, position) + self.eps**2
        return -position / distance_squared**1.5

    def rhs(self, state: np.ndarray) -> np.ndarray:
        position = state[:3]
        velocity = state[3:]
        return np.concatenate((velocity, self.acceleration(position)))


def euler_step(model: KeplerModel, state: np.ndarray, dt: float) -> np.ndarray:
    return state + dt * model.rhs(state)


def heun_step(model: KeplerModel, state: np.ndarray, dt: float) -> np.ndarray:
    slope_initial = model.rhs(state)
    predictor = state + dt * slope_initial
    slope_predictor = model.rhs(predictor)
    return state + 0.5 * dt * (slope_initial + slope_predictor)


def rk4_step(model: KeplerModel, state: np.ndarray, dt: float) -> np.ndarray:
    slope_1 = model.rhs(state)
    slope_2 = model.rhs(state + 0.5 * dt * slope_1)
    slope_3 = model.rhs(state + 0.5 * dt * slope_2)
    slope_4 = model.rhs(state + dt * slope_3)
    return state + (dt / 6.0) * (slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4)


def verlet_step(model: KeplerModel, state: np.ndarray, dt: float) -> np.ndarray:
    position = state[:3]
    velocity = state[3:]
    acceleration_initial = model.acceleration(position)
    velocity_half = velocity + 0.5 * dt * acceleration_initial
    next_position = position + dt * velocity_half
    next_velocity = velocity_half + 0.5 * dt * model.acceleration(next_position)
    return np.concatenate((next_position, next_velocity))


def kepler_acceleration_scalar(position: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = position
    inverse_radius_cubed = (x * x + y * y + z * z) ** -1.5
    return -x * inverse_radius_cubed, -y * inverse_radius_cubed, -z * inverse_radius_cubed


def fast_verlet_step(state: np.ndarray, dt: float) -> np.ndarray:
    x, y, z, vx, vy, vz = (float(value) for value in state)
    ax, ay, az = kepler_acceleration_scalar((x, y, z))
    half_dt = 0.5 * dt
    vx_half = vx + half_dt * ax
    vy_half = vy + half_dt * ay
    vz_half = vz + half_dt * az
    next_x = x + dt * vx_half
    next_y = y + dt * vy_half
    next_z = z + dt * vz_half
    next_ax, next_ay, next_az = kepler_acceleration_scalar((next_x, next_y, next_z))
    return np.array([next_x, next_y, next_z, vx_half + half_dt * next_ax, vy_half + half_dt * next_ay, vz_half + half_dt * next_az])


def fast_rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    x, y, z, vx, vy, vz = (float(value) for value in state)

    def slope(values: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
        px, py, pz, pvx, pvy, pvz = values
        ax, ay, az = kepler_acceleration_scalar((px, py, pz))
        return pvx, pvy, pvz, ax, ay, az

    first = slope((x, y, z, vx, vy, vz))
    second = slope(tuple(value + 0.5 * dt * derivative for value, derivative in zip((x, y, z, vx, vy, vz), first)))
    third = slope(tuple(value + 0.5 * dt * derivative for value, derivative in zip((x, y, z, vx, vy, vz), second)))
    fourth = slope(tuple(value + dt * derivative for value, derivative in zip((x, y, z, vx, vy, vz), third)))
    return np.array([
        value + dt * (first_value + 2.0 * second_value + 2.0 * third_value + fourth_value) / 6.0
        for value, first_value, second_value, third_value, fourth_value in zip((x, y, z, vx, vy, vz), first, second, third, fourth)
    ])


INTEGRATORS = {
    "Euler": euler_step,
    "RK2": heun_step,
    "RK4": rk4_step,
    "Verlet": verlet_step,
}


def circular_initial_state(radius: float) -> np.ndarray:
    return np.array([radius, 0.0, 0.0, 0.0, 1.0 / math.sqrt(radius), 0.0])


def integrate(
    step_function,
    model: KeplerModel,
    initial_state: np.ndarray,
    dt: float,
    final_time: float,
) -> np.ndarray:
    steps = int(round(final_time / dt))
    if not math.isclose(steps * dt, final_time, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("final_time precisa ser multiplo inteiro de dt")

    state = initial_state.copy()
    for _ in range(steps):
        state = step_function(model, state, dt)
    return state


def circular_error(step_function, radius: float, dt: float) -> float:
    model = KeplerModel()
    initial_state = circular_initial_state(radius)
    period = 2.0 * math.pi * radius**1.5
    steps = int(round(period / dt))
    adjusted_dt = period / steps
    state = initial_state.copy()
    maximum_radial_error = 0.0
    for _ in range(steps):
        state = step_function(model, state, adjusted_dt)
        maximum_radial_error = max(
            maximum_radial_error,
            abs(np.linalg.norm(state[:3]) - radius),
        )
    return maximum_radial_error


def observed_order(step_function, radius: float, dt: float) -> float:
    model = KeplerModel()
    period = 2.0 * math.pi * radius**1.5
    initial_state = circular_initial_state(radius)
    steps_dt = int(round(period / dt))
    steps_half = int(round(period / (dt / 2.0)))
    adjusted_dt = period / steps_dt
    adjusted_half = period / steps_half
    error_dt = np.linalg.norm(
        integrate(step_function, model, initial_state, adjusted_dt, period)
        - initial_state
    )
    error_half = np.linalg.norm(
        integrate(step_function, model, initial_state, adjusted_half, period)
        - initial_state
    )
    return math.log(error_dt / error_half, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.02)
    arguments = parser.parse_args()

    print(f"raio={arguments.radius:g}, dt={arguments.dt:g}")
    print("metodo       erro_radial       ordem_observada")
    for name, step_function in INTEGRATORS.items():
        radial_error = circular_error(step_function, arguments.radius, arguments.dt)
        order = observed_order(step_function, arguments.radius, arguments.dt)
        print(f"{name:<12} {radial_error: .6e}      {order: .4f}")


if __name__ == "__main__":
    main()
