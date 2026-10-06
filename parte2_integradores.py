"""Parte 2: energia e reversibilidade de RK4 e Velocity Verlet."""

from __future__ import annotations

import argparse
import math

import numpy as np

from verificacao_kepler import KeplerModel, fast_rk4_step, fast_verlet_step, rk4_step, verlet_step, circular_initial_state


def energy(state: np.ndarray) -> float:
    position = state[:3]
    velocity = state[3:]
    return 0.5 * np.dot(velocity, velocity) - 1.0 / np.linalg.norm(position)


def integrate(step_function, state: np.ndarray, dt: float, duration: float) -> np.ndarray:
    steps = int(round(duration / dt))
    actual_dt = duration / steps
    model = KeplerModel()
    result = state.copy()
    for _ in range(steps):
        result = step_function(model, result, actual_dt)
    return result


def reversibility_error(step_function, dt: float, duration: float) -> float:
    initial = circular_initial_state(1.0)
    forward = integrate(step_function, initial, dt, duration)
    reversed_velocity = forward.copy()
    reversed_velocity[3:] *= -1.0
    backward = integrate(step_function, reversed_velocity, dt, duration)
    backward[3:] *= -1.0
    return float(np.linalg.norm(backward - initial))


def energy_history(step_function, dt: float, duration: float, fast_step=None) -> tuple[np.ndarray, np.ndarray]:
    initial = circular_initial_state(1.0)
    steps = max(1, int(round(duration / dt)))
    actual_dt = duration / steps
    model = KeplerModel()
    state = initial.copy()
    stride = max(1, math.ceil(steps / 4000))
    times = [0.0]
    energies = [energy(state)]
    for step in range(1, steps + 1):
        state = fast_step(state, actual_dt) if fast_step is not None else step_function(model, state, actual_dt)
        if step % stride == 0 or step == steps:
            times.append(step * actual_dt)
            energies.append(energy(state))
    return np.asarray(times), np.asarray(energies)


def fast_integrate(step_function, state: np.ndarray, dt: float, duration: float, fast_step=None) -> np.ndarray:
    steps = max(1, int(round(duration / dt)))
    actual_dt = duration / steps
    result = state.copy()
    model = KeplerModel()
    for _ in range(steps):
        result = fast_step(result, actual_dt) if fast_step is not None else step_function(model, result, actual_dt)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.02)
    arguments = parser.parse_args()

    duration = arguments.periods * 2.0 * math.pi
    initial = circular_initial_state(1.0)
    initial_energy = energy(initial)
    print("metodo       erro_energia_final    erro_reversibilidade")
    for name, step_function, method_dt, fast_step in (("RK4", rk4_step, 4.0 * arguments.dt, fast_rk4_step), ("Verlet", verlet_step, arguments.dt, fast_verlet_step)):
        _, history = energy_history(step_function, method_dt, duration, fast_step)
        final = fast_integrate(step_function, initial, method_dt, duration, fast_step)
        reversed_state = final.copy()
        reversed_state[3:] *= -1.0
        reverse_final = fast_integrate(step_function, reversed_state, method_dt, duration, fast_step)
        reverse_final[3:] *= -1.0
        print(f"{name:<12} dt={method_dt:.6e} deriva_max={np.max(np.abs(history - initial_energy)):.6e} erro_final={abs(history[-1] - initial_energy):.6e} reversibilidade={np.linalg.norm(reverse_final - initial):.6e}")


if __name__ == "__main__":
    main()
