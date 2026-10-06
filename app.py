"""Interface Streamlit para executar as etapas das bacias gravitacionais."""

from __future__ import annotations

import json
import math
import time
from io import StringIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from numba import njit
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp

from parte1_verificacao import elliptic_initial_state, orbital_invariants
from parte2_integradores import energy as kepler_energy
from verificacao_kepler import INTEGRATORS, KeplerModel, circular_error, fast_rk4_step, fast_verlet_step, observed_order

st.set_page_config(page_title="Bacias gravitacionais", layout="wide")

DEFAULT_MASSES = [[0.35], [0.25], [0.20], [0.20]]
DEFAULT_POSITIONS = [
    [-1.2, 0.0, 0.0],
    [1.0, 1.1, 0.3],
    [0.5, -1.4, -0.2],
    [-0.4, 1.5, -0.7],
]

# Experimento fixo da Parte 2, independente da configuracao geral da barra lateral.
PART2_PERIODS = 10_000
PART2_VERLET_DT = 0.02
PART2_MODEL_DESCRIPTION = "N=1, M=1, R=(0,0,0), eps=0, gamma=0"
PART4_DELTA_BENETTIN = 1e-10
PART4_DELTA_MACHINE = 2e-16
PART4_DELTA_TOL = 0.1


def parse_model(masses_text: str, positions_text: str) -> tuple[np.ndarray, np.ndarray]:
    masses = np.asarray(json.loads(masses_text), dtype=np.float64).reshape(-1)
    positions = np.asarray(json.loads(positions_text), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("As posicoes devem ter formato [[x, y, z], ...].")
    if len(masses) != len(positions):
        raise ValueError("Deve existir uma massa para cada posicao.")
    if len(masses) == 0 or masses.sum() <= 0.0:
        raise ValueError("Informe pelo menos uma massa com soma positiva.")
    if np.any(masses <= 0.0) or not np.isfinite(masses).all():
        raise ValueError("As massas devem ser positivas e finitas.")
    if not np.isfinite(positions).all():
        raise ValueError("As posicoes devem ser finitas.")
    return masses / masses.sum(), positions


def parse_float(value: str, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        number = float(value.strip().replace(",", "."))
    except ValueError as error:
        raise ValueError(f"{label} precisa ser um numero.") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} precisa ser finito.")
    if positive and number <= 0.0:
        raise ValueError(f"{label} precisa ser maior que zero.")
    if nonnegative and number < 0.0:
        raise ValueError(f"{label} nao pode ser negativo.")
    return number


def parse_int(value: str, label: str, *, minimum: int = 1) -> int:
    try:
        number = int(value.strip())
    except ValueError as error:
        raise ValueError(f"{label} precisa ser um inteiro.") from error
    if number < minimum:
        raise ValueError(f"{label} precisa ser maior ou igual a {minimum}.")
    return number


def parse_float_list(value: str, label: str, *, positive: bool = False, nonnegative: bool = False) -> list[float]:
    try:
        numbers = [parse_float(item, label, positive=positive, nonnegative=nonnegative) for item in value.split(",") if item.strip()]
    except ValueError:
        raise
    if not numbers:
        raise ValueError(f"{label} precisa conter pelo menos um valor.")
    return numbers


def acceleration(position: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float) -> np.ndarray:
    difference = position[None, :] - positions
    distance_squared = np.sum(difference * difference, axis=1) + eps**2
    return np.sum(-masses[:, None] * difference / distance_squared[:, None] ** 1.5, axis=0)


def rhs(state: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float, gamma: float) -> np.ndarray:
    return np.concatenate((state[3:], acceleration(state[:3], masses, positions, eps) - gamma * state[3:]))


def potential(position: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float) -> float:
    distances = np.linalg.norm(position[None, :] - positions, axis=1)
    return float(-np.sum(masses / np.sqrt(distances**2 + eps**2)))


def total_energy(state: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float) -> float:
    return float(0.5 * np.dot(state[3:], state[3:]) + potential(state[:3], masses, positions, eps))


def dissipation_diagnostics(times: np.ndarray, states: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Calcula E(t), integral de |v|^2 e o residuo da identidade de dissipacao."""
    velocities = states[3:, :].T
    speeds_squared = np.sum(velocities * velocities, axis=1)
    integral = np.zeros(len(times), dtype=np.float64)
    if len(times) > 1:
        integral[1:] = np.cumsum(0.5 * (speeds_squared[1:] + speeds_squared[:-1]) * np.diff(times))
    energies = np.asarray([total_energy(states[:, index], masses, positions, eps) for index in range(states.shape[1])])
    residual = np.abs(energies - energies[0] + gamma * integral)
    return energies, residual


def hermite_state(previous: np.ndarray, current: np.ndarray, dt: float, fraction: float) -> np.ndarray:
    """Dense cubic Hermite state using positions and endpoint velocities."""
    s = fraction
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    dh00 = 6.0 * s2 - 6.0 * s
    dh10 = 3.0 * s2 - 4.0 * s + 1.0
    dh01 = -6.0 * s2 + 6.0 * s
    dh11 = 3.0 * s2 - 2.0 * s
    result = np.empty(6, dtype=np.float64)
    result[:3] = h00 * previous[:3] + h10 * dt * previous[3:] + h01 * current[:3] + h11 * dt * current[3:]
    result[3:] = (dh00 * previous[:3] + dh01 * current[:3]) / dt + dh10 * previous[3:] + dh11 * current[3:]
    return result


def locate_capture_bisection(previous: np.ndarray, current: np.ndarray, attractor: np.ndarray, rcap: float, dt: float, left: float = 0.0, right: float = 1.0, iterations: int = 50, interpolator=None) -> tuple[float, np.ndarray]:
    """Locates sphere crossing by bisection on cubic Hermite dense output."""
    def signed_distance(fraction: float) -> float:
        state = interpolator(fraction) if interpolator is not None else hermite_state(previous, current, dt, fraction)
        displacement = state[:3] - attractor
        return float(np.dot(displacement, displacement) - rcap**2)

    for _ in range(iterations):
        middle = 0.5 * (left + right)
        if signed_distance(middle) >= 0.0:
            left = middle
        else:
            right = middle
    fraction = 0.5 * (left + right)
    event_state = interpolator(fraction) if interpolator is not None else hermite_state(previous, current, dt, fraction)
    return fraction, np.asarray(event_state, dtype=np.float64)


def first_capture_event(previous: np.ndarray, current: np.ndarray, positions: np.ndarray, rcap: float, dt: float, interpolator=None) -> tuple[float, int, np.ndarray] | None:
    """Busca cruzamento em 16 subintervalos do passo (unificado com Numba) e refina por bisseccao."""
    fractions = tuple(index / 16.0 for index in range(17))
    earliest = None
    for index, attractor in enumerate(positions):
        values = []
        for fraction in fractions:
            state = interpolator(fraction) if interpolator is not None else hermite_state(previous, current, dt, fraction)
            displacement = state[:3] - attractor
            values.append(float(np.dot(displacement, displacement) - rcap**2))
        for bracket_index in range(len(fractions) - 1):
            if values[bracket_index] >= 0.0 and values[bracket_index + 1] <= 0.0:
                left, right = fractions[bracket_index], fractions[bracket_index + 1]
                fraction, event_state = locate_capture_bisection(previous, current, attractor, rcap, dt, left, right, interpolator=interpolator)
                if earliest is None or fraction < earliest[0]:
                    earliest = (fraction, index, event_state)
                break
    return earliest


def adaptive_step_size(state: np.ndarray, masses: np.ndarray, positions: np.ndarray, eta: float, dt_initial: float,
                       criterion: str = "dinamico", eps: float = 0.18, gamma: float = 0.0) -> float:
    """Calcula o passo adaptativo segundo criterio dinamico ou cinematico (eta*|v|/|a|)."""
    if criterion == "cinematico":
        speed = float(np.linalg.norm(state[3:]))
        acc = acceleration(state[:3], masses, positions, eps) - gamma * state[3:]
        acc_norm = float(np.linalg.norm(acc))
        if acc_norm > 1e-12 and speed > 1e-12:
            return min(dt_initial, eta * (speed / acc_norm))
        return dt_initial
    distances = np.linalg.norm(state[:3][None, :] - positions, axis=1)
    dynamic_steps = distances**1.5 / np.sqrt(masses)
    return min(dt_initial, eta * float(np.min(dynamic_steps)))


def escape_conditions(state: np.ndarray, positions: np.ndarray, masses: np.ndarray, eps: float, rescape: float) -> bool:
    position = state[:3]
    velocity = state[3:]
    distance = np.linalg.norm(position)
    kinetic = 0.5 * np.dot(velocity, velocity)
    return bool(distance > rescape and np.dot(position, velocity) > 0.0 and kinetic > 1.0 / distance)


def classify_adaptive_detailed(
    initial_state: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    eps: float,
    gamma: float,
    tmax: float,
    rcap: float,
    rescape: float,
    method: str,
    eta: float,
    dt_initial: float,
    dt_min: float,
    detect_events: bool = True,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    store_history: bool = True,
    criterion: str = "dinamico",
) -> dict[str, Any]:
    state = initial_state.astype(np.float64, copy=True)
    initial_time = 0.0
    times = [initial_time] if store_history else None
    states = [state.copy()] if store_history else None
    steps = 0
    result: dict[str, Any] = {
        "estado": "NAO_RESOLVIDA",
        "codigo": -2,
        "tempo_final": 0.0,
        "passos": 0,
        "atrator": None,
        "tempo_captura": None,
        "posicao_captura": None,
        "velocidade_captura": None,
        "times": None,
        "states": None,
    }
    elapsed = 0.0
    initial_distances = np.linalg.norm(positions - state[:3], axis=1)
    initially_captured = np.flatnonzero(initial_distances <= rcap)
    if len(initially_captured):
        index = int(initially_captured[np.argmin(initial_distances[initially_captured])])
        result.update({"estado": "CAPTURADA", "codigo": index, "atrator": index, "tempo_captura": 0.0,
                       "posicao_captura": state[:3].copy(), "velocidade_captura": state[3:].copy()})
    while elapsed < tmax and result["codigo"] == -2:
        proposed_dt = adaptive_step_size(state, masses, positions, eta, dt_initial, criterion=criterion, eps=eps, gamma=gamma)
        remaining = tmax - elapsed
        if proposed_dt < dt_min and remaining > dt_min:
            raise RuntimeError(f"O passo adaptativo requerido ({proposed_dt:.3e}) ficou abaixo de dt_min ({dt_min:.3e}); reduza dt_min para continuar sem violar o criterio.")
        step_dt = min(proposed_dt, remaining)
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise RuntimeError("O passo adaptativo deixou de ser positivo e finito.")
        previous = state.copy()
        dense_interpolator = None
        if method == "RK45":
            local_solution = solve_ivp(
                lambda current_time, current_state: rhs(current_state, masses, positions, eps, gamma),
                (elapsed, elapsed + step_dt), state, method="RK45", rtol=rtol, atol=atol,
                max_step=step_dt, dense_output=True,
            )
            if not local_solution.success:
                raise RuntimeError(local_solution.message)
            next_state = local_solution.y[:, -1]
            dense_interpolator = lambda fraction: local_solution.sol(elapsed + fraction * step_dt)
        elif method == "Verlet":
            next_state = verlet_step(state, step_dt, masses, positions, eps, gamma)
        else:
            next_state = rk4_step(state, step_dt, masses, positions, eps, gamma)
        steps += 1
        captured_event = first_capture_event(previous, next_state, positions, rcap, step_dt, dense_interpolator) if detect_events and result["codigo"] == -2 else None
        naive_capture_index = None
        if result["codigo"] == -2 and not detect_events:
            discrete_distances = np.linalg.norm(positions - next_state[:3], axis=1)
            inside = np.flatnonzero(discrete_distances <= rcap)
            if len(inside):
                naive_capture_index = int(inside[np.argmin(discrete_distances[inside])])
        if captured_event is not None:
            fraction, index, event_state = captured_event
            capture_time = elapsed + fraction * step_dt
            if store_history:
                times.append(capture_time)
                states.append(event_state.copy())
            elapsed = capture_time
            result.update({
                "estado": "CAPTURADA", "codigo": index, "tempo_final": capture_time, "passos": steps,
                "atrator": index, "tempo_captura": capture_time,
                "posicao_captura": event_state[:3].copy(), "velocidade_captura": event_state[3:].copy(),
            })
            break
        elif naive_capture_index is not None:
            result.update({"estado": "CAPTURADA", "codigo": naive_capture_index, "tempo_final": elapsed + step_dt,
                           "passos": steps, "atrator": naive_capture_index, "tempo_captura": elapsed + step_dt,
                           "posicao_captura": next_state[:3].copy(), "velocidade_captura": next_state[3:].copy()})
        state = next_state
        elapsed += step_dt
        if store_history:
            times.append(elapsed)
            states.append(state.copy())
        if result["codigo"] == -2 and escape_conditions(state, positions, masses, eps, rescape):
            result.update({"estado": "ESCAPADA", "codigo": -1, "tempo_final": elapsed, "passos": steps})
            break
        if result["codigo"] >= 0:
            break
    else:
        result.update({"tempo_final": elapsed, "passos": steps})
    result["times"] = np.asarray(times) if store_history else None
    result["states"] = np.asarray(states, dtype=np.float64).T if store_history else None
    return result


def classify_adaptive_pair(initial_state: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float, gamma: float,
                           tmax: float, rcap: float, rescape: float, method: str, eta: float,
                           dt_initial: float, dt_min: float, rtol: float, atol: float) -> tuple[int, int]:
    """One physical integration yields both discrete and event capture labels."""
    if method != "RK45":
        method_code = 1 if method == "Verlet" else 0
        naive_code, event_code, _ = _adaptive_pair_fixed(
            initial_state.astype(np.float64), masses, positions, eps, gamma, tmax, rcap, rescape,
            eta, dt_initial, dt_min, method_code,
        )
        if naive_code == -3 or event_code == -3:
            raise RuntimeError("Passo adaptativo abaixo de dt_min ou estado nao finito; ajuste dt_min e repita.")
        return int(naive_code), int(event_code)

    state = initial_state.astype(np.float64, copy=True)
    initial_distances = np.linalg.norm(positions - state[:3], axis=1)
    initial_inside = np.flatnonzero(initial_distances <= rcap)
    if len(initial_inside):
        initial_attractor = int(initial_inside[np.argmin(initial_distances[initial_inside])])
        return initial_attractor, initial_attractor
    naive_code = -2
    event_code = -2
    elapsed = 0.0
    while elapsed < tmax and (naive_code == -2 or event_code == -2):
        proposed_dt = adaptive_step_size(state, masses, positions, eta, dt_initial)
        remaining = tmax - elapsed
        if proposed_dt < dt_min and remaining > dt_min:
            raise RuntimeError(f"Passo adaptativo {proposed_dt:.3e} < dt_min {dt_min:.3e}; ajuste dt_min.")
        step_dt = min(proposed_dt, remaining)
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise RuntimeError("Passo adaptativo nao positivo ou nao finito.")
        previous = state
        dense_interpolator = None
        if method == "RK45":
            solution = solve_ivp(
                lambda _time, current_state: rhs(current_state, masses, positions, eps, gamma),
                (elapsed, elapsed + step_dt), state, method="RK45", rtol=rtol, atol=atol,
                max_step=step_dt, dense_output=True,
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            next_state = solution.y[:, -1]
            dense_interpolator = lambda fraction: solution.sol(elapsed + fraction * step_dt)
        elif method == "Verlet":
            next_state = verlet_step(state, step_dt, masses, positions, eps, gamma)
        else:
            next_state = rk4_step(state, step_dt, masses, positions, eps, gamma)
        if event_code == -2:
            event = first_capture_event(previous, next_state, positions, rcap, step_dt, dense_interpolator)
            if event is not None:
                event_code = event[1]
        if naive_code == -2:
            discrete_distances = np.linalg.norm(positions - next_state[:3], axis=1)
            inside = np.flatnonzero(discrete_distances <= rcap)
            if len(inside):
                naive_code = int(inside[np.argmin(discrete_distances[inside])])
        state = next_state
        elapsed += step_dt
        if escape_conditions(state, positions, masses, eps, rescape):
            if naive_code == -2:
                naive_code = -1
            if event_code == -2:
                event_code = -1
            break
    return naive_code, event_code


@st.cache_data(show_spinner=False, max_entries=512)
def cached_adaptive_pair_chunk(points: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float, gamma: float,
                               tmax: float, rcap: float, rescape: float, method: str, eta: float,
                               dt_initial: float, dt_min: float, rtol: float, atol: float, version: str = "adaptive-pair-v3") -> tuple[np.ndarray, np.ndarray]:
    naive = np.full(len(points), -2, dtype=np.int8)
    event = np.full(len(points), -2, dtype=np.int8)
    for index, point in enumerate(points):
        initial = np.array([point[0], point[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        naive[index], event[index] = classify_adaptive_pair(initial, masses, positions, eps, gamma, tmax, rcap, rescape, method, eta, dt_initial, dt_min, rtol, atol)
    return naive, event


def adaptive_basin_pair(masses: np.ndarray, positions: np.ndarray, resolution: int, eps: float, gamma: float, tmax: float,
                        rcap: float, rescape: float, method: str, eta: float, dt_initial: float, dt_min: float,
                        xlim: tuple[float, float], ylim: tuple[float, float], rtol: float = 1e-7, atol: float = 1e-9,
                        progress_callback=None, block_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.linspace(xlim[0], xlim[1], resolution)
    y_values = np.linspace(ylim[0], ylim[1], resolution)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    naive_flat = np.full(len(points), -2, dtype=np.int8)
    event_flat = np.full(len(points), -2, dtype=np.int8)
    block_size = max(1, block_size)
    for start in range(0, len(points), block_size):
        stop = min(len(points), start + block_size)
        naive_chunk, event_chunk = cached_adaptive_pair_chunk(
            points[start:stop], masses, positions, eps, gamma, tmax, rcap, rescape, method,
            eta, dt_initial, dt_min, rtol, atol,
        )
        naive_flat[start:stop] = naive_chunk
        event_flat[start:stop] = event_chunk
        if progress_callback is not None:
            progress_callback(stop, len(points))
    return naive_flat.reshape(resolution, resolution), event_flat.reshape(resolution, resolution), x_values, y_values


def vectorized_acceleration(positions_state: np.ndarray, masses: np.ndarray, attractors: np.ndarray, eps: float) -> np.ndarray:
    difference = positions_state[:, None, :] - attractors[None, :, :]
    distance_squared = np.sum(difference * difference, axis=2) + eps**2
    return np.sum(-masses[None, :, None] * difference / distance_squared[:, :, None] ** 1.5, axis=1)


def vectorized_energy(states: np.ndarray, masses: np.ndarray, attractors: np.ndarray, eps: float) -> np.ndarray:
    difference = states[:, None, :3] - attractors[None, :, :]
    distances = np.sqrt(np.sum(difference * difference, axis=2) + eps**2)
    return 0.5 * np.sum(states[:, 3:] * states[:, 3:], axis=1) - np.sum(masses[None, :] / distances, axis=1)


def vectorized_step(states: np.ndarray, dt: float, masses: np.ndarray, attractors: np.ndarray, eps: float, gamma: float, method: str) -> np.ndarray:
    positions_state = states[:, :3]
    velocities = states[:, 3:]
    acceleration_initial = vectorized_acceleration(positions_state, masses, attractors, eps) - gamma * velocities
    if method == "Verlet":
        velocity_half = velocities + 0.5 * dt * acceleration_initial
        next_positions = positions_state + dt * velocity_half
        next_acceleration = vectorized_acceleration(next_positions, masses, attractors, eps) - gamma * velocity_half
        next_velocities = velocity_half + 0.5 * dt * next_acceleration
        return np.column_stack((next_positions, next_velocities))

    def vectorized_rhs(batch: np.ndarray) -> np.ndarray:
        return np.column_stack((batch[:, 3:], vectorized_acceleration(batch[:, :3], masses, attractors, eps) - gamma * batch[:, 3:]))

    slope_1 = vectorized_rhs(states)
    slope_2 = vectorized_rhs(states + 0.5 * dt * slope_1)
    slope_3 = vectorized_rhs(states + 0.5 * dt * slope_2)
    slope_4 = vectorized_rhs(states + dt * slope_3)
    return states + dt * (slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4) / 6.0


def compute_basin_vectorized(masses: np.ndarray, positions: np.ndarray, resolution: int, eps: float, gamma: float, tmax: float, rcap: float, rescape: float, method: str, dt: float, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Calcula a bacia de atracao de forma totalmente vetorizada com deteccao continua de eventos.
    
    A deteccao de eventos de captura verifica se a aproximacao minima durante o segmento de passo continuo
    [previous, current] cruzou a esfera do atrator (min_dist <= rcap), evitando o salto de passo (tunneling)
    de particulas rapidas mesmo sem integracao escalar individual.
    """
    values_x = np.linspace(xlim[0], xlim[1], resolution)
    values_y = np.linspace(ylim[0], ylim[1], resolution)
    grid_x, grid_y = np.meshgrid(values_x, values_y)
    states = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros(resolution * resolution), np.zeros((resolution * resolution, 3))))
    basin = np.full(len(states), -2, dtype=np.int8)
    active = np.ones(len(states), dtype=bool)
    active_fraction = []
    elapsed = 0.0
    while elapsed < tmax and np.any(active):
        step_dt = min(dt, tmax - elapsed)
        previous = states[active].copy()
        states[active] = vectorized_step(previous, step_dt, masses, positions, eps, gamma, method)
        current = states[active]
        active_indices = np.flatnonzero(active)
        captured = np.zeros(len(active_indices), dtype=bool)

        # Deteccao continua de eventos no segmento [previous, current]
        displacement = current[:, :3] - previous[:, :3]
        disp_sq = np.sum(displacement * displacement, axis=1)
        disp_sq_safe = np.maximum(disp_sq, 1e-15)

        for index, attractor in enumerate(positions):
            w = previous[:, :3] - attractor
            s_star = np.clip(-np.sum(w * displacement, axis=1) / disp_sq_safe, 0.0, 1.0)
            closest_points = previous[:, :3] + s_star[:, None] * displacement
            min_dist_sq = np.sum((closest_points - attractor)**2, axis=1)
            new_distance_sq = np.sum((current[:, :3] - attractor)**2, axis=1)
            is_captured = (min_dist_sq < rcap**2) | (new_distance_sq < rcap**2)
            newly_captured = (~captured) & is_captured
            basin[active_indices[newly_captured]] = index
            captured |= newly_captured

        distances = np.linalg.norm(current[:, :3], axis=1)
        escaping = (~captured) & (distances > rescape) & (np.sum(current[:, :3] * current[:, 3:], axis=1) > 0.0) & (vectorized_energy(current, masses, positions, eps) > 0.0)
        basin[active_indices[escaping]] = -1
        finished = captured | escaping
        active[active_indices[finished]] = False
        elapsed += step_dt
        active_fraction.append(float(np.mean(active)))
    return basin.reshape(resolution, resolution), values_x, values_y, active_fraction


def compute_basin_for_ui(masses: np.ndarray, positions: np.ndarray, resolution: int, eps: float, gamma: float, tmax: float, rcap: float, rescape: float, method: str, dt: float, xlim: tuple[float, float], ylim: tuple[float, float], rtol: float, atol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Escolhe o caminho vetorizado para metodos de passo fixo."""
    if method == "RK45" and resolution > 100:
        raise ValueError(f"O metodo RK45 escalar esta bloqueado para resolucao {resolution} > 100 para evitar custo excessivo de chamadas a solve_ivp (>10.000 chamadas). Use RK4 ou Verlet vetorizado.")
    if method in {"RK4", "Verlet"}:
        basin, x_values, y_values, _ = compute_basin_vectorized(masses, positions, resolution, eps, gamma, tmax, rcap, rescape, method, dt, xlim, ylim)
        return basin, x_values, y_values, "vetorizado (passo fixo; captura por cruzamento)"
    basin, x_values, y_values = compute_basin(masses, positions, resolution, eps, gamma, tmax, rcap, rescape, method, dt, xlim, ylim, rtol, atol)
    return basin, x_values, y_values, "RK45 escalar com eventos do solve_ivp"


@st.cache_data(show_spinner=False, max_entries=12)
def cached_basin_for_ui(masses: np.ndarray, positions: np.ndarray, resolution: int, eps: float, gamma: float,
                        tmax: float, rcap: float, rescape: float, method: str, dt: float,
                        xlim: tuple[float, float], ylim: tuple[float, float], rtol: float, atol: float,
                        version: str = "basin-cache-v2") -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Cachea só resultados determinísticos, com todos os parâmetros na chave."""
    return compute_basin_for_ui(masses, positions, resolution, eps, gamma, tmax, rcap, rescape, method, dt, xlim, ylim, rtol, atol)


def scalar_step_batch(states: np.ndarray, dt: float, masses: np.ndarray, positions: np.ndarray,
                      eps: float, gamma: float, method: str) -> np.ndarray:
    """Reference loop: applies the same integrator independently to each state."""
    step = verlet_step if method == "Verlet" else rk4_step
    return np.vstack([step(state, dt, masses, positions, eps, gamma) for state in states])


def benchmark_step_implementations(particles: int, repetitions: int, dt: float, masses: np.ndarray,
                                  positions: np.ndarray, eps: float, gamma: float, method: str,
                                  seed: int = 1234) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    states = np.zeros((particles, 6), dtype=np.float64)
    states[:, :3] = rng.uniform(-3.0, 3.0, (particles, 3))
    states[:, 3:] = rng.uniform(-0.2, 0.2, (particles, 3))

    reference = scalar_step_batch(states, dt, masses, positions, eps, gamma, method)
    vectorized = vectorized_step(states, dt, masses, positions, eps, gamma, method)
    agrees = bool(np.allclose(reference, vectorized, rtol=1e-12, atol=1e-12))
    if not agrees:
        maximum_difference = float(np.max(np.abs(reference - vectorized)))
        raise RuntimeError(f"As implementacoes nao concordam (max abs diff={maximum_difference:.3e}); benchmark cancelado.")

    started = time.perf_counter()
    for _ in range(repetitions):
        scalar_step_batch(states, dt, masses, positions, eps, gamma, method)
    scalar_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repetitions):
        vectorized_step(states, dt, masses, positions, eps, gamma, method)
    vector_seconds = time.perf_counter() - started
    return {
        "allclose": agrees,
        "max_abs_difference": float(np.max(np.abs(reference - vectorized))),
        "scalar_seconds": scalar_seconds,
        "vectorized_seconds": vector_seconds,
        "speedup": scalar_seconds / vector_seconds if vector_seconds > 0.0 else float("inf"),
        "particles": particles,
        "repetitions": repetitions,
        "method": method,
        "dt": dt,
    }


def benchmark_basin_end_to_end(masses: np.ndarray, positions: np.ndarray, resolution: int = 32,
                               eps: float = 0.18, gamma: float = 0.15, tmax: float = 20.0,
                               rcap: float = 0.22, rescape: float = 20.0, method: str = "RK4",
                               dt: float = 0.05, xlim: tuple[float, float] = (-3.0, 3.0),
                               ylim: tuple[float, float] = (-3.0, 3.0)) -> dict[str, Any]:
    """Mede o speedup real de ponta a ponta integrando uma bacia completa escalar vs vetorizada."""
    t0_scalar = time.perf_counter()
    scalar_basin, _, _ = compute_basin(
        masses, positions, resolution, eps, gamma, tmax, rcap, rescape, method, dt,
        xlim=xlim, ylim=ylim, detect_events=True,
    )
    scalar_time = time.perf_counter() - t0_scalar

    t0_vector = time.perf_counter()
    vector_basin, _, _, _ = compute_basin_vectorized(
        masses, positions, resolution, eps, gamma, tmax, rcap, rescape, method, dt,
        xlim=xlim, ylim=ylim,
    )
    vector_time = time.perf_counter() - t0_vector

    speedup = scalar_time / vector_time if vector_time > 0.0 else float("inf")
    pointwise_diff = float(np.mean(scalar_basin != vector_basin))

    return {
        "resolucao": f"{resolution}x{resolution}",
        "total_particulas": resolution * resolution,
        "tempo_escalar_s": scalar_time,
        "tempo_vetorizado_s": vector_time,
        "speedup_end_to_end": speedup,
        "concordancia_pontual": 1.0 - pointwise_diff,
        "diferenca_pontual_Pi": pointwise_diff,
        "metodo": method,
        "dt": dt,
        "tmax": tmax,
    }


def rk4_step(state: np.ndarray, dt: float, masses: np.ndarray, positions: np.ndarray, eps: float, gamma: float) -> np.ndarray:
    slope_1 = rhs(state, masses, positions, eps, gamma)
    slope_2 = rhs(state + 0.5 * dt * slope_1, masses, positions, eps, gamma)
    slope_3 = rhs(state + 0.5 * dt * slope_2, masses, positions, eps, gamma)
    slope_4 = rhs(state + dt * slope_3, masses, positions, eps, gamma)
    return state + dt * (slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4) / 6.0


def verlet_step(state: np.ndarray, dt: float, masses: np.ndarray, positions: np.ndarray, eps: float, gamma: float) -> np.ndarray:
    position = state[:3]
    velocity = state[3:]
    acceleration_initial = acceleration(position, masses, positions, eps) - gamma * velocity
    velocity_half = velocity + 0.5 * dt * acceleration_initial
    next_position = position + dt * velocity_half
    next_acceleration = acceleration(next_position, masses, positions, eps) - gamma * velocity_half
    next_velocity = velocity_half + 0.5 * dt * next_acceleration
    return np.concatenate((next_position, next_velocity))


# ===========================================================================
# [CODIGO MORTO / DEPRECATED]
# Esta funcao foi a implementacao preliminar do expoente de Lyapunov na Parte 4.
# Ela nao e chamada em nenhum ponto da aplicacao e foi totalmente substituida
# por benettin_details() e _benettin_chunk(), que utilizam Numba e reescalonamento no espaco 6D.
# ===========================================================================
def estimate_lyapunov(initial_state: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float, tau: float, dt: float, intervals: int) -> float:
    state = initial_state.copy()
    perturbed = state.copy()
    perturbed[0] += 1e-10
    delta0 = 1e-10
    steps = max(1, int(round(tau / dt)))
    actual_dt = tau / steps
    accumulated = 0.0
    for _ in range(intervals):
        for _ in range(steps):
            state = rk4_step(state, actual_dt, masses, positions, eps, 0.0)
            perturbed = rk4_step(perturbed, actual_dt, masses, positions, eps, 0.0)
        separation_vector = perturbed - state
        separation = np.linalg.norm(separation_vector)
        if separation == 0.0:
            return float("-inf")
        accumulated += math.log(separation / delta0)
        perturbed = state + delta0 * separation_vector / separation
    return accumulated / (intervals * tau)


def predictability_horizon(lambda_value: float, delta0: float, tolerance: float) -> float:
    if lambda_value <= 0.0:
        return float("inf")
    return math.log(tolerance / delta0) / lambda_value


def classify_label(code: int, number_of_attractors: int) -> str:
    if 0 <= code < number_of_attractors:
        return f"Atrator {code + 1}"
    if code == -1:
        return "ESCAPADA"
    return "NAO_RESOLVIDA"


def basin_statistics(basin: np.ndarray, number_of_attractors: int) -> dict[str, float | int]:
    total = basin.size
    statistics: dict[str, float | int] = {"total_pixels": total}
    for index in range(number_of_attractors):
        count = int(np.sum(basin == index))
        statistics[f"capturadas_atrator_{index + 1}"] = count
        statistics[f"fracao_atrator_{index + 1}"] = count / total
    escaped = int(np.sum(basin == -1))
    unresolved = int(np.sum(basin == -2))
    statistics["escapadas"] = escaped
    statistics["fracao_escapadas"] = escaped / total
    statistics["nao_resolvidas"] = unresolved
    statistics["fracao_nao_resolvidas"] = unresolved / total
    return statistics


def find_basin_points(basin: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, number_of_attractors: int) -> tuple[tuple[float, float, int] | None, tuple[float, float, int, int] | None]:
    interior = None
    for row in range(1, basin.shape[0] - 1):
        for column in range(1, basin.shape[1] - 1):
            neighborhood = basin[row - 1:row + 2, column - 1:column + 2]
            if basin[row, column] >= 0 and np.all(neighborhood == basin[row, column]):
                interior = (float(x_values[column]), float(y_values[row]), int(basin[row, column]))
                break
        if interior is not None:
            break

    boundary = None
    for row in range(basin.shape[0]):
        for column in range(basin.shape[1]):
            current = int(basin[row, column])
            for row_neighbor, column_neighbor in ((row + 1, column), (row, column + 1)):
                if row_neighbor < basin.shape[0] and column_neighbor < basin.shape[1]:
                    neighbor = int(basin[row_neighbor, column_neighbor])
                    if current >= 0 and neighbor >= 0 and neighbor != current:
                        boundary = (float((x_values[column] + x_values[column_neighbor]) / 2.0), float((y_values[row] + y_values[row_neighbor]) / 2.0), current, neighbor)
                        break
            if boundary is not None:
                break
        if boundary is not None:
            break
    return interior, boundary


def find_boundary_bracket(basin: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> tuple[tuple[float, float, int], tuple[float, float, int]] | None:
    for row in range(basin.shape[0]):
        for column in range(basin.shape[1]):
            current = int(basin[row, column])
            for row_neighbor, column_neighbor in ((row, column + 1), (row + 1, column)):
                if row_neighbor < basin.shape[0] and column_neighbor < basin.shape[1]:
                    neighbor = int(basin[row_neighbor, column_neighbor])
                    if current >= 0 and neighbor >= 0 and current != neighbor:
                        return ((float(x_values[column]), float(y_values[row]), current),
                                (float(x_values[column_neighbor]), float(y_values[row_neighbor]), neighbor))
    return None


def refine_boundary_candidate(masses: np.ndarray, positions: np.ndarray, eps: float, tmax: float, rcap: float,
                              rescape: float, method: str, eta: float, dt_initial: float, dt_min: float,
                              endpoints: tuple[tuple[float, float, int], tuple[float, float, int]],
                              iterations: int = 6) -> tuple[tuple[float, float, int, int], int, float, bool]:
    """Refines only a bracket between adjacent pixels; stops if midpoint has a third class."""
    left_x, left_y, left_class = endpoints[0]
    right_x, right_y, right_class = endpoints[1]
    initial_width = math.hypot(right_x - left_x, right_y - left_y)
    used = 0
    for _ in range(iterations):
        middle_x = 0.5 * (left_x + right_x)
        middle_y = 0.5 * (left_y + right_y)
        initial = np.array([middle_x, middle_y, 0.0, 0.0, 0.0, 0.0])
        _, middle_class = classify_adaptive_pair(
            initial, masses, positions, eps, 0.0, tmax, rcap, rescape, method, eta, dt_initial, dt_min, 1e-7, 1e-9,
        )
        used += 1
        if middle_class == left_class:
            left_x, left_y = middle_x, middle_y
        elif middle_class == right_class:
            right_x, right_y = middle_x, middle_y
        else:
            break
    middle_x = 0.5 * (left_x + right_x)
    middle_y = 0.5 * (left_y + right_y)
    width = math.hypot(right_x - left_x, right_y - left_y)
    return (middle_x, middle_y, left_class, right_class), used, width, width < initial_width


def validate_interior_candidate(basin: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, x: float, y: float) -> tuple[bool, int]:
    column = int(np.argmin(np.abs(x_values - x)))
    row = int(np.argmin(np.abs(y_values - y)))
    if row == 0 or column == 0 or row == basin.shape[0] - 1 or column == basin.shape[1] - 1:
        return False, int(basin[row, column])
    neighborhood = basin[row - 1:row + 2, column - 1:column + 2]
    code = int(basin[row, column])
    return bool(code >= 0 and np.all(neighborhood == code)), code


def validate_boundary_candidate(basin: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, x: float, y: float) -> tuple[bool, tuple[int, int] | None]:
    column = int(np.argmin(np.abs(x_values - x)))
    row = int(np.argmin(np.abs(y_values - y)))
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if not (0 <= neighbor_row < basin.shape[0] and 0 <= neighbor_column < basin.shape[1]):
                continue
            for delta_row, delta_column in ((0, 1), (1, 0), (1, 1), (1, -1)):
                other_row = neighbor_row + delta_row
                other_column = neighbor_column + delta_column
                if 0 <= other_row < basin.shape[0] and 0 <= other_column < basin.shape[1]:
                    first = int(basin[neighbor_row, neighbor_column])
                    second = int(basin[other_row, other_column])
                    if first >= 0 and second >= 0 and first != second:
                        return True, (first, second)
    return False, None


@njit(cache=True)
def _benettin_acceleration(position: np.ndarray, masses: np.ndarray, attractors: np.ndarray, eps: float) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)
    for index in range(len(masses)):
        dx = position[0] - attractors[index, 0]
        dy = position[1] - attractors[index, 1]
        dz = position[2] - attractors[index, 2]
        inverse_distance_cubed = (dx * dx + dy * dy + dz * dz + eps * eps) ** -1.5
        result[0] -= masses[index] * dx * inverse_distance_cubed
        result[1] -= masses[index] * dy * inverse_distance_cubed
        result[2] -= masses[index] * dz * inverse_distance_cubed
    return result


@njit(cache=True)
def _benettin_rhs(state: np.ndarray, masses: np.ndarray, attractors: np.ndarray, eps: float) -> np.ndarray:
    result = np.empty(6, dtype=np.float64)
    result[:3] = state[3:]
    result[3:] = _benettin_acceleration(state[:3], masses, attractors, eps)
    return result


@njit(cache=True)
def _benettin_rk4(state: np.ndarray, dt: float, masses: np.ndarray, attractors: np.ndarray, eps: float) -> np.ndarray:
    k1 = _benettin_rhs(state, masses, attractors, eps)
    k2 = _benettin_rhs(state + 0.5 * dt * k1, masses, attractors, eps)
    k3 = _benettin_rhs(state + 0.5 * dt * k2, masses, attractors, eps)
    k4 = _benettin_rhs(state + dt * k3, masses, attractors, eps)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True)
def _pair_acceleration(position: np.ndarray, masses: np.ndarray, attractors: np.ndarray, eps: float, gamma: float, velocity: np.ndarray) -> np.ndarray:
    acceleration_value = np.zeros(3, dtype=np.float64)
    for index in range(len(masses)):
        dx = position[0] - attractors[index, 0]
        dy = position[1] - attractors[index, 1]
        dz = position[2] - attractors[index, 2]
        inverse_distance_cubed = (dx * dx + dy * dy + dz * dz + eps * eps) ** -1.5
        acceleration_value[0] -= masses[index] * dx * inverse_distance_cubed
        acceleration_value[1] -= masses[index] * dy * inverse_distance_cubed
        acceleration_value[2] -= masses[index] * dz * inverse_distance_cubed
    acceleration_value -= gamma * velocity
    return acceleration_value


@njit(cache=True)
def _pair_rhs(state: np.ndarray, masses: np.ndarray, attractors: np.ndarray, eps: float, gamma: float) -> np.ndarray:
    result = np.empty(6, dtype=np.float64)
    result[:3] = state[3:]
    result[3:] = _pair_acceleration(state[:3], masses, attractors, eps, gamma, state[3:])
    return result


@njit(cache=True)
def _pair_rk4(state: np.ndarray, dt: float, masses: np.ndarray, attractors: np.ndarray, eps: float, gamma: float) -> np.ndarray:
    k1 = _pair_rhs(state, masses, attractors, eps, gamma)
    k2 = _pair_rhs(state + 0.5 * dt * k1, masses, attractors, eps, gamma)
    k3 = _pair_rhs(state + 0.5 * dt * k2, masses, attractors, eps, gamma)
    k4 = _pair_rhs(state + dt * k3, masses, attractors, eps, gamma)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True)
def _pair_hermite_position(previous: np.ndarray, current: np.ndarray, dt: float, fraction: float) -> np.ndarray:
    s = fraction
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * previous[:3] + h10 * dt * previous[3:] + h01 * current[:3] + h11 * dt * current[3:]


@njit(cache=True)
def _pair_distance_event(previous: np.ndarray, current: np.ndarray, attractor: np.ndarray, rcap: float, dt: float, fraction: float) -> float:
    point = _pair_hermite_position(previous, current, dt, fraction)
    dx = point[0] - attractor[0]
    dy = point[1] - attractor[1]
    dz = point[2] - attractor[2]
    return dx * dx + dy * dy + dz * dz - rcap * rcap


@njit(cache=True)
def _adaptive_pair_fixed(initial: np.ndarray, masses: np.ndarray, attractors: np.ndarray, eps: float, gamma: float,
                         tmax: float, rcap: float, rescape: float, eta: float, dt_initial: float, dt_min: float,
                         method_code: int) -> tuple[int, int, int]:
    state = initial.copy()
    naive_code = -2
    event_code = -2
    initial_distances = np.empty(len(masses), dtype=np.float64)
    for index in range(len(masses)):
        dx = state[0] - attractors[index, 0]
        dy = state[1] - attractors[index, 1]
        dz = state[2] - attractors[index, 2]
        initial_distances[index] = math.sqrt(dx * dx + dy * dy + dz * dz)
    initial_index = -1
    nearest = 1e308
    for index in range(len(masses)):
        if initial_distances[index] <= rcap and initial_distances[index] < nearest:
            nearest = initial_distances[index]
            initial_index = index
    if initial_index >= 0:
        return initial_index, initial_index, 0

    elapsed = 0.0
    steps = 0
    while elapsed < tmax and (naive_code == -2 or event_code == -2):
        local_time = 1e308
        for index in range(len(masses)):
            dx = state[0] - attractors[index, 0]
            dy = state[1] - attractors[index, 1]
            dz = state[2] - attractors[index, 2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            candidate = distance ** 1.5 / math.sqrt(masses[index])
            if candidate < local_time:
                local_time = candidate
        step_dt = min(dt_initial, eta * local_time, tmax - elapsed)
        if step_dt < dt_min and tmax - elapsed > dt_min:
            return -3, -3, steps
        if step_dt <= 0.0 or not math.isfinite(step_dt):
            return -3, -3, steps

        previous = state.copy()
        if method_code == 1:
            acceleration_initial = _pair_acceleration(state[:3], masses, attractors, eps, gamma, state[3:])
            half_dt = 0.5 * step_dt
            velocity_half = state[3:] + half_dt * acceleration_initial
            next_position = state[:3] + step_dt * velocity_half
            next_acceleration = _pair_acceleration(next_position, masses, attractors, eps, gamma, velocity_half)
            next_state = np.empty(6, dtype=np.float64)
            next_state[:3] = next_position
            next_state[3:] = velocity_half + half_dt * next_acceleration
        else:
            next_state = _pair_rk4(state, step_dt, masses, attractors, eps, gamma)
        steps += 1
        for component in range(6):
            if not math.isfinite(next_state[component]):
                return -3, -3, steps

        if event_code == -2:
            earliest_fraction = 2.0
            earliest_attractor = -1
            for index in range(len(masses)):
                previous_fraction = 0.0
                previous_value = _pair_distance_event(previous, next_state, attractors[index], rcap, step_dt, previous_fraction)
                for sample_index in range(1, 17):
                    current_fraction = sample_index / 16.0
                    current_value = _pair_distance_event(previous, next_state, attractors[index], rcap, step_dt, current_fraction)
                    if previous_value >= 0.0 and current_value <= 0.0:
                        left = previous_fraction
                        right = current_fraction
                        for _ in range(42):
                            middle = 0.5 * (left + right)
                            middle_value = _pair_distance_event(previous, next_state, attractors[index], rcap, step_dt, middle)
                            if middle_value >= 0.0:
                                left = middle
                            else:
                                right = middle
                        root = 0.5 * (left + right)
                        if root < earliest_fraction:
                            earliest_fraction = root
                            earliest_attractor = index
                        break
                    previous_fraction = current_fraction
                    previous_value = current_value
            if earliest_attractor >= 0:
                event_code = earliest_attractor

        if naive_code == -2:
            nearest = 1e308
            for index in range(len(masses)):
                dx = next_state[0] - attractors[index, 0]
                dy = next_state[1] - attractors[index, 1]
                dz = next_state[2] - attractors[index, 2]
                distance_squared = dx * dx + dy * dy + dz * dz
                if distance_squared <= rcap * rcap and distance_squared < nearest:
                    nearest = distance_squared
                    naive_code = index

        state = next_state
        elapsed += step_dt
        distance = math.sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2])
        radial_product = state[0] * state[3] + state[1] * state[4] + state[2] * state[5]
        kinetic = 0.5 * (state[3] * state[3] + state[4] * state[4] + state[5] * state[5])
        escaped = distance > rescape and radial_product > 0.0 and kinetic > 1.0 / distance
        if escaped:
            if naive_code == -2:
                naive_code = -1
            if event_code == -2:
                event_code = -1
            break
    return naive_code, event_code, steps


@njit(cache=True)
def _benettin_chunk(reference: np.ndarray, perturbed: np.ndarray, masses: np.ndarray, attractors: np.ndarray,
                    eps: float, dt: float, steps_per_interval: int, interval_count: int,
                    delta_reference: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    logarithms = np.empty(interval_count, dtype=np.float64)
    total_log = 0.0
    for interval in range(interval_count):
        for _ in range(steps_per_interval):
            reference = _benettin_rk4(reference, dt, masses, attractors, eps)
            perturbed = _benettin_rk4(perturbed, dt, masses, attractors, eps)
        difference = perturbed - reference
        separation_squared = 0.0
        for component in range(6):
            separation_squared += difference[component] * difference[component]
        separation = math.sqrt(separation_squared)
        if separation <= 0.0 or not np.isfinite(separation):
            return reference, perturbed, total_log, logarithms[:interval], False
        logarithm = math.log(separation / delta_reference)
        logarithms[interval] = logarithm
        total_log += logarithm
        perturbed = reference + delta_reference * difference / separation
    return reference, perturbed, total_log, logarithms, True


def benettin_details(initial_state: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float, tau: float,
                     dt: float, intervals: int, delta_reference: float = 1e-10,
                     progress_callback=None, chunk_intervals: int = 10) -> dict[str, Any]:
    """Benettin with the full six-state Euclidean norm and gamma fixed to zero."""
    reference = initial_state.astype(np.float64, copy=True)
    rng = np.random.default_rng(42)
    random_dir = rng.standard_normal(len(reference))
    random_dir = random_dir / np.linalg.norm(random_dir) * delta_reference
    perturbed = reference + random_dir
    accumulated = 0.0
    logarithms: list[float] = []
    steps_per_interval = max(1, int(round(tau / dt)))
    actual_dt = tau / steps_per_interval
    completed = 0
    started = time.perf_counter()
    while completed < intervals:
        count = min(max(1, chunk_intervals), intervals - completed)
        reference, perturbed, chunk_log, chunk_values, success = _benettin_chunk(
            reference, perturbed, masses, positions, eps, actual_dt, steps_per_interval, count, delta_reference,
        )
        if not success or len(chunk_values) != count:
            raise RuntimeError(f"Benettin falhou apos {completed + len(chunk_values)} renormalizacoes; resultado incompleto descartado.")
        accumulated += float(chunk_log)
        logarithms.extend(chunk_values.tolist())
        completed += count
        if progress_callback is not None:
            elapsed = time.perf_counter() - started
            remaining = elapsed * (intervals - completed) / max(completed, 1)
            progress_callback(completed, intervals, elapsed, remaining)
    return {
        "lambda": accumulated / (intervals * tau),
        "tau": tau,
        "renormalizacoes": completed,
        "tempo_total": completed * tau,
        "delta_benettin": delta_reference,
        "norma": "Euclidiana do estado completo (x,y,z,vx,vy,vz)",
        "logs": logarithms,
        "completo": completed == intervals,
    }


@njit(cache=True)
def _kepler_force_components(x: float, y: float, z: float) -> tuple[float, float, float]:
    inverse_radius_cubed = (x * x + y * y + z * z) ** -1.5
    return -x * inverse_radius_cubed, -y * inverse_radius_cubed, -z * inverse_radius_cubed


@njit(cache=True)
def _kepler_energy_components(state: np.ndarray) -> float:
    radius = math.sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2])
    speed_squared = state[3] * state[3] + state[4] * state[4] + state[5] * state[5]
    return 0.5 * speed_squared - 1.0 / radius


@njit(cache=True)
def _rk4_kepler_compiled(state: np.ndarray, dt: float) -> np.ndarray:
    x, y, z, vx, vy, vz = state
    ax1, ay1, az1 = _kepler_force_components(x, y, z)
    k1 = np.array([vx, vy, vz, ax1, ay1, az1])
    x2, y2, z2 = x + 0.5 * dt * vx, y + 0.5 * dt * vy, z + 0.5 * dt * vz
    ax2, ay2, az2 = _kepler_force_components(x2, y2, z2)
    k2 = np.array([vx + 0.5 * dt * ax1, vy + 0.5 * dt * ay1, vz + 0.5 * dt * az1, ax2, ay2, az2])
    x3, y3, z3 = x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2]
    ax3, ay3, az3 = _kepler_force_components(x3, y3, z3)
    k3 = np.array([vx + 0.5 * dt * k2[3], vy + 0.5 * dt * k2[4], vz + 0.5 * dt * k2[5], ax3, ay3, az3])
    x4, y4, z4 = x + dt * k3[0], y + dt * k3[1], z + dt * k3[2]
    ax4, ay4, az4 = _kepler_force_components(x4, y4, z4)
    k4 = np.array([vx + dt * k3[3], vy + dt * k3[4], vz + dt * k3[5], ax4, ay4, az4])
    result = np.empty(6, dtype=np.float64)
    for index in range(6):
        result[index] = state[index] + (dt / 6.0) * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
    return result


@njit(cache=True)
def _kepler_chunk(state: np.ndarray, acceleration_state: np.ndarray, dt: float, steps: int,
                  global_start: int, total_steps: int, sample_every: int,
                  method_code: int, energy_initial: float, track_energy: bool):
    sample_capacity = steps // sample_every + 2
    sample_times = np.empty(sample_capacity, dtype=np.float64)
    sample_energies = np.empty(sample_capacity, dtype=np.float64)
    sample_count = 0
    maximum_energy_error = 0.0
    field_evaluations = 0
    current = state.copy()
    current_acceleration = acceleration_state.copy()
    for local_step in range(1, steps + 1):
        if method_code == 0:
            current = _rk4_kepler_compiled(current, dt)
            field_evaluations += 4
        else:
            half_dt = 0.5 * dt
            vx_half = current[3] + half_dt * current_acceleration[0]
            vy_half = current[4] + half_dt * current_acceleration[1]
            vz_half = current[5] + half_dt * current_acceleration[2]
            next_x = current[0] + dt * vx_half
            next_y = current[1] + dt * vy_half
            next_z = current[2] + dt * vz_half
            ax, ay, az = _kepler_force_components(next_x, next_y, next_z)
            field_evaluations += 1
            current = np.array([next_x, next_y, next_z, vx_half + half_dt * ax, vy_half + half_dt * ay, vz_half + half_dt * az])
            current_acceleration = np.array([ax, ay, az])
        for component in range(6):
            if not math.isfinite(current[component]):
                return current, current_acceleration, maximum_energy_error, field_evaluations, sample_times[:sample_count], sample_energies[:sample_count], False
        global_step = global_start + local_step
        if track_energy:
            energy = _kepler_energy_components(current)
            maximum_energy_error = max(maximum_energy_error, abs(energy - energy_initial))
            if global_step % sample_every == 0 or global_step == total_steps:
                sample_times[sample_count] = global_step * dt
                sample_energies[sample_count] = energy
                sample_count += 1
    return current, current_acceleration, maximum_energy_error, field_evaluations, sample_times[:sample_count], sample_energies[:sample_count], True


def _run_part2_direction(initial: np.ndarray, method_code: int, dt: float, total_steps: int,
                         sample_every: int, track_energy: bool, progress_callback=None,
                         phase_name: str = "ida") -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    state = initial.copy()
    energy_initial = float(_kepler_energy_components(state))
    acceleration_state = np.asarray(_kepler_force_components(*state[:3]), dtype=np.float64)
    evaluations = 1 if method_code == 1 else 0
    times = [0.0] if track_energy else []
    energies = [energy_initial] if track_energy else []
    maximum_error = 0.0
    steps_done = 0
    started = time.perf_counter()
    chunk_size = 20_000
    while steps_done < total_steps:
        count = min(chunk_size, total_steps - steps_done)
        state, acceleration_state, chunk_maximum, chunk_evaluations, chunk_times, chunk_energies, finite = _kepler_chunk(
            state, acceleration_state, dt, count, steps_done, total_steps, sample_every,
            method_code, energy_initial, track_energy,
        )
        if not finite:
            raise RuntimeError(f"{phase_name}: estado nao finito no passo {steps_done + count}; simulacao incompleta descartada.")
        maximum_error = max(maximum_error, float(chunk_maximum))
        evaluations += int(chunk_evaluations)
        times.extend(chunk_times.tolist())
        energies.extend(chunk_energies.tolist())
        steps_done += count
        if progress_callback is not None:
            elapsed = time.perf_counter() - started
            remaining = elapsed * (total_steps - steps_done) / max(steps_done, 1)
            progress_callback(phase_name, steps_done, total_steps, elapsed, remaining)
    return state, np.asarray(times), np.asarray(energies), maximum_error, evaluations


def run_part2_experiment(periods: int, verlet_dt: float, sample_every: int, progress_callback=None) -> list[dict[str, Any]]:
    duration = periods * 2.0 * math.pi
    initial = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    energy_initial = float(_kepler_energy_components(initial))
    rows = []
    for name, method_code, requested_dt in (("RK4", 0, 4.0 * verlet_dt), ("Verlet", 1, verlet_dt)):
        steps = max(1, int(round(duration / requested_dt)))
        actual_dt = duration / steps
        final_state, sample_times, sample_energies, maximum_error, forward_evaluations = _run_part2_direction(
            initial, method_code, actual_dt, steps, sample_every, True,
            (lambda phase, done, total, elapsed, remaining: progress_callback(name, phase, done, total, elapsed, remaining)) if progress_callback else None,
            "ida",
        )
        reverse_start = final_state.copy()
        reverse_start[3:] *= -1.0
        reverse_end, _, _, _, reverse_evaluations = _run_part2_direction(
            reverse_start, method_code, actual_dt, steps, sample_every, False,
            (lambda phase, done, total, elapsed, remaining: progress_callback(name, phase, done, total, elapsed, remaining)) if progress_callback else None,
            "reversibilidade",
        )
        reverse_end[3:] *= -1.0
        rows.append({
            "metodo": name,
            "dt": actual_dt,
            "passos_por_direcao": steps,
            "avaliacoes_campo_ida": forward_evaluations,
            "avaliacoes_campo_reversibilidade": reverse_evaluations,
            "avaliacoes_campo_total": forward_evaluations + reverse_evaluations,
            "erro_energia_final": abs(float(sample_energies[-1]) - energy_initial),
            "deriva_energia_maxima": maximum_error,
            "erro_reversibilidade": float(np.linalg.norm(reverse_end - initial)),
            "tempos_grafico": sample_times,
            "energia_relativa_grafico": sample_energies - energy_initial,
        })
    return rows


def integrate_kepler_streaming(step_function, initial_state: np.ndarray, dt: float, duration: float, max_points: int = 4000, fast_step=None, progress_callback=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Integra uma orbita longa sem guardar um estado por passo."""
    steps = max(1, int(round(duration / dt)))
    actual_dt = duration / steps
    model = KeplerModel()
    state = initial_state.copy()
    initial_energy = kepler_energy_state(state)
    sample_stride = max(1, math.ceil(steps / max_points))
    sampled_times = [0.0]
    sampled_energies = [initial_energy]
    maximum_energy_error = 0.0
    selected_step = fast_step or step_function
    for step in range(1, steps + 1):
        state = selected_step(state, actual_dt) if fast_step is not None else selected_step(model, state, actual_dt)
        energy_error = abs(kepler_energy_state(state) - initial_energy)
        maximum_energy_error = max(maximum_energy_error, energy_error)
        if step % sample_stride == 0 or step == steps:
            sampled_times.append(step * actual_dt)
            sampled_energies.append(kepler_energy_state(state))
        if progress_callback is not None and (step % max(1, steps // 100) == 0 or step == steps):
            progress_callback(step / steps)
    return np.asarray(sampled_times), np.asarray(sampled_energies), state, maximum_energy_error


def reversibility_error_streaming(step_function, initial_state: np.ndarray, dt: float, duration: float, fast_step=None, progress_callback=None) -> float:
    _, _, forward, _ = integrate_kepler_streaming(step_function, initial_state, dt, duration, max_points=1, fast_step=fast_step)
    reversed_state = forward.copy()
    reversed_state[3:] *= -1.0
    _, _, backward, _ = integrate_kepler_streaming(step_function, reversed_state, dt, duration, max_points=1, fast_step=fast_step, progress_callback=progress_callback)
    backward[3:] *= -1.0
    return float(np.linalg.norm(backward - initial_state))


def kepler_energy_state(state: np.ndarray) -> float:
    position = state[:3]
    velocity = state[3:]
    return 0.5 * np.dot(velocity, velocity) - 1.0 / np.linalg.norm(position)


def angular_momentum_state(state: np.ndarray) -> np.ndarray:
    return np.cross(state[:3], state[3:])


def lrl_vector(state: np.ndarray) -> np.ndarray:
    position = state[:3]
    velocity = state[3:]
    angular_momentum = np.cross(position, velocity)
    return np.cross(velocity, angular_momentum) - position / np.linalg.norm(position)


def integrate_kepler_history(step_function, initial_state: np.ndarray, dt: float, final_time: float) -> tuple[np.ndarray, np.ndarray]:
    steps = max(1, int(round(final_time / dt)))
    actual_dt = final_time / steps
    model = KeplerModel()
    history = np.empty((steps + 1, 6), dtype=np.float64)
    history[0] = initial_state
    state = initial_state.copy()
    for index in range(steps):
        state = step_function(model, state, actual_dt)
        history[index + 1] = state
    times = np.linspace(0.0, final_time, steps + 1)
    return times, history


def exact_kepler_history(radius: float, eccentricity: float, times: np.ndarray) -> np.ndarray:
    """Solucao analitica no plano xy para a orbita iniciada no periastro."""
    mean_motion = radius ** -1.5
    mean_anomaly = mean_motion * times
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(12):
        eccentric_anomaly -= (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly) / (1.0 - eccentricity * np.cos(eccentric_anomaly))

    denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
    position_x = radius * (np.cos(eccentric_anomaly) - eccentricity)
    position_y = radius * np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly)
    velocity_x = -radius * mean_motion * np.sin(eccentric_anomaly) / denominator
    velocity_y = radius * mean_motion * np.sqrt(1.0 - eccentricity**2) * np.cos(eccentric_anomaly) / denominator
    return np.column_stack((position_x, position_y, np.zeros_like(times), velocity_x, velocity_y, np.zeros_like(times)))


def circular_report_rows(radius: float, dt: float) -> list[dict[str, float | str]]:
    rows = []
    period = 2.0 * math.pi * radius**1.5
    initial_state = elliptic_initial_state(radius, 0.0)
    for name, step_function in INTEGRATORS.items():
        times, history = integrate_kepler_history(step_function, initial_state, dt, period)
        exact_history = exact_kepler_history(radius, 0.0, times)
        initial_energy = kepler_energy_state(initial_state)
        initial_angular_momentum = angular_momentum_state(initial_state)
        rows.append(
            {
                "metodo": name,
                "dt": dt,
                "erro_radial_maximo": float(np.max(np.abs(np.linalg.norm(history[:, :3], axis=1) - radius))),
                "erro_energia_maximo": float(np.max(np.abs([kepler_energy_state(state) - initial_energy for state in history]))),
                "erro_momento_angular_maximo": float(np.max([np.linalg.norm(angular_momentum_state(state) - initial_angular_momentum) for state in history])),
                "erro_LRL_maximo": float(np.max([np.linalg.norm(lrl_vector(state) - lrl_vector(initial_state)) for state in history])),
                "erro_trajetoria_analitica_maximo": float(np.max(np.linalg.norm(history - exact_history, axis=1))),
                "ordem_observada": observed_order(step_function, radius, dt),
            }
        )
    return rows


def lrl_angle_history(history: np.ndarray) -> np.ndarray:
    angles = []
    for state in history:
        vector = lrl_vector(state)
        if np.linalg.norm(vector) <= 1e-14:
            angles.append(float("nan"))
        else:
            angles.append(math.atan2(vector[1], vector[0]))
    angles_array = np.asarray(angles, dtype=np.float64)
    if np.all(np.isnan(angles_array)):
        return angles_array
    valid = ~np.isnan(angles_array)
    angles_array[valid] = np.unwrap(angles_array[valid])
    return angles_array


# ===========================================================================
# [CODIGO LEGADO / DIAGNOSTICO LOCAL]
# Rotina de diagnostico temporal isolado da evolucao do vetor LRL e precessao orbital.
# Mantida especificamente para o grafico auxiliar na Aba Parte 1; os calculos de tabela
# oficiais da Parte 1 utilizam kepler_report_rows() e refinement_rows().
# ===========================================================================
def kepler_time_diagnostics(radius: float, eccentricity: float, method_name: str, dt: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    period = 2.0 * math.pi * radius**1.5
    initial_state = elliptic_initial_state(radius, eccentricity)
    step_function = INTEGRATORS[method_name]
    times, history = integrate_kepler_history(step_function, initial_state, dt, period)
    initial_energy = kepler_energy_state(initial_state)
    initial_angular_momentum = angular_momentum_state(initial_state)
    initial_lrl = lrl_vector(initial_state)
    energy_error = np.asarray([abs(kepler_energy_state(state) - initial_energy) for state in history])
    angular_error = np.asarray([np.linalg.norm(angular_momentum_state(state) - initial_angular_momentum) for state in history])
    lrl_error = np.asarray([np.linalg.norm(lrl_vector(state) - initial_lrl) for state in history])
    angle_history = lrl_angle_history(history)
    if np.all(np.isnan(angle_history)):
        precession = float("nan")
    else:
        valid_angles = angle_history[~np.isnan(angle_history)]
        precession = float((valid_angles[-1] - valid_angles[0]) / period)
    summary = {
        "erro_E_maximo": float(np.max(energy_error)),
        "erro_L_maximo": float(np.max(angular_error)),
        "erro_A_maximo": float(np.max(lrl_error)),
        "precessao_rad_por_periodo": precession * period,
        "taxa_precessao_rad_por_tempo": precession,
    }
    return times, np.column_stack((energy_error, angular_error, lrl_error, angle_history)), summary


def refinement_rows(radius: float, eccentricities: list[float], dt: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for eccentricity in eccentricities:
        initial_state = elliptic_initial_state(radius, eccentricity)
        period = 2.0 * math.pi * radius**1.5
        for method_name, step_function in INTEGRATORS.items():
            errors = []
            dts = [dt, dt / 2.0, dt / 4.0, dt / 8.0]
            for current_dt in dts:
                _, history = integrate_kepler_history(step_function, initial_state, current_dt, period)
                errors.append(float(np.linalg.norm(history[-1] - initial_state)))
            rows.append(
                {
                    "excentricidade": eccentricity,
                    "metodo": method_name,
                    "dt": dts[0],
                    "erro_dt": errors[0],
                    "erro_dt_2": errors[1],
                    "ordem_dt_dt_2": math.log(errors[0] / errors[1], 2.0) if errors[0] > 0 and errors[1] > 0 else float("nan"),
                    "erro_dt_4": errors[2],
                    "ordem_dt_2_dt_4": math.log(errors[1] / errors[2], 2.0) if errors[1] > 0 and errors[2] > 0 else float("nan"),
                    "erro_dt_8": errors[3],
                    "ordem_dt_4_dt_8": math.log(errors[2] / errors[3], 2.0) if errors[2] > 0 and errors[3] > 0 else float("nan"),
                }
            )
    return rows


def kepler_report_rows(radius: float, dt: float, eccentricities: tuple[float, ...] | list[float] = (0.0, 0.5, 0.9)) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    period = 2.0 * math.pi * radius**1.5
    for eccentricity in eccentricities:
        initial_state = elliptic_initial_state(radius, eccentricity)
        initial_energy = kepler_energy_state(initial_state)
        initial_angular_momentum = angular_momentum_state(initial_state)
        initial_lrl = lrl_vector(initial_state)
        for name, step_function in INTEGRATORS.items():
            times, history = integrate_kepler_history(step_function, initial_state, dt, period)
            _, half_history = integrate_kepler_history(step_function, initial_state, dt / 2.0, period)
            exact_history = exact_kepler_history(radius, eccentricity, times)
            final_error = float(np.linalg.norm(history[-1] - initial_state))
            half_error = float(np.linalg.norm(half_history[-1] - initial_state))
            energy_error = np.abs(np.array([kepler_energy_state(state) for state in history]) - initial_energy)
            angular_error = np.array([np.linalg.norm(angular_momentum_state(state) - initial_angular_momentum) for state in history])
            lrl_error = np.array([np.linalg.norm(lrl_vector(state) - initial_lrl) for state in history])
            numerical_energies = np.array([kepler_energy_state(state) for state in history])
            numerical_angular_momenta = np.array([np.linalg.norm(angular_momentum_state(state)) for state in history])
            final_energy = float(numerical_energies[-1])
            final_angular_momentum = float(numerical_angular_momenta[-1])
            measured_semi_major_axis = -1.0 / (2.0 * final_energy)
            measured_eccentricity = math.sqrt(max(0.0, 1.0 + 2.0 * final_energy * final_angular_momentum**2))
            lrl_angles = []
            for state in history:
                current_lrl = lrl_vector(state)
                if np.linalg.norm(initial_lrl) <= 1e-14:
                    lrl_angles.append(float("nan"))
                else:
                    cosine = np.dot(current_lrl, initial_lrl) / max(np.linalg.norm(current_lrl) * np.linalg.norm(initial_lrl), 1e-30)
                    lrl_angles.append(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            rows.append(
                {
                    "excentricidade": eccentricity,
                    "metodo": name,
                    "dt": dt,
                    "periodo": period,
                    "passos": len(history) - 1,
                    "E": final_energy,
                    "modulo_L": final_angular_momentum,
                    "a_medido": measured_semi_major_axis,
                    "e_medido": measured_eccentricity,
                    "erro_estado_final": final_error,
                    "erro_estado_meio_dt": half_error,
                    "ordem_observada": math.log(final_error / half_error, 2.0) if final_error > 0.0 and half_error > 0.0 else float("nan"),
                    "erro_energia_maximo": float(np.max(energy_error)),
                    "erro_momento_angular_maximo": float(np.max(angular_error)),
                    "erro_LRL_maximo": float(np.max(lrl_error)),
                    "erro_trajetoria_analitica_maximo": float(np.max(np.linalg.norm(history - exact_history, axis=1))),
                    "rotacao_LRL_maxima_rad": float(np.nanmax(lrl_angles)) if not np.all(np.isnan(lrl_angles)) else float("nan"),
                    "taxa_rotacao_LRL_rad_por_tempo": (float(np.nanmax(lrl_angles)) / period) if not np.all(np.isnan(lrl_angles)) else float("nan"),
                }
            )
    return rows


def plane_invariant_report(dt: float, duration: float, eps: float, gamma: float) -> dict[str, float | str]:
    masses = np.array([0.4, 0.35, 0.25], dtype=np.float64)
    positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, -1.2, 0.0]], dtype=np.float64)
    state = np.array([0.2, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    steps = max(1, int(round(duration / dt)))
    actual_dt = duration / steps
    maximum_z = 0.0
    maximum_vz = 0.0
    for _ in range(steps):
        state = rk4_step(state, actual_dt, masses, positions, eps, gamma)
        maximum_z = max(maximum_z, abs(float(state[2])))
        maximum_vz = max(maximum_vz, abs(float(state[5])))
    return {"configuracao": "3 massas coplanares, z(0)=vz(0)=0", "tempo": duration, "dt": actual_dt, "max_abs_z": maximum_z, "max_abs_vz": maximum_vz}


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = StringIO()
    columns = list(rows[0])
    output.write(",".join(columns) + "\n")
    for row in rows:
        output.write(",".join(str(row.get(column, "")) for column in columns) + "\n")
    return output.getvalue()


def classify(
    initial_state: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    eps: float,
    gamma: float,
    tmax: float,
    rcap: float,
    rescape: float,
    method: str,
    dt: float,
    return_solution: bool = False,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    detect_events: bool = True,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    if method == "RK45":
        events = []
        for index in range(len(masses)):
            def capture_event(_time: float, state: np.ndarray, index: int = index) -> float:
                return np.dot(state[:3] - positions[index], state[:3] - positions[index]) - rcap**2

            capture_event.terminal = True
            capture_event.direction = -1
            events.append(capture_event)

        def escape_event(_time: float, state: np.ndarray) -> float:
            return np.linalg.norm(state[:3]) - rescape

        escape_event.terminal = False
        escape_event.direction = 1
        solution = solve_ivp(
            lambda _time, state: rhs(state, masses, positions, eps, gamma),
            (0.0, tmax),
            initial_state,
            method="RK45",
            rtol=rtol,
            atol=atol,
            max_step=dt,
            events=[*events, escape_event],
        )
        result = -2
        for index, event_times in enumerate(solution.t_events[:-1]):
            if len(event_times):
                result = index
                break
        if result == -2:
            final_state = solution.y[:, -1]
            distance = np.linalg.norm(final_state[:3])
            energy = total_energy(final_state, masses, positions, eps)
            if np.dot(final_state[:3], final_state[3:]) > 0.0 and energy > 0.0 and distance > rescape:
                result = -1
        if return_solution:
            return result, solution.t, solution.y
        return result, None, None

    state = initial_state.copy()
    times = [0.0]
    states = [state.copy()]
    step_function = verlet_step if method == "Verlet" else rk4_step
    elapsed = 0.0
    result = -2
    while elapsed < tmax:
        step_dt = min(dt, tmax - elapsed)
        next_state = step_function(state, step_dt, masses, positions, eps, gamma)
        event_fraction = None
        for index, attractor in enumerate(positions):
            old_distance = np.linalg.norm(state[:3] - attractor)
            new_distance = np.linalg.norm(next_state[:3] - attractor)
            if old_distance >= rcap and new_distance < rcap:
                if detect_events:
                    fraction, event_state = locate_capture_bisection(state, next_state, attractor, rcap, step_dt)
                else:
                    fraction, event_state = 1.0, next_state
                if event_fraction is None or fraction < event_fraction[0]:
                    event_fraction = (fraction, index, event_state)
        if event_fraction is not None:
            fraction, result, state = event_fraction
            elapsed += fraction * step_dt
            times.append(elapsed)
            states.append(state.copy())
            break
        state = next_state
        elapsed += step_dt
        times.append(elapsed)
        states.append(state.copy())
        distance = np.linalg.norm(state[:3])
        energy = total_energy(state, masses, positions, eps)
        if distance > rescape and np.dot(state[:3], state[3:]) > 0.0 and energy > 0.0:
            result = -1
            break
    if return_solution:
        data = np.asarray(states).T
        return result, np.asarray(times), data
    return result, None, None


def compute_basin(masses: np.ndarray, positions: np.ndarray, resolution: int, eps: float, gamma: float, tmax: float, rcap: float, rescape: float, method: str, dt: float, xlim: tuple[float, float] = (-3.0, 3.0), ylim: tuple[float, float] = (-3.0, 3.0), rtol: float = 1e-7, atol: float = 1e-9, detect_events: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_x = np.linspace(xlim[0], xlim[1], resolution)
    values_y = np.linspace(ylim[0], ylim[1], resolution)
    basin = np.full((resolution, resolution), -2, dtype=np.int8)
    for row, y in enumerate(values_y):
        for column, x in enumerate(values_x):
            initial = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
            basin[row, column] = classify(initial, masses, positions, eps, gamma, tmax, rcap, rescape, method, dt, rtol=rtol, atol=atol, detect_events=detect_events)[0]
    return basin, values_x, values_y


def draw_basin(basin: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, positions: np.ndarray) -> plt.Figure:
    colors = np.full((*basin.shape, 3), 0.5)
    palette = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(10, len(positions))))[:, :3]
    for index in range(len(positions)):
        colors[basin == index] = palette[index]
    colors[basin == -1] = [0.0, 0.0, 0.0]
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(colors, origin="lower", extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]], interpolation="nearest", aspect="equal")
    axis.scatter(positions[:, 0], positions[:, 1], c=palette[: len(positions)], edgecolors="black", s=100)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Bacias de atracao")
    figure.tight_layout()
    return figure


def basin_fractions(basin: np.ndarray, number_of_attractors: int) -> np.ndarray:
    """Fractions for every attractor plus escaped and unresolved pixels."""
    return np.array(
        [(basin == index).mean() for index in range(number_of_attractors)]
        + [float(np.mean(basin == -1)), float(np.mean(basin == -2))],
        dtype=np.float64,
    )


def compare_basin_pair(previous: np.ndarray, current: np.ndarray, number_of_attractors: int) -> tuple[float, float]:
    if previous.shape != current.shape:
        raise ValueError("Pi requer bacias da mesma resolucao e grade.")
    pointwise = float(np.mean(previous != current))
    previous_measure = basin_fractions(previous, number_of_attractors)
    current_measure = basin_fractions(current, number_of_attractors)
    return pointwise, float(np.max(np.abs(previous_measure - current_measure)))


def compute_part5_sensitivity(masses: np.ndarray, positions: np.ndarray, resolution: int, eps: float, gamma: float,
                              tmax: float, rcap: float, rescape: float, method: str, dt: float,
                              xlim: tuple[float, float], ylim: tuple[float, float], rtol: float, atol: float,
                              tests: list[tuple[str, float, float, float, float, float]], progress_callback=None) -> tuple[list[dict[str, float | str]], float]:
    rows = []
    started = time.perf_counter()
    for test_index, (parameter, value, eps_value, gamma_value, tmax_value, rcap_value) in enumerate(tests, start=1):
        basin, _, _, _ = cached_basin_for_ui(
            masses, positions, resolution, eps_value, gamma_value, tmax_value, rcap_value,
            rescape, method, dt, xlim, ylim, rtol, atol,
        )
        statistics = basin_statistics(basin, len(positions))
        captured = sum(float(statistics[f"fracao_atrator_{index + 1}"]) for index in range(len(positions)))
        escaped = float(statistics["fracao_escapadas"])
        unresolved = float(statistics["fracao_nao_resolvidas"])
        rows.append({
            "parametro": parameter,
            "valor": value,
            "fracao_capturada": captured,
            "fracao_escape": escaped,
            "fracao_cinza": unresolved,
            "soma_fracoes": captured + escaped + unresolved,
        })
        if progress_callback is not None:
            elapsed = time.perf_counter() - started
            remaining = elapsed * (len(tests) - test_index) / max(test_index, 1)
            progress_callback(test_index, len(tests), parameter, value, elapsed, remaining)
    return rows, time.perf_counter() - started


def classify_part5_pair(initial_state: np.ndarray, masses: np.ndarray, positions: np.ndarray, eps: float, gamma: float,
                        tmax: float, rcap: float, rescape: float, method: str, dt: float,
                        rtol: float, atol: float) -> tuple[int, int]:
    """Runs one trajectory and returns naive/event labels from that same path."""
    state = initial_state.astype(np.float64, copy=True)
    start_distances = np.linalg.norm(positions - state[:3], axis=1)
    starting_inside = np.flatnonzero(start_distances <= rcap)
    if len(starting_inside):
        attractor = int(starting_inside[np.argmin(start_distances[starting_inside])])
        return attractor, attractor
    naive_code = -2
    event_code = -2
    elapsed = 0.0

    if method == "RK45":
        solution = solve_ivp(
            lambda _time, current_state: rhs(current_state, masses, positions, eps, gamma),
            (0.0, tmax), state, method="RK45", rtol=rtol, atol=atol, max_step=dt, dense_output=True,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        for step_index in range(1, len(solution.t)):
            previous_time = float(solution.t[step_index - 1])
            current_time = float(solution.t[step_index])
            step_dt = current_time - previous_time
            previous_state = solution.y[:, step_index - 1]
            current_state = solution.y[:, step_index]
            dense = lambda fraction, t0=previous_time, h=step_dt: solution.sol(t0 + fraction * h)
            if event_code == -2:
                capture = first_capture_event(previous_state, current_state, positions, rcap, step_dt, dense)
                if capture is not None:
                    event_code = capture[1]
            if naive_code == -2:
                distances = np.linalg.norm(positions - current_state[:3], axis=1)
                inside = np.flatnonzero(distances <= rcap)
                if len(inside):
                    naive_code = int(inside[np.argmin(distances[inside])])
            if escape_conditions(current_state, positions, masses, eps, rescape):
                if naive_code == -2:
                    naive_code = -1
                if event_code == -2:
                    event_code = -1
            elapsed = current_time
            if naive_code != -2 and event_code != -2:
                break
        return naive_code, event_code

    while elapsed < tmax and (naive_code == -2 or event_code == -2):
        step_dt = min(dt, tmax - elapsed)
        previous = state
        current = verlet_step(state, step_dt, masses, positions, eps, gamma) if method == "Verlet" else rk4_step(state, step_dt, masses, positions, eps, gamma)
        if event_code == -2:
            capture = first_capture_event(previous, current, positions, rcap, step_dt)
            if capture is not None:
                event_code = capture[1]
        if naive_code == -2:
            distances = np.linalg.norm(positions - current[:3], axis=1)
            inside = np.flatnonzero(distances <= rcap)
            if len(inside):
                naive_code = int(inside[np.argmin(distances[inside])])
        state = current
        elapsed += step_dt
        if escape_conditions(state, positions, masses, eps, rescape):
            if naive_code == -2:
                naive_code = -1
            if event_code == -2:
                event_code = -1
    return naive_code, event_code


def compute_part5_event_basins(masses: np.ndarray, positions: np.ndarray, resolution: int, eps: float, gamma: float,
                               tmax: float, rcap: float, rescape: float, method: str, dt: float,
                               xlim: tuple[float, float], ylim: tuple[float, float], rtol: float, atol: float,
                               progress_callback=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.linspace(xlim[0], xlim[1], resolution)
    y_values = np.linspace(ylim[0], ylim[1], resolution)
    initial_x, initial_y = np.meshgrid(x_values, y_values)
    initial_points = np.column_stack((initial_x.ravel(), initial_y.ravel()))
    naive = np.full(len(initial_points), -2, dtype=np.int8)
    event = np.full(len(initial_points), -2, dtype=np.int8)
    for index, (x, y) in enumerate(initial_points):
        initial_state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        naive[index], event[index] = classify_part5_pair(initial_state, masses, positions, eps, gamma, tmax, rcap, rescape, method, dt, rtol, atol)
        if progress_callback is not None and ((index + 1) % resolution == 0 or index + 1 == len(initial_points)):
            progress_callback(index + 1, len(initial_points))
    return naive.reshape(resolution, resolution), event.reshape(resolution, resolution), x_values, y_values


def sidebar_parameters() -> dict[str, Any]:
    st.sidebar.header("Configuracao")
    masses_text = st.sidebar.text_area("Massas", json.dumps(DEFAULT_MASSES), height=80)
    positions_text = st.sidebar.text_area("Posicoes 3D", json.dumps(DEFAULT_POSITIONS), height=150)
    st.sidebar.caption("Use ponto ou virgula decimal. Nao ha limite superior artificial; valores extremos podem custar mais tempo ou causar instabilidade.")
    resolution_text = st.sidebar.text_input("Resolucao da imagem", "24")
    eps_text = st.sidebar.text_input("eps", "0.18")
    gamma_text = st.sidebar.text_input("gamma", "0.15")
    tmax_text = st.sidebar.text_input("tmax", "20.0")
    rcap_text = st.sidebar.text_input("raio de captura", "0.22")
    rescape_text = st.sidebar.text_input("raio de escape", "20.0")
    dt_text = st.sidebar.text_input("passo dt", "0.05")
    rtol_text = st.sidebar.text_input("tolerancia relativa RK45", "1e-7")
    atol_text = st.sidebar.text_input("tolerancia absoluta RK45", "1e-9")
    xmin_text = st.sidebar.text_input("x minimo do corte", "-3.0")
    xmax_text = st.sidebar.text_input("x maximo do corte", "3.0")
    ymin_text = st.sidebar.text_input("y minimo do corte", "-3.0")
    ymax_text = st.sidebar.text_input("y maximo do corte", "3.0")
    method = st.sidebar.selectbox("Integrador da simulacao", ["RK4", "Verlet", "RK45"])
    return {
        "masses_text": masses_text,
        "positions_text": positions_text,
        "resolution_text": resolution_text,
        "eps_text": eps_text,
        "gamma_text": gamma_text,
        "tmax_text": tmax_text,
        "rcap_text": rcap_text,
        "rescape_text": rescape_text,
        "dt_text": dt_text,
        "rtol_text": rtol_text,
        "atol_text": atol_text,
        "xmin_text": xmin_text,
        "xmax_text": xmax_text,
        "ymin_text": ymin_text,
        "ymax_text": ymax_text,
        "method": method,
    }


def parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
    parsed = {
        "resolution": parse_int(settings["resolution_text"], "Resolucao", minimum=2),
        "eps": parse_float(settings["eps_text"], "eps", nonnegative=True),
        "gamma": parse_float(settings["gamma_text"], "gamma", nonnegative=True),
        "tmax": parse_float(settings["tmax_text"], "tmax", nonnegative=True),
        "rcap": parse_float(settings["rcap_text"], "raio de captura", nonnegative=True),
        "rescape": parse_float(settings["rescape_text"], "raio de escape", positive=True),
        "dt": parse_float(settings["dt_text"], "dt", positive=True),
        "rtol": parse_float(settings["rtol_text"], "tolerancia relativa", positive=True),
        "atol": parse_float(settings["atol_text"], "tolerancia absoluta", positive=True),
        "xmin": parse_float(settings["xmin_text"], "x minimo"),
        "xmax": parse_float(settings["xmax_text"], "x maximo"),
        "ymin": parse_float(settings["ymin_text"], "y minimo"),
        "ymax": parse_float(settings["ymax_text"], "y maximo"),
    }
    if parsed["xmax"] <= parsed["xmin"] or parsed["ymax"] <= parsed["ymin"]:
        raise ValueError("Os limites maximos do corte devem ser maiores que os limites minimos.")
    return parsed


def main() -> None:
    st.title("Bacias de Atracao Gravitacionais")
    st.caption("Laboratorio interativo das Partes 0 a 6")
    settings = sidebar_parameters()
    try:
        masses, positions = parse_model(settings["masses_text"], settings["positions_text"])
        numeric = parse_settings(settings)
        st.sidebar.success(f"Configuracao valida: {len(masses)} massas; soma={masses.sum():.3f}")

        # [1] Verificacao do produto misto (nao-coplanaridade)
        if len(positions) >= 4:
            mixed_volume = float(np.dot(positions[1] - positions[0], np.cross(positions[2] - positions[0], positions[3] - positions[0])))
            if abs(mixed_volume) < 1e-4:
                st.sidebar.error(f"Alerta: Massas coplanares! Volume do tetraedro = {abs(mixed_volume):.4e} ≈ 0. O corte 2D pode ser invariante.")
            else:
                st.sidebar.info(f"Produto misto (nao-coplanaridade): {abs(mixed_volume):.4f} (dinamica 3D garantida)")

        # [8] Warning no sidebar se eps == 0 ou eps > 1.0
        if numeric["eps"] == 0.0:
            st.sidebar.warning("Aviso (modelo): eps = 0 remove o amaciamento gravitacional, podendo gerar singularidade (|r - R| -> 0) e instabilidade numerica.")
        elif numeric["eps"] > 1.0:
            st.sidebar.warning(f"Aviso (modelo): eps = {numeric['eps']:.2g} e muito grande (> 1.0), distorcendo a atracao gravitacional newtoniana.")

        # [7] Bloqueio do RK45 para resolucao > 100
        if settings["method"] == "RK45" and numeric["resolution"] > 100:
            st.sidebar.error(f"Bloqueio: Metodo RK45 selecionado com resolucao={numeric['resolution']} > 100. Altere o metodo para RK4 ou Verlet vetorizado para evitar sobrecarga excessiva no solve_ivp.")
    except (ValueError, json.JSONDecodeError) as error:
        st.sidebar.error(str(error))
        st.stop()

    tabs = st.tabs(["Parte 0", "Parte 1", "Parte 2", "Parte 3", "Parte 4", "Parte 5", "Parte 6", "Itens Bônus"])

    with tabs[0]:
        st.header("Analise no papel")
        st.markdown("Consulte e complete as deducoes em `parte0_analise.md`. Esta aba resume as verificacoes matematicas que devem acompanhar os experimentos.")
        st.latex(r"\frac{dE}{dt}=-\gamma |v|^2")
        st.latex(r"\mathcal{R}(t)=|E(t)-E(0)+\gamma\int_0^t|v|^2dt|")

    with tabs[1]:
        st.header("Verificacao de Kepler")
        st.markdown("A execucao abaixo gera diretamente os numeros da Parte 1 para copiar ou baixar no relatorio.")
        radius_text = st.text_input("Raio orbital a", "1.0", key="kepler_radius")
        kepler_dt_text = st.text_input("dt da tabela de ordem", numeric["dt"].__format__(".16g"), key="kepler_dt")
        eccentricities_text = st.text_input("Excentricidades, separadas por virgula", "0, 0.5, 0.9", key="kepler_eccentricities")
        plane_duration_text = st.text_input("Tempo do teste do plano invariante", "100.0", key="plane_duration")
        if st.button("Gerar dados completos da Parte 1"):
            radius = parse_float(radius_text, "raio da orbita", positive=True)
            kepler_dt = parse_float(kepler_dt_text, "dt de Kepler", positive=True)
            eccentricities = parse_float_list(eccentricities_text, "excentricidades", nonnegative=True)
            if any(value >= 1.0 for value in eccentricities):
                st.error("As excentricidades devem satisfazer 0 <= e < 1 para orbitas elipticas.")
            else:
                circular_rows = circular_report_rows(radius, kepler_dt)
                st.subheader("Orbita circular: e=0")
                st.dataframe(circular_rows, use_container_width=True)
                st.download_button("Baixar dados da orbita circular (CSV)", rows_to_csv(circular_rows), "parte1_orbita_circular.csv", "text/csv")

                circular_figure, circular_axes = plt.subplots(1, 4, figsize=(17, 4.5))
                circular_columns = ["erro_radial_maximo", "erro_energia_maximo", "erro_momento_angular_maximo", "erro_LRL_maximo"]
                circular_titles = ["Erro radial", "Erro de energia", "Erro de momento angular", "Erro LRL"]
                for axis, column, title in zip(circular_axes, circular_columns, circular_titles):
                    axis.bar([row["metodo"] for row in circular_rows], [row[column] for row in circular_rows])
                    axis.set_title(title)
                    axis.tick_params(axis="x", rotation=45)
                    axis.set_yscale("log")
                    axis.grid(True, which="both", alpha=0.25)
                circular_figure.tight_layout()
                st.pyplot(circular_figure)
                plt.close(circular_figure)

                elliptic_eccentricities = [value for value in eccentricities if value > 0.0]
                elliptic_rows = [row for row in kepler_report_rows(radius, kepler_dt, elliptic_eccentricities)]
                st.subheader("Orbitas elipticas: e=0,5 e e=0,9")
                st.dataframe(elliptic_rows, use_container_width=True)
                st.download_button("Baixar dados das orbitas elipticas (CSV)", rows_to_csv(elliptic_rows), "parte1_orbitas_elipticas.csv", "text/csv")

                elliptic_figure, elliptic_axes = plt.subplots(1, 4, figsize=(17, 4.5))
                elliptic_columns = ["erro_energia_maximo", "erro_momento_angular_maximo", "erro_LRL_maximo", "erro_trajetoria_analitica_maximo"]
                elliptic_titles = ["Erro de energia", "Erro de momento angular", "Erro LRL", "Erro de trajetoria analitica"]
                for axis, column, title in zip(elliptic_axes, elliptic_columns, elliptic_titles):
                    for method_name in INTEGRATORS:
                        method_rows = [row for row in elliptic_rows if row["metodo"] == method_name]
                        axis.plot([row["excentricidade"] for row in method_rows], [row[column] for row in method_rows], marker="o", label=method_name)
                    axis.set_title(title)
                    axis.set_xlabel("e")
                    axis.set_yscale("log")
                    axis.grid(True, which="both", alpha=0.25)
                elliptic_axes[0].legend(fontsize="small")
                elliptic_figure.tight_layout()
                st.pyplot(elliptic_figure)
                plt.close(elliptic_figure)

                st.subheader("Refinamento temporal: dt -> dt/2")
                refinement = refinement_rows(radius, eccentricities, kepler_dt)
                st.dataframe(refinement, use_container_width=True)
                st.download_button("Baixar refinamento temporal (CSV)", rows_to_csv(refinement), "parte1_refinamento_dt.csv", "text/csv")
                refinement_figure, refinement_axis = plt.subplots(figsize=(8, 4.5))
                for eccentricity in eccentricities:
                    for method_name in INTEGRATORS:
                        row = next(item for item in refinement if item["excentricidade"] == eccentricity and item["metodo"] == method_name)
                        refinement_axis.loglog([kepler_dt, kepler_dt / 2.0, kepler_dt / 4.0, kepler_dt / 8.0], [row["erro_dt"], row["erro_dt_2"], row["erro_dt_4"], row["erro_dt_8"]], marker="o", label=f"{method_name}, e={eccentricity:g}")
                refinement_axis.set_xlabel("dt")
                refinement_axis.set_ylabel("erro do estado ao fim de um periodo")
                refinement_axis.set_title("Refinamento temporal, separado da dificuldade da orbita")
                refinement_axis.grid(True, which="both", alpha=0.25)
                refinement_axis.legend(fontsize="small", ncol=2)
                refinement_figure.tight_layout()
                st.pyplot(refinement_figure)
                plt.close(refinement_figure)

                st.subheader("Precessao e comportamento do vetor A")
                diagnostic_eccentricity = eccentricities[-1] if eccentricities else 0.5
                diagnostic_method = "RK4"
                diagnostic_times, diagnostic_data, diagnostic_summary = kepler_time_diagnostics(radius, diagnostic_eccentricity, diagnostic_method, kepler_dt)
                st.dataframe([diagnostic_summary], use_container_width=True)
                diagnostic_figure, diagnostic_axes = plt.subplots(1, 2, figsize=(12, 4.5))
                diagnostic_axes[0].semilogy(diagnostic_times, diagnostic_data[:, 2], label="|A(t)-A(0)|")
                diagnostic_axes[0].set_xlabel("tempo")
                diagnostic_axes[0].set_ylabel("erro no vetor A")
                diagnostic_axes[0].set_title(f"Comportamento de A, RK4, e={diagnostic_eccentricity:g}")
                diagnostic_axes[0].grid(True, which="both", alpha=0.25)
                diagnostic_axes[1].plot(diagnostic_times, diagnostic_data[:, 3], label="angulo de A")
                diagnostic_axes[1].set_xlabel("tempo")
                diagnostic_axes[1].set_ylabel("angulo (rad)")
                diagnostic_axes[1].set_title("Precessao do periélio")
                diagnostic_axes[1].grid(True, alpha=0.25)
                diagnostic_figure.tight_layout()
                st.pyplot(diagnostic_figure)
                plt.close(diagnostic_figure)

                plane_duration = parse_float(plane_duration_text, "tempo do plano", positive=True)
                plane_row = plane_invariant_report(kepler_dt, plane_duration, numeric["eps"], numeric["gamma"])
                st.subheader("Tabela 2: teste do plano invariante")
                st.dataframe([plane_row], use_container_width=True)
                st.download_button("Baixar teste do plano (CSV)", rows_to_csv([plane_row]), "parte1_plano_invariante.csv", "text/csv")
                st.info("Para e=0, a direcao do vetor LRL nao existe; por isso a rotacao angular aparece como NaN. Use erro_LRL_maximo para esse caso.")

    with tabs[2]:
        st.header("Energia e reversibilidade")
        st.info(
            f"Experimento oficial isolado: {PART2_MODEL_DESCRIPTION}. "
            f"A configuracao geral da barra lateral nao e usada nesta aba. "
            "O padrão e 10.000 periodos; reduza esse valor para teste preliminar. "
            "RK4 usa quatro vezes o passo do Verlet para igualar o custo de forca."
        )
        part2_periods_text = st.text_input("Periodos", str(PART2_PERIODS), key="part2_periods")
        part2_dt_text = st.text_input("dt do Velocity Verlet", f"{PART2_VERLET_DT:g}", key="part2_verlet_dt")
        sample_every_text = st.text_input("Amostrar grafico a cada N passos", "1000", key="part2_sample_every")
        if st.button("Executar Parte 2"):
            try:
                part2_periods = parse_int(part2_periods_text, "periodos", minimum=1)
                part2_dt = parse_float(part2_dt_text, "dt do Verlet", positive=True)
                sample_every = parse_int(sample_every_text, "frequencia de amostragem", minimum=1)
                cache_key = (part2_periods, part2_dt, sample_every, PART2_MODEL_DESCRIPTION, "RK4-Verlet-cost-v2")
                cached_key = st.session_state.get("part2_result_key")
                if cached_key == cache_key:
                    st.info("Usando o resultado deterministico ja calculado para estes mesmos parametros.")
                else:
                    progress = st.progress(0.0, text="Preparando kernels compilados...")
                    progress_status = st.empty()
                    duration = part2_periods * 2.0 * math.pi
                    rk4_steps = max(1, int(round(duration / (4.0 * part2_dt))))
                    verlet_steps = max(1, int(round(duration / part2_dt)))
                    total_work_steps = 2 * (rk4_steps + verlet_steps)
                    start_time = time.perf_counter()

                    def update_part2_progress(method, phase, done, total, elapsed, remaining):
                        method_index = 0 if method == "RK4" else 1
                        phase_index = 0 if phase == "ida" else 1
                        method_offset = 0 if method_index == 0 else 2 * rk4_steps
                        phase_offset = 0 if phase_index == 0 else total
                        completed = method_offset + phase_offset + done
                        fraction = completed / total_work_steps
                        elapsed_global = time.perf_counter() - start_time
                        eta_global = elapsed_global * (total_work_steps - completed) / max(completed, 1)
                        progress.progress(min(fraction, 1.0), text=f"{method} | {phase} | {completed:,}/{total_work_steps:,} passos")
                        progress_status.info(f"Metodo: {method} | Fase: {phase} | Passos: {done:,}/{total:,} | Decorrido: {elapsed_global:.1f} s | Restante estimado: {eta_global:.1f} s")

                    result_rows = run_part2_experiment(part2_periods, part2_dt, sample_every, update_part2_progress)
                    elapsed_total = time.perf_counter() - start_time
                    st.session_state["part2_result_key"] = cache_key
                    st.session_state["part2_result_rows"] = result_rows
                    st.session_state["part2_elapsed"] = elapsed_total
                result_rows = st.session_state.get("part2_result_rows")
                if st.session_state.get("part2_result_key") == cache_key and result_rows:
                    energy_figure, energy_axis = plt.subplots(figsize=(9, 4.5))
                    table_rows = []
                    for row in result_rows:
                        energy_axis.plot(row["tempos_grafico"], row["energia_relativa_grafico"], label=f"{row['metodo']} (dt={row['dt']:.6g})")
                        table_rows.append({key: value for key, value in row.items() if key not in {"tempos_grafico", "energia_relativa_grafico"}})
                    energy_axis.set_xlabel("tempo")
                    energy_axis.set_ylabel("E(t)-E(0)")
                    energy_axis.set_title(f"Energia em {part2_periods:,} periodos, custo comparavel")
                    energy_axis.legend()
                    energy_axis.grid(True, alpha=0.25)
                    energy_figure.tight_layout()
                    st.pyplot(energy_figure)
                    plt.close(energy_figure)
                    st.dataframe(table_rows, use_container_width=True)
                    st.write(f"Tempo total de calculo: {st.session_state.get('part2_elapsed', 0.0):.2f} s")
            except Exception as error:
                st.error(f"Falha na Parte 2. Nenhum resultado incompleto foi apresentado: {error}")

    with tabs[3]:
        st.header("Parte 3 - passo adaptativo e eventos")
        eta_text = st.text_input("eta do passo adaptativo", "0.05", key="part3_eta")
        dt_initial_text = st.text_input("dt inicial/maximo", numeric["dt"].__format__(".16g"), key="part3_dt_initial")
        dt_min_text = st.text_input("dt minimo", "1e-5", key="part3_dt_min")
        part3_resolution_text = st.text_input("Resolucao da comparacao (ex.: 12, 20, 40, 80)", "12", key="part3_resolution")
        x0_text = st.text_input("x inicial", "0.0", key="event_x")
        y0_text = st.text_input("y inicial", "0.0", key="event_y")
        step_criterion = st.selectbox("Criterio de passo adaptativo", ["Tempo dinamico local (eta*min_i(r^(3/2)/sqrt(M)))", "Cinematico (eta*|v|/|a|)"], key="part3_criterion")
        st.caption("RK45 usa controle adaptativo de erro e eta limita o passo maximo local. RK4/Verlet usam o criterio selecionado (tempo dinamico ou cinematico).")
        if st.button("Integrar trajetoria adaptativa", key="event_button"):
            eta = parse_float(eta_text, "eta", positive=True)
            dt_initial = parse_float(dt_initial_text, "dt inicial", positive=True)
            dt_min = parse_float(dt_min_text, "dt minimo", positive=True)
            x0 = parse_float(x0_text, "x inicial")
            y0 = parse_float(y0_text, "y inicial")
            criterion_code = "cinematico" if "Cinematico" in step_criterion else "dinamico"
            if dt_min > dt_initial:
                st.error("dt minimo deve ser menor ou igual a dt inicial.")
            else:
                initial = np.array([x0, y0, 0.0, 0.0, 0.0, 0.0])
                try:
                    detailed = classify_adaptive_detailed(initial, masses, positions, numeric["eps"], numeric["gamma"], numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], eta, dt_initial, dt_min, True, numeric["rtol"], numeric["atol"], criterion=criterion_code)
                except Exception as error:
                    st.error(f"Falha na trajetória da Parte 3; não será exibido resultado incompleto: {error}")
                    st.stop()
                figure = plt.figure(figsize=(8, 6))
                axis = figure.add_subplot(111, projection="3d")
                states = detailed["states"]
                axis.plot(states[0], states[1], states[2], label="trajetoria")
                axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="red", edgecolors="black", label="atratores")
                plane_vertices = [[[numeric["xmin"], numeric["ymin"], 0.0], [numeric["xmax"], numeric["ymin"], 0.0], [numeric["xmax"], numeric["ymax"], 0.0], [numeric["xmin"], numeric["ymax"], 0.0]]]
                axis.add_collection3d(Poly3DCollection(plane_vertices, alpha=0.15, facecolor="cyan", edgecolor="cyan"))
                axis.scatter([x0], [y0], [0.0], color="magenta", s=70, label="inicio")
                axis.scatter([states[0, -1]], [states[1, -1]], [states[2, -1]], color="black", s=60, label="ponto final")
                if detailed["posicao_captura"] is not None:
                    capture_position = detailed["posicao_captura"]
                    axis.scatter([capture_position[0]], [capture_position[1]], [capture_position[2]], color="yellow", edgecolors="black", s=80, label="captura")
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.set_zlabel("z")
                axis.set_title(f"{detailed['estado']} | atrator={detailed['atrator']}")
                axis.legend()
                st.pyplot(figure)
                plt.close(figure)
                st.dataframe([{
                    "estado_final": detailed["estado"], "tempo_final": detailed["tempo_final"], "passos": detailed["passos"],
                    "atrator": detailed["atrator"], "tempo_captura": detailed["tempo_captura"],
                    "posicao_captura": detailed["posicao_captura"], "velocidade_captura": detailed["velocidade_captura"],
                }], use_container_width=True)
                energies, residual = dissipation_diagnostics(detailed["times"], detailed["states"], masses, positions, numeric["eps"], numeric["gamma"])
                st.write({"residuo_maximo": float(np.max(residual)), "eta": eta, "dt_inicial": dt_initial, "dt_minimo": dt_min})

        st.subheader("Deteccao ingenua x deteccao por evento")
        if st.button("Gerar as duas bacias da Parte 3", key="part3_basins_button"):
            try:
                eta = parse_float(eta_text, "eta", positive=True)
                dt_initial = parse_float(dt_initial_text, "dt inicial", positive=True)
                dt_min = parse_float(dt_min_text, "dt minimo", positive=True)
                resolution = parse_int(part3_resolution_text, "resolucao da Parte 3", minimum=2)
                if dt_min > dt_initial:
                    raise ValueError("dt minimo deve ser menor ou igual a dt inicial.")
                part3_key = (resolution, masses.tobytes(), positions.tobytes(), numeric["eps"], numeric["gamma"], numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], eta, dt_initial, dt_min, numeric["rtol"], numeric["atol"], numeric["xmin"], numeric["xmax"], numeric["ymin"], numeric["ymax"], "pair-v3")
                if st.session_state.get("part3_result_key") == part3_key:
                    st.info("Reutilizando classificacoes calculadas para estes mesmos parametros.")
                    naive_basin, event_basin, x_values, y_values = st.session_state["part3_result"]
                    elapsed_total = st.session_state["part3_elapsed"]
                else:
                    total_points = resolution * resolution
                    progress = st.progress(0.0, text=f"Preparando grade {resolution} x {resolution}...")
                    status = st.empty()
                    started = time.perf_counter()

                    def update_basin_progress(done, total):
                        elapsed = time.perf_counter() - started
                        remaining = elapsed * (total - done) / max(done, 1)
                        progress.progress(done / total, text=f"Pontos: {done:,}/{total:,}")
                        status.info(f"Integracao compartilhada + duas classificacoes | Decorrido: {elapsed:.1f} s | Restante estimado: {remaining:.1f} s")

                    naive_basin, event_basin, x_values, y_values = adaptive_basin_pair(
                        masses, positions, resolution, numeric["eps"], numeric["gamma"], numeric["tmax"],
                        numeric["rcap"], numeric["rescape"], settings["method"], eta, dt_initial, dt_min,
                        (numeric["xmin"], numeric["xmax"]), (numeric["ymin"], numeric["ymax"]),
                        numeric["rtol"], numeric["atol"], update_basin_progress, 64,
                    )
                    elapsed_total = time.perf_counter() - started
                    st.session_state["part3_result_key"] = part3_key
                    st.session_state["part3_result"] = (naive_basin, event_basin, x_values, y_values)
                    st.session_state["part3_elapsed"] = elapsed_total
                if naive_basin.shape != event_basin.shape:
                    raise RuntimeError("As duas classificacoes nao usam a mesma grade.")
                difference = naive_basin != event_basin
                total_pixels = int(difference.size)
                changed_pixels = int(np.count_nonzero(difference))
                columns = st.columns(2)
                with columns[0]:
                    naive_figure = draw_basin(naive_basin, x_values, y_values, positions)
                    st.pyplot(naive_figure)
                    plt.close(naive_figure)
                    st.caption("Bacia com deteccao ingenua")
                with columns[1]:
                    event_figure = draw_basin(event_basin, x_values, y_values, positions)
                    st.pyplot(event_figure)
                    plt.close(event_figure)
                    st.caption("Bacia com deteccao precisa por evento")
                st.write({"resolucao": f"{resolution} x {resolution}", "total_pixels": total_pixels,
                          "pixels_alterados": changed_pixels, "Pi": changed_pixels / total_pixels,
                          "percentual_alterado": 100.0 * changed_pixels / total_pixels,
                          "tempo_calculo_compartilhado_s": elapsed_total,
                          "fracao_nao_resolvida_ingenua": float(np.mean(naive_basin == -2)),
                          "fracao_nao_resolvida_evento": float(np.mean(event_basin == -2))})
                rows = []
                for label, basin in (("ingenua", naive_basin), ("evento", event_basin)):
                    row = {"metodo": label, "Pi": changed_pixels / total_pixels,
                           "tempo_calculo_compartilhado_s": elapsed_total}
                    row.update(basin_statistics(basin, len(positions)))
                    rows.append(row)
                st.dataframe(rows, use_container_width=True)
                st.caption("O tempo exibido e da execucao emparelhada: a trajetoria fisica e integrada uma vez e usada pelas duas regras de classificacao. Isso evita duplicar integrações; nao e um benchmark isolado de cada classificador.")
            except Exception as error:
                st.error(f"Falha na Parte 3; resultados incompletos descartados: {error}")

    with tabs[4]:
        st.header("Parte 4 - caos e expoente de Lyapunov")
        st.info("gamma=0. Benettin: delta=1e-10. Horizonte: delta_machine=2e-16 e Delta_tol=0.1. A norma usada e Euclidiana no estado completo (posicao e velocidade).")
        point_mode = st.radio("Selecao dos pontos", ["Manual", "Automaticos localizados"], horizontal=True, key="part4_point_mode")
        chaos_x_text = st.text_input("x do ponto interior", "-2.0", key="chaos_x")
        chaos_y_text = st.text_input("y do ponto interior", "-2.0", key="chaos_y")
        boundary_x_text = st.text_input("x do ponto proximo a fronteira", "0.0", key="chaos_boundary_x")
        boundary_y_text = st.text_input("y do ponto proximo a fronteira", "0.0", key="chaos_boundary_y")
        resolution_text = st.text_input("Resolucao inicial da busca (12 recomendado; refine para 20/40)", "12", key="chaos_resolution")
        eta_text = st.text_input("eta da busca adaptativa", "0.05", key="chaos_eta")
        dt_initial_text = st.text_input("dt inicial/maximo da busca", numeric["dt"].__format__(".16g"), key="chaos_dt_initial")
        dt_min_text = st.text_input("dt minimo da busca", "1e-5", key="chaos_dt_min")
        total_time_text = st.text_input("Tempo total de integracao por ponto", "10.0", key="chaos_total_time")
        taus_text = st.text_input("Valores tau para teste de sensibilidade", "0.5, 1.0", key="chaos_taus")

        if st.button("Encontrar pontos automaticamente", key="find_boundary_button"):
            try:
                eta = parse_float(eta_text, "eta", positive=True)
                dt_initial = parse_float(dt_initial_text, "dt inicial", positive=True)
                dt_min = parse_float(dt_min_text, "dt minimo", positive=True)
                resolution = parse_int(resolution_text, "resolucao", minimum=3)
                progress = st.progress(0.0, text="Buscando classes na grade...")
                def update_part4_grid(done, total):
                    progress.progress(done / total, text=f"Pontos classificados: {done:,}/{total:,}")
                naive_grid, boundary_basin, boundary_x_values, boundary_y_values = adaptive_basin_pair(
                    masses, positions, resolution, numeric["eps"], 0.0, numeric["tmax"], numeric["rcap"], numeric["rescape"],
                    settings["method"], eta, dt_initial, dt_min, (numeric["xmin"], numeric["xmax"]),
                    (numeric["ymin"], numeric["ymax"]), numeric["rtol"], numeric["atol"], update_part4_grid, 64,
                )
                interior_point, boundary_point = find_basin_points(boundary_basin, boundary_x_values, boundary_y_values, len(positions))
                bracket = find_boundary_bracket(boundary_basin, boundary_x_values, boundary_y_values)
                refinement_count = 0
                refined_width = float("nan")
                refinement_success = False
                if bracket is not None:
                    boundary_point, refinement_count, refined_width, refinement_success = refine_boundary_candidate(
                        masses, positions, numeric["eps"], numeric["tmax"], numeric["rcap"], numeric["rescape"],
                        settings["method"], eta, dt_initial, dt_min, bracket,
                    )
                if interior_point is None or boundary_point is None or bracket is None:
                    st.session_state.pop("part4_auto_points", None)
                    st.warning("A grade nao encontrou um interior capturado e uma fronteira entre atratores. Aumente a resolucao ou reveja o dominio/tempo.")
                else:
                    st.session_state["part4_auto_points"] = (interior_point, boundary_point)
                    st.session_state["part4_grid_key"] = (resolution, masses.tobytes(), positions.tobytes(), numeric["eps"], 0.0, numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], eta, dt_initial, dt_min, numeric["rtol"], numeric["atol"], numeric["xmin"], numeric["xmax"], numeric["ymin"], numeric["ymax"], "pair-v3")
                    st.write({"interior_xy": interior_point[:2], "classe_interior": classify_label(interior_point[2], len(positions)),
                              "fronteira_xy_refinada" if refinement_success else "fronteira_xy_candidata": boundary_point[:2],
                              "classes_vizinhas": (classify_label(boundary_point[2], len(positions)), classify_label(boundary_point[3], len(positions))),
                              "refinamentos_locais": refinement_count, "largura_final_do_intervalo": refined_width,
                              "refinamento_convergiu": refinement_success})
                    figure = draw_basin(boundary_basin, boundary_x_values, boundary_y_values, positions)
                    axis = figure.axes[0]
                    axis.scatter([interior_point[0]], [interior_point[1]], color="white", edgecolors="black", label="interior")
                    axis.scatter([boundary_point[0]], [boundary_point[1]], color="yellow", edgecolors="black", label="fronteira")
                    axis.legend()
                    st.pyplot(figure)
                    plt.close(figure)
            except Exception as error:
                st.error(f"Falha ao localizar os pontos da Parte 4: {error}")

        if st.button("Executar Parte 4", key="part4_run_button"):
            try:
                eta = parse_float(eta_text, "eta", positive=True)
                dt_initial = parse_float(dt_initial_text, "dt inicial", positive=True)
                dt_min = parse_float(dt_min_text, "dt minimo", positive=True)
                resolution = parse_int(resolution_text, "resolucao", minimum=3)
                total_time = parse_float(total_time_text, "tempo total", positive=True)
                taus = parse_float_list(taus_text, "taus", positive=True)
                grid_key = (resolution, masses.tobytes(), positions.tobytes(), numeric["eps"], 0.0, numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], eta, dt_initial, dt_min, numeric["rtol"], numeric["atol"], numeric["xmin"], numeric["xmax"], numeric["ymin"], numeric["ymax"], "pair-v3")
                if point_mode == "Manual":
                    interior = (parse_float(chaos_x_text, "x interior"), parse_float(chaos_y_text, "y interior"))
                    boundary = (parse_float(boundary_x_text, "x fronteira"), parse_float(boundary_y_text, "y fronteira"))
                    if interior == boundary:
                        raise ValueError("Os pontos interior e fronteira devem ser diferentes; o teste nao foi iniciado.")
                else:
                    auto_points = st.session_state.get("part4_auto_points")
                    if auto_points is None or st.session_state.get("part4_grid_key") != grid_key:
                        raise ValueError("Localize novamente os pontos automaticos: parametros alterados ou candidatos nao encontrados.")
                    interior = (auto_points[0][0], auto_points[0][1])
                    boundary = (auto_points[1][0], auto_points[1][1])
                    if interior == boundary:
                        raise ValueError("Os pontos automaticos resultaram iguais; nenhum teste foi iniciado.")
                progress = st.progress(0.0, text="Validando bacia para localizar os pontos...")
                def update_part4_grid(done, total):
                    progress.progress(done / total, text=f"Pontos classificados: {done:,}/{total:,}")
                _, basin, x_values, y_values = adaptive_basin_pair(
                    masses, positions, resolution, numeric["eps"], 0.0, numeric["tmax"], numeric["rcap"], numeric["rescape"],
                    settings["method"], eta, dt_initial, dt_min, (numeric["xmin"], numeric["xmax"]),
                    (numeric["ymin"], numeric["ymax"]), numeric["rtol"], numeric["atol"], update_part4_grid, 64,
                )
                interior_valid, interior_code = validate_interior_candidate(basin, x_values, y_values, *interior)
                boundary_valid, boundary_classes = validate_boundary_candidate(basin, x_values, y_values, *boundary)
                if not interior_valid:
                    raise ValueError("O ponto interior nao foi confirmado: sua vizinhanca 3x3 nao pertence a uma unica bacia capturada. Refine a grade ou escolha outro ponto.")
                if not boundary_valid:
                    raise ValueError("O ponto indicado nao esta proximo de pixels de dois atratores diferentes. Use a localizacao automatica ou refine a grade.")

                point_specs = [("Interior", interior), ("Fronteira", boundary)]
                classifications = {}
                for point_name, point in point_specs:
                    initial = np.array([point[0], point[1], 0.0, 0.0, 0.0, 0.0])
                    point_result = classify_adaptive_detailed(
                        initial, masses, positions, numeric["eps"], 0.0, numeric["tmax"], numeric["rcap"], numeric["rescape"],
                        settings["method"], eta, dt_initial, dt_min, True, numeric["rtol"], numeric["atol"], False,
                    )
                    classifications[point_name] = (classify_label(point_result["codigo"], len(positions)), point_result["codigo"])
                if classifications["Interior"][1] != interior_code:
                    raise ValueError("A classificacao direta do ponto interior divergiu da bacia usada para valida-lo.")
                if classifications["Fronteira"][1] >= 0 and classifications["Fronteira"][1] not in boundary_classes:
                    raise ValueError("A classificacao direta do ponto de fronteira nao coincide com nenhuma das classes vizinhas detectadas.")

                cache = st.session_state.setdefault("part4_lyapunov_cache", {})
                result_rows = []
                convergence_rows = []
                lyapunov_status = st.empty()
                for point_name, point in point_specs:
                    point_class, point_code = classifications[point_name]
                    for tau in taus:
                        intervals = int(math.floor(total_time / tau + 1e-12))
                        if intervals < 1:
                            raise ValueError(f"Tempo total deve ser pelo menos tau={tau}.")
                        actual_time = intervals * tau
                        result_key = (point, masses.tobytes(), positions.tobytes(), numeric["eps"], 0.0, tau, numeric["dt"], intervals, PART4_DELTA_BENETTIN, "state-norm-v2")
                        details = cache.get(result_key)
                        if details is None:
                            progress = st.progress(0.0, text=f"Benettin {point_name}, tau={tau:g}...")
                            def update_lyapunov(done, total, elapsed, remaining):
                                progress.progress(done / total, text=f"Renormalizacoes: {done}/{total}")
                                eta_text_value = f"{remaining:.1f} s restantes" if np.isfinite(remaining) else "estimativa indisponivel"
                                lyapunov_status.info(f"{point_name}, tau={tau:g} | Renormalizacoes: {done}/{total} | Decorrido: {elapsed:.1f} s | {eta_text_value}")
                            details = benettin_details(
                                np.array([point[0], point[1], 0.0, 0.0, 0.0, 0.0]), masses, positions,
                                numeric["eps"], tau, numeric["dt"], intervals, PART4_DELTA_BENETTIN,
                                update_lyapunov,
                            )
                            if not details["completo"] or details["renormalizacoes"] != intervals:
                                raise RuntimeError("Teste de Benettin incompleto; nenhum lambda parcial foi apresentado.")
                            cache[result_key] = details
                        lambda_value = float(details["lambda"])
                        horizon = predictability_horizon(lambda_value, PART4_DELTA_MACHINE, PART4_DELTA_TOL) if lambda_value > 0.0 else None
                        convergence_rows.append({"ponto": point_name, "tau": tau, "lambda": lambda_value, "tempo_integrado": details["tempo_total"]})
                        if tau == taus[0]:
                            result_rows.append({"ponto": point_name, "x": point[0], "y": point[1], "classificacao": point_class,
                                                "lambda": lambda_value, "tau": tau, "renormalizacoes": details["renormalizacoes"],
                                                "tempo_integracao": details["tempo_total"],
                                                "t_prev": f"{horizon:.6g}" if horizon is not None else "nao aplicavel para lambda <= 0"})
                st.write({"gamma_do_teste": 0.0, "delta_Benettin": PART4_DELTA_BENETTIN,
                          "delta_machine_horizonte": PART4_DELTA_MACHINE, "Delta_tol": PART4_DELTA_TOL,
                          "norma": "Euclidiana do estado completo (x,y,z,vx,vy,vz)"})
                st.dataframe(result_rows, use_container_width=True)
                st.subheader("Sensibilidade de lambda a tau")
                st.dataframe(convergence_rows, use_container_width=True)
                if len(taus) >= 2:
                    for point_name in ("Interior", "Fronteira"):
                        values = [row["lambda"] for row in convergence_rows if row["ponto"] == point_name]
                        if len(values) >= 2 and abs(values[0] - values[1]) > 0.2 * max(abs(values[0]), abs(values[1]), 1e-12):
                            st.warning("O expoente de Lyapunov apresenta sensibilidade ao intervalo de renormalizacao; aumente o tempo de integracao ou revise os parametros.")
                if result_rows[0]["lambda"] > 0.0 or result_rows[1]["lambda"] <= 0.0:
                    st.warning("Resultado nao consistente com o comportamento esperado; verifique localizacao, resolucao, tempo e parametros. Lambda nao foi alterado.")
                figure = draw_basin(basin, x_values, y_values, positions)
                axis = figure.axes[0]
                axis.scatter([interior[0]], [interior[1]], color="white", edgecolors="black", label="interior")
                axis.scatter([boundary[0]], [boundary[1]], color="yellow", edgecolors="black", label="fronteira")
                axis.legend()
                axis.set_title(f"Pontos Benettin; vizinhos da fronteira: {boundary_classes}")
                st.pyplot(figure)
                plt.close(figure)
                convergence_figure, convergence_axis = plt.subplots(figsize=(7, 4))
                for point_name in ("Interior", "Fronteira"):
                    point_rows = [row for row in convergence_rows if row["ponto"] == point_name]
                    convergence_axis.plot([row["tau"] for row in point_rows], [row["lambda"] for row in point_rows], marker="o", label=point_name)
                convergence_axis.set_xlabel("tau")
                convergence_axis.set_ylabel("lambda")
                convergence_axis.set_title("Convergencia/sensibilidade do expoente")
                convergence_axis.grid(True, alpha=0.25)
                convergence_axis.legend()
                st.pyplot(convergence_figure)
                plt.close(convergence_figure)
            except Exception as error:
                st.error(f"Falha na Parte 4; resultados parciais descartados: {error}")

    with tabs[5]:
        st.header("Imagem das bacias")
        st.warning("O custo cresce com resolucao^2 e com tmax/dt. Valores grandes podem exigir bastante memoria e tempo.")
        if st.button("Gerar bacia", key="basin_button"):
            started = time.perf_counter()
            with st.spinner("Integrando as trajetorias..."):
                basin, x_values, y_values, execution_mode = compute_basin_for_ui(masses, positions, numeric["resolution"], numeric["eps"], numeric["gamma"], numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], numeric["dt"], (numeric["xmin"], numeric["xmax"]), (numeric["ymin"], numeric["ymax"]), numeric["rtol"], numeric["atol"])
            st.pyplot(draw_basin(basin, x_values, y_values, positions))
            st.write(f"Tempo: {time.perf_counter() - started:.2f} s")
            st.write(f"Modo: {execution_mode}")
            st.write({"capturadas": int(np.sum(basin >= 0)), "escape": int(np.sum(basin == -1)), "pendentes": int(np.sum(basin == -2))})

        st.subheader("Convergencia em dt")
        convergence_text = st.text_input(
            "Passos a comparar, separados por virgula",
            f"{numeric['dt']:.16g}, {numeric['dt'] / 2.0:.16g}, {numeric['dt'] / 4.0:.16g}, {numeric['dt'] / 8.0:.16g}",
            key="convergence_dts",
        )
        if st.button("Executar convergencia em dt", key="convergence_button"):
            try:
                dts = parse_float_list(convergence_text, "passos dt", positive=True)
                if len(dts) != 4 or not all(np.isclose(dts[index + 1], dts[index] / 2.0, rtol=1e-10, atol=0.0) for index in range(3)):
                    raise ValueError("Informe exatamente dt, dt/2, dt/4 e dt/8, nessa ordem.")
                convergence_key = (numeric["resolution"], masses.tobytes(), positions.tobytes(), numeric["eps"], numeric["gamma"], numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], tuple(dts), numeric["xmin"], numeric["xmax"], numeric["ymin"], numeric["ymax"], numeric["rtol"], numeric["atol"], "part5-full-classes-v2")
                cached_convergence = st.session_state.get("part5_convergence")
                if cached_convergence is not None and cached_convergence[0] == convergence_key:
                    basins, x_values, y_values = cached_convergence[1:]
                    elapsed_total = 0.0
                    st.info("Reutilizando as quatro bacias calculadas para os mesmos parametros.")
                else:
                    basins = []
                    total_runs = 4 * numeric["resolution"] ** 2
                    progress = st.progress(0.0, text="Iniciando convergencia dt...")
                    status = st.empty()
                    started = time.perf_counter()
                    completed_runs = 0
                    for step_index, current_dt in enumerate(dts):
                        current_basin, x_values, y_values, _ = cached_basin_for_ui(
                            masses, positions, numeric["resolution"], numeric["eps"], numeric["gamma"], numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], current_dt,
                            (numeric["xmin"], numeric["xmax"]), (numeric["ymin"], numeric["ymax"]), numeric["rtol"], numeric["atol"],
                        )
                        basins.append(current_basin)
                        completed_runs += numeric["resolution"] ** 2
                        elapsed = time.perf_counter() - started
                        remaining = elapsed * (total_runs - completed_runs) / max(completed_runs, 1)
                        progress.progress(completed_runs / total_runs, text=f"dt[{step_index + 1}/4]={current_dt:g} | {completed_runs:,}/{total_runs:,} pontos")
                        status.info(f"Decorrido: {elapsed:.1f} s | Restante estimado: {remaining:.1f} s")
                    elapsed_total = time.perf_counter() - started
                    st.session_state["part5_convergence"] = (convergence_key, basins, x_values, y_values)
                rows = []
                for index in range(1, 4):
                    pointwise, measure = compare_basin_pair(basins[index - 1], basins[index], len(positions))
                    rows.append({"comparacao": f"dt/{2 ** (index - 1)} x dt/{2 ** index}" if index > 1 else "dt x dt/2",
                                 "dt_anterior": dts[index - 1], "dt_atual": dts[index], "Pi": pointwise, "Delta_mu": measure,
                                 "pixels_alterados": int(np.count_nonzero(basins[index - 1] != basins[index])),
                                 "classes_Delta_mu": "todos atratores + escape + cinza"})
                st.dataframe(rows, use_container_width=True)
                for index in range(1, 4):
                    difference = basins[index - 1] != basins[index]
                    changed = int(np.count_nonzero(difference))
                    if changed == 0:
                        st.info(f"Comparacao dt[{index}] x dt[{index + 1}]: 0 pixels alterados na grade {numeric['resolution']}x{numeric['resolution']}.")
                    difference_figure, difference_axis = plt.subplots(figsize=(6, 4))
                    difference_axis.imshow(difference, origin="lower", extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]], cmap="gray_r", interpolation="nearest")
                    difference_axis.set_title(f"Pixels alterados: dt[{index}] x dt[{index + 1}] ({changed}/{difference.size})")
                    difference_axis.set_xlabel("x")
                    difference_axis.set_ylabel("y")
                    difference_figure.tight_layout()
                    st.pyplot(difference_figure)
                    plt.close(difference_figure)
                st.write(f"Tempo total: {elapsed_total:.2f} s")
                st.pyplot(draw_basin(basins[-1], x_values, y_values, positions))
            except Exception as error:
                st.error(f"Falha na convergencia da Parte 5; resultados incompletos descartados: {error}")

        st.subheader("Efeito da deteccao de eventos")
        if st.button("Comparar teste ingenuo e bissecao", key="event_comparison_button"):
            try:
                event_key = (numeric["resolution"], masses.tobytes(), positions.tobytes(), numeric["eps"], numeric["gamma"], numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], numeric["dt"], numeric["rtol"], numeric["atol"], numeric["xmin"], numeric["xmax"], numeric["ymin"], numeric["ymax"], "paired-event-v2")
                cached_event_pair = st.session_state.get("part5_event_pair")
                if cached_event_pair is not None and cached_event_pair[0] == event_key:
                    naive_basin, event_basin, event_x, event_y = cached_event_pair[1:]
                    event_elapsed = st.session_state.get("part5_event_elapsed", 0.0)
                    st.info("Reutilizando as classificacoes de eventos para os mesmos parametros.")
                else:
                    total = numeric["resolution"] ** 2
                    progress = st.progress(0.0, text="Calculando classificacoes ingênua e de evento na mesma grade...")
                    status = st.empty()
                    started = time.perf_counter()
                    def update_event_progress(done, point_total):
                        elapsed = time.perf_counter() - started
                        remaining = elapsed * (point_total - done) / max(done, 1)
                        progress.progress(done / point_total, text=f"Pontos: {done:,}/{point_total:,}")
                        status.info(f"Trajetoria compartilhada | Decorrido: {elapsed:.1f} s | Restante estimado: {remaining:.1f} s")
                    naive_basin, event_basin, event_x, event_y = compute_part5_event_basins(
                        masses, positions, numeric["resolution"], numeric["eps"], numeric["gamma"], numeric["tmax"],
                        numeric["rcap"], numeric["rescape"], settings["method"], numeric["dt"],
                        (numeric["xmin"], numeric["xmax"]), (numeric["ymin"], numeric["ymax"]),
                        numeric["rtol"], numeric["atol"], update_event_progress,
                    )
                    event_elapsed = time.perf_counter() - started
                    st.session_state["part5_event_pair"] = (event_key, naive_basin, event_basin, event_x, event_y)
                    st.session_state["part5_event_elapsed"] = event_elapsed
                expected_x = np.linspace(numeric["xmin"], numeric["xmax"], numeric["resolution"])
                expected_y = np.linspace(numeric["ymin"], numeric["ymax"], numeric["resolution"])
                if naive_basin.shape != event_basin.shape or not np.array_equal(event_x, expected_x) or not np.array_equal(event_y, expected_y):
                    raise RuntimeError("As bacias nao usam a mesma grade; Pi cancelado.")
                difference = naive_basin != event_basin
                total_pixels = int(difference.size)
                changed_pixels = int(np.count_nonzero(difference))
                event_pi = changed_pixels / total_pixels
                columns = st.columns(2)
                with columns[0]:
                    figure = draw_basin(naive_basin, event_x, event_y, positions)
                    st.pyplot(figure)
                    plt.close(figure)
                    st.caption("Bacia com deteccao ingenua")
                with columns[1]:
                    figure = draw_basin(event_basin, event_x, event_y, positions)
                    st.pyplot(figure)
                    plt.close(figure)
                    st.caption("Bacia com deteccao por evento")
                if changed_pixels == 0:
                    st.info(f"0 pixels alterados entre as duas classificacoes nesta grade {numeric['resolution']}x{numeric['resolution']}.")
                st.write({"total_pixels": total_pixels, "pixels_alterados": changed_pixels,
                          "Pi": event_pi, "percentual_alterado": 100.0 * event_pi,
                          "fracao_nao_resolvida_ingenua": float(np.mean(naive_basin == -2)),
                          "fracao_nao_resolvida_evento": float(np.mean(event_basin == -2)),
                          "tempo_execucao_compartilhada_s": event_elapsed})
                difference_figure, difference_axis = plt.subplots(figsize=(8, 6))
                difference_axis.imshow(difference, origin="lower", extent=[event_x[0], event_x[-1], event_y[0], event_y[-1]], cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
                difference_axis.set_title(f"Pixels alterados pela deteccao de eventos: {changed_pixels}")
                difference_axis.set_xlabel("x")
                difference_axis.set_ylabel("y")
                difference_figure.tight_layout()
                st.pyplot(difference_figure)
                plt.close(difference_figure)
                rows = []
                for label, basin in (("ingenua", naive_basin), ("evento", event_basin)):
                    row = {"metodo": label, "Pi": event_pi, "tempo_compartilhado_s": event_elapsed}
                    row.update(basin_statistics(basin, len(positions)))
                    fractions_sum = sum(float(row[key]) for key in row if key.startswith("fracao_"))
                    row["soma_fracoes"] = fractions_sum
                    rows.append(row)
                st.dataframe(rows, use_container_width=True)
                st.caption("O tempo e compartilhado: cada trajetoria e integrada uma vez para obter ambos os rotulos; a dinamica fisica e identica nos dois metodos.")
            except Exception as error:
                st.error(f"Falha na comparacao de eventos; mapas incompletos descartados: {error}")

        st.subheader("Sensibilidade dos parametros de modelo")
        sensitivity_eps = st.text_input("Valores de epsilon", "0.09, 0.18, 0.36", key="sensitivity_eps")
        sensitivity_gamma = st.text_input("Valores de gamma", "0.075, 0.15, 0.30", key="sensitivity_gamma")
        sensitivity_rcap = st.text_input("Valores de Rcap", "0.11, 0.22", key="sensitivity_rcap")
        sensitivity_tmax = st.text_input("Valores de tmax", "10, 20, 40", key="sensitivity_tmax")
        if st.button("Executar sensibilidade", key="sensitivity_button"):
            try:
                eps_values = parse_float_list(sensitivity_eps, "valores de epsilon", nonnegative=True)
                gamma_values = parse_float_list(sensitivity_gamma, "valores de gamma", nonnegative=True)
                rcap_values = parse_float_list(sensitivity_rcap, "valores de Rcap", nonnegative=True)
                tmax_values = parse_float_list(sensitivity_tmax, "valores de tmax", nonnegative=True)
                tests = [("epsilon", value, value, numeric["gamma"], numeric["tmax"], numeric["rcap"]) for value in eps_values]
                tests += [("gamma", value, numeric["eps"], value, numeric["tmax"], numeric["rcap"]) for value in gamma_values]
                tests += [("Rcap", value, numeric["eps"], numeric["gamma"], numeric["tmax"], value) for value in rcap_values]
                tests += [("tmax", value, numeric["eps"], numeric["gamma"], value, numeric["rcap"]) for value in tmax_values]
                sensitivity_key = (numeric["resolution"], masses.tobytes(), positions.tobytes(), tuple(tests), numeric["rescape"], settings["method"], numeric["dt"], (numeric["xmin"], numeric["xmax"]), (numeric["ymin"], numeric["ymax"]), numeric["rtol"], numeric["atol"], "sensitivity-v2")
                cached_sensitivity = st.session_state.get("part5_sensitivity")
                if cached_sensitivity is not None and cached_sensitivity[0] == sensitivity_key:
                    sensitivity_rows, sensitivity_elapsed = cached_sensitivity[1:]
                    st.info("Reutilizando os testes de sensibilidade para estes mesmos parametros.")
                else:
                    sensitivity_rows = []
                    progress = st.progress(0.0, text="Iniciando sensibilidade dos parametros...")
                    status = st.empty()
                    started = time.perf_counter()
                    for test_index, (parameter, value, eps_value, gamma_value, tmax_value, rcap_value) in enumerate(tests, start=1):
                        current, _, _, _ = cached_basin_for_ui(
                            masses, positions, numeric["resolution"], eps_value, gamma_value, tmax_value, rcap_value,
                            numeric["rescape"], settings["method"], numeric["dt"],
                            (numeric["xmin"], numeric["xmax"]), (numeric["ymin"], numeric["ymax"]), numeric["rtol"], numeric["atol"],
                        )
                        stats = basin_statistics(current, len(positions))
                        captured_fraction = sum(float(stats[f"fracao_atrator_{index + 1}"]) for index in range(len(positions)))
                        fraction_sum = captured_fraction + float(stats["fracao_escapadas"]) + float(stats["fracao_nao_resolvidas"])
                        sensitivity_rows.append({"parametro": parameter, "valor": value,
                                                 "fracao_capturada": captured_fraction,
                                                 "fracao_escape": float(stats["fracao_escapadas"]),
                                                 "fracao_cinza": float(stats["fracao_nao_resolvidas"]),
                                                 "soma_fracoes": fraction_sum})
                        elapsed = time.perf_counter() - started
                        remaining = elapsed * (len(tests) - test_index) / test_index
                        progress.progress(test_index / len(tests), text=f"Teste {test_index}/{len(tests)}: {parameter}={value:g}")
                        status.info(f"Decorrido: {elapsed:.1f} s | Restante estimado: {remaining:.1f} s")
                    sensitivity_elapsed = time.perf_counter() - started
                    st.session_state["part5_sensitivity"] = (sensitivity_key, sensitivity_rows, sensitivity_elapsed)
                st.dataframe(sensitivity_rows, use_container_width=True)
                st.write(f"Tempo total: {sensitivity_elapsed:.2f} s")
                if any(not np.isclose(row["soma_fracoes"], 1.0, atol=1e-12) for row in sensitivity_rows):
                    st.warning("A soma das tres fracoes difere de 1; revise a classificacao de estados.")
            except Exception as error:
                st.error(f"Falha na sensibilidade da Parte 5; resultados incompletos descartados: {error}")

    with tabs[6]:
        st.header("Desempenho")
        particles_text = st.text_input("Numero de particulas para o benchmark", "1000", key="part6_particles")
        repetitions_text = st.text_input("Repeticoes do passo", "3", key="part6_repetitions")
        benchmark_method = st.selectbox("Metodo para comparar", ["RK4", "Verlet"], key="part6_method")
        if st.button("Validar e comparar escalar x vetorizado", key="part6_benchmark_button"):
            try:
                particles = parse_int(particles_text, "numero de particulas", minimum=1)
                repetitions = parse_int(repetitions_text, "repeticoes", minimum=1)
                result = benchmark_step_implementations(
                    particles, repetitions, numeric["dt"], masses, positions,
                    numeric["eps"], numeric["gamma"], benchmark_method,
                )
                st.success(f"Validacao numerica aprovada por np.allclose(): {result['allclose']}; erro absoluto maximo={result['max_abs_difference']:.3e}.")
                st.metric("Tempo nao vetorizado", f"{result['scalar_seconds']:.6f} s")
                st.metric("Tempo vetorizado", f"{result['vectorized_seconds']:.6f} s")
                st.metric("Speedup", f"{result['speedup']:.3f}x")
                st.write({key: value for key, value in result.items() if key not in {"scalar_seconds", "vectorized_seconds", "speedup", "allclose", "max_abs_difference"}})
            except Exception as error:
                st.error(f"Benchmark cancelado; os resultados escalares e vetorizados nao foram comparados: {error}")
        if st.button("Executar bacia vetorizada", key="vectorized_basin_button"):
            started = time.perf_counter()
            with st.spinner("Integrando todas as particulas ativas em conjunto..."):
                vectorized_basin, vector_x, vector_y, active_history = compute_basin_vectorized(
                    masses, positions, numeric["resolution"], numeric["eps"], numeric["gamma"], numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"], numeric["dt"],
                    (numeric["xmin"], numeric["xmax"]), (numeric["ymin"], numeric["ymax"]),
                )
            basin_figure = draw_basin(vectorized_basin, vector_x, vector_y, positions)
            st.pyplot(basin_figure)
            plt.close(basin_figure)
            st.write(f"Tempo vetorizado: {time.perf_counter() - started:.2f} s")
            active_figure, active_axis = plt.subplots(figsize=(8, 3.5))
            active_axis.plot(np.arange(len(active_history)) * numeric["dt"], active_history)
            active_axis.set_xlabel("tempo")
            active_axis.set_ylabel("fracao ativa")
            active_axis.set_ylim(0.0, 1.05)
            active_axis.grid(True, alpha=0.25)
            active_figure.tight_layout()
            st.pyplot(active_figure)
            plt.close(active_figure)

    with tabs[7]:
        st.header("Itens Bônus (Seção 6)")
        st.markdown(
            "Experimentos adicionais opcionais (até **+15%** na nota final):\n"
            "- **T2 & T3:** Dimensão Fractal da Fronteira por Box-Counting e Expoente de Incerteza $\\alpha = 2 - D$.\n"
            "- **T4:** Integrador Simplético de 4ª Ordem via Composição de Yoshida."
        )
        st.subheader("Bônus T4 - Integrador Simplético de 4ª Ordem (Yoshida)")
        st.latex(r"w_1 = \frac{1}{2 - 2^{1/3}}, \qquad w_0 = -2^{1/3} w_1")
        st.markdown(r"Composição de 3 sub-passos de Velocity Verlet com passos $w_1 \Delta t$, $w_0 \Delta t$, $w_1 \Delta t$. É simplético **e** de ordem 4.")
        yoshida_periods = st.number_input("Períodos de Kepler para teste de reversibilidade", min_value=1, max_value=10000, value=100, step=10, key="yoshida_periods")
        yoshida_dt = st.number_input("Passo dt do Yoshida", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.4f", key="yoshida_dt")
        if st.button("Executar Verificação do Yoshida (T4)", key="btn_run_yoshida"):
            from bonus_itens import verify_yoshida
            with st.spinner("Integrando órbitas com Yoshida..."):
                res = verify_yoshida(periods=int(yoshida_periods), dt=float(yoshida_dt))
            st.success(f"Ordem observada p = {res['ordem_observada']:.4f} (esperado ~4.0)")
            col_y1, col_y2 = st.columns(2)
            col_y1.metric("Erro de Reversibilidade", f"{res['erro_reversibilidade']:.3e}")
            col_y2.metric("Deriva de Energia (|E(T) - E(0)|)", f"{res['deriva_energia']:.3e}")
            st.dataframe([res], use_container_width=True)

        st.subheader("Bônus T2 & T3 - Dimensão Fractal da Fronteira (Box-Counting) e Incerteza")
        st.markdown(r"Mede a complexidade fractal da fronteira das bacias geradas: $N(\delta) \sim \delta^{-D} \implies D = -\frac{d\ln N}{d\ln \delta}$.")
        if st.button("Calcular Dimensão Fractal da Bacia Atual", key="btn_run_fractal"):
            from bonus_itens import extract_boundary, box_counting, plot_fractal_fit
            basin_found = None
            if "part5_convergence" in st.session_state:
                basin_found = st.session_state["part5_convergence"][1][-1]
            elif "part3_result" in st.session_state:
                basin_found = st.session_state["part3_result"][1]

            if basin_found is None:
                st.info("Gerando uma bacia para a análise fractal...")
                basin_found, _, _, _ = compute_basin_for_ui(
                    masses, positions, numeric["resolution"], numeric["eps"], numeric["gamma"],
                    numeric["tmax"], numeric["rcap"], numeric["rescape"], settings["method"],
                    numeric["dt"], (numeric["xmin"], numeric["xmax"]), (numeric["ymin"], numeric["ymax"]),
                    numeric["rtol"], numeric["atol"]
                )

            boundary = extract_boundary(basin_found)
            box_res = box_counting(boundary)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dimensão Fractal D", f"{box_res['D']:.4f}")
                st.metric("Expoente de Incerteza alpha (2 - D)", f"{box_res['alpha']:.4f}")
                st.metric("Coeficiente R²", f"{box_res['r_squared']:.4f}")
                st.write({
                    "tamanhos_caixa_delta": box_res["sizes"],
                    "contagens_N": box_res["counts"],
                    "pixels_fronteira": int(np.count_nonzero(boundary)),
                })
            with col2:
                fig = plot_fractal_fit(box_res)
                st.pyplot(fig)
                plt.close(fig)


if __name__ == "__main__":
    main()
