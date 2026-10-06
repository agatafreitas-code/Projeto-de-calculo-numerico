"""Bacias de atracao gravitacionais.

Execucao:
    python projeto.py --quick
    python projeto.py --resolution 100 --update-every 100

As imagens sao salvas em imagens_bacias/, ao lado deste arquivo.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Empty, Queue

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Configuracao do modelo adimensional
# ---------------------------------------------------------------------------

MASS = np.array([0.35, 0.25, 0.20, 0.20], dtype=np.float64)
POSITIONS = np.array(
    [
        [-1.2, 0.0, 0.0],
        [1.0, 1.1, 0.3],
        [0.5, -1.4, -0.2],
        [-0.4, 1.5, -0.7],
    ],
    dtype=np.float64,
)
COLORS = np.array(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.25, 0.90],
        [0.10, 0.70, 0.20],
        [1.00, 0.55, 0.05],
    ],
    dtype=np.float64,
)


class Parameters:
    """Parametros numericos e de modelo da simulacao."""

    def __init__(
        self,
        resolution: int = 80,
        eps: float = 0.18,
        gamma: float = 0.15,
        tmax: float = 100.0,
        rcap: float = 0.22,
        r_escape: float = 20.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step: float = 0.05,
        update_every: int = 50,
        workers: int | None = None,
    ) -> None:
        self.resolution = resolution
        self.eps = eps
        self.gamma = gamma
        self.tmax = tmax
        self.rcap = rcap
        self.r_escape = r_escape
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.update_every = max(1, update_every)
        self.workers = workers or max(1, min(32, os.cpu_count() or 1))


def acceleration(position: np.ndarray, parameters: Parameters) -> np.ndarray:
    """Calcula o campo gravitacional amaciado no ponto informado."""
    difference = position[None, :] - POSITIONS
    distance_squared = np.sum(difference * difference, axis=1) + parameters.eps**2
    contributions = -MASS[:, None] * difference / distance_squared[:, None] ** 1.5
    return np.sum(contributions, axis=0)


def right_hand_side(
    _time: float,
    state: np.ndarray,
    parameters: Parameters,
) -> np.ndarray:
    """Sistema de primeira ordem y' = (v, g(r) - gamma*v)."""
    position = state[:3]
    velocity = state[3:]
    total_acceleration = acceleration(position, parameters) - parameters.gamma * velocity
    return np.concatenate((velocity, total_acceleration))


def capture_event_factory(index: int, parameters: Parameters):
    """Cria o evento terminal de cruzamento da esfera de captura."""
    def capture_event(_time: float, state: np.ndarray) -> float:
        distance_squared = np.sum((state[:3] - POSITIONS[index]) ** 2)
        return distance_squared - parameters.rcap**2

    capture_event.terminal = True
    capture_event.direction = -1
    return capture_event


def escape_event_factory(parameters: Parameters):
    """Cria o evento terminal de alcance do raio de escape."""
    def escape_event(_time: float, state: np.ndarray) -> float:
        return np.linalg.norm(state[:3]) - parameters.r_escape

    # Cruzar R_escape sozinho nao prova escape; a energia e a direcao radial
    # ainda precisam ser verificadas no estado final.
    escape_event.terminal = False
    escape_event.direction = 1
    return escape_event


def classify_trajectory(
    x: float,
    y: float,
    parameters: Parameters,
    save_path: Path | None = None,
) -> int:
    """Integra uma particula e retorna atrator, -1 escape ou -2 pendente."""
    initial_state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    events = [
        capture_event_factory(index, parameters)
        for index in range(len(MASS))
    ]
    events.append(escape_event_factory(parameters))

    solution = solve_ivp(
        lambda current_time, state: right_hand_side(current_time, state, parameters),
        (0.0, parameters.tmax),
        initial_state,
        method="RK45",
        rtol=parameters.rtol,
        atol=parameters.atol,
        max_step=parameters.max_step,
        events=events,
    )

    result = -2
    for index in range(len(MASS)):
        if len(solution.t_events[index]) > 0:
            result = index
            break

    if result == -2:
        final_position = solution.y[:3, -1]
        final_velocity = solution.y[3:, -1]
        distance = np.linalg.norm(final_position)
        radial_velocity = np.dot(final_position, final_velocity) / max(distance, 1e-15)
        potential = -np.sum(MASS / np.sqrt(np.sum((final_position[None, :] - POSITIONS) ** 2, axis=1) + parameters.eps**2))
        energy = 0.5 * np.dot(final_velocity, final_velocity) + potential
        if distance > parameters.r_escape and radial_velocity > 0.0 and energy > 0.0:
            result = -1

    if save_path is not None:
        save_trajectory_plot(solution, x, y, result, save_path)

    return result


def make_image(basin: np.ndarray) -> np.ndarray:
    """Converte os codigos dos destinos em uma imagem RGB."""
    image = np.full((*basin.shape, 3), 0.5, dtype=np.float64)
    for index, color in enumerate(COLORS):
        image[basin == index] = color
    image[basin == -1] = [0.0, 0.0, 0.0]
    return image


def save_basin_image(
    basin: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Salva uma imagem 2D da classificacao atual das particulas."""
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(
        make_image(basin),
        origin="lower",
        extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
        interpolation="nearest",
        aspect="equal",
    )
    axis.scatter(
        POSITIONS[:, 0],
        POSITIONS[:, 1],
        s=130,
        c=COLORS,
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_trajectory_plot(
    solution,
    x: float,
    y: float,
    result: int,
    output_path: Path,
) -> None:
    """Salva uma trajetoria 3D ilustrativa."""
    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(solution.y[0], solution.y[1], solution.y[2], color="black", linewidth=1.5)
    axis.scatter(
        POSITIONS[:, 0],
        POSITIONS[:, 1],
        POSITIONS[:, 2],
        s=110,
        c=COLORS,
        edgecolors="black",
    )
    axis.scatter([x], [y], [0.0], color="magenta", s=70, label="inicio")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(f"Trajetoria ilustrativa | resultado = {result}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_grid(parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """Cria a grade 2D de condicoes iniciais no plano z=0."""
    values = np.linspace(-3.0, 3.0, parameters.resolution)
    return values, values.copy()


def run_basin_simulation(
    parameters: Parameters,
    output_directory: Path,
    prefix: str = "bacias",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula a imagem com progresso por particula e snapshots."""
    output_directory.mkdir(parents=True, exist_ok=True)
    x_values, y_values = build_grid(parameters)
    basin = np.full((parameters.resolution, parameters.resolution), -2, dtype=np.int8)
    total = parameters.resolution**2
    progress_queue: Queue[tuple[int, int, int]] = Queue()

    def worker(row: int, column: int) -> tuple[int, int, int]:
        result = classify_trajectory(x_values[column], y_values[row], parameters)
        progress_queue.put((row, column, result))
        return row, column, result

    futures = set()
    completed = 0
    last_snapshot = 0
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=parameters.workers) as executor:
        for row in range(parameters.resolution):
            for column in range(parameters.resolution):
                futures.add(executor.submit(worker, row, column))

        while futures:
            while True:
                try:
                    row, column, result = progress_queue.get_nowait()
                except Empty:
                    break

                basin[row, column] = result
                completed += 1
                elapsed = time.perf_counter() - started
                print(
                    f"\rParticula {completed:>{len(str(total))}d}/{total} "
                    f"({100.0 * completed / total:6.2f}%) "
                    f"pos=({column + 1:03d},{row + 1:03d}) "
                    f"resultado={result:2d} tempo={elapsed:8.1f}s",
                    end="",
                    flush=True,
                )

                if completed - last_snapshot >= parameters.update_every:
                    snapshot_path = output_directory / f"{prefix}_progresso_{completed:06d}.png"
                    save_basin_image(
                        basin,
                        x_values,
                        y_values,
                        snapshot_path,
                        f"Bacias em construcao: {completed}/{total} particulas",
                    )
                    last_snapshot = completed

            done, futures = wait(futures, timeout=0.05, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()

    # Os eventos finais podem ter sido enfileirados depois da ultima leitura.
    while True:
        try:
            row, column, result = progress_queue.get_nowait()
        except Empty:
            break
        basin[row, column] = result
        completed += 1

    print()
    final_path = output_directory / f"{prefix}_final.png"
    save_basin_image(
        basin,
        x_values,
        y_values,
        final_path,
        f"Bacias de atracao gravitacionais ({completed}/{total})",
    )
    return basin, x_values, y_values


def save_configuration_image(output_path: Path) -> None:
    """Salva a configuracao das massas e o plano de amostragem."""
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        POSITIONS[:, 0],
        POSITIONS[:, 1],
        POSITIONS[:, 2],
        s=150,
        c=COLORS,
        edgecolors="black",
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("Etapa 1 - configuracao dos atratores")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_convergence_image(
    basin_coarse: np.ndarray,
    basin_fine: np.ndarray,
    output_path: Path,
) -> None:
    """Salva o mapa dos pixels que mudaram entre duas resolucoes."""
    difference = basin_coarse != basin_fine
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(difference, origin="lower", cmap="gray_r", interpolation="nearest")
    axis.set_title(
        "Etapa 4 - pixels diferentes entre duas execucoes "
        f"({100.0 * difference.mean():.3f}%)"
    )
    axis.set_xlabel("coluna")
    axis.set_ylabel("linha")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_project(parameters: Parameters, output_directory: Path) -> None:
    """Executa as etapas e salva as imagens na pasta de saida."""
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Imagens serao salvas em: {output_directory.resolve()}")

    save_configuration_image(output_directory / "01_configuracao.png")
    print("[1/4] Configuracao salva.")

    sample_points = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    for index, (x, y) in enumerate(sample_points, start=1):
        sample_path = output_directory / f"02_trajetoria_{index:02d}.png"
        classify_trajectory(x, y, parameters, sample_path)
    print("[2/4] Trajetorias ilustrativas salvas.")

    basin, x_values, y_values = run_basin_simulation(parameters, output_directory)
    print("[3/4] Imagem final e snapshots de progresso salvos.")

    coarse_parameters = Parameters(
        resolution=max(8, parameters.resolution // 2),
        eps=parameters.eps,
        gamma=parameters.gamma,
        tmax=parameters.tmax,
        rcap=parameters.rcap,
        r_escape=parameters.r_escape,
        rtol=parameters.rtol,
        atol=parameters.atol,
        max_step=parameters.max_step,
        update_every=max(1, parameters.update_every),
        workers=parameters.workers,
    )
    coarse_basin, _, _ = run_basin_simulation(
        coarse_parameters,
        output_directory,
        prefix="convergencia_coarse",
    )
    fine_for_comparison = basin[::2, ::2]
    if fine_for_comparison.shape == coarse_basin.shape:
        save_convergence_image(
            coarse_basin,
            fine_for_comparison,
            output_directory / "04_convergencia.png",
        )
    print("[4/4] Analise de convergencia salva.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=80)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.quick:
        parameters = Parameters(
            resolution=16,
            tmax=15.0,
            max_step=0.10,
            update_every=4,
            workers=arguments.workers,
        )
    else:
        parameters = Parameters(
            resolution=arguments.resolution,
            update_every=arguments.update_every,
            workers=arguments.workers,
        )

    output_directory = arguments.output
    if output_directory is None:
        output_directory = Path(__file__).resolve().parent / "imagens_bacias"

    run_project(parameters, output_directory)


if __name__ == "__main__":
    main()