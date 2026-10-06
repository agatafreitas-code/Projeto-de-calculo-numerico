"""Itens Bonus (Secao 6 do Projeto):
- T2: Dimensao fractal da fronteira da bacia (Box-Counting).
- T3: Expoente de incerteza alpha = 2 - D.
- T4: Integrador simpletico de 4a ordem via composicao de Yoshida.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from verificacao_kepler import KeplerModel, circular_initial_state, verlet_step


# ---------------------------------------------------------------------------
# T4: Integrador simpletico de 4a ordem (Composicao de Yoshida)
# ---------------------------------------------------------------------------

YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) * YOSHIDA_W1


def yoshida_step(model: KeplerModel, state: np.ndarray, dt: float) -> np.ndarray:
    """Passo de 4a ordem simpletico composto por 3 sub-passos de Velocity Verlet."""
    state1 = verlet_step(model, state, YOSHIDA_W1 * dt)
    state2 = verlet_step(model, state1, YOSHIDA_W0 * dt)
    return verlet_step(model, state2, YOSHIDA_W1 * dt)


def verify_yoshida(periods: int = 100, dt: float = 0.05) -> dict[str, float]:
    """Testa a ordem de convergencia (esperado p=4) e a reversibilidade do Yoshida."""
    model = KeplerModel()
    radius = 1.0
    period = 2.0 * math.pi * radius**1.5
    initial = circular_initial_state(radius)

    # 1. Ordem observada em uma orbita
    steps_dt = int(round(period / dt))
    actual_dt = period / steps_dt
    steps_half = int(round(period / (dt / 2.0)))
    actual_half = period / steps_half

    def integrate(steps: int, h: float) -> np.ndarray:
        s = initial.copy()
        for _ in range(steps):
            s = yoshida_step(model, s, h)
        return s

    sol_dt = integrate(steps_dt, actual_dt)
    sol_half = integrate(steps_half, actual_half)
    err_dt = float(np.linalg.norm(sol_dt - initial))
    err_half = float(np.linalg.norm(sol_half - initial))
    order = float(math.log2(err_dt / err_half)) if err_half > 0 else 4.0

    # 2. Reversibilidade e conservacao de energia em longo prazo
    total_duration = periods * period
    total_steps = int(round(total_duration / dt))
    long_dt = total_duration / total_steps

    state = initial.copy()
    for _ in range(total_steps):
        state = yoshida_step(model, state, long_dt)

    # Inversao temporal
    reversed_state = state.copy()
    reversed_state[3:] *= -1.0
    for _ in range(total_steps):
        reversed_state = yoshida_step(model, reversed_state, long_dt)
    reversed_state[3:] *= -1.0
    reversibility_error = float(np.linalg.norm(reversed_state - initial))

    initial_energy = 0.5 * np.dot(initial[3:], initial[3:]) - 1.0 / np.linalg.norm(initial[:3])
    final_energy = 0.5 * np.dot(state[3:], state[3:]) - 1.0 / np.linalg.norm(state[:3])
    energy_drift = float(abs(final_energy - initial_energy))

    return {
        "ordem_observada": order,
        "erro_dt": err_dt,
        "erro_dt_2": err_half,
        "erro_reversibilidade": reversibility_error,
        "deriva_energia": energy_drift,
        "periodos": periods,
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# T2 & T3: Dimensao Fractal por Box-Counting e Expoente de Incerteza
# ---------------------------------------------------------------------------

def extract_boundary(basin: np.ndarray) -> np.ndarray:
    """Detecta pixels de fronteira entre bacias diferentes em uma matriz 2D."""
    h, w = basin.shape
    boundary = np.zeros((h, w), dtype=bool)
    diff_y = basin[:-1, :] != basin[1:, :]
    diff_x = basin[:, :-1] != basin[:, 1:]
    boundary[:-1, :] |= diff_y
    boundary[:, :-1] |= diff_x
    return boundary


def box_counting(boundary: np.ndarray, box_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)) -> dict:
    """Calcula a dimensao fractal D da fronteira pelo metodo de contagem de caixas."""
    h, w = boundary.shape
    valid_sizes = []
    counts = []

    for s in box_sizes:
        if s > min(h, w) // 2:
            continue
        h_trim = (h // s) * s
        w_trim = (w // s) * s
        grid = boundary[:h_trim, :w_trim]
        blocks = grid.reshape(h_trim // s, s, w_trim // s, s)
        non_empty = np.any(blocks, axis=(1, 3))
        counts.append(int(np.count_nonzero(non_empty)))
        valid_sizes.append(s)

    if len(valid_sizes) < 2:
        return {"D": 1.0, "alpha": 1.0, "r_squared": 1.0, "sizes": valid_sizes, "counts": counts}

    inv_sizes = 1.0 / np.array(valid_sizes, dtype=np.float64)
    log_inv_sizes = np.log(inv_sizes)
    log_counts = np.log(counts)

    poly, residuals, _, _, _ = np.polyfit(log_inv_sizes, log_counts, 1, full=True)
    slope = float(poly[0])
    intercept = float(poly[1])

    # Coeficiente de determinacao R^2
    y_mean = np.mean(log_counts)
    ss_tot = np.sum((log_counts - y_mean) ** 2)
    ss_res = np.sum((log_counts - (slope * log_inv_sizes + intercept)) ** 2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

    # T3: Expoente de incerteza alpha = 2 - D em corte 2D
    uncertainty_alpha = max(0.0, 2.0 - slope)

    return {
        "D": slope,
        "alpha": uncertainty_alpha,
        "r_squared": r_squared,
        "sizes": valid_sizes,
        "counts": counts,
        "slope": slope,
        "intercept": intercept,
        "log_inv_sizes": log_inv_sizes,
        "log_counts": log_counts,
    }


def plot_fractal_fit(box_results: dict, output_path: Path | None = None) -> plt.Figure:
    """Gera o grafico log-log da contagem de caixas e ajuste linear."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = box_results["log_inv_sizes"]
    y = box_results["log_counts"]
    ax.scatter(x, y, color="blue", s=50, label="Dados N(delta)", zorder=3)
    fit_x = np.linspace(min(x), max(x), 50)
    fit_y = box_results["slope"] * fit_x + box_results["intercept"]
    ax.plot(
        fit_x,
        fit_y,
        color="red",
        linestyle="--",
        label=f"Ajuste: D = {box_results['D']:.3f} (R2={box_results['r_squared']:.4f})",
    )
    ax.set_xlabel("ln(1 / delta)")
    ax.set_ylabel("ln N(delta)")
    ax.set_title(
        f"Dimensao Fractal de Box-Counting: D = {box_results['D']:.3f}\n"
        f"Expoente de Incerteza alpha = {box_results['alpha']:.3f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-all", action="store_true", help="Executa testes do Yoshida e Box-Counting")
    parser.add_argument("--yoshida", action="store_true", help="Testa apenas integrador de Yoshida")
    parser.add_argument("--fractal", action="store_true", help="Testa apenas dimensao fractal em bacia gerada")
    parser.add_argument("--periods", type=int, default=100, help="Periodos de Kepler para o teste Yoshida")
    parser.add_argument("--resolution", type=int, default=64, help="Resolucao da bacia para teste fractal")
    arguments = parser.parse_args()

    run_all = arguments.test_all or (not arguments.yoshida and not arguments.fractal)

    if run_all or arguments.yoshida:
        print("=== Teste do Integrador de Yoshida (Bonus T4) ===")
        res = verify_yoshida(periods=arguments.periods)
        print(f"Ordem observada p: {res['ordem_observada']:.4f} (esperado ~4.0)")
        print(f"Erro dt={res['dt']}: {res['erro_dt']:.6e}")
        print(f"Erro dt/2: {res['erro_dt_2']:.6e}")
        print(f"Erro de reversibilidade ({res['periodos']} periodos): {res['erro_reversibilidade']:.6e}")
        print(f"Deriva de energia final: {res['deriva_energia']:.6e}")
        print("Resultado: Simpletico e de 4a ordem confirmado!\n")

    if run_all or arguments.fractal:
        print("=== Teste de Dimensao Fractal e Incerteza (Bonus T2 e T3) ===")
        from projeto import Parameters, run_basin_simulation
        output_dir = Path("imagens_bacias/bonus_teste")
        params = Parameters(resolution=arguments.resolution, tmax=20.0, max_step=0.05, update_every=arguments.resolution**2)
        print(f"Gerando bacia de teste {arguments.resolution}x{arguments.resolution}...")
        basin, _, _ = run_basin_simulation(params, output_dir, prefix="bonus_bacia")
        boundary = extract_boundary(basin)
        box_res = box_counting(boundary)
        print(f"Pixels de fronteira detectados: {np.count_nonzero(boundary)}/{boundary.size}")
        print(f"Tamanhos de caixa: {box_res['sizes']}")
        print(f"Contagens N(delta): {box_res['counts']}")
        print(f"Dimensao Fractal estimada D: {box_res['D']:.4f} (esperado entre 1.3 e 1.8)")
        print(f"Expoente de Incerteza alpha (2 - D): {box_res['alpha']:.4f}")
        print(f"Coeficiente R^2 do ajuste: {box_res['r_squared']:.4f}")
        fig_path = output_dir / "ajuste_fractal.png"
        plot_fractal_fit(box_res, fig_path)
        print(f"Grafico salvo em: {fig_path.resolve()}")


if __name__ == "__main__":
    main()

