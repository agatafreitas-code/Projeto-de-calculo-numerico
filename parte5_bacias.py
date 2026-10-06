"""Parte 5: convergencia pontual e convergencia em medida das bacias."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from projeto import Parameters, run_basin_simulation


def basin_fractions(basin: np.ndarray, number_of_attractors: int = 4) -> np.ndarray:
    total = basin.size
    return np.array([(basin == index).sum() / total for index in range(number_of_attractors)])


def compare_basins(coarse: np.ndarray, fine: np.ndarray, number_of_attractors: int = 4) -> tuple[float, float]:
    if coarse.shape == fine.shape:
        pointwise = float(np.mean(coarse != fine))
    elif fine.shape[0] == 2 * coarse.shape[0] and fine.shape[1] == 2 * coarse.shape[1]:
        fine_sample = fine[::2, ::2]
        pointwise = float(np.mean(coarse != fine_sample))
    else:
        raise ValueError("As grades devem ter mesma dimensao (para refinamento em dt) ou dimensao 2x (para refinamento espacial).")
    measure = float(np.max(np.abs(basin_fractions(coarse, number_of_attractors) - basin_fractions(fine, number_of_attractors))))
    return pointwise, measure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--dts", type=str, default="0.05,0.025,0.0125,0.00625")
    parser.add_argument("--output", type=Path, default=Path("imagens_bacias/parte5"))
    arguments = parser.parse_args()

    dts = [float(value.strip()) for value in arguments.dts.split(",")]
    basins = []
    for index, current_dt in enumerate(dts):
        parameters = Parameters(resolution=arguments.resolution, max_step=current_dt, update_every=arguments.resolution**2 + 1)
        basin, _, _ = run_basin_simulation(parameters, arguments.output, prefix=f"parte5_dt_{index:02d}")
        basins.append(basin)
    for index in range(1, len(basins)):
        pointwise, measure = compare_basins(basins[index - 1], basins[index], 4)
        print(f"dt_anterior={dts[index - 1]:.12g} dt_atual={dts[index]:.12g} Pi={pointwise:.6e} Delta_mu={measure:.6e}")


if __name__ == "__main__":
    main()
