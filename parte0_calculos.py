"""Calculos numericos auxiliares da Parte 0 para a configuracao atual."""

from __future__ import annotations

import argparse
import math

import numpy as np

from projeto import POSITIONS as DEFAULT_POSITIONS


def mixed_volume(positions: np.ndarray) -> float:
    if len(positions) < 4:
        return float("nan")
    return float(np.dot(positions[1] - positions[0], np.cross(positions[2] - positions[0], positions[3] - positions[0])))


def reference_length(positions: np.ndarray) -> float:
    distances = [np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i)]
    return float(np.mean(distances))


def radial_free_fall_time(radius: float, gravitational_parameter: float = 1.0) -> float:
    return math.pi / (2.0 * math.sqrt(2.0 * gravitational_parameter)) * radius**1.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=2.0)
    arguments = parser.parse_args()
    positions = np.asarray(DEFAULT_POSITIONS, dtype=np.float64)
    print(f"L_medio={reference_length(positions):.12g}")
    print(f"produto_misto={mixed_volume(positions):.12g}")
    print("M_total_adimensional=1")
    print(f"tempo_queda_radial_de_{arguments.radius:g}={radial_free_fall_time(arguments.radius):.12g}")
    print("Tc_fisico=sqrt(L^3/(G*M_total)); substitua L e M em unidades fisicas do grupo")


if __name__ == "__main__":
    main()
