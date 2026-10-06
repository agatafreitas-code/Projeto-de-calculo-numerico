"""Parte 3: eventos de captura e escape na simulacao principal."""

from __future__ import annotations

import argparse
from pathlib import Path

from projeto import Parameters, classify_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    parameters = Parameters(tmax=arguments.tmax)
    result = classify_trajectory(arguments.x, arguments.y, parameters, arguments.output)
    labels = {-2: "pendente", -1: "escape"}
    label = labels.get(result, f"capturada pelo atrator {result}")
    print(f"ponto=({arguments.x}, {arguments.y}, 0), resultado={label}")


if __name__ == "__main__":
    main()
