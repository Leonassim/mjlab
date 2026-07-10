from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class PlotTorquesConfig:
  input_path: Path
  """Path to the saved torque npz file."""

  series: str = "env0"
  """Series to plot: 'env0' or 'mean'."""

  pattern: str = ".*"
  """Regex used to filter actuator names."""

  output_path: Path | None = None
  """Output png path. Defaults to <input_stem>_<series>.png."""

  max_cols: int = 3
  """Maximum number of subplot columns."""

  smoothing: int = 1
  """Moving-average window. Use 1 to disable smoothing."""


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
  if window <= 1:
    return values
  kernel = np.ones(window, dtype=np.float32) / float(window)
  return np.convolve(values, kernel, mode="same")


def main() -> None:
  cfg = tyro.cli(PlotTorquesConfig)

  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise SystemExit(
      "matplotlib is required for plot_torques. Install it, then rerun the command."
    ) from exc

  data = np.load(cfg.input_path)
  actuator_names = [str(name) for name in data["actuator_names"]]
  time_s = data["time_s"]

  if cfg.series == "env0":
    torques = data["torques_env0"]
  elif cfg.series == "mean":
    torques = data["torques_mean"]
  else:
    raise SystemExit(f"Invalid series '{cfg.series}'. Use 'env0' or 'mean'.")

  pattern = re.compile(cfg.pattern)
  indices = [i for i, name in enumerate(actuator_names) if pattern.search(name)]
  if not indices:
    raise SystemExit(f"No actuator matched pattern '{cfg.pattern}'.")

  nplots = len(indices)
  ncols = min(cfg.max_cols, nplots)
  nrows = math.ceil(nplots / ncols)
  fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(5.5 * ncols, 3.2 * nrows),
    squeeze=False,
    sharex=True,
  )
  fig.suptitle(f"Actuator Torques ({cfg.series})", fontsize=14)

  for ax, actuator_idx in zip(axes.flat, indices, strict=False):
    raw = torques[:, actuator_idx]
    series = _moving_average(raw, cfg.smoothing)
    ax.plot(time_s, raw, alpha=0.25, linewidth=1.0, color="#8aa1b1")
    ax.plot(time_s, series, linewidth=1.8, color="#0b7285")
    ax.set_title(actuator_names[actuator_idx])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("torque")
    ax.grid(alpha=0.25)

  for ax in axes.flat[nplots:]:
    ax.axis("off")

  fig.tight_layout()
  output_path = cfg.output_path
  if output_path is None:
    output_path = cfg.input_path.with_name(
      f"{cfg.input_path.stem}_{cfg.series}_torques.png"
    )
  fig.savefig(output_path, dpi=180, bbox_inches="tight")
  print(output_path)


if __name__ == "__main__":
  main()
