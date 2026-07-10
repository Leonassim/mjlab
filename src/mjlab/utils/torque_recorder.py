from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class TorqueRecorder:
  """Accumulate actuator torques over time and dump them to disk."""

  actuator_names: tuple[str, ...]
  step_dt: float
  steps: list[int] = field(default_factory=list)
  torques_env0: list[np.ndarray] = field(default_factory=list)
  torques_mean: list[np.ndarray] = field(default_factory=list)

  def append(self, step: int, actuator_force: torch.Tensor) -> None:
    if actuator_force.ndim != 2:
      raise ValueError(
        f"Expected actuator_force with shape (num_envs, num_actuators), got {actuator_force.shape}"
      )
    self.steps.append(step)
    self.torques_env0.append(
      actuator_force[0].detach().cpu().numpy().astype(np.float32)
    )
    self.torques_mean.append(
      actuator_force.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    )

  def dump(self, path: str | Path) -> Path | None:
    if not self.steps:
      return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(self.steps, dtype=np.int64)
    np.savez_compressed(
      output_path,
      actuator_names=np.asarray(self.actuator_names),
      steps=steps,
      time_s=steps.astype(np.float32) * np.float32(self.step_dt),
      torques_env0=np.stack(self.torques_env0, axis=0),
      torques_mean=np.stack(self.torques_mean, axis=0),
    )
    return output_path
