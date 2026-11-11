"""Actor with the height scan compressed to a latent before the MLP (cf. Extreme Parkour's scan encoder).

The scan is the last term of the actor group (stairs.py). RHPS1_SCAN_ENC="128,32": encoder hidden dims, then
latent size. Trained end to end with PPO, so the latent keeps what the reward needs (tread edges) rather than
what reconstructs the map. Exports (JIT/ONNX) take the same concatenated input as MLPModel.
"""

from __future__ import annotations

import copy
import os

import torch
from torch import nn

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP


def _enc_dims() -> tuple[list[int], int]:
  d = [int(x) for x in os.environ.get("RHPS1_SCAN_ENC", "128,32").split(",")]
  return d[:-1], d[-1]


class ScanEncoderMLPModel(MLPModel):
  def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 256, 256), activation="elu",
               obs_normalization=False, distribution_cfg=None) -> None:
    from mjlab.tasks.tracking.config.rhps1 import stairs

    hid, lat = _enc_dims()
    object.__setattr__(self, "_scan_dim", stairs.scan_dim())
    object.__setattr__(self, "_scan_latent", lat)
    super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                     distribution_cfg)
    assert self.obs_dim > self._scan_dim, (self.obs_dim, self._scan_dim)
    self.scan_encoder = MLP(self._scan_dim, lat, hid, activation)

  def _get_latent_dim(self) -> int:
    return self.obs_dim - self._scan_dim + self._scan_latent

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    x = super().get_latent(obs, masks, hidden_state)
    return torch.cat([x[..., : -self._scan_dim], self.scan_encoder(x[..., -self._scan_dim :])], dim=-1)

  def as_jit(self) -> nn.Module:
    return _ExportScanModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    m = _ExportScanModel(self)
    m.verbose = verbose
    return m


class _ExportScanModel(nn.Module):
  is_recurrent: bool = False

  def __init__(self, model: ScanEncoderMLPModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.scan_encoder = copy.deepcopy(model.scan_encoder)
    self.mlp = copy.deepcopy(model.mlp)
    self.scan_dim = model._scan_dim
    self.input_size = model.obs_dim
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    z = torch.cat([x[..., : -self.scan_dim], self.scan_encoder(x[..., -self.scan_dim :])], dim=-1)
    return self.deterministic_output(self.mlp(z))

  @torch.jit.export
  def reset(self) -> None:
    pass

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
