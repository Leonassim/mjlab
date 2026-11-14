"""Distiller l'experte de suivi de mouvement vers une politique COMMANDABLE.

L'experte (tache tracking) observe la reference : angles cibles, erreur d'ancre.
Rien de tout ca n'existe sur le robot. L'eleve observe ce que mc_rtc fournit --
proprioception plus une commande de vitesse a 3 dims -- et apprend a reproduire
les actions de l'experte.

DAGGER, pas de la regression naive. Si l'eleve n'apprend que sur les etats
visites par l'EXPERTE, il ne voit jamais ceux ou lui-meme derive, et la moindre
erreur l'emmene hors distribution sans qu'il sache se rattraper. On melange donc
progressivement ses propres actions dans le deroulement (beta decroissant), tout
en continuant d'etiqueter avec l'experte.

  uv run python scripts/tools/distill.py <run> <checkpoint.pt> [--iters 2000]
"""

import argparse
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy"


class Student(torch.nn.Module):
  """Meme largeur que l'experte : on ne change que ce qu'elle observe."""

  def __init__(self, n_in: int, n_out: int) -> None:
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(n_in, 512), torch.nn.ELU(),
      torch.nn.Linear(512, 256), torch.nn.ELU(),
      torch.nn.Linear(256, 128), torch.nn.ELU(),
      torch.nn.Linear(128, n_out),
    )
    self.register_buffer("mean", torch.zeros(n_in))
    self.register_buffer("var", torch.ones(n_in))
    self.register_buffer("count", torch.tensor(1e-4))

  def update_norm(self, x: torch.Tensor) -> None:
    bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    d = bm - self.mean
    tot = self.count + bc
    self.mean += d * bc / tot
    m_a = self.var * self.count
    m_b = bv * bc
    self.var = (m_a + m_b + d.pow(2) * self.count * bc / tot) / tot
    self.count += bc

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net((x - self.mean) / torch.sqrt(self.var + 1e-8))


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("run"); p.add_argument("checkpoint")
  p.add_argument("--iters", type=int, default=2000)
  p.add_argument("--envs", type=int, default=1024)
  p.add_argument("--steps", type=int, default=24, help="pas collectes par iteration")
  p.add_argument("--motion", default="docs/bwc_motion.npz")
  p.add_argument("--root", default="logs/rsl_rl/rhps1_tracking")
  p.add_argument("--out", default="logs/distill")
  p.add_argument("--beta-end", type=float, default=0.0,
                 help="part des actions de l'EXPERTE en fin d'entrainement")
  a = p.parse_args()
  dev = "cuda:0" if torch.cuda.is_available() else "cpu"

  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = a.envs
  cfg.commands["motion"].motion_file = a.motion
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=load_rl_cfg(TASK).clip_actions
  )
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
  runner.load(str(Path(a.root) / a.run / a.checkpoint),
              load_cfg={"actor": True}, strict=True, map_location=dev)
  expert = runner.get_inference_policy(device=dev)

  obs = env.get_observations()
  n_in = obs["student"].shape[1]
  n_out = env.unwrapped.action_manager.total_action_dim
  student = Student(n_in, n_out).to(dev)
  opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)

  out = Path(a.out) / a.run
  out.mkdir(parents=True, exist_ok=True)
  hist = deque(maxlen=50)
  print(f"eleve {n_in} -> {n_out}   experte {obs['actor'].shape[1]} dims   "
        f"{a.envs} envs   {a.iters} iterations   ({dev})", flush=True)

  for it in range(a.iters):
    # beta : part des actions de l'experte dans le deroulement. 1 au depart
    # (l'eleve n'a rien appris), puis on lui laisse la main pour qu'il visite
    # SES propres etats -- c'est tout l'interet de DAgger.
    beta = max(a.beta_end, 1.0 - it / (0.6 * a.iters))
    xs, ys = [], []
    with torch.no_grad():
      for _ in range(a.steps):
        a_exp = expert(obs)
        xs.append(obs["student"].clone())
        ys.append(a_exp.clone())
        a_stu = student(obs["student"])
        mix = torch.rand(a.envs, 1, device=dev) < beta
        obs, _, _, _ = env.step(torch.where(mix, a_exp, a_stu))
    x = torch.cat(xs); y = torch.cat(ys)
    student.update_norm(x)
    for _ in range(5):
      loss = torch.nn.functional.mse_loss(student(x), y)
      opt.zero_grad(); loss.backward()
      torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
      opt.step()
    sched.step()
    hist.append(loss.detach().item())
    if it % 25 == 0 or it == a.iters - 1:
      print(f"  it {it:5d}  beta {beta:4.2f}  mse {sum(hist)/len(hist):.5f}  "
            f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
    if it % 200 == 0 or it == a.iters - 1:
      torch.save({"model": student.state_dict(), "n_in": n_in, "n_out": n_out},
                 out / f"student_{it}.pt")
  print(f"ecrit {out}/student_{a.iters - 1}.pt")


if __name__ == "__main__":
  main()
