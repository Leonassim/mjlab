"""Le robot traverse-t-il le sol, ou pietine-t-il ? Mesure sur la video.

La camera des videos d'entrainement suit le robot (viewer.origin_type =
ASSET_BODY) : le robot reste au centre de l'image quoi qu'il arrive. Donc on ne
peut pas juger de son avance en le regardant lui -- il faut regarder le SOL.

Le sol est un damier. S'il defile sous le robot, le robot avance ; s'il est fige,
le robot pietine. On mesure ce defilement par correlation de phase entre images
successives, sur deux bandes laterales choisies loin du robot pour qu'il ne
pollue pas la mesure.

La sortie est un deplacement en PIXELS cumules, pas en metres : la question est
binaire (le sol defile ou non) et l'echelle importe peu.

  uv run python scripts/tools/video_ground_motion.py <video.mp4> [<video2.mp4> ...]
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np


def frames(path: str, stride: int = 5) -> np.ndarray:
  """Decoder la video en niveaux de gris via ffmpeg, une image sur `stride`."""
  probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
    capture_output=True, text=True, check=True,
  ).stdout.strip()
  w, h = (int(x) for x in probe.split("x"))
  raw = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=not(mod(n\\,{stride}))",
     "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
    capture_output=True, check=True,
  ).stdout
  n = len(raw) // (w * h)
  return np.frombuffer(raw[: n * w * h], dtype=np.uint8).reshape(n, h, w).astype(float)


def shift_1d(a: np.ndarray, b: np.ndarray) -> float:
  """Decalage horizontal de b par rapport a a, par correlation de phase."""
  a = a - a.mean()
  b = b - b.mean()
  fa = np.fft.rfft(a)
  fb = np.fft.rfft(b)
  cross = fa * np.conj(fb)
  mag = np.abs(cross)
  cross = np.where(mag > 1e-12, cross / mag, 0)
  corr = np.fft.irfft(cross, n=a.size)
  k = int(np.argmax(corr))
  return k - a.size if k > a.size // 2 else k


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)

  for path in sys.argv[1:]:
    f = frames(path)
    n, h, w = f.shape
    # Bandes laterales, moitie basse : le sol, sans le robot qui est au centre.
    left = f[:, int(0.60 * h) : int(0.95 * h), : int(0.22 * w)]
    right = f[:, int(0.60 * h) : int(0.95 * h), int(0.78 * w) :]

    total = 0.0
    per_frame = []
    for band in (left, right):
      # Profil horizontal moyen : le damier y apparait comme une ondulation.
      prof = band.mean(axis=1)
      d = [shift_1d(prof[i], prof[i + 1]) for i in range(n - 1)]
      per_frame.append(float(np.median(np.abs(d))))
      total += float(np.sum(d))

    texture = float(
      np.mean([np.std(left.mean(axis=1)), np.std(right.mean(axis=1))])
    )
    print(f"\n{path}")
    print(f"  {n} images analysees, contraste du sol (ecart-type) {texture:.1f}")
    print(f"  decalage median par image : gauche {per_frame[0]:.2f} px, "
          f"droite {per_frame[1]:.2f} px")
    print(f"  defilement cumule (somme des deux bandes) : {total:+.0f} px")
    verdict = "LE SOL DEFILE -> le robot avance" if abs(total) > 40 else (
      "sol quasi fige -> le robot ne traverse pas le decor")
    print(f"  {verdict}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
