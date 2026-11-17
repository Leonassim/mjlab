# depth4_1500: depth-image stairs student (DAgger from s50e_50hz_15000)

- **Inputs** (ONNX `policy.onnx`):
  - `obs` [1, 126]: the first 126 entries of the s50e actor observation, i.e. everything except the 325-value
    height scan. They are unnormalized: the student holds the teacher's normalizer, frozen.
  - `depth` [1, 3072]: a 48 x 64 depth image, row-major with rows top to bottom. Each value is range / 2.88 m, and 0
    means no return (Azure invalid). Pixels outside the 120 deg WFOV circle are always 0.
  - `h_in` [1, 1, 256]: GRU state, zero at arming. Feed `h_out` back at every 50 Hz step.
- **Camera**:
  - Azure depth frame on HEAD_P_LINK, at `CAM_POS` / `CAM_RPY` of `vision_map.py`; head held at 25 deg.
  - Virtual pinhole: 70 deg vertical, 4:3 (about 86 deg horizontal), tilted 25 deg down about the camera x axis
    inside the WFOV.
  - Rays: `depth_student.HeadDepthPatternCfg`.
  - Frames at 25 Hz, policy reading the last delivered frame (trained with 80 ms +- 40 ms of latency).
- **Output**: `actions` [1, 30], the same meaning as the teacher's. `h_out`.
- **Training**: `RHPS1_DEPTH_RANDLEVEL=1`. Each episode draws its own degradations: noise 0-2 cm, scale +-1 %, bias
  +-1 cm, shift +-1 px, 0-10 % missing pixels, 0-0.8 mixed pixels at edges, 0.5 s blackouts.
- **mjlab bench** (`depth_eval.sh`, student alone, 32 envs x 24 s, stairs only):

  | | arrivals | falls |
  |---|---|---|
  | degraded image | 27 | 4 |
  | clean image | 22 | 9 |
  | teacher, head held, exact scan | 28 | 2 |

- **Known weakness**: climbs. The student hesitates mid-flight, and the training curriculum's climb level collapsed
  (see `docs/coordination/rtx4080.md`).
