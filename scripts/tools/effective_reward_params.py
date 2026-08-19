"""Effective reward parameters: explicit cfg params PLUS the defaults in play.

The env.yaml diff only records params a RewardTermCfg sets explicitly, so a
function whose default changed between two trees shows no difference there.
This prints every parameter each reward actually runs with.
"""
import inspect
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Mjlab-Velocity-Flat-RHPS1")
for name, t in sorted(cfg.rewards.items()):
    f = t.func
    # A class-based reward is stored as the class itself, so type(f).__call__ is
    # the metaclass and introspects as (*args, **kwargs). Use its own __call__.
    if inspect.isclass(f):
        tgt = f.__call__
    elif inspect.isfunction(f):
        tgt = f
    else:
        tgt = type(f).__call__
    try:
        sig = inspect.signature(tgt)
    except Exception:
        print(f"{name}\tweight={t.weight}\t<no signature>")
        continue
    eff = []
    for p in sig.parameters.values():
        if p.name in ("self", "env") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in t.params:
            v, src = t.params[p.name], "cfg"
        elif p.default is not inspect.Parameter.empty:
            v, src = p.default, "def"
        else:
            v, src = "<REQUIRED>", "!!"
        s = str(v)
        if "SceneEntityCfg" in s:
            s = "SceneEntityCfg(...)"
        eff.append(f"{p.name}={s}[{src}]")
    print(f"{name}\tweight={t.weight}\t" + " ".join(eff))
