"""Episode records, dataset adapters, and leakage-safe splits.

An Episode is one policy rollout (or subtask) to be judged: a few frames,
the language instruction, and a binary outcome (1 = success, 0 = failure).
"""
from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Episode:
    uid: str
    images: list[str]
    instruction: str
    label: int  # 1 = success, 0 = failure
    domain: str  # e.g. "rlbench", "bridge", "ur5"
    task: str  # grouping key for splits and clustered CIs
    failure_mode: str | None = None
    meta: dict = field(default_factory=dict)


def save_manifest(episodes: list[Episode], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(asdict(ep)) + "\n")


def load_manifest(path: str | Path) -> list[Episode]:
    with Path(path).open() as f:
        return [Episode(**json.loads(line)) for line in f if line.strip()]


_INSTRUCTION_KEYS = ("instruction", "task_instruction", "high_level_goal", "goal", "prompt", "task_description")
_TASK_KEYS = ("task", "task_name", "task_id", "env_name")


def _first(sample: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in sample and sample[k] not in (None, ""):
            return sample[k]
    return None


def from_guardian_jsonl(
    path: str | Path,
    domain: str,
    image_root: str | Path | None = None,
    label_key: str = "execution_reward",
) -> list[Episode]:
    """Adapter for Guardian-style metadata jsonl (RLBench-Fail, BridgeDataV2-Fail, UR5-Fail).

    Documented fields: images (list of paths), execution_reward (1/0), failure_mode.
    Instruction and task field names vary, so several candidates are tried.
    Verify against the dataset card before trusting the output.
    """
    path = Path(path)
    root = Path(image_root) if image_root else path.parent
    episodes, n_no_instr, n_no_task = [], 0, 0
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            s = json.loads(line)
            instr = _first(s, _INSTRUCTION_KEYS)
            if instr is None:
                n_no_instr += 1
                instr = ""
            task = _first(s, _TASK_KEYS)
            if task is None:
                n_no_task += 1
                task = instr or f"unknown_{i}"
            imgs = s["images"] if isinstance(s["images"], list) else [s["images"]]
            episodes.append(
                Episode(
                    uid=f"{domain}:{s.get('id', i)}",
                    images=[str(root / p) for p in imgs],
                    instruction=str(instr),
                    label=int(s[label_key]),
                    domain=domain,
                    task=str(task),
                    failure_mode=s.get("failure_mode"),
                )
            )
    if n_no_instr:
        warnings.warn(f"{path}: {n_no_instr} samples had no instruction field; check _INSTRUCTION_KEYS")
    if n_no_task:
        warnings.warn(f"{path}: {n_no_task} samples had no task field; grouping by instruction instead")
    return episodes


def grouped_split(
    episodes: list[Episode],
    fractions: dict[str, float],
    seed: int = 0,
    group_key: str = "task",
) -> dict[str, list[Episode]]:
    """Split so that no group (default: task) appears in more than one split.

    Random episode-level splits leak scene and task identity into the test set
    and inflate every metric. Grouping by task measures generalization to new tasks.
    """
    assert abs(sum(fractions.values()) - 1.0) < 1e-6, "fractions must sum to 1"
    groups: dict[str, list[Episode]] = {}
    for ep in episodes:
        groups.setdefault(getattr(ep, group_key), []).append(ep)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = len(episodes)
    names = list(fractions)
    out: dict[str, list[Episode]] = {n: [] for n in names}
    targets = {n: fractions[n] * total for n in names}
    # Greedy fill: each group goes to the split furthest below its target.
    for k in keys:
        name = max(names, key=lambda n: targets[n] - len(out[n]))
        out[name].extend(groups[k])
    return out
