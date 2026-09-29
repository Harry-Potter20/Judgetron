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

# `taskvar` first, verified against all three real Guardian datasets. It is the only grouping key
# present in every one, and the other candidates are actively wrong here:
#   - rlbench and ur5 have no task/task_name/task_id/env_name at all, so the adapter fell back to
#     grouping by INSTRUCTION. On ur5 that is 36 instruction groups spanning only 7 real taskvars,
#     so episodes of one taskvar would land in different splits -- the exact leakage grouped_split
#     exists to prevent.
#   - bridge does have task_name, but only 13 distinct values over 7,830 episodes (it names the
#     scene, not the task), which would make the clustered bootstrap absurdly coarse. Its
#     instructions are deliberately perturbed per episode (4,320 distinct over 7,830), so grouping
#     by instruction there is close to episode-level and leaks just as badly.
_TASK_KEYS = ("taskvar", "task", "task_name", "task_id", "env_name")

# Path prefixes seen in the real metadata that point at the dataset author's own filesystem. Image
# paths are stored relative to the extracted tarball root, which always begins at `records/`.
_RECORDS_ANCHOR = "records/"


def _first(sample: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in sample and sample[k] not in (None, ""):
            return sample[k]
    return None


def normalise_image_path(p: str) -> str:
    """Strip any leading path that precedes the tarball root.

    bdv2fail stores paths like `data/failure_forge/data/bdv2fail_val_dataset/records/...`, which is
    where the files lived on the machine that built the dataset. rlbench and ur5 store them relative
    to `records/` already. Anchoring on `records/` makes all three resolve against the same
    extracted directory.
    """
    i = p.find(_RECORDS_ANCHOR)
    return p[i:] if i >= 0 else p


# Viewpoint preference, most preferred first. One viewpoint is chosen per episode so that every
# domain contributes the same number of frames; see select_start_end.
_VIEW_PREFERENCE = ("front", "0", "left", "right", "wrist", "1", "2")

_START_END = {"start": -1, "end": 1 << 30}      # sort keys for the named timesteps


def _parse_frame(path: str) -> tuple[object, str] | None:
    """(timestep, viewpoint) from a frame filename, or None if it does not look like one.

    Handles both real naming schemes: `start_img_viewpoint_front.png` / `end_img_viewpoint_left.png`
    (rlbench, bridge) and `1_img_viewpoint_0.png` / `6_img_viewpoint_2.png` (ur5, numeric timestep).
    """
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if "_img_viewpoint_" not in stem:
        return None
    ts, view = stem.split("_img_viewpoint_", 1)
    if ts in _START_END:
        return _START_END[ts], view
    return (int(ts), view) if ts.isdigit() else None


def select_start_end(images: list[str]) -> list[str]:
    """Reduce an episode's frames to [start, end] from ONE viewpoint.

    The raw datasets carry a different number of frames per domain -- 8 for rlbench (4 viewpoints x
    start/end), 6 for ur5 (3 x 2), 2 for bridge -- and the judge consumes every image it is handed.
    Left alone that is both a 4x compute difference and, worse, a confound: view count would vary
    with domain, so an in-domain vs OOD gap would partly measure how many views the judge saw rather
    than deployment shift, which is the whole question. It also breaks the prompt, since
    view_description(8) tells the model it is seeing "8 moments in time, in order" when it is seeing
    4 viewpoints at 2 times.

    Falls back to the original list if the filenames do not parse, so an unfamiliar dataset degrades
    to previous behaviour rather than silently dropping frames.
    """
    parsed = [(p, _parse_frame(p)) for p in images]
    if any(v is None for _, v in parsed):
        return images
    views = {v for _, (_, v) in parsed}
    pick = next((v for v in _VIEW_PREFERENCE if v in views), sorted(views)[0])
    chosen = sorted([(t, p) for p, (t, v) in parsed if v == pick])
    if len(chosen) < 2:
        return images
    return [chosen[0][1], chosen[-1][1]]


def from_guardian_jsonl(
    path: str | Path,
    domain: str,
    image_root: str | Path | None = None,
    label_key: str = "execution_reward",
    frames: str = "start_end",
) -> list[Episode]:
    """Adapter for Guardian-style metadata jsonl (RLBench-Fail, BridgeDataV2-Fail, UR5-Fail).

    Documented fields: images (list of paths), execution_reward (1/0), failure_mode.
    Instruction and task field names vary, so several candidates are tried.

    Verified against the real datasets (val splits) rather than the card alone:
      rlbench  n=1000  taskvar=12   task_instruction present  8 images/ep (4 views x start/end)
      bridge   n=1000  taskvar=332  task_instruction present  2 images/ep
      ur5      n=  30  taskvar=7    task_instruction present  6 images/ep (3 views x 2 times)

    `frames="start_end"` (default) reduces each episode to two frames from one viewpoint, so every
    domain contributes the same number of images; `frames="all"` keeps the raw list. See
    select_start_end for why the raw lists are not comparable across domains.
    """
    path = Path(path)
    root = Path(image_root) if image_root else path.parent
    # Identify the SOURCE FILE in the uid. `--ood` is repeatable so one domain can pool several
    # jsonls, and both episode_id and the line index restart in each file -- so "ur5:0:59" was
    # produced by both the train and test files, for genuinely different episodes with different
    # tasks and labels. Six such collisions occurred in test_ood and two in cal_ood.
    src = path.parent.name or path.stem
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
            imgs = select_start_end([normalise_image_path(p) for p in imgs]) if frames == "start_end" \
                else [normalise_image_path(p) for p in imgs]
            episodes.append(
                Episode(
                    uid=f"{domain}:{src}:{s.get('id', s.get('episode_id', i))}:{i}",
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
