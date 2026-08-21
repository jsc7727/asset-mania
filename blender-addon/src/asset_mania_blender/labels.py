# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic portable labels for private Blender datablock names.

A portable artifact never carries a real datablock name. Labels are assigned by sorting
the names inside one kind, so the same scene always produces the same labels while the
label itself reveals nothing but an ordinal.
"""

from collections.abc import Iterable, Mapping

KINDS = ("camera", "mesh", "armature", "action", "bone")


def assign_labels(kind: str, names: Iterable[str]) -> dict[str, str]:
    """Map each private name to `kind-N`, ordered by the sorted private name."""
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not a labelled datablock kind")
    ordered = sorted(set(names))
    return {name: f"{kind}-{index}" for index, name in enumerate(ordered, start=1)}


def label_for(labels: Mapping[str, str], name: str) -> str:
    try:
        return labels[name]
    except KeyError:
        raise ValueError("the requested datablock is not in the labelled inventory") from None


def sorted_labels(*label_maps: Mapping[str, str]) -> list[str]:
    """Every assigned label, sorted, as the response's `portable_labels` array."""
    labels: set[str] = set()
    for label_map in label_maps:
        labels.update(label_map.values())
    return sorted(labels)
