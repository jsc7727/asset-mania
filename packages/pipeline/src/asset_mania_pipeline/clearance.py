"""The fail-closed engine-clearance gate.

An inference engine is unusable here until a user has recorded, for the engine code, the
model weights, the preprocessing model, and *every* runtime dependency: an immutable
revision, a content digest, a license identifier, a license URL, a download receipt, and a
commercial-use state.

The design decision worth stating: `unknown` fails exactly as `prohibited` does. An
unchecked license is not a smaller problem than a forbidding one -- the reason this project's
v0.1 documentation described two engines as permissively licensed when their dependency
closures were not is that nobody had checked. A gate that accepts `unknown` reproduces that
error with extra steps.
"""

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from asset_mania_contracts import (
    CLEARANCE_COMPONENT_ROLES,
    canonical_digest,
)

NOT_CLEARED = "ENGINE_NOT_CLEARED"
LICENSE_UNCLEARED = "ENGINE_LICENSE_UNCLEARED"
REQUIRED_RECORD_FIELDS = (
    "name",
    "revision",
    "content_sha256",
    "license_spdx",
    "license_url",
    "commercial_use",
    "download_receipt_sha256",
)


class EngineNotCleared(Exception):
    """The clearance artifact is absent, incomplete, expired, or not user-issued."""


class EngineLicenseUncleared(Exception):
    """A component or dependency is not cleared for use."""


def _refuse(detail: str) -> None:
    raise EngineNotCleared(f"{NOT_CLEARED}: {detail}")


def _refuse_license(detail: str) -> None:
    raise EngineLicenseUncleared(f"{LICENSE_UNCLEARED}: {detail}")


def _check_record(record: Mapping[str, Any], *, label: str) -> None:
    missing = [field for field in REQUIRED_RECORD_FIELDS if not record.get(field)]
    if missing:
        _refuse(f"{label} is missing {sorted(missing)}")

    state = record["commercial_use"]
    if state != "cleared":
        # `unknown` and `prohibited` are both refusals; see the module docstring.
        _refuse_license(f"{label} records commercial_use {state!r}")


def verify_engine_clearance(
    clearance: Mapping[str, Any] | None,
    *,
    engine: str,
    now: datetime,
) -> str:
    """Verify a clearance and return its digest, or refuse.

    Returns the digest so a caller can bind a plan to the exact artifact it verified.
    """
    if clearance is None:
        _refuse("no clearance artifact was supplied")

    if clearance.get("schema_id") != "asset-mania/engine-clearance":
        _refuse("the artifact is not an engine clearance")
    if clearance.get("engine") != engine:
        _refuse(f"the clearance covers {clearance.get('engine')!r}, not {engine!r}")
    if clearance.get("cleared_by") != "user":
        _refuse(
            f"clearance issued by {clearance.get('cleared_by')!r}; only the user may accept "
            "a third party's license terms"
        )

    preimage = {key: value for key, value in clearance.items() if key != "clearance_sha256"}
    digest = canonical_digest(preimage)
    if digest != clearance.get("clearance_sha256"):
        _refuse("the clearance digest does not match its content")

    try:
        cleared_at = datetime.fromisoformat(clearance["cleared_at"])
        expires_at = datetime.fromisoformat(clearance["expires_at"])
    except (KeyError, ValueError) as error:
        _refuse("the clearance validity window is unreadable")
        raise AssertionError from error  # pragma: no cover - _refuse always raises
    if now < cleared_at:
        _refuse("the clearance is not yet in effect")
    if now > expires_at:
        _refuse("the clearance has expired")

    components = list(clearance.get("components") or [])
    roles = [item.get("role") for item in components]
    if roles != list(CLEARANCE_COMPONENT_ROLES):
        _refuse(f"components must cover exactly {list(CLEARANCE_COMPONENT_ROLES)}, found {roles}")

    dependencies = list(clearance.get("runtime_dependencies") or [])
    if not dependencies:
        _refuse(
            "the clearance lists no runtime dependency, which is never true for an inference engine"
        )
    names = [item.get("name") for item in dependencies]
    if names != sorted(names):
        _refuse("runtime dependencies must be name-sorted")
    if len(set(names)) != len(names):
        _refuse("a runtime dependency is listed twice")

    for component in components:
        _check_record(component, label=f"component {component.get('role')!r}")
    for dependency in dependencies:
        _check_record(dependency, label=f"dependency {dependency.get('name')!r}")

    return digest


def run_if_cleared(
    *,
    clearance: Mapping[str, Any] | None,
    engine: str,
    now: datetime,
    run: Callable[[str], Any],
) -> tuple[str, Any]:
    """Verify clearance, then run. The engine is unreachable until the gate passes.

    Composing the two here is what makes the ordering testable: an uncleared engine raises
    before `run` is ever called, so no weight is loaded and no subprocess starts on a path
    that was never cleared.
    """
    digest = verify_engine_clearance(clearance, engine=engine, now=now)
    return digest, run(digest)


def clearance_summary(clearance: Mapping[str, Any]) -> dict[str, Any]:
    """A log-safe summary. It carries no license URL and no download receipt."""
    components = list(clearance.get("components") or [])
    dependencies = list(clearance.get("runtime_dependencies") or [])
    states: dict[str, int] = {}
    for record in (*components, *dependencies):
        state = str(record.get("commercial_use"))
        states[state] = states.get(state, 0) + 1
    return {
        "engine": clearance.get("engine"),
        "component_count": len(components),
        "dependency_count": len(dependencies),
        "commercial_use_states": dict(sorted(states.items())),
        "cleared_by": clearance.get("cleared_by"),
    }


def uncleared_entries(clearance: Mapping[str, Any]) -> list[str]:
    """Which entries block this clearance, by label only.

    Useful for telling a user what to go and clear, without echoing a license URL or a
    receipt digest into a log.
    """
    blocked: list[str] = []
    for component in clearance.get("components") or []:
        if component.get("commercial_use") != "cleared":
            blocked.append(f"component:{component.get('role')}")
    for dependency in clearance.get("runtime_dependencies") or []:
        if dependency.get("commercial_use") != "cleared":
            blocked.append(f"dependency:{dependency.get('name')}")
    return sorted(blocked)


def require_mask_or_audited_remover(
    *,
    mask_sha256: str | None,
    background_removal_clearance: Mapping[str, Any] | None,
    engine: str,
    now: datetime,
) -> tuple[str | None, str | None]:
    """A mask, or an audited background remover pinned by digest. Never neither.

    A single-image reconstructor handed a full scene reconstructs the scene, so an absent
    mask is a different job rather than a permissive default. And an unpinned remover is
    refused outright: a background model that arrives without a digest is the classic way
    an uncleared dependency slips into a pipeline.
    """
    from asset_mania_contracts import DiagnosticCode

    if mask_sha256:
        return mask_sha256, None

    if background_removal_clearance is None:
        raise ValueError(
            f"{DiagnosticCode.MASK_REQUIRED.value}: supply a mask or an audited "
            "background-removal clearance"
        )

    preprocessing = next(
        (
            item
            for item in background_removal_clearance.get("components") or []
            if item.get("role") == "preprocessing_model"
        ),
        None,
    )
    if preprocessing is None or not preprocessing.get("content_sha256"):
        raise ValueError(
            f"{DiagnosticCode.BACKGROUND_REMOVAL_UNPINNED.value}: the background-removal "
            "model has no pinned content digest"
        )

    digest = verify_engine_clearance(background_removal_clearance, engine=engine, now=now)
    return None, digest
