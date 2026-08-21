# Research and landscape

Asset Mania will evaluate research as evidence for bounded product claims, not as proof that a
workflow is ready. Each candidate must be assessed for output reliability, bias, applicant or
user acceptance where relevant, licensing, privacy, reproducibility, and operating cost.

Primary references to assess for future generic image-to-3D work include
[InstantMesh](https://github.com/TencentARC/InstantMesh) and
[TRELLIS](https://github.com/microsoft/TRELLIS).

**Licensing correction.** Earlier wording treated these stock runtimes as permissively
licensed. That was wrong at the level that matters: a permissive top-level license does not
make a runtime usable, because the dependency closure carries its own terms. Both stacks
pull in components under non-commercial or custom licenses, and model weights are licensed
separately from code. They therefore remain **research-only** in Asset Mania: no evaluated
adapter may ship until those dependencies are replaced or each one is independently cleared
and recorded in `THIRD_PARTY_NOTICES.md` with its exact license and redistribution
evidence. Nothing in this repository bundles, vendors, or downloads either runtime or its
weights.

Future scene-guided generation will evaluate the applicable provider documentation at
implementation time. Face/head reconstruction remains
research-only and must not be marketed as exact likeness, anonymity, biometric safety, or legal
clearance.

See [the roadmap](roadmap.md) for the staged decision sequence and
[security and privacy](security-and-privacy.md) for non-negotiable approval gates.
