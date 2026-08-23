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

The turntable experiment keeps GPT-generated views distinct from camera observations. Seven
inferred yaw views may improve geometric coverage for local consensus, but they cannot reveal the
person's real unseen appearance. The structural audit checks dimensions, mask containment,
centering, area continuity, and duplicate pixels; it performs no identity embedding or biometric
comparison. Yaw-aware TripoSR voxel voting is a research consensus method, not photogrammetry and
not a face-accuracy benchmark.

The private live turntable experiment confirmed that a geometrically valid result can still be a
visual failure. Eight generated yaws passed structural checks and six of eight TripoSR meshes were
closed at resolution 256. Cleanup produced one watertight, winding-consistent, positive-volume
GLB, but majority voting removed recognizable facial detail. Identity consistency remains
unmeasured.

`face-anchor-visual-hull-v1` is the bounded follow-up. It preserves one observed-front TripoSR
anchor and carves a robust seven-of-eight silhouette hull for side and rear completion. It adds no
model or provider. The verified CUDA path reduces iteration time, but GPU acceleration changes
runtime rather than likeness. The private run passed its numeric gates: one closed component,
151,564 triangles, minimum/mean silhouette IoU 0.861/0.944, 96.1% front-volume retention, and 96.0%
color coverage. Blender review nevertheless failed because voxel resurfacing replaced the
recognizable front face with stepped bands. Removing front hull clipping and copying anchor colors
from either X hemisphere did not restore it.

That repeated failure closes this TripoSR voxel branch rather than inviting more threshold tuning.
The next experiment uses a private out-of-tree DAD-3DHeads plugin pinned at
`68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7`. The public integration has a closed process protocol,
synthetic fake-plugin E2E, create-only OBJ/GLB conversion, redaction checks, and Blender comparison
orchestration.

DAD is CC BY-NC-SA 4.0 non-commercial research software, not a permissive or commercially cleared
engine. Asset Mania distributes only its own Apache adapter; the external source, checkpoint,
FLAME assets, runtime, patches, and outputs stay under ignored private storage. Identity
consistency remains unmeasured.

The private live CUDA run completed on the pinned checkpoint under PyTorch `2.13.0+cu130`.
Inference, including model construction in the plugin process, took 0.946 seconds. The fixed output
contained 5,023 vertices and 9,976 triangles across the head and two equal eye shells. It had 62
boundary edges in two neck/eye-related loops, zero non-manifold edges, and consistent winding.
Observed-front vertex colour covered 30.6 percent of vertices.

Blender review passed the experiment's bounded geometry criterion: the DAD result preserved a
coherent head, nose, lips, chin, cheeks, eye sockets, ears, and side profiles and was clearly better
than both prior voxel surfaces. It did not reconstruct hair or real rear appearance, and the colour
projection remains incomplete. This result therefore supports only a face-specific geometry
improvement claim; identity consistency remains unmeasured.

Compatibility required two private lazy-import edits that keep Hydra training modules out of the
inference import path, plus process-local Python 3.12/Chumpy aliases. No architecture, checkpoint,
input resolution, topology, PyTorch build, CUDA device, or model fallback changed.

See [the roadmap](roadmap.md) for the staged decision sequence and
[security and privacy](security-and-privacy.md) for non-negotiable approval gates.

## Planned clay-first face geometry comparison

The textured DAD result is not accepted as evidence that person-specific 3D face geometry was
recovered. A corrected-axis untextured audit showed a coherent but generalized FLAME face; most
recognizability came from texture. The next local experiment therefore compares MICA identity
geometry, DECA coarse/detail geometry, their bounded fusion, and corrected DAD using identical
texture-free Blender clay renders.

The new adapters persist only 5,023 FLAME positions, 9,976 triangles, source projection, and
per-vertex detail displacement. MICA's identity feature and all detector crops and landmarks remain
transient in a network-denied process. No external source, weight, FLAME asset, detector model, or
real-person result is bundled. This capability remains planned and unverified until a separately
approved private model acquisition and clay evaluation records a result.
