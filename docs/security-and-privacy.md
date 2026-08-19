# Security and privacy

v0.1 is local-first: it inspects supported local inputs without network requests, uploads,
model downloads, Blender invocation, rendering, GPU jobs, or provider calls. Source inspection
is read-only; it may write a new run directory but must never overwrite or embed the input.

Any future action that leaves the machine or can incur cost requires fresh explicit approval for
that run. Before asking, the tool must disclose the provider/model/revision, data leaving the
machine, known or unknown retention and region, estimated cost or download/runtime needs, output
paths, overwrite behavior, and relevant privacy/license implications.

Face/head work is a separately gated future workflow. External processing will require the
user's confirmation of rights and consent. No face input belongs in fixtures, telemetry,
training data, galleries, or bug reports by default. See [the roadmap](roadmap.md) and
[security reporting](../SECURITY.md).
