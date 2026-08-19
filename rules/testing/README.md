# Testing and verification

Use red-green-refactor for behavior code. A new behavior starts with a focused failing test, then
gains the smallest implementation, followed by refactoring under green tests.

Verification must cover the appropriate layer: package tests, deterministic output comparison,
schema validation, source-integrity checks, Skill structure, and release checks. Public fixtures
must be tiny, synthetic or redistributable, and inventoried; never use real-person imagery.

Run the applicable canonical Make targets when their backing implementation exists. Report the
command and outcome instead of claiming success from inspection alone.
