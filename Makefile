.PHONY: setup check test skill-check release-check license-check

setup:
	uv sync --locked --all-packages --dev

check:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

skill-check:
	uv run python scripts/validate_skill.py skills/asset-mania

release-check:
	uv run python scripts/check_release.py

license-check:
	uv run python scripts/check_license_boundary.py
