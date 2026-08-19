#!/usr/bin/env python3
"""Validate repository-owned Agent Skill structure using only the standard library."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_FRONTMATTER = re.compile(r"\A---\r?\n(?P<content>.*?)\r?\n---(?:\r?\n|\Z)", re.DOTALL)
_FIELD = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):(?P<value>.*)$")
_SCAFFOLD_MARKERS = ("[TODO:", "TODO:", "PLACEHOLDER")
_TEXT_SUFFIXES = {".json", ".md", ".py", ".yaml", ".yml"}


def _scalar(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] in {'"', "'"} and stripped[-1] == stripped[0]:
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped
        return parsed if isinstance(parsed, str) else stripped
    return stripped


def _frontmatter_fields(content: str) -> tuple[dict[str, str] | None, str]:
    match = _FRONTMATTER.match(content)
    if match is None:
        return None, content
    fields: dict[str, str] = {}
    for line in match.group("content").splitlines():
        field = _FIELD.match(line)
        if field is not None:
            fields[field.group("key")] = _scalar(field.group("value"))
    return fields, content[match.end() :]


def validate_skill(skill: Path) -> list[str]:
    findings: list[str] = []
    skill = Path(skill)
    skill_file = skill / "SKILL.md"
    skill_content = ""
    has_frontmatter = False

    if not skill_file.is_file():
        findings.append("SKILL.md: missing")
    else:
        skill_content = skill_file.read_text(encoding="utf-8")
        fields, _body = _frontmatter_fields(skill_content)
        if fields is None:
            findings.append("SKILL.md: missing YAML frontmatter")
        else:
            has_frontmatter = True
            name = fields.get("name", "").strip()
            description = fields.get("description", "").strip()
            if not name:
                findings.append("SKILL.md: missing name")
            elif name != skill.name:
                findings.append(f"SKILL.md: name '{name}' does not match folder '{skill.name}'")
            if not description:
                findings.append("SKILL.md: missing description")

    if not (skill / "agents" / "openai.yaml").is_file():
        findings.append("agents/openai.yaml: missing")

    if skill.is_dir():
        for path in sorted(skill.rglob("*")):
            if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            for marker in _SCAFFOLD_MARKERS:
                if marker in content:
                    relative = path.relative_to(skill).as_posix()
                    findings.append(f"{relative}: unfinished scaffold marker '{marker}'")
                    break

    references = skill / "references"
    if has_frontmatter and references.is_dir():
        for reference in sorted(path for path in references.rglob("*") if path.is_file()):
            relative = reference.relative_to(skill).as_posix()
            if relative not in skill_content:
                findings.append(f"{relative}: not discoverable from SKILL.md")

    return sorted(set(findings))


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 1:
        print("usage: validate_skill.py <skill-directory>", file=sys.stderr)
        return 2
    findings = validate_skill(Path(arguments[0]))
    for finding in findings:
        print(finding)
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
