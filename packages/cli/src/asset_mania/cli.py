"""Command-line interface and stable stdout/stderr mapping."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from asset_mania_contracts import canonical_json

from asset_mania.service import InspectRequest, execute_inspect


class _UsageError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self._print_message(f"{self.prog}: error: invalid command usage\n", sys.stderr)
        raise _UsageError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI without emitting tracebacks for expected failures."""
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except _UsageError:
        return 2
    except SystemExit as exit_signal:
        return exit_signal.code if isinstance(exit_signal.code, int) else 2

    try:
        result = execute_inspect(
            InspectRequest(
                input_path=arguments.input,
                output_parent=arguments.output_parent,
                workflow=arguments.workflow,
                kind=arguments.kind,
            )
        )
    except ValueError as error:
        try:
            parser.error(str(error))
        except _UsageError:
            return 2
    except Exception:  # noqa: BLE001 - the CLI boundary must never expose a traceback
        sys.stderr.write("INTERNAL_ERROR\n")
        return 4

    if result.exit_code == 73:
        sys.stderr.write(f"{result.primary_diagnostic or 'OUTPUT_STORAGE_UNAVAILABLE'}\n")
        return 73

    if result.report is None:
        sys.stderr.write("OUTPUT_STORAGE_UNAVAILABLE\n")
        return 73

    if arguments.output_format == "json":
        sys.stdout.write(canonical_json(result.report))
    else:
        sys.stdout.write(_text_report(result.report))

    if result.exit_code in {3, 4}:
        sys.stderr.write(f"{result.primary_diagnostic or 'INTERNAL_ERROR'}\n")
    return result.exit_code


def entrypoint() -> int:
    """Installed console-script entrypoint."""
    return main()


def _build_parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="asset-mania")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subcommands.add_parser("inspect", help="inspect a local image or .blend file")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.add_argument(
        "--workflow",
        choices=("image-to-3d", "scene-to-image"),
    )
    inspect_parser.add_argument(
        "--kind",
        choices=("object", "character", "face-head"),
    )
    inspect_parser.add_argument("--out", dest="output_parent", type=Path)
    inspect_parser.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "text"),
        default="json",
    )
    return parser


def _text_report(report: dict[str, object]) -> str:
    result = report["result"]
    parameters = report["parameters"]
    inputs = report["inputs"]
    diagnostics = result["diagnostics"]
    warnings = report["warnings"]
    advisories = report["advisories"]

    lines = [
        f"Asset Mania inspection: {result['status']}",
        f"Workflow: {parameters['workflow']}",
    ]
    if "kind" in parameters:
        lines.append(f"Kind: {parameters['kind']}")
    if inputs:
        lines.append(f"Input: {inputs[0]['label']}")
        lines.append(f"Media type: {inputs[0]['media_type']}")
        lines.append(f"SHA-256: {inputs[0]['sha256']}")
    lines.append(f"Diagnostics: {', '.join(diagnostics) if diagnostics else 'none'}")
    lines.append(f"Warnings: {', '.join(warnings) if warnings else 'none'}")
    for advisory in advisories:
        lines.append(f"Advisory: {advisory['message']}")
    return "\n".join(lines) + "\n"
