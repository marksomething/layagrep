"""Command-line semantic grep backed by Laya's ``noul`` question type."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

QUESTION_KEY = "match"
LAYA_CALIBRATION_WARNING = (
    r"^laya: this checkpoint ships invalid temperatures or values outside "
    r".*using choice:11\+="
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layagrep",
        add_help=False,
        description=(
            "Search lines by meaning. Laya evaluates each line against your "
            "description and returns a noul probability that it matches."
        ),
        epilog=(
            "Example: layagrep 'mentions a cancelled subscription' app.log\n"
            "The first run downloads the Laya checkpoint. Use --scores to print "
            "the noul probability for each matching line."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--help", action="help", help="show this help message and exit")
    parser.add_argument("description", help="natural-language description of matching lines")
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="FILE",
        help="files to search; use '-' for stdin (default: stdin)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        metavar="PROBABILITY",
        help="minimum noul yes-probability to match (default: 0.5)",
    )
    parser.add_argument("-n", "--line-number", action="store_true", help="prefix output with line numbers")
    filenames = parser.add_mutually_exclusive_group()
    filenames.add_argument("-H", "--with-filename", action="store_true", help="always print filenames")
    filenames.add_argument("-h", "--no-filename", action="store_true", help="never print filenames")
    parser.add_argument("-v", "--invert-match", action="store_true", help="select lines below the threshold")
    parser.add_argument("-c", "--count", action="store_true", help="print the number of selected lines per input")
    parser.add_argument("-l", "--files-with-matches", action="store_true", help="print each input with a match")
    parser.add_argument("-q", "--quiet", action="store_true", help="stop after the first match; print nothing")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--scores", action="store_true", help="include each match's noul probability")
    output.add_argument("--json", action="store_true", help="write matching lines as JSON Lines, including noul scores")
    return parser


def validate_probability(value: float, option: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{option} must be a number between 0 and 1")


def noul_probability(result: dict[str, Any]) -> float:
    """Read and validate the yes-probability returned for the noul answer."""
    try:
        value = float(result["answers"][QUESTION_KEY]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Laya response did not contain answers.match.noul") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Laya returned an invalid noul probability: {value!r}")
    return value


def matches(probability: float, threshold: float, invert: bool = False) -> bool:
    selected = probability >= threshold
    return not selected if invert else selected


def format_match(
    *,
    path: str,
    line_number: int,
    text: str,
    probability: float,
    show_filename: bool,
    args: argparse.Namespace,
) -> str:
    if args.json:
        return json.dumps(
            {"file": path, "line": line_number, "score": probability, "text": text},
            ensure_ascii=False,
        )

    prefix = ""
    if show_filename:
        prefix += f"{path}:"
    if args.line_number:
        prefix += f"{line_number}:"
    if args.scores:
        prefix = f"{probability:.3f}\t{prefix}"
    return f"{prefix}{text}"


def iter_lines(stream: TextIO) -> Iterable[tuple[int, str]]:
    for number, raw_line in enumerate(stream, start=1):
        # Remove only line terminators; preserve all other whitespace in the match.
        yield number, raw_line.rstrip("\r\n")


def search_stream(
    stream: TextIO,
    *,
    path: str,
    description: str,
    router: Any,
    args: argparse.Namespace,
    show_filename: bool,
) -> tuple[int, float | None]:
    """Search one stream, printing selected lines and returning count/first score."""
    question = {
        QUESTION_KEY: {
            "type": "noul",
            "instructions": (
                f"{description}"
            ),
        }
    }
    count = 0
    first_probability: float | None = None
    for line_number, text in iter_lines(stream):
        # Laya currently emits this known warning for its unused 11+-option choice
        # calibration bucket. Keep it from cluttering grep output without muting other
        # RuntimeWarnings from Laya or the application.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=LAYA_CALIBRATION_WARNING,
                category=RuntimeWarning,
            )
            result = router.predict(text, question)
        probability = noul_probability(result)
        if not matches(probability, args.threshold, args.invert_match):
            continue

        count += 1
        if first_probability is None:
            first_probability = probability
        if args.quiet:
            return count, first_probability
        if args.files_with_matches:
            print(path)
            return count, first_probability
        if not args.count:
            print(
                format_match(
                    path=path,
                    line_number=line_number,
                    text=text,
                    probability=probability,
                    show_filename=show_filename,
                    args=args,
                )
            )

    if args.files_with_matches and count:
        print(path)
    elif args.count:
        prefix = f"{path}:" if show_filename else ""
        print(f"{prefix}{count}")
    return count, first_probability


def _load_router() -> Any:
    try:
        from laya import Router
    except ImportError as exc:
        raise RuntimeError("Could not import Laya. Install the project with `uv sync`.") from exc
    return Router()


def run(args: argparse.Namespace, router: Any | None = None) -> int:
    validate_probability(args.threshold, "--threshold")
    paths = args.paths or ["-"]
    show_filename = args.with_filename or (len(paths) > 1 and not args.no_filename)

    try:
        router = router if router is not None else _load_router()
    except (ImportError, RuntimeError) as exc:
        print(f"layagrep: {exc}", file=sys.stderr)
        return 2

    any_match = False
    had_error = False
    for path in paths:
        if path == "-":
            stream = sys.stdin
            close_stream = False
        else:
            try:
                stream = Path(path).open("r", encoding="utf-8", errors="replace")
                close_stream = True
            except OSError as exc:
                print(f"layagrep: {path}: {exc}", file=sys.stderr)
                had_error = True
                continue

        try:
            count, _ = search_stream(
                stream,
                path=path,
                description=args.description,
                router=router,
                args=args,
                show_filename=show_filename,
            )
        except Exception as exc:
            print(f"layagrep: {path}: {exc}", file=sys.stderr)
            had_error = True
            count = 0
        finally:
            if close_stream:
                stream.close()

        any_match = any_match or count > 0
        if args.quiet and any_match:
            return 0

    if had_error:
        return 2
    return 0 if any_match else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
