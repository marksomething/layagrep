"""Command-line interface for :mod:`layagrep.engine`.

Only presentation lives here: argument parsing, reading inputs, formatting
matches, and exit statuses. All matching logic is in :mod:`layagrep.engine`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from layagrep.engine import (
    DEFAULT_THRESHOLD,
    MODE_NOUL,
    MODES,
    SORT_LINE,
    SORT_SCORE,
    SORTS,
    Match,
    Matcher,
    create_router,
    iter_lines,
    rank_matches,
)

EXIT_MATCH = 0
EXIT_NO_MATCH = 1
EXIT_ERROR = 2
EXIT_BROKEN_PIPE = 128 + int(signal.SIGPIPE)  # 141, the status grep gets from SIGPIPE


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
        default=DEFAULT_THRESHOLD,
        metavar="PROBABILITY",
        help=f"minimum noul yes-probability to match (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=MODES,
        default=MODE_NOUL,
        help=(
            "question type used for matching: 'noul' (calibrated yes/no probability) or "
            "'choice' (two-option choice, immune to noul's false/true label bias)"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        metavar="N",
        help="score N lines per batched call (faster on large inputs; less lazy)",
    )
    parser.add_argument(
        "-n", "--line-number", action="store_true", help="prefix output with line numbers"
    )
    filenames = parser.add_mutually_exclusive_group()
    filenames.add_argument(
        "-H", "--with-filename", action="store_true", help="always print filenames"
    )
    filenames.add_argument(
        "-h", "--no-filename", action="store_true", help="never print filenames"
    )
    parser.add_argument(
        "-v", "--invert-match", action="store_true", help="select lines below the threshold"
    )
    parser.add_argument(
        "-c", "--count", action="store_true", help="print the number of selected lines per input"
    )
    parser.add_argument(
        "-l", "--files-with-matches", action="store_true", help="print each input with a match"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="stop after the first match; print nothing"
    )
    parser.add_argument(
        "--sort",
        choices=SORTS,
        default=SORT_LINE,
        help="output order: 'line' (default) or 'score' (best matches first)",
    )
    parser.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="print only the N best matches (implies --sort score; collects matches first)",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--scores", action="store_true", help="include each match's noul probability"
    )
    output.add_argument(
        "--json",
        action="store_true",
        help="write matching lines as JSON Lines with score, confidence, and routing metadata",
    )
    return parser


def check_options(args: argparse.Namespace) -> str | None:
    """Validate flag combinations; return an error message or ``None``."""
    if not 0.0 <= args.threshold <= 1.0:
        return "--threshold must be a number between 0 and 1"
    if args.batch_size is not None and args.batch_size < 1:
        return "--batch-size must be at least 1"
    if args.top is not None and args.top < 1:
        return "--top must be at least 1"
    if args.top is not None or args.sort == SORT_SCORE:
        for flag, enabled in (
            ("-c/--count", args.count),
            ("-l/--files-with-matches", args.files_with_matches),
            ("-q/--quiet", args.quiet),
        ):
            if enabled:
                return f"--sort score/--top cannot be combined with {flag}"
    return None


def format_match(
    match: Match,
    *,
    show_filename: bool,
    show_line_number: bool,
    show_scores: bool,
    as_json: bool,
) -> str:
    """Render one match for output."""
    if as_json:
        return json.dumps(
            {
                "file": match.source,
                "line": match.line_number,
                "text": match.text,
                "score": match.score,
                "confidence": match.confidence,
                "routing": match.routing,
            },
            ensure_ascii=False,
        )

    prefix = ""
    if show_filename:
        prefix += f"{match.source}:"
    if show_line_number:
        prefix += f"{match.line_number}:"
    if show_scores:
        prefix = f"{match.score:.3f}\t{prefix}"
    return f"{prefix}{match.text}"


def search_stream(
    stream: Any,
    *,
    path: str,
    matcher: Matcher,
    args: argparse.Namespace,
    show_filename: bool,
) -> int:
    """Search one stream, print selected lines as found, return how many matched."""
    matches = matcher.search(
        iter_lines(stream),
        source=path,
        threshold=args.threshold,
        invert=args.invert_match,
    )

    count = 0
    for match in matches:
        count += 1
        if args.quiet:
            return count
        if args.files_with_matches:
            print(path)
            return count
        if not args.count:
            print(
                format_match(
                    match,
                    show_filename=show_filename,
                    show_line_number=args.line_number,
                    show_scores=args.scores,
                    as_json=args.json,
                )
            )

    if args.count:
        prefix = f"{path}:" if show_filename else ""
        print(f"{prefix}{count}")
    return count


def run(args: argparse.Namespace, predictor: Any | None = None) -> int:
    problem = check_options(args)
    if problem is not None:
        print(f"layagrep: {problem}", file=sys.stderr)
        return EXIT_ERROR

    paths = args.paths or ["-"]
    show_filename = args.with_filename or (len(paths) > 1 and not args.no_filename)
    ranked = args.top is not None or args.sort == SORT_SCORE

    try:
        matcher = Matcher(
            args.description,
            predictor if predictor is not None else create_router(),
            mode=args.mode,
            batch_size=args.batch_size,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"layagrep: {exc}", file=sys.stderr)
        return EXIT_ERROR

    any_match = False
    had_error = False
    ranked_matches: list[Match] = []
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
            if ranked:
                count = 0
                for match in matcher.search(
                    iter_lines(stream),
                    source=path,
                    threshold=args.threshold,
                    invert=args.invert_match,
                ):
                    ranked_matches.append(match)
                    count += 1
            else:
                count = search_stream(
                    stream,
                    path=path,
                    matcher=matcher,
                    args=args,
                    show_filename=show_filename,
                )
        except BrokenPipeError:
            raise  # `layagrep ... | head`: not an error, see main()
        except Exception as exc:
            print(f"layagrep: {path}: {exc}", file=sys.stderr)
            had_error = True
            count = 0
        finally:
            if close_stream:
                stream.close()

        any_match = any_match or count > 0
        if args.quiet and any_match:
            return EXIT_MATCH

    if ranked:
        order = SORT_SCORE if (args.top is not None or args.sort == SORT_SCORE) else SORT_LINE
        for match in rank_matches(ranked_matches, top=args.top, sort=order):
            print(
                format_match(
                    match,
                    show_filename=show_filename,
                    show_line_number=args.line_number,
                    show_scores=args.scores,
                    as_json=args.json,
                )
            )

    if had_error:
        return EXIT_ERROR
    return EXIT_MATCH if any_match else EXIT_NO_MATCH


def handle_broken_pipe() -> int:
    """Silence the broken stdout and exit as grep does under SIGPIPE (141).

    Redirecting stdout to devnull keeps the interpreter's shutdown flush from
    printing "Exception ignored ... BrokenPipeError" after the pipe reader left.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        finally:
            os.close(devnull)
    except (OSError, ValueError):
        pass
    return EXIT_BROKEN_PIPE


def main() -> int:
    # Die on SIGPIPE like grep does, so `layagrep ... | head` is silent. Python
    # otherwise ignores SIGPIPE and turns it into BrokenPipeError at some write
    # (or only at shutdown flush, which prints "Exception ignored" noise).
    if hasattr(signal, "SIGPIPE"):
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        except ValueError:  # not in the main thread
            pass
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except BrokenPipeError:
        return handle_broken_pipe()


if __name__ == "__main__":
    raise SystemExit(main())
