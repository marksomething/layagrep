from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from test_engine import FakeBatchRouter, FakeChoiceRouter, FakeRouter

from layagrep.cli import (
    EXIT_BROKEN_PIPE,
    build_parser,
    format_match,
    handle_broken_pipe,
    main,
    run,
    search_stream,
)
from layagrep.engine import Match, Matcher


def make_match() -> Match:
    return Match(
        source="app.log",
        line_number=7,
        text="charged twice",
        score=0.91,
        confidence=0.91,
        routing={"model": "english"},
    )


class BrokenPipeOut(io.StringIO):
    def write(self, text: str) -> int:
        raise BrokenPipeError


class FormatTests(unittest.TestCase):
    @staticmethod
    def render(
        match: Match,
        *,
        filename: bool = False,
        numbers: bool = False,
        scores: bool = False,
        as_json: bool = False,
    ) -> str:
        return format_match(
            match,
            show_filename=filename,
            show_line_number=numbers,
            show_scores=scores,
            as_json=as_json,
        )

    def test_plain_text(self) -> None:
        match = make_match()
        self.assertEqual(self.render(match), "charged twice")
        self.assertEqual(
            self.render(match, filename=True, numbers=True),
            "app.log:7:charged twice",
        )
        self.assertEqual(
            self.render(match, numbers=True, scores=True),
            "0.910\t7:charged twice",
        )

    def test_json_includes_confidence_and_routing(self) -> None:
        rendered = json.loads(self.render(make_match(), as_json=True))
        self.assertEqual(
            rendered,
            {
                "file": "app.log",
                "line": 7,
                "text": "charged twice",
                "score": 0.91,
                "confidence": 0.91,
                "routing": {"model": "english"},
            },
        )


class CliTests(unittest.TestCase):
    def run_search(self, argv: list[str], lines: str, router: FakeRouter) -> tuple[int, str]:
        args = build_parser().parse_args(argv)
        output = io.StringIO()
        with patch("layagrep.cli.sys.stdin", io.StringIO(lines)), redirect_stdout(output):
            status = run(args, predictor=router)
        return status, output.getvalue()

    def test_exit_statuses_and_output(self) -> None:
        status, output = self.run_search(
            ["an outage", "-", "--scores", "-n"],
            "service outage\nordinary line\n",
            FakeRouter({"service outage": 0.8, "ordinary line": 0.2}),
        )
        self.assertEqual(status, 0)
        self.assertEqual(output, "0.800\t1:service outage\n")

        status, output = self.run_search(
            ["an outage"],
            "ordinary line\n",
            FakeRouter({"ordinary line": 0.2}),
        )
        self.assertEqual(status, 1)
        self.assertEqual(output, "")

    def test_invert_count_and_quiet(self) -> None:
        status, output = self.run_search(
            ["anything", "-", "-v"],
            "good\nbad\n",
            FakeRouter({"good": 0.8, "bad": 0.1}),
        )
        self.assertEqual(status, 0)
        self.assertEqual(output, "bad\n")

        status, output = self.run_search(
            ["anything", "-", "-c"],
            "good\nbad\n",
            FakeRouter({"good": 0.8, "bad": 0.1}),
        )
        self.assertEqual((status, output), (0, "1\n"))

        status, output = self.run_search(
            ["anything", "-", "-q"],
            "bad\ngood\n",
            FakeRouter({"good": 0.8, "bad": 0.1}),
        )
        self.assertEqual((status, output), (0, ""))

    def test_files_with_matches_prints_path_once(self) -> None:
        status, output = self.run_search(
            ["anything", "-", "-l"],
            "good\nalso good\n",
            FakeRouter({"good": 0.8, "also good": 0.9}),
        )
        self.assertEqual((status, output), (0, "-\n"))

    def test_choice_mode_end_to_end(self) -> None:
        status, output = self.run_search(
            ["anything", "-", "--mode", "choice", "--scores"],
            "good\nbad\n",
            FakeChoiceRouter({"good": 0.8, "bad": 0.1}),
        )
        self.assertEqual(status, 0)
        self.assertEqual(output, "0.800\tgood\n")

    def test_batch_mode_matches_streaming_output(self) -> None:
        router = FakeBatchRouter({"a": 0.9, "b": 0.1, "c": 0.7, "d": 0.2})
        status, output = self.run_search(
            ["anything", "-", "--batch-size", "2", "--scores", "-n"],
            "a\nb\nc\nd\n",
            router,
        )
        self.assertEqual(status, 0)
        self.assertEqual(output, "0.900\t1:a\n0.700\t3:c\n")
        self.assertEqual(router.batch_calls, [["a", "b"], ["c", "d"]])
        self.assertEqual(router.calls, [])

    def test_top_prints_best_matches_first(self) -> None:
        status, output = self.run_search(
            ["anything", "-", "--top", "2", "--scores"],
            "a\nb\nc\n",
            FakeRouter({"a": 0.6, "b": 0.9, "c": 0.7}),
        )
        self.assertEqual(status, 0)
        self.assertEqual(output, "0.900\tb\n0.700\tc\n")

    def test_sort_score_orders_all_matches(self) -> None:
        status, output = self.run_search(
            ["anything", "-", "--sort", "score", "--scores"],
            "a\nb\n",
            FakeRouter({"a": 0.6, "b": 0.9}),
        )
        self.assertEqual(status, 0)
        self.assertEqual(output, "0.900\tb\n0.600\ta\n")

    def test_invalid_options_are_errors(self) -> None:
        cases = [
            ["anything", "-", "-t", "1.5"],
            ["anything", "-", "--top", "0"],
            ["anything", "-", "--batch-size", "0"],
            ["anything", "-", "--top", "2", "-c"],
            ["anything", "-", "--sort", "score", "-q"],
            ["anything", "-", "--top", "2", "-l"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), patch("layagrep.cli.sys.stderr", io.StringIO()) as stderr:
                status = run(build_parser().parse_args(argv), predictor=FakeRouter({}))
            self.assertEqual(status, 2, argv)
            self.assertIn("layagrep:", stderr.getvalue())

    def test_missing_file_is_an_error_status(self) -> None:
        with patch("layagrep.cli.sys.stderr", io.StringIO()):
            status = run(
                build_parser().parse_args(["anything", "no-such-file"]),
                predictor=FakeRouter({}),
            )
        self.assertEqual(status, 2)

    def test_search_stream_prints_filename_for_multiple_inputs(self) -> None:
        args = build_parser().parse_args(["anything", "--json"])
        router = FakeRouter({"good": 0.8})
        output = io.StringIO()
        with redirect_stdout(output):
            count = search_stream(
                io.StringIO("good\n"),
                path="a.log",
                matcher=Matcher(args.description, router),
                args=args,
                show_filename=True,
            )
        self.assertEqual(count, 1)
        self.assertEqual(json.loads(output.getvalue())["file"], "a.log")

    def test_run_propagates_broken_pipe(self) -> None:
        args = build_parser().parse_args(["anything", "-", "--scores"])
        with (
            patch("layagrep.cli.sys.stdin", io.StringIO("good\n")),
            patch("sys.stdout", BrokenPipeOut()),
        ):
            with self.assertRaises(BrokenPipeError):
                run(args, predictor=FakeRouter({"good": 0.8}))

    def test_main_exits_like_grep_on_broken_pipe(self) -> None:
        with patch("layagrep.cli.run", side_effect=BrokenPipeError):
            with patch("sys.argv", ["layagrep", "anything", "file"]):
                self.assertEqual(main(), EXIT_BROKEN_PIPE)

    def test_handle_broken_pipe_is_safe_without_real_stdout(self) -> None:
        with patch("sys.stdout", io.StringIO()):
            self.assertEqual(handle_broken_pipe(), EXIT_BROKEN_PIPE)


class QuietFlagTests(unittest.TestCase):
    def test_quiet_stops_after_first_match(self) -> None:
        router = FakeRouter({"bad": 0.1, "good": 0.8, "also good": 0.9})
        args = build_parser().parse_args(["anything", "-", "-q"])
        stdin = io.StringIO("bad\ngood\nalso good\n")
        with patch("layagrep.cli.sys.stdin", stdin), redirect_stderr(io.StringIO()):
            status = run(args, predictor=router)
        self.assertEqual(status, 0)
        self.assertEqual(len(router.calls), 2)  # no scoring beyond the first match


if __name__ == "__main__":
    unittest.main()
