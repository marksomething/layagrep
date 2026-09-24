from __future__ import annotations

import io
import unittest
import warnings
from contextlib import redirect_stdout
from unittest.mock import patch

from layagrep.cli import build_parser, matches, noul_probability, run, search_stream


class FakeRouter:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.calls: list[tuple[str, dict[str, object]]] = []

    def predict(self, state: str, questions: dict[str, object]) -> dict[str, object]:
        self.calls.append((state, questions))
        return {"answers": {"match": {"noul": self.probabilities[state]}}}


class WarningRouter(FakeRouter):
    def __init__(self, message: str) -> None:
        super().__init__({"some line": 0.9})
        self.message = message

    def predict(self, state: str, questions: dict[str, object]) -> dict[str, object]:
        warnings.warn(self.message, RuntimeWarning)
        return super().predict(state, questions)


class LayagrepTests(unittest.TestCase):
    def test_matches_threshold_and_inversion(self) -> None:
        self.assertTrue(matches(0.7, 0.7))
        self.assertFalse(matches(0.69, 0.7))
        self.assertTrue(matches(0.69, 0.7, invert=True))
        self.assertFalse(matches(0.7, 0.7, invert=True))

    def test_noul_probability_is_validated(self) -> None:
        self.assertEqual(noul_probability({"answers": {"match": {"noul": 0.42}}}), 0.42)
        with self.assertRaises(ValueError):
            noul_probability({"answers": {"match": {"noul": 2}}})
        with self.assertRaises(ValueError):
            noul_probability({"answers": {}})

    def test_search_stream_uses_noul_description_and_scores(self) -> None:
        args = build_parser().parse_args(
            ["billing issue", "-", "--scores", "-n", "--threshold", "0.6"]
        )
        router = FakeRouter({"charged twice": 0.91, "thanks": 0.1})
        output = io.StringIO()
        with redirect_stdout(output):
            count, first_score = search_stream(
                io.StringIO("charged twice\nthanks\n"),
                path="-",
                description=args.description,
                router=router,
                args=args,
                show_filename=False,
            )

        self.assertEqual(count, 1)
        self.assertEqual(first_score, 0.91)
        self.assertEqual(output.getvalue(), "0.910\t1:charged twice\n")
        self.assertEqual(len(router.calls), 2)
        question = router.calls[0][1]["match"]
        self.assertEqual(question["type"], "noul")
        self.assertIn("billing issue", question["instructions"])

    def test_search_suppresses_only_known_laya_calibration_warning(self) -> None:
        args = build_parser().parse_args(["description", "-"])
        expected_warning = (
            "laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; "
            "using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the "
            "affected entries as uncalibrated."
        )
        router = WarningRouter(expected_warning)
        output = io.StringIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with redirect_stdout(output):
                count, _ = search_stream(
                    io.StringIO("some line\n"),
                    path="-",
                    description=args.description,
                    router=router,
                    args=args,
                    show_filename=False,
                )

        self.assertEqual(count, 1)
        self.assertEqual(output.getvalue(), "some line\n")
        self.assertEqual(caught, [])

        unrelated = WarningRouter("an unrelated runtime warning")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with redirect_stdout(io.StringIO()):
                search_stream(
                    io.StringIO("some line\n"),
                    path="-",
                    description=args.description,
                    router=unrelated,
                    args=args,
                    show_filename=False,
                )
        self.assertEqual(len(caught), 1)
        self.assertEqual(str(caught[0].message), "an unrelated runtime warning")

    def test_run_reports_grep_exit_statuses(self) -> None:
        args = build_parser().parse_args(["an outage"])
        router = FakeRouter({"service outage": 0.8})
        output = io.StringIO()
        with patch("layagrep.cli.sys.stdin", io.StringIO("service outage\n")), redirect_stdout(output):
            self.assertEqual(run(args, router=router), 0)
        self.assertEqual(output.getvalue(), "service outage\n")

        router = FakeRouter({"ordinary line": 0.2})
        with patch("layagrep.cli.sys.stdin", io.StringIO("ordinary line\n")), redirect_stdout(io.StringIO()):
            self.assertEqual(run(args, router=router), 1)


if __name__ == "__main__":
    unittest.main()
