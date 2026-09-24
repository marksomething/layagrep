from __future__ import annotations

import io
import unittest
import warnings
from typing import Any

from layagrep.engine import (
    FALSE_OPTION,
    TRUE_OPTION,
    Match,
    Matcher,
    answer_confidence,
    build_question,
    is_selected,
    iter_lines,
    match_probability,
    rank_matches,
    routing_decision,
    validate_probability,
)


class FakeRouter:
    """Canned Laya-shaped results keyed by state text; answers in either mode."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _result(self, state: str) -> dict[str, Any]:
        p = self.probabilities[state]
        return {
            "answers": {"match": self._answer(p)},
            "routing": {"model": "english", "reason": "fake"},
        }

    @staticmethod
    def _answer(p: float) -> dict[str, Any]:
        return {
            "type": "noul",
            "noul": p,
            "answer_confidence": max(p, 1.0 - p),
        }

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        return self._result(state)


class FakeChoiceRouter(FakeRouter):
    @staticmethod
    def _answer(p: float) -> dict[str, Any]:
        return {
            "type": "choice",
            "choice": TRUE_OPTION if p >= 0.5 else FALSE_OPTION,
            "probabilities": {TRUE_OPTION: p, FALSE_OPTION: 1.0 - p},
            "answer_confidence": max(p, 1.0 - p),
        }


class FakeBatchRouter(FakeRouter):
    def __init__(self, probabilities: dict[str, float]) -> None:
        super().__init__(probabilities)
        self.batch_calls: list[list[str]] = []

    def predict_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        states = [str(request["state"]) for request in requests]
        self.batch_calls.append(states)
        return [self._result(state) for state in states]


class WarningRouter(FakeRouter):
    def __init__(self, message: str) -> None:
        super().__init__({"some line": 0.9})
        self.message = message

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        warnings.warn(self.message, RuntimeWarning, stacklevel=2)
        return super().predict(state, questions)


def make_match(score: float, line_number: int = 1) -> Match:
    return Match(
        source="app.log",
        line_number=line_number,
        text=f"line {line_number}",
        score=score,
        confidence=max(score, 1.0 - score),
        routing={"model": "english"},
    )


class QuestionTests(unittest.TestCase):
    def test_noul_question(self) -> None:
        question = build_question("mentions a billing issue")
        self.assertEqual(question["match"]["type"], "noul")
        self.assertEqual(question["match"]["instructions"], "mentions a billing issue")

    def test_choice_question_uses_neutral_keys(self) -> None:
        question = build_question("mentions a billing issue", mode="choice")
        self.assertEqual(question["match"]["type"], "choice")
        criteria = question["match"]["criteria"]
        self.assertEqual(set(criteria), {TRUE_OPTION, FALSE_OPTION})
        self.assertIn("yes", criteria[TRUE_OPTION])
        self.assertIn("no", criteria[FALSE_OPTION])

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_question("x", mode="sentiment")


class ExtractionTests(unittest.TestCase):
    def test_match_probability_noul_and_choice(self) -> None:
        noul_result = {"answers": {"match": FakeRouter._answer(0.42)}}
        self.assertEqual(match_probability(noul_result, "noul"), 0.42)

        choice_result = {"answers": {"match": FakeChoiceRouter._answer(0.42)}}
        self.assertEqual(match_probability(choice_result, "choice"), 0.42)

        with self.assertRaises(ValueError):
            match_probability({"answers": {"match": {"noul": 2}}})
        with self.assertRaises(ValueError):
            match_probability({"answers": {}}, "noul")
        with self.assertRaises(ValueError):
            match_probability({"answers": {"match": {"type": "noul"}}}, "noul")

    def test_answer_confidence_and_routing(self) -> None:
        result = FakeRouter({"x": 0.42})._result("x")
        self.assertAlmostEqual(answer_confidence(result) or 0.0, 0.58)
        self.assertEqual(routing_decision(result), {"model": "english", "reason": "fake"})
        self.assertIsNone(answer_confidence({"answers": {"match": {"noul": 0.5}}}))
        self.assertIsNone(routing_decision({"answers": {"match": {"noul": 0.5}}}))

    def test_validate_probability(self) -> None:
        for value in (0.0, 0.5, 1.0):
            validate_probability(value)
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_probability(value, "--threshold")


class SearchTests(unittest.TestCase):
    def test_is_selected_threshold_and_inversion(self) -> None:
        self.assertTrue(is_selected(0.7, 0.7))
        self.assertFalse(is_selected(0.69, 0.7))
        self.assertTrue(is_selected(0.69, 0.7, invert=True))
        self.assertFalse(is_selected(0.7, 0.7, invert=True))

    def test_iter_lines_numbers_and_trims_only_terminators(self) -> None:
        self.assertEqual(
            list(iter_lines(io.StringIO("alpha\r\n  beta  \n"))),
            [(1, "alpha"), (2, "  beta  ")],
        )

    def test_search_yields_matches_with_diagnostics(self) -> None:
        router = FakeRouter({"charged twice": 0.91, "thanks": 0.1})
        matcher = Matcher("billing issue", router)

        matches = list(
            matcher.search([(1, "charged twice"), (2, "thanks")], source="app.log", threshold=0.6)
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source, "app.log")
        self.assertEqual(matches[0].line_number, 1)
        self.assertEqual(matches[0].text, "charged twice")
        self.assertEqual(matches[0].score, 0.91)
        self.assertEqual(matches[0].confidence, 0.91)
        self.assertEqual(matches[0].routing, {"model": "english", "reason": "fake"})
        self.assertIn("billing issue", router.calls[0][1]["match"]["instructions"])

    def test_search_choice_mode_uses_option_probability(self) -> None:
        router = FakeChoiceRouter({"charged twice": 0.8, "thanks": 0.1})
        matcher = Matcher("billing issue", router, mode="choice")

        matches = list(matcher.search([(1, "charged twice"), (2, "thanks")], threshold=0.5))

        self.assertEqual([(m.text, m.score) for m in matches], [("charged twice", 0.8)])
        question = router.calls[0][1]["match"]
        self.assertEqual(question["type"], "choice")

    def test_search_is_lazy_for_early_exit(self) -> None:
        router = FakeRouter({"first": 0.9, "second": 0.8})
        matcher = Matcher("description", router)

        first = next(matcher.search([(1, "first"), (2, "second")]))

        self.assertEqual(first.text, "first")
        self.assertEqual(len(router.calls), 1)

    def test_batched_search_calls_predict_batch_per_chunk(self) -> None:
        router = FakeBatchRouter({f"line {i}": 0.9 for i in range(5)})
        matcher = Matcher("description", router, batch_size=2)

        matches = list(matcher.search((i, f"line {i}") for i in range(5)))

        self.assertEqual([m.line_number for m in matches], [0, 1, 2, 3, 4])
        self.assertEqual(
            router.batch_calls,
            [["line 0", "line 1"], ["line 2", "line 3"], ["line 4"]],
        )
        self.assertEqual(router.calls, [])

    def test_batched_search_falls_back_without_predict_batch(self) -> None:
        router = FakeRouter({f"line {i}": 0.9 for i in range(3)})
        matcher = Matcher("description", router, batch_size=2)

        matches = list(matcher.search((i, f"line {i}") for i in range(3)))

        self.assertEqual([m.line_number for m in matches], [0, 1, 2])
        self.assertEqual(len(router.calls), 3)

    def test_score_suppresses_only_known_laya_calibration_warning(self) -> None:
        known = (
            "laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; "
            "using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the "
            "affected entries as uncalibrated."
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Matcher("description", WarningRouter(known)).score("some line")
        self.assertEqual(caught, [])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Matcher("description", WarningRouter("an unrelated runtime warning")).score("some line")
        self.assertEqual(len(caught), 1)
        self.assertEqual(str(caught[0].message), "an unrelated runtime warning")


class RankTests(unittest.TestCase):
    def test_rank_matches_orders_by_score_stably(self) -> None:
        matches = [make_match(0.2, 1), make_match(0.9, 2), make_match(0.9, 3), make_match(0.5, 4)]

        ranked = rank_matches(matches, sort="score")

        self.assertEqual(
            [(m.score, m.line_number) for m in ranked],
            [(0.9, 2), (0.9, 3), (0.5, 4), (0.2, 1)],
        )

    def test_rank_matches_top_keeps_input_order_by_default(self) -> None:
        matches = [make_match(0.2, 1), make_match(0.9, 2), make_match(0.5, 3)]

        self.assertEqual([m.line_number for m in rank_matches(matches, top=2)], [1, 2])
        self.assertEqual(
            [m.line_number for m in rank_matches(matches, top=2, sort="score")],
            [2, 3],
        )

    def test_rank_matches_validates_arguments(self) -> None:
        with self.assertRaises(ValueError):
            rank_matches([], sort="popularity")
        with self.assertRaises(ValueError):
            rank_matches([], top=0)


if __name__ == "__main__":
    unittest.main()
