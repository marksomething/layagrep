"""Semantic line matching backed by Laya decision models.

This module is the search engine: it scores text against a natural-language
description and yields matches. It has no command-line or output concerns, so
the same engine can back a CLI, a server, or a library caller.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, TextIO

QUESTION_KEY = "match"
DEFAULT_THRESHOLD = 0.5

# ``noul`` is the cheapest question type, but its rendered ``false:``/``true:``
# option labels can dominate the answer on the English checkpoint (Laya model
# card, issue #156). ``choice`` mode asks the same question as a two-option
# choice with neutral keys and the yes/no wording in the option descriptions,
# which is the mitigation Laya's model card recommends.
MODE_NOUL = "noul"
MODE_CHOICE = "choice"
MODES = (MODE_NOUL, MODE_CHOICE)

TRUE_OPTION = "A"
FALSE_OPTION = "B"
CHOICE_CRITERIA = {
    TRUE_OPTION: "yes, the line matches the description",
    FALSE_OPTION: "no, the line does not match the description",
}

SORT_LINE = "line"
SORT_SCORE = "score"
SORTS = (SORT_LINE, SORT_SCORE)

# Laya currently emits this known warning for its unused 11+-option choice
# calibration bucket (e.g. ``choice:11+=0.100... -> 0.5``). ``layagrep`` never
# asks questions with 11+ options, so the warning is noise; suppress exactly this
# one without muting other RuntimeWarnings from Laya or the caller.
LAYA_CALIBRATION_WARNING = (
    r"^laya: this checkpoint ships invalid temperatures or values outside "
    r".*using choice:11\+="
)


class Predictor(Protocol):
    """Anything that answers Laya-style questions, e.g. ``laya.Router``.

    Objects that also implement ``predict_batch(requests)`` are used for batched
    scoring when ``Matcher(batch_size=...)`` is set; others fall back to one
    ``predict`` call per line.
    """

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Prediction:
    """One scored line: its match probability plus Laya's diagnostics."""

    score: float
    confidence: float | None = None
    routing: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Match:
    """One selected line."""

    source: str
    line_number: int
    text: str
    score: float
    confidence: float | None = None
    routing: Mapping[str, Any] | None = None


@contextmanager
def _silence_known_laya_warning() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=LAYA_CALIBRATION_WARNING,
            category=RuntimeWarning,
        )
        yield


def build_question(description: str, mode: str = MODE_NOUL) -> dict[str, Any]:
    """Build the question asking whether a line matches a description.

    ``noul`` mode asks directly for a calibrated yes-probability. ``choice``
    mode uses neutral option keys so the rendered ``false:``/``true:`` labels
    cannot bias the answer (Laya issue #156).
    """
    if mode == MODE_NOUL:
        return {QUESTION_KEY: {"type": "noul", "instructions": description}}
    if mode == MODE_CHOICE:
        return {
            QUESTION_KEY: {
                "type": "choice",
                "instructions": description,
                "criteria": dict(CHOICE_CRITERIA),
            }
        }
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def validate_probability(value: float, name: str = "probability") -> None:
    """Raise ``ValueError`` unless ``value`` is a finite number in [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1")


def _answer(result: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        answer = result["answers"][QUESTION_KEY]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Laya response did not contain answers.{QUESTION_KEY}") from exc
    if not isinstance(answer, Mapping):
        raise ValueError(f"Laya response answers.{QUESTION_KEY} was not an object")
    return answer


def match_probability(result: Mapping[str, Any], mode: str = MODE_NOUL) -> float:
    """Extract and validate the match probability from a Laya response.

    For ``noul`` questions this is the calibrated P(true); for ``choice``
    questions it is P(A) where option A means "the line matches".
    """
    answer = _answer(result)
    if mode == MODE_NOUL:
        raw: Any = answer.get("noul")
    elif mode == MODE_CHOICE:
        probabilities = answer.get("probabilities")
        raw = probabilities.get(TRUE_OPTION) if isinstance(probabilities, Mapping) else None
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Laya response did not contain a usable {mode} probability") from exc
    validate_probability(value, "Laya probability")
    return value


def answer_confidence(result: Mapping[str, Any]) -> float | None:
    """Laya's calibrated confidence (``answer_confidence``), when reported.

    This is the probability mass on the reported answer -- the quantity Laya's
    model card recommends gating on (the card names it ``confidence``; the field
    is called ``answer_confidence`` in laya >= 0.3 because plain ``confidence``
    for choice/score answers is normalized entropy instead).
    """
    answer = _answer(result)
    for key in ("answer_confidence", "confidence"):
        if key in answer:
            value = float(answer[key])
            validate_probability(value, f"Laya {key}")
            return value
    return None


def routing_decision(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The Router's routing metadata (checkpoint chosen and why), when present."""
    routing = result.get("routing")
    return routing if isinstance(routing, Mapping) else None


def is_selected(probability: float, threshold: float, invert: bool = False) -> bool:
    """Whether a score passes the threshold, optionally inverted like grep -v."""
    selected = probability >= threshold
    return not selected if invert else selected


def iter_lines(stream: TextIO) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, text)`` with line terminators removed.

    All other whitespace is preserved, since the caller may want to echo the
    line exactly as it appeared in the source.
    """
    for number, raw_line in enumerate(stream, start=1):
        yield number, raw_line.rstrip("\r\n")


def rank_matches(
    matches: Iterable[Match],
    *,
    top: int | None = None,
    sort: str = SORT_LINE,
) -> list[Match]:
    """Order matches for ranked output.

    ``sort="score"`` orders by descending score (ties keep input order);
    ``sort="line"`` keeps input order. ``top`` keeps only the first ``top``
    after ordering.
    """
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}, got {sort!r}")
    if top is not None and top < 1:
        raise ValueError("top must be at least 1")
    ordered = list(matches)
    if sort == SORT_SCORE:
        # sort() is stable: equal scores keep their input order even reversed.
        ordered.sort(key=lambda m: m.score, reverse=True)
    return ordered if top is None else ordered[:top]


class Matcher:
    """Scores lines against a description with Laya and yields matches.

    Matching is lazy: lines are scored only as the caller consumes the
    iteration, so callers can stop early (e.g. "first match wins") without
    paying for the rest of the input.

    With ``batch_size`` set, lines are scored in chunks through the predictor's
    ``predict_batch`` (which packs states into shared forward passes), trading
    per-chunk latency of laziness for much higher throughput.
    """

    def __init__(
        self,
        description: str,
        predictor: Predictor | None = None,
        *,
        mode: str = MODE_NOUL,
        batch_size: int | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.description = description
        self.mode = mode
        self.batch_size = batch_size
        self._question = build_question(description, mode)
        self._predictor: Predictor = predictor if predictor is not None else create_router()

    def predict(self, text: str) -> Prediction:
        """Score one line and return its probability plus Laya's diagnostics."""
        with _silence_known_laya_warning():
            result = self._predictor.predict(text, self._question)
        return self._prediction(result)

    def score(self, text: str) -> float:
        """Return the probability that ``text`` matches the description."""
        return self.predict(text).score

    def search(
        self,
        lines: Iterable[tuple[int, str]],
        *,
        source: str = "<stdin>",
        threshold: float = DEFAULT_THRESHOLD,
        invert: bool = False,
    ) -> Iterator[Match]:
        """Yield a :class:`Match` for every line selected by the threshold."""
        validate_probability(threshold, "threshold")
        limit = self.batch_size or 1
        chunk: list[tuple[int, str]] = []
        for entry in lines:
            chunk.append(entry)
            if len(chunk) >= limit:
                yield from self._select(chunk, source, threshold, invert)
                chunk.clear()
        if chunk:
            yield from self._select(chunk, source, threshold, invert)

    def _select(
        self,
        chunk: list[tuple[int, str]],
        source: str,
        threshold: float,
        invert: bool,
    ) -> Iterator[Match]:
        predictions = self._predict_many([text for _, text in chunk])
        for (line_number, text), prediction in zip(chunk, predictions, strict=True):
            if is_selected(prediction.score, threshold, invert):
                yield Match(
                    source=source,
                    line_number=line_number,
                    text=text,
                    score=prediction.score,
                    confidence=prediction.confidence,
                    routing=prediction.routing,
                )

    def _predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        predict_batch = getattr(self._predictor, "predict_batch", None)
        if self.batch_size is None or predict_batch is None:
            return [self.predict(text) for text in texts]
        requests = [{"state": text, "questions": self._question} for text in texts]
        with _silence_known_laya_warning():
            results = predict_batch(requests)
        if len(results) != len(texts):
            raise ValueError(
                f"predict_batch returned {len(results)} results for {len(texts)} lines"
            )
        return [self._prediction(result) for result in results]

    def _prediction(self, result: Mapping[str, Any]) -> Prediction:
        return Prediction(
            score=match_probability(result, self.mode),
            confidence=answer_confidence(result),
            routing=routing_decision(result),
        )


def create_router() -> Any:
    """Create the default Laya router (downloads the checkpoint on first use)."""
    try:
        from laya import Router
    except ImportError as exc:
        raise RuntimeError("Could not import Laya. Install the project with `uv sync`.") from exc
    return Router()
