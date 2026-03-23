from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .base import RewardContext, RewardTermResult


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "\\text{s}",
    "\\text{.}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    cursor = idx
    right_brace_idx = None
    open_count = 0
    while cursor < len(string):
        if string[cursor] == "{":
            open_count += 1
        if string[cursor] == "}":
            open_count -= 1
            if open_count == 0:
                right_brace_idx = cursor
                break
        cursor += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(value: str) -> str:
    left = "\\boxed{"
    if not value.startswith(left) or not value.endswith("}"):
        raise ValueError(f"Invalid boxed value: {value}")
    return value[len(left) : -1]


def normalize_final_answer(final_answer: str) -> str:
    normalized = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        normalized = normalized.replace(before, after)
    for expression in REMOVED_EXPRESSIONS:
        normalized = normalized.replace(expression, "")
    normalized = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", normalized)
    normalized = re.sub(r"(\\text\{)(.*?)(\})", "\\2", normalized)
    normalized = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", normalized)
    normalized = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", normalized)
    normalized = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", normalized)
    normalized = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", normalized)
    normalized = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", normalized)
    normalized = normalized.replace("$", "")
    if normalized.replace(",", "").isdigit():
        normalized = normalized.replace(",", "")
    return normalized.strip()


def extract_prediction(solution: str, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)") -> str:
    matches = re.findall(answer_pattern, solution)
    extracted = matches[-1] if matches else "[INVALID]"
    return normalize_final_answer(extracted)


def compute_math_accuracy(solution: str, ground_truth: str) -> tuple[bool, str]:
    prediction = extract_prediction(solution[-300:])
    normalized_truth = normalize_final_answer(ground_truth)
    return prediction == normalized_truth, prediction


@dataclass(slots=True)
class MathAccuracyReward:
    name: str = "accuracy"

    def compute(self, context: RewardContext) -> RewardTermResult:
        ground_truth = context.trajectory.ground_truth or ""
        correct, prediction = compute_math_accuracy(context.trajectory.response, ground_truth)
        score = 1.0 if correct else -1.0
        return RewardTermResult(score=score, metrics={"acc": float(correct), "pred": prediction})


@dataclass(slots=True)
class BoxedFormatReward:
    name: str = "boxed_format"

    def compute(self, context: RewardContext) -> RewardTermResult:
        boxed = last_boxed_only_string(context.trajectory.response)
        has_box = boxed is not None
        return RewardTermResult(score=1.0 if has_box else 0.0, metrics={"has_boxed_answer": has_box})
