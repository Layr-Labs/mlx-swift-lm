#!/usr/bin/env python3
"""Compare the native Swift port with the independent Python MLX reference.

Run in an environment containing laya-mlx, with an already downloaded checkpoint.
No model download, remote inference, or production registration is performed.
"""
import argparse
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from laya_mlx import load
from laya_mlx.agent import collate_items


def fixtures():
    mixed = {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "payments and refunds", "technical": "bugs and outages", "sales": "new purchases"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "soon", "critical"]},
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    }
    cases = [
        ("mixed", "I was billed twice. Please refund the duplicate today.", mixed),
        ("single-and-padding", {"ticket": "The site is down.", "history": ["failed login", "unavailable"]}, {
            "single": {"type": "choice", "instructions": "Choose the available action.", "criteria": {"escalate": None}},
            "outage": {"type": "noul", "instructions": "Is the product unavailable?", "criteria": {"false": "working", "true": "unavailable"}},
        }),
        ("structured-unicode", [{"role": "user", "content": "Bonjour, je veux un remboursement. [MASK] 🌸"}], {
            "decision": {"type": "choice", "instructions": {"question": "Which department?", "hint": ["réclamation", "technical"]},
                         "criteria": {"billing": {"description": "Refunds"}, "support": ["Product help"]}},
        }),
        ("context-limit", "unimportant background " * 1000 + "refund needed", {
            "truncated": {"type": "noul", "instructions": "Does this mention a refund?"},
        }),
        ("cardinality", "Please route my refund request to billing.", {
            "department": {"type": "choice", "instructions": "Which department?", "criteria": {
                name: None for name in ["billing", "sales", "engineering", "support", "security", "legal", "people", "marketing", "operations", "finance", "other"]}},
        }),
        ("numeric-json", {"exponent": 100.0, "tiny": 1e-7, "negative_integer_zero": 0,
                          "negative_float_zero": -0.0, "decimal": 1.23, "large": 1e16,
                          "large_integer": 123456789012345678901234567890}, {
            "positive": {"type": "noul", "instructions": {"question": "Is exponent positive?", "threshold": 1e-6}},
        }),
        ("batch-boundary", "Please refund the duplicate payment.", {
            f"q{i}": value for i, value in enumerate([*mixed.values()] * 6)
        }),
    ]
    return [(name, {"model": "laya", "state": state, "questions": questions}) for name, state, questions in cases]


def numeric_error(expected, actual):
    """Compare response semantics while allowing fp16 rounding variation."""
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"Answer keys differ: {set(expected)} vs {set(actual)}")
        return max((numeric_error(value, actual[key]) for key, value in expected.items()), default=0)
    if isinstance(expected, (int, float)):
        if not np.isfinite(actual):
            raise AssertionError("Nonfinite response")
        return abs(expected - actual)
    if expected != actual:
        raise AssertionError(f"Response label/legend differs: {expected!r} vs {actual!r}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logit-atol", type=float, default=0.05)
    parser.add_argument("--action-rtol", type=float, default=0.002,
                        help="FP16 action logits reach thousands; compare relative rounding error too")
    parser.add_argument("--answer-atol", type=float, default=0.005)
    args = parser.parse_args()
    reference = load(str(args.checkpoint), dtype="float16")
    results = []
    with tempfile.TemporaryDirectory(prefix="laya-qualification-") as temporary:
        for name, request in fixtures():
            path = Path(temporary) / (name + ".json")
            encoded = json.dumps(request, ensure_ascii=False)
            if name == "numeric-json":
                # Exercise lexical forms that json.loads/dumps normalizes before tokenization.
                encoded = encoded.replace('"exponent": 100.0', '"exponent": 1e2')
                encoded = encoded.replace('"negative_integer_zero": 0', '"negative_integer_zero": -0')
                encoded = encoded.replace('"decimal": 1.23', '"decimal": 1.23000')
            path.write_text(encoded)
            command = [str(args.probe.resolve()), str(args.checkpoint.resolve()), str(path)]
            diagnostic = json.loads(subprocess.run(command + ["--diagnostics"], check=True, capture_output=True, text=True).stdout)
            items, _ = reference.prepare(request["state"], request["questions"])
            for key, expected in {
                "tokenIDs": [item["ids"] for item in items],
                "markerPositions": [item["markers"] for item in items],
                "questionTypes": [item["qtype"] for item in items],
            }.items():
                if diagnostic[key] != expected:
                    raise AssertionError(f"{name}: {key} differs from the reference")
            logit_error = 0.0
            action_error = 0.0
            action_close = True
            for start in range(0, len(items), reference.batch_size):
                chunk = items[start:start + reference.batch_size]
                logits, action = reference.forward(collate_items(chunk, reference.tok.pad_token_id))
                for row, item in enumerate(chunk):
                    k = len(item["markers"])
                    logit_error = max(logit_error, float(np.max(np.abs(np.asarray(logits)[row, :k]
                        - np.asarray(diagnostic["logits"][start + row])[:k]))))
                    action_error = max(action_error, float(np.max(np.abs(np.asarray(action)[row]
                        - np.asarray(diagnostic["actionLogits"][start + row])))))
                    action_close = action_close and bool(np.allclose(np.asarray(action)[row],
                        np.asarray(diagnostic["actionLogits"][start + row]),
                        rtol=args.action_rtol, atol=args.logit_atol))
            expected = reference.predict(request["state"], request["questions"])
            actual = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            if actual["model"] != request["model"] or actual["usage"] != expected["usage"]:
                raise AssertionError(f"{name}: model identity or usage differs")
            answer_error = numeric_error(expected["answers"], actual["answers"])
            record = {"case": name, "questions": len(items), "input_tokens": actual["usage"]["input_tokens"],
                      "max_logit_abs_error": logit_error, "max_action_logit_abs_error": action_error,
                      "max_answer_abs_error": answer_error, "exact_tokens": True,
                      "action_logits_within_tolerance": action_close,
                      "passed": logit_error <= args.logit_atol and action_close and answer_error <= args.answer_atol}
            results.append(record)
            print(json.dumps(record), flush=True)
    report = {"platform": platform.platform(), "python_mlx_version": mx.__version__, "dtype": "float16",
              "checkpoint": str(args.checkpoint), "logit_atol": args.logit_atol,
              "action_rtol": args.action_rtol, "answer_atol": args.answer_atol,
              "results": results, "passed": all(result["passed"] for result in results)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("Laya numerical qualification failed")


if __name__ == "__main__":
    main()
