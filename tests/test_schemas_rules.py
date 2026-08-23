from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from qc_pipeline.rules import evaluate_rules
from qc_pipeline.schemas import Interval, QCJob, Verdict, atlas_v1_rules


def valid_payload() -> dict:
    return {
        "schema_version": "qc-job-v1",
        "job_id": "test-job",
        "source": {
            "bucket": "source",
            "items": [
                {
                    "item_id": "one",
                    "key": "clips/one.mp4",
                    "size_bytes": 10,
                    "etag": '"abc"',
                }
            ],
        },
        "target": {"bucket": "target", "prefix": "reports"},
    }


def test_example_job_validates() -> None:
    path = Path(__file__).parents[1] / "examples" / "job.example.json"
    job = QCJob.model_validate_json(path.read_text())
    assert job.schema_version == "qc-job-v1"
    assert job.source.items[0].etag == "replace-with-r2-etag"
    assert job.target.prefix == "atlas/qc/"


def test_unknown_signal_and_unsafe_key_fail() -> None:
    payload = valid_payload()
    payload["rules"] = [
        {
            "id": "bad",
            "signal": "unknown.signal",
            "operator": ">=",
            "threshold": 0.5,
            "severity": "reject",
        }
    ]
    with pytest.raises(ValidationError, match="unsupported signal"):
        QCJob.model_validate(payload)
    payload = valid_payload()
    payload["source"]["items"][0]["key"] = "../secret"
    with pytest.raises(ValidationError, match="safe"):
        QCJob.model_validate(payload)


def signals(hand: float = 0.8, idle: float = 0.2) -> dict[str, float | bool]:
    return {
        "hand.visibility_fraction": hand,
        "hand.motion_speed_p95": 0.1,
        "idle.fraction": idle,
        "motion.repetition_score": 0.1,
        "camera.corrupt": False,
        "camera.black_covered": False,
        "camera.blur_fraction": 0.0,
        "camera.exposure_fraction": 0.0,
        "camera.shake_p95_translation": 0.0,
        "camera.shake_p95_rotation": 0.0,
    }


def test_verdict_good_bad_and_uncertain() -> None:
    rules = atlas_v1_rules()
    verdict, _, _ = evaluate_rules(
        rules,
        signals(),
        {
            "hand.visibility_fraction": Interval(low=0.7, high=0.9),
            "idle.fraction": Interval(low=0.1, high=0.3),
        },
    )
    assert verdict == Verdict.GOOD

    verdict, _, reasons = evaluate_rules(
        rules,
        signals(hand=0.3),
        {
            "hand.visibility_fraction": Interval(low=0.2, high=0.4),
            "idle.fraction": Interval(low=0.1, high=0.3),
        },
    )
    assert verdict == Verdict.BAD
    assert "hands-visible" in reasons

    verdict, _, reasons = evaluate_rules(
        rules,
        signals(hand=0.6),
        {
            "hand.visibility_fraction": Interval(low=0.55, high=0.65),
            "idle.fraction": Interval(low=0.1, high=0.3),
        },
    )
    assert verdict == Verdict.UNCERTAIN
    assert "hands-visible" in reasons


def test_warning_does_not_reject() -> None:
    values = signals()
    values["camera.blur_fraction"] = 0.9
    verdict, results, reasons = evaluate_rules(
        atlas_v1_rules(),
        values,
        {
            "hand.visibility_fraction": Interval(low=0.7, high=0.9),
            "idle.fraction": Interval(low=0.1, high=0.3),
        },
    )
    assert verdict == Verdict.GOOD
    assert "blur-warning" in reasons
    assert next(result for result in results if result.rule_id == "blur-warning").outcome == "fail"


def test_speed_repetition_and_shake_are_rejecting_boundaries() -> None:
    rules = atlas_v1_rules()
    values = signals()
    values["hand.motion_speed_p95"] = 0.75
    values["motion.repetition_score"] = 0.85
    values["camera.shake_p95_translation"] = 0.03
    values["camera.shake_p95_rotation"] = 5.0
    verdict, _, _ = evaluate_rules(
        rules,
        values,
        {
            "hand.visibility_fraction": Interval(low=0.7, high=0.9),
            "hand.motion_speed_p95": Interval(low=0.70, high=0.75),
            "idle.fraction": Interval(low=0.1, high=0.3),
        },
    )
    assert verdict == Verdict.GOOD

    values["hand.motion_speed_p95"] = 0.76
    verdict, _, reasons = evaluate_rules(rules, values, {})
    assert verdict == Verdict.BAD
    assert reasons == ["hand-speed"]

    values = signals()
    values["motion.repetition_score"] = 0.86
    verdict, _, reasons = evaluate_rules(rules, values, {})
    assert verdict == Verdict.BAD
    assert reasons == ["repetitive-motion"]

    values = signals()
    values["camera.shake_p95_rotation"] = 5.01
    verdict, _, reasons = evaluate_rules(rules, values, {})
    assert verdict == Verdict.BAD
    assert reasons == ["rotation-shake"]


def test_definite_reject_does_not_mislabel_an_unknown_second_reason() -> None:
    verdict, _, reasons = evaluate_rules(
        atlas_v1_rules(),
        signals(hand=0.0, idle=0.0),
        {
            "hand.visibility_fraction": Interval(low=0.0, high=0.0),
            "idle.fraction": Interval(low=0.0, high=1.0),
        },
    )
    assert verdict == Verdict.BAD
    assert reasons == ["hands-visible"]
