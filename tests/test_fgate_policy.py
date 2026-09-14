from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fgate.checks import PolicyEvaluator
from fgate.models import CheckRequest, EvaluationContext, Interval, Outcome, Verdict
from fgate.policy import PolicyRegistry, load_config, load_policy

ROOT = Path(__file__).parents[1]


def context(**updates: object) -> EvaluationContext:
    values = {
        "camera_covered": False,
        "hands_visible_percent": 95.0,
        "worker_idle_percent": 10.0,
        "translation_percent": 1.0,
        "rotation_degrees_per_second": 1.0,
        "repetition_score": 0.2,
        "blurry_percent": 5.0,
        "bad_exposure_percent": 5.0,
        "hands_interval": {"low": 90.0, "high": 98.0},
        "idle_interval": {"low": 5.0, "high": 20.0},
    }
    values.update(updates)
    return EvaluationContext.model_validate(values)


def test_locked_standard_policy_contract() -> None:
    policy = load_policy(ROOT / "policies" / "standard.yaml")
    assert policy.name == "standard"
    assert policy.checks.camera_covered.reject_after_seconds == 5
    assert policy.checks.hands_visible.minimum_percent == 60
    assert policy.checks.worker_idle.maximum_percent == 50
    assert policy.checks.worker_idle.minimum_segment_seconds == 10
    assert policy.checks.camera_steadiness.maximum_translation_percent == 3
    assert policy.checks.camera_steadiness.maximum_rotation_degrees_per_second == 5
    assert policy.checks.repetitive_motion.maximum_score == 0.85
    assert policy.warnings.blurry.maximum_percent == 20
    assert policy.warnings.bad_exposure.maximum_percent == 20
    assert PolicyRegistry(ROOT / "policies").hashes == {"standard": policy.sha256}


def test_app_config_is_strict_and_cloud_requires_endpoint(tmp_path: Path) -> None:
    assert load_config(ROOT / "config.yaml").api_url == "http://127.0.0.1:8787"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("version: 1\nbackend: cloud\npolicy: standard\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="endpoint"):
        load_config(invalid)
    invalid.write_text(
        "version: 1\nbackend: local\npolicy: standard\nextra: true\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="extra"):
        load_config(invalid)


def test_check_request_requires_https_video_urls() -> None:
    with pytest.raises(ValidationError, match="URL scheme"):
        CheckRequest(video_urls=("http://example.com/video.mp4",))


def test_hand_interval_controls_pass_fail_and_uncertain() -> None:
    evaluator = PolicyEvaluator(load_policy(ROOT / "policies" / "standard.yaml"))
    verdict, checks, _ = evaluator.evaluate(
        context(hands_visible_percent=63, hands_interval=Interval(low=61, high=70))
    )
    assert verdict == Verdict.GOOD
    assert checks[1].outcome == Outcome.PASS

    verdict, checks, _ = evaluator.evaluate(
        context(hands_visible_percent=30, hands_interval=Interval(low=20, high=40))
    )
    assert verdict == Verdict.BAD
    assert checks[1].outcome == Outcome.FAIL
    assert checks[2].outcome == Outcome.NOT_APPLICABLE
    assert checks[4].outcome == Outcome.NOT_APPLICABLE

    verdict, checks, _ = evaluator.evaluate(
        context(hands_visible_percent=63, hands_interval=Interval(low=45, high=81))
    )
    assert verdict == Verdict.UNCERTAIN
    assert checks[1].outcome == Outcome.UNCERTAIN


def test_covered_camera_disables_dependent_results() -> None:
    evaluator = PolicyEvaluator(load_policy(ROOT / "policies" / "standard.yaml"))
    verdict, checks, warnings = evaluator.evaluate(context(camera_covered=True))
    assert verdict == Verdict.BAD
    assert checks[0].outcome == Outcome.FAIL
    assert checks[1].outcome == Outcome.NOT_APPLICABLE
    assert checks[2].outcome == Outcome.NOT_APPLICABLE
    assert checks[4].outcome == Outcome.NOT_APPLICABLE
    assert {warning.outcome for warning in warnings} == {Outcome.NOT_APPLICABLE}


def test_warning_does_not_change_verdict() -> None:
    evaluator = PolicyEvaluator(load_policy(ROOT / "policies" / "standard.yaml"))
    verdict, _, warnings = evaluator.evaluate(context(blurry_percent=21, bad_exposure_percent=100))
    assert verdict == Verdict.GOOD
    assert {warning.outcome for warning in warnings} == {Outcome.FAIL}


def test_idle_steadiness_and_repetition_boundaries() -> None:
    evaluator = PolicyEvaluator(load_policy(ROOT / "policies" / "standard.yaml"))
    verdict, checks, _ = evaluator.evaluate(
        context(
            worker_idle_percent=50,
            idle_interval=Interval(low=40, high=50),
            translation_percent=3,
            rotation_degrees_per_second=5,
            repetition_score=0.85,
        )
    )
    assert verdict == Verdict.GOOD
    assert all(check.outcome == Outcome.PASS for check in checks)

    verdict, checks, _ = evaluator.evaluate(context(rotation_degrees_per_second=5.01))
    assert verdict == Verdict.BAD
    assert checks[3].outcome == Outcome.FAIL
