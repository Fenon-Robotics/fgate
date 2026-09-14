from __future__ import annotations

from .interfaces import Check
from .models import CheckResult, EvaluationContext, Interval, Outcome, Verdict
from .policy import Policy


def _minimum_outcome(value: float, limit: float, interval: Interval | None) -> Outcome:
    if interval is None:
        return Outcome.PASS if value >= limit else Outcome.FAIL
    if interval.low >= limit:
        return Outcome.PASS
    if interval.high < limit:
        return Outcome.FAIL
    return Outcome.UNCERTAIN


def _maximum_outcome(value: float, limit: float, interval: Interval | None) -> Outcome:
    if interval is None:
        return Outcome.PASS if value <= limit else Outcome.FAIL
    if interval.high <= limit:
        return Outcome.PASS
    if interval.low > limit:
        return Outcome.FAIL
    return Outcome.UNCERTAIN


class CameraCoveredCheck(Check):
    def __init__(self, policy: Policy):
        self._policy = policy.checks.camera_covered

    @property
    def name(self) -> str:
        return "camera_covered"

    def evaluate(self, context: EvaluationContext) -> CheckResult:
        return CheckResult(
            name=self.name,
            outcome=Outcome.FAIL if context.camera_covered else Outcome.PASS,
            measurements={"covered": context.camera_covered},
            limits={"reject_after_seconds": self._policy.reject_after_seconds},
        )


class HandsVisibleCheck(Check):
    def __init__(self, policy: Policy):
        self._policy = policy.checks.hands_visible

    @property
    def name(self) -> str:
        return "hands_visible"

    def evaluate(self, context: EvaluationContext) -> CheckResult:
        interval = context.hands_interval
        return CheckResult(
            name=self.name,
            outcome=_minimum_outcome(
                context.hands_visible_percent, self._policy.minimum_percent, interval
            ),
            measurements={"visible_percent": context.hands_visible_percent},
            limits={"minimum_percent": self._policy.minimum_percent},
            intervals={"visible_percent": interval} if interval else {},
        )


class WorkerIdleCheck(Check):
    def __init__(self, policy: Policy):
        self._policy = policy.checks.worker_idle

    @property
    def name(self) -> str:
        return "worker_idle"

    def evaluate(self, context: EvaluationContext) -> CheckResult:
        interval = context.idle_interval
        return CheckResult(
            name=self.name,
            outcome=_maximum_outcome(
                context.worker_idle_percent, self._policy.maximum_percent, interval
            ),
            measurements={"idle_percent": context.worker_idle_percent},
            limits={
                "maximum_percent": self._policy.maximum_percent,
                "minimum_segment_seconds": self._policy.minimum_segment_seconds,
            },
            intervals={"idle_percent": interval} if interval else {},
        )


class CameraSteadinessCheck(Check):
    def __init__(self, policy: Policy):
        self._policy = policy.checks.camera_steadiness

    @property
    def name(self) -> str:
        return "camera_steadiness"

    def evaluate(self, context: EvaluationContext) -> CheckResult:
        passed = (
            context.translation_percent <= self._policy.maximum_translation_percent
            and context.rotation_degrees_per_second
            <= self._policy.maximum_rotation_degrees_per_second
        )
        return CheckResult(
            name=self.name,
            outcome=Outcome.PASS if passed else Outcome.FAIL,
            measurements={
                "translation_percent": context.translation_percent,
                "rotation_degrees_per_second": context.rotation_degrees_per_second,
            },
            limits={
                "maximum_translation_percent": self._policy.maximum_translation_percent,
                "maximum_rotation_degrees_per_second": (
                    self._policy.maximum_rotation_degrees_per_second
                ),
            },
        )


class RepetitiveMotionCheck(Check):
    def __init__(self, policy: Policy):
        self._policy = policy.checks.repetitive_motion

    @property
    def name(self) -> str:
        return "repetitive_motion"

    def evaluate(self, context: EvaluationContext) -> CheckResult:
        return CheckResult(
            name=self.name,
            outcome=(
                Outcome.PASS
                if context.repetition_score <= self._policy.maximum_score
                else Outcome.FAIL
            ),
            measurements={"score": context.repetition_score},
            limits={"maximum_score": self._policy.maximum_score},
        )


class BlurWarning(Check):
    def __init__(self, policy: Policy):
        self._policy = policy.warnings.blurry

    @property
    def name(self) -> str:
        return "blurry"

    def evaluate(self, context: EvaluationContext) -> CheckResult:
        return CheckResult(
            name=self.name,
            outcome=(
                Outcome.PASS
                if context.blurry_percent <= self._policy.maximum_percent
                else Outcome.FAIL
            ),
            measurements={"percent": context.blurry_percent},
            limits={"maximum_percent": self._policy.maximum_percent},
        )


class ExposureWarning(Check):
    def __init__(self, policy: Policy):
        self._policy = policy.warnings.bad_exposure

    @property
    def name(self) -> str:
        return "bad_exposure"

    def evaluate(self, context: EvaluationContext) -> CheckResult:
        return CheckResult(
            name=self.name,
            outcome=(
                Outcome.PASS
                if context.bad_exposure_percent <= self._policy.maximum_percent
                else Outcome.FAIL
            ),
            measurements={"percent": context.bad_exposure_percent},
            limits={"maximum_percent": self._policy.maximum_percent},
        )


class PolicyEvaluator:
    def __init__(self, policy: Policy):
        self._camera_covered = CameraCoveredCheck(policy)
        self._hands_visible = HandsVisibleCheck(policy)
        self._worker_idle = WorkerIdleCheck(policy)
        self._camera_steadiness = CameraSteadinessCheck(policy)
        self._repetitive_motion = RepetitiveMotionCheck(policy)
        self._warnings: tuple[Check, ...] = (BlurWarning(policy), ExposureWarning(policy))

    def evaluate(
        self, context: EvaluationContext
    ) -> tuple[Verdict, tuple[CheckResult, ...], tuple[CheckResult, ...]]:
        covered = self._camera_covered.evaluate(context)
        steady = self._camera_steadiness.evaluate(context)
        if covered.outcome == Outcome.FAIL:
            checks = (
                covered,
                self._not_applicable(self._hands_visible.name),
                self._not_applicable(self._worker_idle.name),
                steady,
                self._not_applicable(self._repetitive_motion.name),
            )
            warnings = tuple(self._not_applicable(check.name) for check in self._warnings)
            return Verdict.BAD, checks, warnings

        hands = self._hands_visible.evaluate(context)
        if hands.outcome != Outcome.PASS:
            checks = (
                covered,
                hands,
                self._not_applicable(self._worker_idle.name),
                steady,
                self._not_applicable(self._repetitive_motion.name),
            )
        else:
            checks = (
                covered,
                hands,
                self._worker_idle.evaluate(context),
                steady,
                self._repetitive_motion.evaluate(context),
            )
        warnings = tuple(check.evaluate(context) for check in self._warnings)
        if any(check.outcome == Outcome.FAIL for check in checks):
            verdict = Verdict.BAD
        elif any(check.outcome == Outcome.UNCERTAIN for check in checks):
            verdict = Verdict.UNCERTAIN
        else:
            verdict = Verdict.GOOD
        return verdict, checks, warnings

    @staticmethod
    def _not_applicable(name: str) -> CheckResult:
        return CheckResult(name=name, outcome=Outcome.NOT_APPLICABLE)
