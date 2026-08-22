from __future__ import annotations

from collections.abc import Mapping

from .schemas import Interval, Operator, RuleConfig, RuleResult, Severity, Verdict


def compare(value: float | bool, operator: Operator, threshold: float | bool) -> bool:
    if operator == Operator.GTE:
        return float(value) >= float(threshold)
    if operator == Operator.LTE:
        return float(value) <= float(threshold)
    if operator == Operator.GT:
        return float(value) > float(threshold)
    if operator == Operator.LT:
        return float(value) < float(threshold)
    return value == threshold


def interval_outcome(rule: RuleConfig, value: float | bool, interval: Interval | None) -> str:
    if interval is None or isinstance(value, bool):
        return "pass" if compare(value, rule.operator, rule.threshold) else "fail"

    threshold = float(rule.threshold)
    if rule.operator == Operator.GTE:
        if interval.low >= threshold:
            return "pass"
        if interval.high < threshold:
            return "fail"
        return "uncertain"
    if rule.operator == Operator.GT:
        if interval.low > threshold:
            return "pass"
        if interval.high <= threshold:
            return "fail"
        return "uncertain"
    if rule.operator == Operator.LTE:
        if interval.high <= threshold:
            return "pass"
        if interval.low > threshold:
            return "fail"
        return "uncertain"
    if rule.operator == Operator.LT:
        if interval.high < threshold:
            return "pass"
        if interval.low >= threshold:
            return "fail"
        return "uncertain"
    return "pass" if compare(value, rule.operator, rule.threshold) else "fail"


def evaluate_rules(
    rules: list[RuleConfig],
    signals: Mapping[str, float | bool],
    intervals: Mapping[str, Interval],
) -> tuple[Verdict, list[RuleResult], list[str]]:
    results: list[RuleResult] = []
    rejecting_fail = False
    rejecting_uncertain = False

    for rule in rules:
        if rule.signal not in signals:
            raise KeyError(f"missing signal {rule.signal!r} for rule {rule.id!r}")
        value = signals[rule.signal]
        interval = intervals.get(rule.signal)
        outcome = interval_outcome(rule, value, interval)
        result = RuleResult(
            rule_id=rule.id,
            signal=rule.signal,
            severity=rule.severity,
            outcome=outcome,
            value=value,
            interval=interval,
        )
        results.append(result)
        if rule.severity == Severity.REJECT:
            rejecting_fail |= outcome == "fail"
            rejecting_uncertain |= outcome == "uncertain"

    if rejecting_fail:
        verdict = Verdict.BAD
    elif rejecting_uncertain:
        verdict = Verdict.UNCERTAIN
    else:
        verdict = Verdict.GOOD
    reasons = [
        result.rule_id
        for result in results
        if result.outcome == "fail"
        or (
            result.outcome == "uncertain"
            and result.severity == Severity.REJECT
            and not rejecting_fail
        )
    ]
    return verdict, results, reasons
