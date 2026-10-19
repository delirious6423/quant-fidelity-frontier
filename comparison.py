"""Evidence-only comparison of published measurements; no publisher heuristics.

Protocol schema (all fields are optional, and omission means *unknown*):

    {"protocol": {
        "base_model": {"value": "org/model@revision", "evidence": [{"url": ...}]},
        ...,
        "metric_definition": {"value": {
            "name": "KLD", "direction": "lower", "unit": "nats",
            "aggregation": "arithmetic mean over evaluated tokens"
        }, "evidence": [{"url": ...}]},
        "metrics": {"kld": {"metric_definition": {...}, "dataset": {...}}},
        "equivalences": [{"condition": "implementation", "values": [a, b],
                          "evidence": [{"url": ...}]}]
    }}

Each assertion can instead be a list of assertions. Conflicting values are
reported as contradictions, never resolved by selecting a convenient source.
Metric overlays replace the corresponding shared assertions. An equivalence is
an evidenced, direct, unordered pair of condition values; it is never chained.
An opaque identity for a documented shared experiment is permitted, provided
the evidence establishes that condition's shared identity. Missing metadata may
not be filled with an opaque identity merely to obtain equality.
Such assertions carry ``identity_scope: experiment_id``: equality establishes
relative identity within that scope only. Different scopes (or scoped versus
absolute identity) imply unverified equivalence, never a confirmed mismatch.

Copied claims use ``claim_provenance.kind = copied|quoted`` and require
``original_attribution = {verified: True, evidence: [...]}`` before ranking.
Quantization and measurement ownership are deliberately absent from the rules.
The classifier validates evidence structure, not the truth of cited sources;
adapters and reviewers must ensure citations support their exact assertions.
``protocol.limitations`` carries visible caveats that do not negate evidenced
relative compatibility. Nonempty ``protocol.issues`` records validation failures
or unresolved source conflicts and always prevents direct ranking.
"""

from __future__ import annotations

from itertools import combinations
from bisect import bisect_right
import json
import math
from urllib.parse import urlparse


CONDITIONS = (
    "base_model", "reference_checkpoint", "dataset", "token_selection",
    "metric_definition", "context_length", "implementation", "settings",
)
LABELS = {
    "same_protocol": "Same Protocol — Comparable",
    "different_or_unverified_protocol": "Different or Unverified Protocol — Not Directly Comparable",
    "different_base_model": "Different Base Model — Fidelity Not Directly Comparable",
}
_METRIC_FIELDS = ("name", "direction", "unit", "aggregation")


def _meaningful(value):
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "unknown", "not stated", "unspecified", "not separately documented", "n/a")
    if isinstance(value, (dict, list)):
        members = value.values() if isinstance(value, dict) else value
        return bool(value) and any(_meaningful(member) for member in members)
    return not isinstance(value, (float, int)) or math.isfinite(value)


def _key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _unique(values):
    seen, result = set(), []
    for value in values:
        key = _key(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _evidence(raw):
    """Only externally resolvable source URLs count as supporting evidence."""
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            continue
        url = urlparse(item["url"])
        if url.scheme in ("https", "http") and url.netloc:
            result.append(item)
    return _unique(result)


def _protocol(measurement, metric):
    shared = measurement.get("protocol", {})
    if not isinstance(shared, dict):
        return {}
    result = dict(shared)
    overlays = shared.get("metrics", {})
    if metric is not None and isinstance(overlays, dict):
        overlay = overlays.get(metric, {})
        if isinstance(overlay, dict):
            result.update(overlay)
            # Both shared and metric-scoped direct equivalences remain useful.
            for field in ("equivalences", "limitations", "issues"):
                base_entries = shared.get(field, [])
                metric_entries = overlay.get(field, [])
                if field == "issues":
                    # Malformed nonempty issue metadata is itself an issue;
                    # never drop a validation failure while applying overlays.
                    base_entries = base_entries if isinstance(base_entries, list) else ([{"reason": "Invalid shared methodology issues metadata"}] if base_entries else [])
                    metric_entries = metric_entries if isinstance(metric_entries, list) else ([{"reason": "Invalid metric methodology issues metadata"}] if metric_entries else [])
                result[field] = (base_entries if isinstance(base_entries, list) else []) + (metric_entries if isinstance(metric_entries, list) else [])
        else:
            issues = result.get("issues", [])
            result["issues"] = (issues if isinstance(issues, list) else []) + [{"reason": "Invalid metric protocol overlay"}]
    return result


def _assertion(protocol, condition):
    raw = protocol.get(condition)
    assertions = raw if isinstance(raw, list) else [raw]
    values, evidence, missing_evidence, scopes = [], [], False, []
    for assertion in assertions:
        if not isinstance(assertion, dict) or "value" not in assertion:
            continue
        value = assertion["value"]
        if not _meaningful(value):
            continue
        values.append(value)
        scopes.append(assertion.get("identity_scope"))
        sources = _evidence(assertion.get("evidence"))
        evidence.extend(sources)
        missing_evidence |= not bool(sources)
    values = _unique(values)
    if len(values) > 1:
        return {"state": "contradictory", "values": values, "evidence": _unique(evidence)}
    if not values:
        return {"state": "unknown", "reason": "Missing methodology assertion", "evidence": []}
    scopes = _unique(scopes)
    if len(scopes) > 1:
        return {"state": "unknown", "reason": "Methodology assertion mixes distinct identity scopes", "evidence": _unique(evidence)}
    value = values[0]
    if condition == "metric_definition" and (
        not isinstance(value, dict)
        or any(field not in value or (not _meaningful(value[field]) and not (field == "unit" and value[field] == "")) for field in _METRIC_FIELDS)
    ):
        return {"state": "unknown", "reason": "Metric definition must specify name, direction, unit, and aggregation", "value": value, "evidence": _unique(evidence)}
    if condition == "metric_definition":
        direction = value['direction']
        optimization = direction.get('optimization') if isinstance(direction, dict) else direction
        if optimization not in ("lower", "higher", "minimize", "maximize", "lower_is_better", "higher_is_better"):
            return {"state": "unknown", "reason": "Metric optimization direction is not established", "value": value, "evidence": _unique(evidence)}
    if condition == "context_length" and (not isinstance(value, int) or value <= 0):
        return {"state": "unknown", "reason": "Context length must be a positive integer", "value": value, "evidence": _unique(evidence)}
    if missing_evidence or not evidence:
        return {"state": "unknown", "reason": "Methodology assertion lacks source evidence", "value": value, "evidence": _unique(evidence)}
    return {"state": "verified", "value": value, "evidence": _unique(evidence), "identity_scope": scopes[0]}


def _equivalent(left, right, condition, protocols, scopes=(None, None)):
    if scopes[0] == scopes[1] and _key(left) == _key(right):
        return True, []
    pair = {_key([left, scopes[0]]), _key([right, scopes[1]])}
    for protocol in protocols:
        entries = protocol.get("equivalences", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("condition") != condition:
                continue
            values = entry.get("values")
            sources = _evidence(entry.get("evidence"))
            identity_scopes = entry.get("identity_scopes", [None, None])
            if (isinstance(values, list) and len(values) == 2
                    and isinstance(identity_scopes, list) and len(identity_scopes) == 2
                    and {_key([v, s]) for v, s in zip(values, identity_scopes)} == pair and sources):
                return True, sources
    return False, []


def _provenance_issue(measurement):
    provenance = measurement.get("claim_provenance", {})
    if not isinstance(provenance, dict) or provenance.get("kind") not in ("copied", "quoted"):
        return None
    origin = provenance.get("original_attribution", {})
    if not isinstance(origin, dict) or origin.get("verified") is not True or not _evidence(origin.get("evidence")):
        return "Copied or quoted claim has no verified original measurement attribution"
    return None


def _result(code, verified, differences, unknowns, evidence, metric=None, limitations=None):
    if code == "same_protocol":
        explanation = "Evidence establishes matching or directly equivalent conditions for direct numerical comparison. Publisher identity is not a compatibility condition."
    elif code == "different_base_model":
        explanation = "Evidence identifies different base models or reference distributions. Values may be shown side by side, but their numerical order is not a cross-model quality or fidelity ranking."
    else:
        explanation = "Methodology differs, is contradictory, or is not sufficiently documented. Direct numerical ranking is not permitted."
    return {
        "classification": code, "label": LABELS[code], "explanation": explanation,
        "evidence": _unique(evidence), "verified_matching_conditions": verified,
        "differences": _unique(differences), "unknowns": _unique(unknowns),
        "ranking_permitted": code == "same_protocol", "metric": metric,
        "limitations": _unique(limitations or []),
    }


def classify_pair(left, right, metric=None):
    """Compare two measurement records using source-backed condition identities."""
    protocols = (_protocol(left, metric), _protocol(right, metric))
    verified, differences, unknowns, evidence = [], [], [], []
    contradictory = False
    limitations = []
    for side, protocol in zip(("left", "right"), protocols):
        if isinstance(protocol.get("limitations"), list):
            limitations.extend(protocol["limitations"])
        issues = protocol.get("issues", [])
        if issues:
            contradictory = True
            if not isinstance(issues, list):
                issues = [{"reason": "Invalid methodology issues metadata"}]
            for issue in issues:
                if not isinstance(issue, dict):
                    issue = {"reason": str(issue)}
                differences.append({**issue, "condition": issue.get("condition", "protocol"), "reason": issue.get("reason", "Unresolved methodology evidence issue"), "side": side})
                evidence.extend(_evidence(issue.get("evidence")))
    for condition in CONDITIONS:
        assertions = [_assertion(p, condition) for p in protocols]
        for side, assertion in zip(("left", "right"), assertions):
            evidence.extend(assertion.get("evidence", []))
            if assertion["state"] == "contradictory":
                contradictory = True
                differences.append({"condition": condition, "reason": "Contradictory methodology assertions", "side": side, "values": assertion["values"]})
            elif assertion["state"] == "unknown":
                unknowns.append({"condition": condition, "reason": assertion["reason"], "side": side})
        if any(a["state"] != "verified" for a in assertions):
            continue
        a, b = [a["value"] for a in assertions]
        scopes = [assertion.get("identity_scope") for assertion in assertions]
        match, sources = _equivalent(a, b, condition, protocols, scopes)
        evidence.extend(sources)
        if match:
            verified.append(condition)
        elif scopes[0] != scopes[1] and any(scopes):
            unknowns.append({"condition": condition, "reason": "Separate experiment identities; cross-experiment equivalence is unverified", "left_scope": scopes[0], "right_scope": scopes[1]})
        else:
            differences.append({"condition": condition, "reason": "Confirmed methodology mismatch", "left": a, "right": b})
    for side, measurement in zip(("left", "right"), (left, right)):
        issue = _provenance_issue(measurement)
        if issue:
            unknowns.append({"condition": "claim_provenance", "reason": issue, "side": side})
        provenance = measurement.get("claim_provenance", {})
        if isinstance(provenance, dict) and isinstance(provenance.get("original_attribution"), dict):
            evidence.extend(_evidence(provenance["original_attribution"].get("evidence")))
    identity_mismatch = any(d["condition"] in ("base_model", "reference_checkpoint") and d["reason"] == "Confirmed methodology mismatch" for d in differences)
    code = "same_protocol"
    if contradictory:
        code = "different_or_unverified_protocol"
    elif identity_mismatch:
        code = "different_base_model"
    elif differences or unknowns:
        code = "different_or_unverified_protocol"
    return _result(code, verified, differences, unknowns, evidence, metric, limitations)


def classify_group(measurements, metric=None):
    """Require every pair to agree; connected components are never sufficient.

    Caller selects records reporting the requested metric. Exact repeated
    profiles are evaluated once with multiplicity, never merged by compatibility
    edges. Both pair orientations are retained when profiles are interleaved.
    The result is bounded to five failure examples; pair_count remains the full
    original all-pairs count, including pairs within duplicate profiles.
    """
    measurements = list(measurements)
    pair_count = len(measurements) * (len(measurements) - 1) // 2
    if not pair_count:
        result = _result("different_or_unverified_protocol", [], [], [
            {"condition": "comparison_group", "reason": "At least two measurements are required to establish a comparison"}
        ], [], metric)
        result.update(pair_count=0, comparable_pair_count=0, failure_examples=[])
        return result
    profiles = {}
    for index, measurement in enumerate(measurements):
        protocol = _protocol(measurement, metric)
        resolved = {key: protocol[key] for key in CONDITIONS + ("equivalences", "issues", "limitations") if key in protocol}
        provenance = measurement.get("claim_provenance", {})
        relevant_provenance = {key: provenance.get(key) for key in ("kind", "original_attribution")} if isinstance(provenance, dict) else {}
        profile = _key([resolved, relevant_provenance])
        profiles.setdefault(profile, []).append(index)
    # A directed profile pair represents precisely the original i < j pairs in
    # that orientation. Sort its first pair to retain original evidence order.
    tasks = []
    members = list(profiles.values())
    for indices in members:
        if len(indices) >= 2:
            tasks.append(((indices[0], indices[1]), len(indices) * (len(indices) - 1) // 2, indices, indices))
    for a, b in combinations(members, 2):
        for left, right in ((a, b), (b, a)):
            count = sum(len(right) - bisect_right(right, i) for i in left)
            if count:
                tasks.append(((left[0], right[bisect_right(right, left[0])]), count, left, right))
    tasks.sort(key=lambda task: task[0])
    verified, differences, unknowns, evidence, limitations = set(CONDITIONS), [], [], [], []
    failures, compatible, has_identity_mismatch = [], 0, False
    for (i, j), multiplicity, left_indices, right_indices in tasks:
        pair = classify_pair(measurements[i], measurements[j], metric)
        verified.intersection_update(pair["verified_matching_conditions"])
        differences.extend(pair["differences"])
        unknowns.extend(pair["unknowns"])
        evidence.extend(pair["evidence"])
        limitations.extend(pair["limitations"])
        compatible += multiplicity if pair["ranking_permitted"] else 0
        has_identity_mismatch |= pair["classification"] == "different_base_model"
        if not pair["ranking_permitted"]:
            # Only the first five original pairs of this profile can possibly
            # belong to the first five failures globally. No quadratic list.
            examples = 0
            for left_index in left_indices:
                for right_index in right_indices[bisect_right(right_indices, left_index):]:
                    failures.append({"pair": [left_index, right_index], "classification": pair["classification"], "differences": pair["differences"], "unknowns": pair["unknowns"]})
                    examples += 1
                    if examples == 5:
                        break
                if examples == 5:
                    break
    failures.sort(key=lambda failure: failure["pair"])
    failures = failures[:5]
    code = "same_protocol" if compatible == pair_count else ("different_base_model" if has_identity_mismatch else "different_or_unverified_protocol")
    result = _result(code, [c for c in CONDITIONS if c in verified], differences, unknowns, evidence, metric, limitations)
    result.update(pair_count=pair_count, comparable_pair_count=compatible, failure_examples=failures)
    return result


def classify_by_metric(measurements, metrics):
    """Classify the actual metric population, never unrelated absent values.

    Supports site points (``m``) and claims rows (``metrics``). A metric exists
    only when its scalar or ``v`` claim is non-null; no metric-name aliases are
    used to infer protocol definitions.
    """
    measurements = list(measurements)
    results = {}
    for metric in metrics:
        selected = []
        for measurement in measurements:
            claims = measurement.get("m", measurement.get("metrics", {}))
            if not isinstance(claims, dict) or metric not in claims:
                continue
            claim = claims[metric]
            value = claim.get("v") if isinstance(claim, dict) else claim
            if value is not None:
                selected.append(measurement)
        results[metric] = classify_group(selected, metric)
    return results


def summarize_comparisons(results):
    """Conservatively summarize a mapping of metric IDs to classifications.

    Every reported metric must permit ranking for the aggregate to permit it.
    This summary never replaces the individual per-metric ranking gates.
    """
    if not results:
        return _result("different_or_unverified_protocol", [], [], [
            {"condition": "comparison_group", "reason": "No metric comparison is available"}
        ], [])
    items = list(results.items())
    comparable = all(result.get("ranking_permitted") is True and result.get("classification") == "same_protocol" for _, result in items)
    code = "same_protocol" if comparable else (
        "different_base_model" if any(result.get("classification") == "different_base_model" for _, result in items)
        else "different_or_unverified_protocol"
    )
    verified = set(CONDITIONS)
    differences, unknowns, evidence, limitations = [], [], [], []
    for metric, result in items:
        verified.intersection_update(result.get("verified_matching_conditions", []))
        differences.extend({**issue, "metric": metric} for issue in result.get("differences", []))
        unknowns.extend({**issue, "metric": metric} for issue in result.get("unknowns", []))
        evidence.extend(result.get("evidence", []))
        limitations.extend(result.get("limitations", []))
    summary = _result(code, [c for c in CONDITIONS if c in verified], differences, unknowns, evidence, limitations=limitations)
    summary["metric_count"] = len(items)
    summary["comparable_metrics"] = [metric for metric, result in items if result.get("ranking_permitted")]
    return summary
