"""Bind reviewed protocol assertions to exact cached sources at build time.

Legacy methodology text is retained in data.json but is not promoted into an
exact checkpoint, dataset or implementation identity by guessing. The registry
contains assertions, never comparison labels or numerical measurement claims.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from comparison import classify_group, summarize_comparisons

CODE = Path(__file__).resolve().parent.parent   # repo root; this file lives in web/


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_sources(value, code=CODE):
    """Reject missing/changed evidence, including altered line ranges or quotes."""
    if isinstance(value, list):
        for item in value:
            _validate_sources(item, code)
    elif isinstance(value, dict):
        if "url" in value:
            path = (code / value.get("path", "")).resolve()
            if not path.is_relative_to(code.resolve()) or not path.is_file():
                raise ValueError("Protocol evidence source is missing or outside the checkout")
            if _digest(path) != value.get("sha256"):
                raise ValueError("Protocol evidence source hash no longer matches")
            span = value.get("lines")
            if not isinstance(span, list) or len(span) != 2 or not all(isinstance(n, int) for n in span):
                raise ValueError("Protocol evidence needs an exact source line range")
            lines = path.read_text().splitlines()
            if not 1 <= span[0] <= span[1] <= len(lines):
                raise ValueError("Protocol evidence line range is outside the source")
            if value.get("quote") != "\n".join(lines[span[0]-1:span[1]]):
                raise ValueError("Protocol evidence quote differs from its cited source lines")
        else:
            for item in value.values():
                _validate_sources(item, code)


def load_bindings(root, idx, code=CODE):
    root = Path(root)
    file = code / "protocol_evidence" / (root.name + ".json")
    if not file.exists():
        return {}
    registry = json.loads(file.read_text())
    if registry.get("version") != 1:
        raise ValueError("Unsupported protocol evidence schema")
    bindings = {}
    for binding in registry.get("bindings", []):
        repo = binding["measurement_repo"]
        meta = idx["repos"].get(repo)
        if not meta or _digest(root / meta["file"]) != binding["source_sha256"]:
            # Stale evidence never grants compatibility. Keep the rows and show why.
            bindings.setdefault(repo, []).append({"protocol": {"issues": [{
                "condition": "source_binding", "reason": "Reviewed protocol evidence is stale; source snapshot changed"}]}})
            continue
        try:
            _validate_sources(binding["protocol"], code)
        except (ValueError, OSError) as error:
            bindings.setdefault(repo, []).append({"protocol": {"issues": [{
                "condition": "source_binding", "reason": str(error)}]}})
            continue
        bindings.setdefault(repo, []).append(binding)
    return bindings


def _merge_assertions(target, incoming):
    """Retain contradictions instead of resolving them by overwrite order."""
    for key, value in incoming.items():
        if key not in target:
            target[key] = deepcopy(value)
        elif key == "metrics":
            for metric, overlay in value.items():
                _merge_assertions(target[key].setdefault(metric, {}), overlay)
        elif key in ("issues", "limitations", "equivalences"):
            target[key] += deepcopy(value)
        elif target[key] != value:
            existing = target[key] if isinstance(target[key], list) else [target[key]]
            target[key] = existing + deepcopy(value if isinstance(value, list) else [value])


def attach_protocols(rows, root, model, idx, code=CODE, metric_mapper=None):
    bindings = load_bindings(root, idx, code)
    config_url = "https://github.com/delirious6423/quant-fidelity-frontier/blob/main/models/" + Path(root).name + "/model.json"
    for row in rows:
        identity = [model, row["measured_by"], row.get("sha"), row.get("L"), row["repo"], row["variant"], row.get("panel")]
        row["measurement_id"] = "m-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
        row["quantization_owner"] = row["repo"].split("/")[0]
        provenance = "quoted" if row["kind"] == "quoted" else "copied" if row.get("copied_from") else "original"
        row["measurement_owner"] = row["measured_by"] if provenance == "original" else None
        row["claim_provenance"] = {"kind": provenance, "claim_publisher": row["measured_by"], "source": row["src"]}
        if row.get("copied_from"):
            row["claim_provenance"]["reported_origin"] = row["copied_from"]
        protocol = {"base_model": {"value": model, "evidence": [{"url": config_url, "description": "Catalog base-model identity; other protocol fields require measurement-source evidence"}]}}
        for binding in bindings.get(row["measured_by"], []):
            if binding.get("panel") is not None and row.get("panel") != binding["panel"]:
                continue
            _merge_assertions(protocol, binding["protocol"])
            # A card-level experiment must not swallow a separately documented row protocol.
            if row.get("methodology_override") and binding.get("panel") is None:
                protocol.setdefault("issues", []).append({"condition": "row_methodology", "reason": "Row has separate methodology; the card-level evidence binding has not verified this override"})
        if row.get("weights_modified"):
            protocol.setdefault("issues", []).append({"condition": "base_model", "reason": "Modified weights: exact base/reference lineage needs separate protocol verification"})
        # Alias only the existing build's source-column mapping. This carries
        # reviewed evidence to the raw claim key without inventing definitions.
        if metric_mapper:
            for series, source in metric_mapper(row).items():
                if series in protocol.get("metrics", {}):
                    protocol["metrics"].setdefault(source, deepcopy(protocol["metrics"][series]))
        row["protocol"] = protocol


def _classify(points, metric):
    # The shared engine memoizes exact profiles while preserving full all-pairs
    # counts, contradictions, and failure examples for website and MCP alike.
    result = classify_group(points, metric=metric)
    result["measurement_count"] = len(points)
    return result



def classify_sets(sets):
    compatible = {}
    for group in sets:
        comparisons = {}
        for metric in group["metrics"]:
            points = [p for p in group["points"] if p.get("m", {}).get(metric, {}).get("v") is not None]
            comparisons[metric] = _classify(points, metric)
            if comparisons[metric]["ranking_permitted"] and group["id"] != "all":
                for point in points:
                    compatible.setdefault(point.get("measurement_id"), set()).add(group["id"])
        group["comparisons"] = comparisons
        group["comparison"] = summarize_comparisons(comparisons)
        group["comparable"] = group["comparison"]["ranking_permitted"]
        group["ranking_permitted"] = group["comparable"]
    for group in sets:
        for point in group["points"]:
            point["compatible_sets"] = sorted(compatible.get(point.get("measurement_id"), set()) - {group["id"]})
