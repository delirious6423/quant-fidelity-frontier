"""Add source-backed cross-publisher views without changing existing views.

Groups use deterministic greedy complete-link covers over exact protocol
profiles. Compatibility is checked directly between every profile in a group;
neither shared publishers nor connected compatibility paths establish an edge.
Overlapping groups are permitted when equivalence is non-transitive.
"""
from copy import deepcopy
import hashlib
import json
import math

from comparison import classify_pair, classify_group


def _key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _profile(point):
    provenance = point.get("claim_provenance", {})
    relevant = {key: provenance.get(key) for key in ("kind", "original_attribution")} if isinstance(provenance, dict) else {}
    return _key([point.get("protocol"), relevant])


def _publisher(point):
    owner = point.get("measurement_owner") if "measurement_owner" in point else point.get("measured_by")
    return owner.split("/")[0] if isinstance(owner, str) and owner else None


def _known_identity(value):
    return isinstance(value, str) and value.strip().casefold() not in ("", "unknown", "not stated", "unspecified", "not documented", "not separately documented", "n/a")


def add_verified_cross_publisher_sets(sets):
    """Append verified per-metric comparison views; return the appended views.

    Original sets, point coordinates, claims, protocols, and source citations
    remain untouched. Missing identities or size bases never imply equality.
    Duplicate measurement IDs with conflicting representations are excluded
    from generated views rather than resolved by source iteration order.
    """
    buckets = {}
    for source in sets:
        if source.get("id") == "all" or source.get("verified_protocol"):
            continue
        axis = source.get("x", {})
        if not isinstance(axis, dict) or not _known_identity(axis.get("field")) or not _known_identity(axis.get("unit")):
            continue
        for metric in source.get("metrics", []):
            for point in source.get("points", []):
                identity = point.get("measurement_id")
                basis, x = point.get("basis"), point.get("x")
                claim = point.get("m", {}).get(metric)
                if (not identity or not _known_identity(basis) or not _publisher(point) or point.get("is_ref")
                        or not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
                        or not isinstance(claim, dict) or claim.get("v") is None):
                    continue
                # The raw quoted/copied marker cannot be bypassed by dropping
                # structured provenance on an otherwise evidenced protocol.
                provenance = point.get("claim_provenance", {})
                if (point.get("kind") in ("copied", "quoted") or point.get("copied_from")) and (
                    not isinstance(provenance, dict) or provenance.get("kind") not in ("copied", "quoted")
                ):
                    continue
                bucket = buckets.setdefault((metric, axis["unit"], axis["field"], basis), {})
                representation = _key([_profile(point), x, claim, point.get("src"), _publisher(point)])
                if identity not in bucket:
                    bucket[identity] = {"point": point, "sources": [source], "representation": representation, "conflict": False}
                else:
                    entry = bucket[identity]
                    entry["conflict"] |= entry["representation"] != representation
                    entry["sources"].append(source)

    generated = []
    existing_ids = {source.get("id") for source in sets}
    for boundary, entries in sorted(buckets.items()):
        metric, unit, field, basis = boundary
        profiles, valid = {}, {}
        for identity, entry in sorted(entries.items()):
            if entry["conflict"]:
                continue
            point = entry["point"]
            signature = _profile(point)
            if signature not in valid:
                valid[signature] = classify_pair(point, point, metric)["ranking_permitted"]
            if valid[signature]:
                profiles.setdefault(signature, []).append((identity, entry))
        ordered = sorted(profiles, key=lambda signature: profiles[signature][0][0])
        pair_cache = {}

        def compatible(a, b):
            if a == b:
                return True  # Already verified by an actual self-comparison.
            key = tuple(sorted((a, b)))
            if key not in pair_cache:
                pair_cache[key] = classify_pair(profiles[a][0][1]["point"], profiles[b][0][1]["point"], metric)["ranking_permitted"]
            return pair_cache[key]

        candidates = {}
        for seed in ordered:
            clique = [seed]
            for signature in ordered:
                if signature != seed and all(compatible(signature, member) for member in clique):
                    clique.append(signature)
            members = sorted([member for signature in clique for member in profiles[signature]], key=lambda member: member[0])
            if len({_publisher(entry["point"]) for _, entry in members}) < 2:
                continue
            identities = frozenset(identity for identity, _ in members)
            candidates.setdefault(identities, members)
        for identities, members in sorted(candidates.items(), key=lambda item: tuple(sorted(item[0]))):
            if any(identities < other for other in candidates):
                continue
            points = []
            for identity, entry in members:
                point = deepcopy(entry["point"])
                point["qid"] = str(point.get("qid", "measurement")) + "|" + identity
                points.append(point)
            result = classify_group(points, metric)
            if not result["ranking_permitted"]:
                continue  # Defensive exact final check, never graph inference.
            suffix = hashlib.sha256(_key([boundary, sorted(identities)]).encode()).hexdigest()[:16]
            set_id = "protocol-" + suffix
            if set_id in existing_ids:
                continue
            sources = {source["id"]: source for _, entry in members for source in entry["sources"]}

            def description(field_name):
                claims = sorted({source.get(field_name, "") for source in sources.values()} - {""})
                return claims[0] if len(claims) == 1 else "See the cited source protocols in methodology details."

            publishers = sorted({_publisher(point) for point in points})
            view = {
                "id": set_id, "title": "Verified protocol · " + metric + " · " + " + ".join(publishers),
                "reference": description("reference"), "data": description("data"),
                "x": deepcopy(members[0][1]["sources"][0]["x"]), "metrics": [metric], "refs": {},
                "points": points, "comparable": True, "ranking_permitted": True,
                "comparison": result, "comparisons": {metric: result}, "auto": False,
                "verified_protocol": True, "linked": sorted(sources),
                "note": "Every measurement pair has source-backed compatible protocol conditions. Measurement publishers: " + ", ".join(publishers) + ". Quantization publishers are listed separately.",
            }
            generated.append(view)
            existing_ids.add(set_id)
    sets.extend(generated)
    return generated
