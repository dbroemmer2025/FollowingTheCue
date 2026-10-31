"""Provenance checks for E46 follow-up replay caches."""
import hashlib
import json


def followup_fingerprint(rows):
    payload = [{k: r.get(k) for k in ("index", "arm", "prompt2", "text2", "ids2")}
               for r in sorted(rows, key=lambda r: (r["index"], r["arm"]))]
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def unique_rows(rows, fields):
    by = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        if key in by:
            raise ValueError(f"Duplicate E46 key {fields}: {key}")
        by[key] = row
    return by


def validate_followups(fu, resid):
    by = unique_rows(fu["rows"], ("index", "arm"))
    unique_rows(resid["rows"], ("index", "arm", "anchor"))
    if len(resid["R"]) != len(resid["rows"]):
        raise ValueError("Residual vectors and row metadata have different lengths")
    for field in ("model", "hf", "hf_revision", "replication_design_id", "seed", "substrate", "cue_type", "prompt_regime",
                  "cot_counterbalance", "adaptive_cue", "followup_question", "sampler"):
        a, b = fu["metadata"].get(field), resid["metadata"].get(field)
        if a != b:
            raise ValueError(f"Follow-up provenance mismatch: {field}: {a!r} != {b!r}")
    expected = resid["metadata"].get("source_followups_sha256")
    digest = followup_fingerprint(fu["rows"])
    if expected and expected != digest:
        raise ValueError("Follow-up content fingerprint does not match residual source")
    for r in resid["rows"]:
        key = (r["index"], r["arm"])
        if key not in by or "text2" not in by[key]:
            raise ValueError(f"Residual row has no generated follow-up partner: {key}")
        for field in ("stratum", "answer", "cue_letter", "correct_letter"):
            if r.get(field) != by[key].get(field):
                raise ValueError(f"Follow-up row mismatch for {key}: {field}")
    covered = {(r["index"], r["arm"]) for r in resid["rows"]}
    return {"status": "content_verified" if expected else "legacy_metadata_only",
            "source_followups_sha256": digest,
            "followup_rows": len(by), "residual_rows": len(resid["rows"]),
            "followups_without_residuals": [list(k) for k in by if k not in covered],
            "limitation": None if expected else
            "Old cache has no content fingerprint; prompt/token identity cannot be fully verified."}
