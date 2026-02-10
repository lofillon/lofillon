from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any
import json
import re

import polars as pl
import httpx

from impactlens.llm.ollama import OllamaClient, extract_json_object
from impactlens.schemas.extraction import ExtractionOutput
from impactlens.storage import append_parquet, read_parquet, write_manifest
from impactlens.utils import utc_now
from impactlens.validation import has_errors, validate_extraction


PROMPT_VERSION = "v1"


def _coerce_llm_output(project_id: str, obj: dict[str, Any]) -> dict[str, Any]:
    """Best-effort adapter from LLM JSON to `ExtractionOutput` schema.

    Models occasionally omit required fields or use slightly different top-level keys.
    We prefer to be permissive (store a mostly-empty ExtractionOutput) rather than crash.
    """
    # Required field for our schema
    if not obj.get("project_id"):
        obj["project_id"] = project_id

    # Common mismatch: model returns `results` instead of `results_indicators`
    if "results_indicators" not in obj and isinstance(obj.get("results"), list):
        coerced: list[dict[str, Any]] = []
        for item in obj.get("results") or []:
            if not isinstance(item, dict):
                continue
            pillar = item.get("pillar")
            indicator = item.get("indicator") or item.get("name") or item.get("result")
            name = indicator or pillar
            if pillar and indicator and pillar not in str(indicator):
                name = f"{pillar}: {indicator}"

            def _num_fact(x: Any) -> dict[str, Any] | None:
                if x is None:
                    return None
                if isinstance(x, (int, float)):
                    return {"value": float(x), "year": None, "citations": []}
                if isinstance(x, dict):
                    # allow {value, year} or similar shapes
                    value = x.get("value", x.get("val", None))
                    year = x.get("year", x.get("date", None))
                    out: dict[str, Any] = {"citations": []}
                    out["value"] = float(value) if isinstance(value, (int, float)) else None
                    out["year"] = int(year) if isinstance(year, int) else None
                    return out
                return None

            coerced.append(
                {
                    "name": str(name) if name is not None else "Unknown indicator",
                    "unit": item.get("unit"),
                    "baseline": _num_fact(item.get("baseline")),
                    "target": _num_fact(item.get("target")),
                    "achieved": _num_fact(item.get("achieved")),
                    "citations": [],
                }
            )
        obj["results_indicators"] = coerced

    # Enforce schema type: insufficient_evidence must be list[str]
    ie = obj.get("insufficient_evidence")
    if isinstance(ie, list):
        fixed: list[str] = []
        for x in ie:
            if isinstance(x, str):
                fixed.append(x)
            elif isinstance(x, dict):
                fixed.append(json.dumps(x, ensure_ascii=False))
            elif x is None:
                continue
            else:
                fixed.append(str(x))
        obj["insufficient_evidence"] = fixed
    elif isinstance(ie, dict):
        obj["insufficient_evidence"] = [json.dumps(ie, ensure_ascii=False)]
    elif ie is None:
        pass
    else:
        obj["insufficient_evidence"] = [str(ie)]

    return obj


_DASHBOARD_KEYWORDS = [
    # Basic information / key dates
    "BASIC INFORMATION",
    "Project ID",
    "Project Name",
    "Country",
    "Financing Instrument",
    "Bank Approval Date",
    "Effectiveness Date",
    "Closing Date",
    "Implementing Agency",
    "Borrower",
    # Ratings
    "Overall Ratings",
    "Progress towards achievement of PDO",
    "Overall Implementation Progress",
    "Overall Risk Rating",
    "Outcome",
    # Results tables
    "Results",
    "PDO Indicators",
    "Intermediate Results Indicators",
    "Baseline",
    "Actual (Current)",
    "End Target",
    "Results Framework",
]


def _keyword_score(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for k in _DASHBOARD_KEYWORDS if k.lower() in t)


def _project_id_score(text: str, project_id: str) -> int:
    """Boost chunks that appear to be the project's own datasheet/table rows."""
    t = (text or "")
    score = 0
    if project_id and project_id in t:
        # frequent header mention is common; give a modest boost
        score += 2
    # Strong boost only when the project_id appears as a standalone table value line.
    # This avoids boosting multi-project headers that mention multiple IDs.
    if project_id and ("Project ID" in t or "Project Id" in t):
        if f"\n{project_id}\n" in t or f"\r\n{project_id}\r\n" in t:
            score += 20
    return score


def _first_matching_chunk(project_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick a chunk that is likely to contain the project's own datasheet."""
    for ch in chunks:
        txt = str(ch.get("text") or "")
        if ("DATA SHEET" not in txt) or ("BASIC INFORMATION" not in txt) or ("Project ID" not in txt):
            continue
        # Guard against multi-project headers: require the project_id to appear as a standalone table value line.
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if project_id not in lines:
            continue
        return ch
    return None


def _first_pid_chunk(project_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick a chunk likely to be a PID 'Basic Information' table."""
    for ch in chunks:
        txt = str(ch.get("text") or "")
        if "BASIC INFORMATION" not in txt:
            continue
        if "Operation ID" not in txt and "Project ID" not in txt:
            continue
        if project_id and project_id not in txt:
            continue
        return ch
    return None


def _extract_overview_from_pid_chunk(project_id: str, ch: dict[str, Any]) -> dict[str, Any]:
    """Parse PID Concept Stage 'BASIC INFORMATION' tables."""
    txt = str(ch.get("text") or "")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

    def _scalar(value: str, quote: str | None = None) -> dict[str, Any]:
        v = (value or "").strip()
        q = (quote or v).strip()
        return _make_scalar_fact(ch, v, quote=q) if v else {"value": "", "citations": []}

    out: dict[str, Any] = {}

    def _slice_after(label: str, *, stop_at: set[str]) -> list[str]:
        if label not in lines:
            return []
        i = lines.index(label)
        seg: list[str] = []
        for ln in lines[i + 1 :]:
            if ln in stop_at:
                break
            seg.append(ln)
        return seg

    # BASIC INFO: infer (country, op_id, op_name) by locating the op_id value near the "Operation Name" header.
    op_seg = _slice_after("Operation Name", stop_at={"Region", "Estimated Appraisal Date", "Estimated Approval Date"})
    if op_seg:
        op_id_idx = None
        for j, ln in enumerate(op_seg):
            if re.fullmatch(r"P\d{6}", ln):
                op_id_idx = j
                break
        if op_id_idx is not None:
            op_id_val = op_seg[op_id_idx]
            if project_id and op_id_val != project_id:
                return {}
            if op_id_idx - 1 >= 0:
                out["country"] = _scalar(op_seg[op_id_idx - 1], quote=op_seg[op_id_idx - 1])
            if op_id_idx + 1 < len(op_seg):
                out["project_name"] = _scalar(op_seg[op_id_idx + 1], quote=op_seg[op_id_idx + 1])

    # Region + estimated dates block is typically 3 columns: Region / Estimated Appraisal Date / Estimated Approval Date
    region_seg = _slice_after("Estimated Approval Date", stop_at={"Financing Instrument", "Borrower(s)", "Implementing Agency", "Proposed Development Objective(s)"})
    if region_seg:
        # Heuristic: first non-date is region; first two dates are appraisal/approval.
        date_pat = re.compile(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b")
        dates = [ln for ln in region_seg if date_pat.search(ln)]
        # Region sometimes wraps across multiple lines in ALL CAPS (e.g., "EASTERN AND SOUTHERN" + "AFRICA").
        reg_parts: list[str] = []
        for ln in region_seg:
            if date_pat.search(ln):
                break
            if ln in {"Region", "Estimated Appraisal Date", "Estimated Approval Date"}:
                continue
            reg_parts.append(ln)
        region_val = " ".join(reg_parts).strip()
        if region_val:
            # Pick a quote that is actually present in the chunk text (newlines often split regions).
            reg_quote = "\n".join(reg_parts).strip()
            if reg_quote and reg_quote not in txt:
                reg_quote = " ".join(reg_parts).strip()
            if reg_quote and reg_quote not in txt:
                reg_quote = (reg_parts[0] if reg_parts else region_val).strip()
            out["region"] = _scalar(region_val, quote=reg_quote)
        if len(dates) >= 1:
            out["effectiveness_date"] = _scalar(dates[0], quote=dates[0])
        if len(dates) >= 2:
            out["approval_date"] = _scalar(dates[1], quote=dates[1])

    # Proposed Development Objective(s) block
    if "Proposed Development Objective(s)" in lines:
        i = lines.index("Proposed Development Objective(s)")
        pdo_lines: list[str] = []
        for ln in lines[i + 1 :]:
            if ln.startswith("@#&OPS~") or "PROJECT FINANCING DATA" in ln:
                break
            pdo_lines.append(ln)
            if len(" ".join(pdo_lines)) > 800:
                break
        pdo_text = " ".join(pdo_lines).strip()
        if pdo_text:
            out["_pid_objective_text"] = pdo_text
            # Preserve a quote that's likely to exist verbatim in the chunk text.
            pdo_quote = "\n".join(pdo_lines).strip()
            if pdo_quote and pdo_quote not in txt:
                pdo_quote = (pdo_lines[0] if pdo_lines else pdo_text).strip()
            out["_pid_objective_quote"] = pdo_quote

    return out


def _extract_financials_from_pid_chunk(project_id: str, ch: dict[str, Any]) -> dict[str, Any]:
    """Parse PID 'PROJECT FINANCING DATA (US$, Millions)' summary amounts."""
    txt = str(ch.get("text") or "")
    if not txt or "PROJECT FINANCING DATA" not in txt:
        return {}
    if project_id and project_id not in txt:
        return {}

    def _to_usd_from_millions(s: str) -> str | None:
        try:
            x = float((s or "").replace(",", "").strip())
        except Exception:
            return None
        usd = int(round(x * 1_000_000))
        return f"{usd:,}"

    out: dict[str, Any] = {}
    # Look for:
    # Total Operation Cost\n353.00
    # Total Financing\n353.00
    # of which IBRD/IDA\n100.00
    for label, key in [
        ("Total Operation Cost", "total_project_cost_usd"),
        ("Total Financing", "total_commitment_usd"),
        ("of which IBRD/IDA", "world_bank_financing_usd"),
    ]:
        m = re.search(rf"{re.escape(label)}\s*[\r\n]+([0-9][0-9,]*\.\d+)", txt, flags=re.IGNORECASE)
        if not m:
            continue
        usd = _to_usd_from_millions(m.group(1))
        if not usd:
            continue
        out[key] = _make_scalar_fact(ch, usd, quote=m.group(0))

    # Back-compat: if we got world bank financing but not total_commitment_usd, fill it.
    if out.get("world_bank_financing_usd") and not out.get("total_commitment_usd"):
        out["total_commitment_usd"] = out["world_bank_financing_usd"]

    return out


def _extract_overview_from_chunk(project_id: str, ch: dict[str, Any]) -> dict[str, Any]:
    """Heuristic parser for World Bank datasheet-style tables.

    Returns a dict shaped like `ProjectOverview` (ScalarFact objects as dicts).
    """
    txt = str(ch.get("text") or "")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

    def _citation(quote: str) -> dict[str, Any]:
        return {
            "doc_id": ch.get("doc_id"),
            "chunk_id": ch.get("chunk_id"),
            "page": int(ch.get("page") or 0) or None,
            "quote": quote.strip(),
        }

    def _scalar(value: str, quote: str | None = None) -> dict[str, Any]:
        v = value.strip()
        q = (quote or v).strip()
        return {"value": v, "citations": [_citation(q)]}

    # Project name appears right after Project ID / Project Name block in these datasheets.
    # We'll parse it as: find the project_id line, then take the next non-empty line as name.
    project_name = None
    for i, ln in enumerate(lines):
        if ln.strip() == project_id:
            if i + 1 < len(lines):
                project_name = lines[i + 1]
            break

    # Country + Financing Instrument commonly appear as a 2x2 table:
    # Country / Financing Instrument (labels), then India / Investment Project Financing (values).
    country = None
    fin_instr = None
    try:
        i_country = lines.index("Country")
        i_fin = lines.index("Financing Instrument")
        if i_fin == i_country + 1 and i_fin + 2 < len(lines):
            country = lines[i_fin + 1]
            fin_instr = lines[i_fin + 2]
    except ValueError:
        pass

    # Organizations section similarly: Borrower / Implementing Agency labels then values.
    borrower = None
    impl = None
    try:
        i_b = lines.index("Borrower")
        i_ia = lines.index("Implementing Agency")
        if i_ia == i_b + 1:
            # Collect all lines after Implementing Agency until next section
            j = i_ia + 1
            org_lines: list[str] = []
            while j < len(lines) and lines[j] not in {"Project Development Objective (PDO)", "Project Development Objective"}:
                org_lines.append(lines[j])
                j += 1
            if org_lines:
                # Heuristic: last line is implementing agency, prior lines are borrower (often multi-line)
                impl = org_lines[-1]
                borrower = " ".join(org_lines[:-1]).strip() or None
    except ValueError:
        pass

    out: dict[str, Any] = {}
    if project_name:
        out["project_name"] = _scalar(project_name)
    if country:
        out["country"] = _scalar(country)
    if fin_instr:
        out["financing_instrument"] = _scalar(fin_instr)
    if borrower:
        out["borrower"] = _scalar(borrower)
    if impl:
        out["implementing_agency"] = _scalar(impl)
    return out


def _extract_objective_from_chunk(ch: dict[str, Any]) -> dict[str, Any] | None:
    txt = str(ch.get("text") or "")
    # Prefer the "Original PDO" paragraph if present (be robust to formatting quirks).
    lo = txt.lower()
    start = lo.find("original pdo")
    if start == -1:
        return None
    start = start + len("original pdo")
    end = lo.find("revised pdo", start)
    segment = txt[start:end] if end != -1 else txt[start:]
    obj_txt = " ".join(line.strip() for line in segment.splitlines() if line.strip())
    obj_txt = obj_txt.strip()
    if not obj_txt:
        return None
    quote = obj_txt[:300]
    return {
        "text": obj_txt,
        "citations": [
            {
                "doc_id": ch.get("doc_id"),
                "chunk_id": ch.get("chunk_id"),
                "page": int(ch.get("page") or 0) or None,
                "quote": quote,
            }
        ],
    }


def _clean_rating(s: str) -> str:
    # Remove common bullet glyphs and normalize whitespace
    s = (s or "").replace("", "").replace("•", "").strip()
    s = " ".join(s.split())
    return s


def _make_citation(ch: dict[str, Any], quote: str) -> dict[str, Any]:
    return {
        "doc_id": ch.get("doc_id"),
        "chunk_id": ch.get("chunk_id"),
        "page": int(ch.get("page") or 0) or None,
        "quote": quote.strip(),
    }


def _make_scalar_fact(ch: dict[str, Any], value: str, quote: str | None = None) -> dict[str, Any]:
    v = (value or "").strip()
    q = (quote or v).strip()
    return {"value": v, "citations": [_make_citation(ch, q)]} if v else {"value": "", "citations": []}


def _extract_financials_from_chunk(project_id: str, ch: dict[str, Any]) -> dict[str, Any]:
    """Extract common financing facts from PAD/datasheet-like chunks.

    We intentionally require the explicit "Project ID: <id>" pattern to avoid
    picking up multi-project documents (e.g., ICRs covering several projects).
    """
    txt = str(ch.get("text") or "")
    if not txt or not project_id:
        return {}

    if not re.search(rf"\bProject ID:\s*{re.escape(project_id)}\b", txt):
        return {}

    out: dict[str, Any] = {}

    def _to_usd_from_millions(s: str) -> str | None:
        try:
            x = float((s or "").replace(",", "").strip())
        except Exception:
            return None
        usd = int(round(x * 1_000_000))
        return f"{usd:,}"

    # Example:
    # "Total Bank financing (US$m): 1,100.00"
    m = re.search(
        r"Total\s+Bank\s+financing\s*\(US\$\s*m\)\s*:\s*([0-9][0-9,]*\.\d+)",
        txt,
        flags=re.IGNORECASE,
    )
    if m:
        usd = _to_usd_from_millions(m.group(1))
        if usd:
            # Preserve exact matched quote from the chunk text.
            out["world_bank_financing_usd"] = _make_scalar_fact(ch, usd, quote=m.group(0))
            # Back-compat: our UI already has this field.
            out["total_commitment_usd"] = _make_scalar_fact(ch, usd, quote=m.group(0))

    # Financing Plan table (US$m), capture borrower total and overall total.
    # We cite the exact matched multi-line snippet for validation support.
    m_b = re.search(
        r"Borrower:\s*\n\s*([0-9][0-9,]*\.\d+)\s*\n\s*([0-9][0-9,]*\.\d+)\s*\n\s*([0-9][0-9,]*\.\d+)\s*",
        txt,
        flags=re.IGNORECASE,
    )
    if m_b:
        usd = _to_usd_from_millions(m_b.group(3))
        if usd:
            out["borrower_contribution_usd"] = _make_scalar_fact(ch, usd, quote=m_b.group(0))

    m_t = re.search(
        r"Total:\s*\n\s*([0-9][0-9,]*\.\d+)\s*\n\s*([0-9][0-9,]*\.\d+)\s*\n\s*([0-9][0-9,]*\.\d+)\s*",
        txt,
        flags=re.IGNORECASE,
    )
    if m_t:
        usd = _to_usd_from_millions(m_t.group(3))
        if usd:
            out["total_project_cost_usd"] = _make_scalar_fact(ch, usd, quote=m_t.group(0))

    return out


def _extract_key_dates_from_chunk(project_id: str, ch: dict[str, Any]) -> dict[str, Any]:
    """Parse ISR/ICR-style KEY DATES tables into overview fields."""
    txt = str(ch.get("text") or "")
    if not txt or "KEY DATES" not in txt:
        return {}
    if project_id and project_id not in txt:
        return {}

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    try:
        i = lines.index("KEY DATES")
    except ValueError:
        return {}

    # collect labels until first date-like value
    labels: list[str] = []
    j = i + 1
    while j < len(lines):
        ln = lines[j]
        if re.search(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b", ln):
            break
        if ln.isupper() and ln not in {"KEY DATES"} and len(labels) >= 2:
            break
        labels.append(ln)
        j += 1

    dates: list[str] = []
    k = j
    while k < len(lines) and len(dates) < len(labels) and len(dates) < 12:
        ln = lines[k]
        if ln.isupper() and ln not in {"KEY DATES"} and dates:
            break
        if re.search(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b", ln):
            dates.append(ln)
        k += 1

    if len(labels) < 2 or len(dates) < 2:
        return {}

    pairs = {labels[idx].strip().lower(): dates[idx].strip() for idx in range(min(len(labels), len(dates)))}

    out: dict[str, Any] = {}
    if "approval" in pairs:
        out["approval_date"] = _make_scalar_fact(ch, pairs["approval"], quote=pairs["approval"])
    if "effectiveness" in pairs:
        out["effectiveness_date"] = _make_scalar_fact(ch, pairs["effectiveness"], quote=pairs["effectiveness"])

    if "actual closing" in pairs:
        out["closing_date"] = _make_scalar_fact(ch, pairs["actual closing"], quote=pairs["actual closing"])
    elif "original closing" in pairs:
        out["closing_date"] = _make_scalar_fact(ch, pairs["original closing"], quote=pairs["original closing"])

    return out


def _extract_region_practice_from_chunk(ch: dict[str, Any]) -> dict[str, Any]:
    """Extract region/practice area from ISR headers like:
    'SOUTH ASIA | India | Transport & ICT Global Practice |'
    """
    txt = str(ch.get("text") or "")
    if not txt:
        return {}
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    for ln in lines:
        if "Global Practice" not in ln or "|" not in ln:
            continue
        parts = [p.strip() for p in ln.split("|") if p.strip()]
        if len(parts) < 3:
            continue
        region_raw = parts[0]
        practice_raw = parts[2]
        practice_raw = practice_raw.replace("Global Practice", "").strip()
        if not region_raw and not practice_raw:
            continue

        # Normalize region a bit for display while keeping citation quote verbatim.
        region_val = region_raw.title() if region_raw.isupper() else region_raw
        out: dict[str, Any] = {}
        if region_val:
            out["region"] = _make_scalar_fact(ch, region_val, quote=ln)
        if practice_raw:
            out["practice_area"] = _make_scalar_fact(ch, practice_raw, quote=ln)
        return out
    return {}


def _merge_missing(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Merge `src` into `dst` without overwriting non-empty values."""

    def _is_empty(x: Any) -> bool:
        if x in (None, "", {}, []):
            return True
        if isinstance(x, dict):
            if "value" in x:
                return not str(x.get("value") or "").strip()
            if "text" in x:
                return not str(x.get("text") or "").strip()
            # Defensive: treat citation-only or schema-incomplete dicts as empty.
            # This prevents partially formed objects (e.g. {"citations":[...]}) from
            # blocking later deterministic/LLM fills or failing schema validation.
            if "citations" in x:
                return True
        return False

    for k, v in (src or {}).items():
        if k not in dst or _is_empty(dst.get(k)):
            dst[k] = v
            continue
        if isinstance(dst.get(k), dict) and isinstance(v, dict):
            dst[k] = _merge_missing(dst[k], v)
        elif isinstance(dst.get(k), list) and isinstance(v, list):
            # For results indicators, try to append unique items rather than refusing to grow.
            if k == "results_indicators":
                existing = dst.get(k) if isinstance(dst.get(k), list) else []
                incoming = v
                seen = set()
                for it in existing:
                    if isinstance(it, dict):
                        seen.add((it.get("name"), it.get("unit")))
                for it in incoming:
                    if not isinstance(it, dict):
                        continue
                    key = (it.get("name"), it.get("unit"))
                    if key in seen:
                        continue
                    existing.append(it)
                    seen.add(key)
                dst[k] = existing
            elif len(dst[k]) == 0:
                dst[k] = v

    return dst


def _slice_top_keys(obj: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Keep only allowed top-level keys (plus project_id)."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict) and obj.get("project_id"):
        out["project_id"] = obj.get("project_id")
    for k in allowed:
        if isinstance(obj, dict) and k in obj:
            out[k] = obj[k]
    return out


def _sanitize_extraction_obj(obj: dict[str, Any]) -> dict[str, Any]:
    """Remove/repair partially-formed fields before Pydantic validation."""
    if not isinstance(obj, dict):
        return {"project_id": ""}

    obj.setdefault("insufficient_evidence", [])
    if not isinstance(obj.get("insufficient_evidence"), list):
        obj["insufficient_evidence"] = [str(obj.get("insufficient_evidence"))]

    def _flag(path: str) -> None:
        if path and path not in obj["insufficient_evidence"]:
            obj["insufficient_evidence"].append(path)

    # objective must have non-empty text if present
    if "objective" in obj:
        o = obj.get("objective")
        if not isinstance(o, dict) or not str(o.get("text") or "").strip():
            obj["objective"] = None
            _flag("objective")

    # list text fields: keep only items with text
    for k in ["theory_of_change", "risks_and_limitations"]:
        v = obj.get(k)
        if v is None:
            continue
        if not isinstance(v, list):
            obj[k] = []
            _flag(k)
            continue
        cleaned = [it for it in v if isinstance(it, dict) and str(it.get("text") or "").strip()]
        if cleaned != v:
            obj[k] = cleaned
            if not cleaned:
                _flag(k)

    # overview/ratings scalar facts must have value if present
    for top in ["overview", "ratings"]:
        d = obj.get(top)
        if not isinstance(d, dict):
            continue
        for fk, fv in list(d.items()):
            if fv is None:
                continue
            if isinstance(fv, dict) and "value" in fv and not str(fv.get("value") or "").strip():
                d[fk] = None

    return obj


def _parse_isr_overall_ratings(ch: dict[str, Any]) -> dict[str, Any]:
    """Parse ISR-style Overall Ratings block into ProjectRatings dict."""
    txt = str(ch.get("text") or "")
    # Keep stripped lines, but preserve special glyphs like "" for citation quotes.
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    out: dict[str, Any] = {}

    mapping = [
        ("Progress towards achievement of PDO", "pdo_rating"),
        ("Overall Implementation Progress", "ip_rating"),
        ("Overall Risk Rating", "overall_risk_rating"),
        ("Outcome", "outcome_rating"),
    ]

    # Format A (table):
    # label line, then previous rating line, then current rating line.
    for label, key in mapping:
        for i, ln in enumerate(lines):
            if not ln.lower().startswith(label.lower()):
                continue
            # Collect the next two rating-ish lines (often start with bullet glyph)
            vals_clean: list[str] = []
            vals_raw: list[str] = []
            j = i + 1
            while j < len(lines) and len(vals_raw) < 2:
                raw = lines[j]
                clean = _clean_rating(raw)
                if clean and clean.lower() not in {"name", "previous rating", "current rating"}:
                    vals_raw.append(raw)
                    vals_clean.append(clean)
                j += 1
            if vals_clean:
                # Prefer "current" rating line when both are present.
                current_clean = vals_clean[1] if len(vals_clean) >= 2 else vals_clean[0]
                current_quote = vals_raw[1] if len(vals_raw) >= 2 else vals_raw[0]
                out[key] = _make_scalar_fact(ch, current_clean, quote=current_quote)
            break

    # Format B (inline on one line) fallback
    if not out:
        for label, key in mapping:
            m = re.search(rf"{re.escape(label)}\s+(.+)", txt, flags=re.IGNORECASE)
            if not m:
                continue
            rest = _clean_rating(m.group(1))
            if rest:
                out[key] = _make_scalar_fact(ch, rest, quote=m.group(0))

    return out


def _parse_isr_indicator_blocks(ch: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse ISR results tables into ResultsIndicator list.

    We map:
    - baseline = Baseline Value (year from Baseline Date)
    - achieved = Actual (Current) Value (year from Actual Current Date)
    - target = End Target Value (year from End Target Date)
    """
    txt = str(ch.get("text") or "")
    if "Baseline" not in txt or "End Target" not in txt or "Value" not in txt:
        return []

    out: list[dict[str, Any]] = []

    def _to_float(s: str) -> float | None:
        s = (s or "").strip()
        if s in {"--", "—", "-", ""}:
            return None
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None

    def _year_from_date(s: str) -> int | None:
        m = re.search(r"(\d{4})", s or "")
        return int(m.group(1)) if m else None

    # Split into indicator blocks on the ISR glyphs and known section markers.
    # ISR PDFs often use " " for bullet indicators.
    blocks = re.split(r"(?:\n\s*\s*|\n\s*►\s*)", txt)
    for b in blocks[1:]:
        blines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not blines:
            continue
        header = blines[0]
        name = header
        unit = None
        m = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", header)
        if m:
            name = m.group(1).strip()
            unit = m.group(2).split(",")[0].strip()

        # Find indices of Value and Date labels
        try:
            i_val = blines.index("Value")
        except ValueError:
            continue
        vals = blines[i_val + 1 : i_val + 5]
        if len(vals) < 4:
            continue

        base_s, prev_s, cur_s, tgt_s = vals[0], vals[1], vals[2], vals[3]
        base_v, cur_v, tgt_v = _to_float(base_s), _to_float(cur_s), _to_float(tgt_s)
        if base_v is None and cur_v is None and tgt_v is None:
            continue

        years = [None, None, None, None]
        try:
            i_date = blines.index("Date")
            dates = blines[i_date + 1 : i_date + 5]
            if len(dates) >= 4:
                years = [_year_from_date(dates[0]), _year_from_date(dates[1]), _year_from_date(dates[2]), _year_from_date(dates[3])]
        except ValueError:
            pass

        quote_val = "Value\n" + "\n".join(vals)
        citation = _make_citation(ch, quote_val)

        def _num_fact(v: float | None, year: int | None) -> dict[str, Any] | None:
            if v is None and year is None:
                return None
            return {"value": v, "year": year, "citations": [citation]}

        out.append(
            {
                "name": name,
                "unit": unit,
                "baseline": _num_fact(base_v, years[0]),
                "achieved": _num_fact(cur_v, years[2]),
                "target": _num_fact(tgt_v, years[3]),
                "citations": [],
            }
        )

    return out


def _build_prompt(project_id: str, chunks: list[dict[str, Any]]) -> tuple[str, str]:
    system = """You are an information extraction system.

Rules:
- Use ONLY the provided evidence chunks.
- Output MUST be a single JSON object (no markdown).
- Every numeric value or date MUST have at least one citation.
- Each citation must include: doc_id, chunk_id, page, and an exact quote copied from the chunk text.
- If evidence is missing, set the value to null and add the JSONPath-like field path to insufficient_evidence.
- You MUST include project_id exactly as provided.
- For every non-null field in the output, include at least one citation (even for text).
- When citing, copy a short exact quote (5–30 words) from the chunk text.
"""

    user = {
        "task": "Extract decision-ready facts for a World Bank project.",
        "project_id": project_id,
        "output_schema": {
            "project_id": "string",
            "overview": {
                "project_name": {
                    "value": "string",
                    "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}],
                },
                "country": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "region": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "practice_area": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "financing_instrument": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "borrower": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "implementing_agency": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "approval_date": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "effectiveness_date": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "closing_date": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "total_commitment_usd": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "world_bank_financing_usd": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "borrower_contribution_usd": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "total_project_cost_usd": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
            },
            "ratings": {
                "pdo_rating": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "ip_rating": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "overall_risk_rating": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
                "outcome_rating": {"value": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
            },
            "objective": {"text": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]},
            "theory_of_change": [{"text": "string", "citations": [{"doc_id": "str", "chunk_id": "str", "page": "int", "quote": "str"}]}],
            "results_indicators": [
                {
                    "name": "string",
                    "unit": "string|null",
                    "baseline": {"value": "number|null", "year": "int|null", "citations": ["Citation[]"]},
                    "target": {"value": "number|null", "year": "int|null", "citations": ["Citation[]"]},
                    "achieved": {"value": "number|null", "year": "int|null", "citations": ["Citation[]"]},
                    "citations": ["Citation[]"]
                }
            ],
            "risks_and_limitations": [{"text": "string", "citations": ["Citation[]"]}],
            "insufficient_evidence": ["string[]"]
        },
        "evidence_chunks": chunks,
        "instructions": [
            "Prefer indicators that contain baseline/target/achieved.",
            "Do not invent numbers or dates.",
            "If unsure, leave fields null and record insufficient_evidence.",
        ],
    }
    import json

    return system, json.dumps(user, ensure_ascii=False)


@dataclass(frozen=True)
class ExtractProjectResult:
    dataset_version_id: str
    project_id: str
    extraction_id: str
    status: str
    n_chunks_used: int
    n_validation_errors: int


async def extract_project(
    *,
    dataset_version_id: str,
    project_id: str,
    max_chunks: int = 30,
    max_chars_per_chunk: int = 2000,
    temperature: float = 0.0,
    analysis_mode: str = "full",
) -> ExtractProjectResult:
    chunks_df = read_parquet(layer="silver", name="document_text_chunks", dataset_version_id=dataset_version_id)
    chunks_df = chunks_df.filter(pl.col("project_id") == project_id)
    if chunks_df.height == 0:
        raise ValueError(f"No chunks found for project_id={project_id} (did you run process-docs?)")

    # Join document metadata so we can prefer newer/more relevant sources.
    try:
        docs_df = read_parquet(layer="silver", name="documents", dataset_version_id=dataset_version_id).filter(
            pl.col("project_id") == project_id
        )
        docs_df = docs_df.select(["doc_id", "doc_class", "doc_type", "publication_date", "doc_score"])
        docs_df = docs_df.with_columns(
            pl.col("publication_date").str.strptime(pl.Datetime, strict=False).alias("publication_dt")
        )
        chunks_df = chunks_df.join(docs_df, on="doc_id", how="left")
    except Exception:  # noqa: BLE001
        chunks_df = chunks_df.with_columns(pl.lit(None).cast(pl.Datetime).alias("publication_dt"))

    # Rank chunks: prefer "project details" + results pages first, then numeric density.
    # This is important because dashboard-style fields often live in "Basic information / Key Dates / Ratings" tables.
    chunks_df = chunks_df.with_columns(
        [
            pl.col("text")
            .map_elements(lambda s: int(_keyword_score(str(s or ""))), return_dtype=pl.Int64)
            .alias("keyword_score"),
            pl.col("text")
            .map_elements(lambda s: int(_project_id_score(str(s or ""), project_id)), return_dtype=pl.Int64)
            .alias("project_id_score"),
        ]
    )
    ranked_all = chunks_df.sort(
        by=["project_id_score", "keyword_score", "section_hint_score", "digit_ratio", "text_length", "publication_dt"],
        descending=[True, True, True, True, True, True],
        nulls_last=True,
    )

    # Evidence readiness heuristics (fail-fast for under-documented projects).
    # This prevents long, unproductive LLM calls when the corpus is clearly thin.
    def _count_hits(needle: str) -> int:
        try:
            return int(ranked_all.select(pl.col("text").str.contains(needle).sum()).item())
        except Exception:  # noqa: BLE001
            return 0

    n_chunks_total = int(ranked_all.height)
    try:
        n_docs_total = int(ranked_all.select(pl.col("doc_id").n_unique()).item())
    except Exception:  # noqa: BLE001
        n_docs_total = 0
    try:
        total_chars = int(ranked_all.select(pl.col("text_length").sum()).item())
    except Exception:  # noqa: BLE001
        total_chars = 0

    hits_objective = _count_hits("Project Development Objective") + _count_hits("Original PDO")
    hits_results = _count_hits("Baseline") + _count_hits("End Target") + _count_hits("Results Framework")
    hits_risks = _count_hits("lessons learned") + _count_hits("mitigation") + _count_hits("risk")

    # "Thin" means we likely cannot extract anything useful even from a PID.
    # Do NOT treat single-document projects as thin (PID-only projects are common).
    thin_corpus = (n_docs_total == 0) or (n_chunks_total < 3) or (total_chars < 1500)
    # "fast" mode always applies stricter gating; "full" mode still short-circuits if corpus is clearly thin.
    mode = (analysis_mode or "full").strip().lower()
    if mode not in {"fast", "full"}:
        mode = "full"

    # Map chunk_id -> full text for citation validation
    chunks_by_id: dict[str, str] = {}
    for row in ranked_all.iter_rows(named=True):
        chunks_by_id[str(row["chunk_id"])] = str(row.get("text") or "")

    client = OllamaClient()
    extraction_id = str(uuid.uuid4())
    created_at = utc_now().isoformat()
    status = "ok"
    output = ExtractionOutput(project_id=project_id)
    issues = []
    n_llm_chunks_used = 0

    try:
        # Deterministic-first object. We'll fill missing fields from a focused LLM pass.
        obj: dict[str, Any] = {"project_id": project_id}

        # Deterministic bootstrapping for exhaustivity/exactitude:
        # If we can parse datasheet-style fields directly, fill them with exact citations.
        ranked_head = [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "page": int(r["page_start"]),
                "text": str(r.get("text") or ""),
            }
            for r in ranked_all.head(min(160, ranked_all.height)).iter_rows(named=True)
        ]
        seed_chunk = _first_matching_chunk(project_id, ranked_head)
        if seed_chunk:
            overview_seed = _extract_overview_from_chunk(project_id, seed_chunk)
            if overview_seed:
                obj.setdefault("overview", {})
                if isinstance(obj.get("overview"), dict):
                    for k, v in overview_seed.items():
                        existing = obj["overview"].get(k)
                        def _is_empty_scalar(x: Any) -> bool:
                            if x in (None, "", {}):
                                return True
                            if isinstance(x, dict):
                                vv = (x.get("value") or "").strip()
                                cits = x.get("citations") or []
                                return (not vv) or (not isinstance(cits, list)) or (len(cits) == 0)
                            return False

                        if _is_empty_scalar(existing):
                            obj["overview"][k] = v
            if not obj.get("objective"):
                obj["objective"] = _extract_objective_from_chunk(seed_chunk)

        # PID-style bootstrapping (projects in progress often only have PID/ESRS docs).
        if not seed_chunk:
            pid_chunk = _first_pid_chunk(project_id, ranked_head)
            if pid_chunk:
                pid_seed = _extract_overview_from_pid_chunk(project_id, pid_chunk)
                if pid_seed:
                    obj.setdefault("overview", {})
                    if isinstance(obj.get("overview"), dict):
                        # Pull out PID objective text if present.
                        pid_obj_txt = pid_seed.pop("_pid_objective_text", None)
                        pid_obj_quote = pid_seed.pop("_pid_objective_quote", None)
                        for k, v in pid_seed.items():
                            if obj["overview"].get(k) in (None, "", {}):
                                obj["overview"][k] = v
                        if (not obj.get("objective")) and isinstance(pid_obj_txt, str) and pid_obj_txt.strip():
                            # Cite the exact PDO text line(s) we extracted from the chunk.
                            obj["objective"] = {
                                "text": pid_obj_txt.strip(),
                                "citations": [
                                    {
                                        "doc_id": pid_chunk.get("doc_id"),
                                        "chunk_id": pid_chunk.get("chunk_id"),
                                        "page": int(pid_chunk.get("page") or 0) or None,
                                        "quote": (pid_obj_quote or pid_obj_txt).strip()[:500],
                                    }
                                ],
                            }

                # PID financing table is usually on the same page/chunk.
                obj.setdefault("overview", {})
                if isinstance(obj.get("overview"), dict):
                    pid_fin = _extract_financials_from_pid_chunk(project_id, pid_chunk)
                    for k, v in pid_fin.items():
                        if obj["overview"].get(k) in (None, "", {}):
                            obj["overview"][k] = v

        # Deterministic finance parsing (PAD tables are often more reliable than LLM).
        # Only fill missing finance fields, and only from chunks that explicitly label the project id.
        obj.setdefault("overview", {})
        if isinstance(obj.get("overview"), dict):
            finance_keys = {"total_commitment_usd", "world_bank_financing_usd", "borrower_contribution_usd", "total_project_cost_usd"}

            def _is_empty_scalar(x: Any) -> bool:
                if x in (None, "", {}):
                    return True
                if isinstance(x, dict):
                    vv = (x.get("value") or "").strip()
                    cits = x.get("citations") or []
                    return (not vv) or (not isinstance(cits, list)) or (len(cits) == 0)
                return False

            need_any = any(_is_empty_scalar(obj["overview"].get(k)) for k in finance_keys)
            if need_any:
                for r in ranked_all.iter_rows(named=True):
                    txt = str(r.get("text") or "")
                    if not txt:
                        continue
                    if "Total Bank financing" not in txt and "Project Financing Data" not in txt and "Financing Plan" not in txt:
                        continue
                    ch = {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "page": int(r["page_start"]), "text": txt}
                    parsed_fin = _extract_financials_from_chunk(project_id, ch)
                    if not parsed_fin:
                        continue
                    for k, v in parsed_fin.items():
                        if k not in finance_keys:
                            continue
                        if _is_empty_scalar(obj["overview"].get(k)):
                            obj["overview"][k] = v
                    # Stop early if all filled
                    if not any(_is_empty_scalar(obj["overview"].get(k)) for k in finance_keys):
                        break

        # Deterministic KEY DATES parsing (approval/effectiveness/closing)
        if isinstance(obj.get("overview"), dict):
            date_keys = {"approval_date", "effectiveness_date", "closing_date"}
            def _is_empty_scalar(x: Any) -> bool:
                if x in (None, "", {}):
                    return True
                if isinstance(x, dict):
                    vv = (x.get("value") or "").strip()
                    cits = x.get("citations") or []
                    return (not vv) or (not isinstance(cits, list)) or (len(cits) == 0)
                return False
            if any(_is_empty_scalar(obj["overview"].get(k)) for k in date_keys):
                for r in ranked_all.iter_rows(named=True):
                    txt = str(r.get("text") or "")
                    if "KEY DATES" not in txt:
                        continue
                    ch = {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "page": int(r["page_start"]), "text": txt}
                    parsed = _extract_key_dates_from_chunk(project_id, ch)
                    if not parsed:
                        continue
                    for k, v in parsed.items():
                        if k in date_keys and _is_empty_scalar(obj["overview"].get(k)):
                            obj["overview"][k] = v
                    if not any(_is_empty_scalar(obj["overview"].get(k)) for k in date_keys):
                        break

        # Deterministic region/practice area parsing from ISR-style headers.
        if isinstance(obj.get("overview"), dict):
            def _is_empty_scalar(x: Any) -> bool:
                if x in (None, "", {}):
                    return True
                if isinstance(x, dict):
                    vv = (x.get("value") or "").strip()
                    cits = x.get("citations") or []
                    return (not vv) or (not isinstance(cits, list)) or (len(cits) == 0)
                return False
            if _is_empty_scalar(obj["overview"].get("region")) or _is_empty_scalar(obj["overview"].get("practice_area")):
                for r in ranked_all.iter_rows(named=True):
                    txt = str(r.get("text") or "")
                    if "Global Practice" not in txt or "|" not in txt:
                        continue
                    ch = {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "page": int(r["page_start"]), "text": txt}
                    parsed = _extract_region_practice_from_chunk(ch)
                    if not parsed:
                        continue
                    if _is_empty_scalar(obj["overview"].get("region")) and parsed.get("region"):
                        obj["overview"]["region"] = parsed["region"]
                    if _is_empty_scalar(obj["overview"].get("practice_area")) and parsed.get("practice_area"):
                        obj["overview"]["practice_area"] = parsed["practice_area"]
                    if (not _is_empty_scalar(obj["overview"].get("region"))) and (not _is_empty_scalar(obj["overview"].get("practice_area"))):
                        break

        # Deterministic ISR parsing: ratings + numeric indicators from results tables.
        # Scan ranked chunks (already biased towards "results" and "datasheet" pages).
        for r in ranked_all.iter_rows(named=True):
            txt = str(r.get("text") or "")
            if not txt:
                continue
            ch = {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "page": int(r["page_start"]), "text": txt}
            if "Overall Ratings" in txt or "Progress towards achievement of PDO" in txt or "Overall Risk Rating" in txt:
                parsed = _parse_isr_overall_ratings(ch)
                if parsed:
                    obj.setdefault("ratings", {})
                    if isinstance(obj.get("ratings"), dict):
                        for k, v in parsed.items():
                            if obj["ratings"].get(k) in (None, "", {}):
                                obj["ratings"][k] = v

            if "Baseline" in txt and "End Target" in txt and "Value" in txt and ("" in txt or "►" in txt):
                inds = _parse_isr_indicator_blocks(ch)
                if inds:
                    # Append, but avoid duplicates by (name, unit)
                    existing = obj.get("results_indicators") if isinstance(obj.get("results_indicators"), list) else []
                    seen = {(i.get("name"), i.get("unit")) for i in existing if isinstance(i, dict)}
                    for ind in inds:
                        key = (ind.get("name"), ind.get("unit"))
                        if key in seen:
                            continue
                        existing.append(ind)
                        seen.add(key)
                    obj["results_indicators"] = existing

        # Focused LLM extraction: retrieve evidence for remaining/narrative fields from *all* chunks.
        llm_chunk_ids_used: set[str] = set()
        llm_passes_run = 0
        max_llm_passes = 1 if mode == "fast" else 4
        # Hard cap runtime spent waiting on any single pass (independent of Ollama's configured timeout).
        pass_timeout_s = 20.0 if mode == "fast" else 60.0
        # Extra gating: in full mode, avoid spending time if the corpus is clearly thin.
        allow_llm = (mode == "full" and not thin_corpus) or (mode == "fast" and not thin_corpus)

        async def _run_llm_pass(*, pass_name: str, terms: list[str], allowed_keys: set[str], limit: int) -> None:
            nonlocal n_llm_chunks_used, obj, llm_chunk_ids_used
            nonlocal llm_passes_run

            if not allow_llm:
                return
            if llm_passes_run >= max_llm_passes:
                return

            def _focus_score(t: str) -> int:
                tl = (t or "").lower()
                return sum(1 for kw in terms if kw.lower() in tl)

            ranked_focus = (
                ranked_all.with_columns(
                    pl.col("text")
                    .map_elements(lambda s: int(_focus_score(str(s or ""))), return_dtype=pl.Int64)
                    .alias("focus_kw")
                )
                .filter(pl.col("focus_kw") > 0)
                .with_columns(
                    (
                        pl.col("focus_kw") * 10
                        + pl.col("keyword_score")
                        + pl.col("project_id_score")
                        + pl.col("section_hint_score")
                    ).alias("focus_score")
                )
                .sort(["focus_score", "publication_dt", "text_length"], descending=[True, True, True], nulls_last=True)
                .head(limit)
            )
            if ranked_focus.height == 0:
                return

            # Minimum evidence threshold per pass: skip when retrieval is too weak.
            min_focus = 2 if pass_name in {"objective_toc", "risks"} else 1
            if int(ranked_focus.height) < min_focus:
                return

            llm_chunks: list[dict[str, Any]] = []
            for row in ranked_focus.iter_rows(named=True):
                text = str(row.get("text") or "")
                if len(text) > max_chars_per_chunk:
                    text = text[:max_chars_per_chunk] + "\n[TRUNCATED]"
                llm_chunks.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "page": int(row["page_start"]),
                        "text": text,
                    }
                )
                llm_chunk_ids_used.add(str(row["chunk_id"]))

            system, user = _build_prompt(project_id, llm_chunks)
            try:
                resp = await asyncio.wait_for(
                    client.chat_json(system=system, user=user, temperature=temperature),
                    timeout=pass_timeout_s,
                )
            except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ReadTimeout):
                # Don't fail the entire extraction if a single pass times out.
                # Best-effort retry with fewer chunks.
                if len(llm_chunks) > 12:
                    llm_chunks2 = llm_chunks[: max(12, len(llm_chunks) // 2)]
                    system2, user2 = _build_prompt(project_id, llm_chunks2)
                    try:
                        resp = await asyncio.wait_for(
                            client.chat_json(system=system2, user=user2, temperature=temperature),
                            timeout=pass_timeout_s,
                        )
                    except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ReadTimeout):
                        return
                else:
                    return

            llm_obj = extract_json_object(resp.content)
            llm_obj = _coerce_llm_output(project_id, llm_obj)
            llm_obj = _slice_top_keys(llm_obj, allowed_keys | {"insufficient_evidence"})
            obj = _merge_missing(obj, llm_obj)
            n_llm_chunks_used = len(llm_chunk_ids_used)
            llm_passes_run += 1

        # Pass A: Overview gaps (region/practice area/borrower/etc.) and general project facts.
        if not isinstance(obj.get("overview"), dict) or not obj.get("overview"):
            await _run_llm_pass(
                pass_name="overview",
                terms=[
                    "Project Name",
                    "Country",
                    "Region",
                    "Practice Area",
                    "Global Practice",
                    "Borrower",
                    "Implementing Agency",
                    "Lending Instrument",
                    "Financing Instrument",
                ],
                allowed_keys={"overview"},
                limit=max(12, max_chunks // 2),
            )

        # Pass B: Objective + theory of change.
        if (hits_objective > 0) and (not obj.get("objective") or not (obj.get("theory_of_change") or [])):
            await _run_llm_pass(
                pass_name="objective_toc",
                terms=[
                    "Project Development Objective",
                    "Original PDO",
                    "Revised PDO",
                    "PDO",
                    "theory of change",
                    "results chain",
                    "results framework",
                    "outcome",
                    "outputs",
                    "intermediate outcomes",
                    "components",
                    "activities",
                ],
                allowed_keys={"objective", "theory_of_change"},
                limit=max_chunks,
            )

        # Pass C: Risks & limitations narrative.
        if (hits_risks > 0) and (not (obj.get("risks_and_limitations") or [])):
            await _run_llm_pass(
                pass_name="risks",
                terms=[
                    "risk",
                    "mitigation",
                    "challenge",
                    "delay",
                    "issue",
                    "implementation challenges",
                    "constraints",
                    "limitations",
                    "procurement",
                    "financial management",
                    "fiduciary",
                    "safeguards",
                    "resettlement",
                    "environmental",
                    "social",
                    "lessons learned",
                ],
                allowed_keys={"risks_and_limitations"},
                limit=max_chunks,
            )

        # Pass D: Results indicators (backup/augmentation).
        if (hits_results > 0) and (not (obj.get("results_indicators") or [])):
            await _run_llm_pass(
                pass_name="results",
                terms=[
                    "PDO Indicators",
                    "Intermediate Results Indicators",
                    "Results Framework",
                    "Baseline",
                    "End Target",
                    "Actual (Current)",
                    "Value",
                    "Date",
                ],
                allowed_keys={"results_indicators"},
                limit=max_chunks,
            )

        obj = _sanitize_extraction_obj(obj)
        output = ExtractionOutput.model_validate(obj)
        issues = validate_extraction(output, chunks_by_id)
        if has_errors(issues):
            status = "validation_failed"
    finally:
        await client.aclose()

    # Persist
    import json

    out_df = pl.from_dicts(
        [
            {
                "extraction_id": extraction_id,
                "project_id": project_id,
                "dataset_version_id": dataset_version_id,
                "prompt_version": PROMPT_VERSION,
                "ollama_model": client.model,
                "created_at": created_at,
                "status": status,
                "n_chunks_used": int(n_llm_chunks_used),
                "output_json": output.model_dump_json(),
                "validation_issues_json": json.dumps([i.model_dump() for i in issues], ensure_ascii=False),
            }
        ]
    )
    append_parquet(out_df, layer="gold", name="extractions", dataset_version_id=dataset_version_id)

    write_manifest(
        dataset_version_id,
        {
            "pipeline": "extract_project",
            "project_id": project_id,
            "extraction_id": extraction_id,
            "prompt_version": PROMPT_VERSION,
            "status": status,
            "counts": {
                "chunks_used": int(n_llm_chunks_used),
                "validation_errors": sum(1 for i in issues if i.severity == "error"),
            },
        },
    )

    return ExtractProjectResult(
        dataset_version_id=dataset_version_id,
        project_id=project_id,
        extraction_id=extraction_id,
        status=status,
        n_chunks_used=int(n_llm_chunks_used),
        n_validation_errors=sum(1 for i in issues if i.severity == "error"),
    )

