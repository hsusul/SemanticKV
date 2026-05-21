from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from semantic_kv.cache.models import SemanticType
from semantic_kv.traces.models import ServingTrace, TracePromptBlock, TraceRequest
from semantic_kv.utils.hashing import stable_hash


class TraceLoadError(ValueError):
    pass


def load_trace(path: str | Path) -> ServingTrace:
    trace_path = Path(path)
    if not trace_path.exists():
        raise TraceLoadError(f"Trace file does not exist: {trace_path}")
    suffix = trace_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_trace(trace_path)
    if suffix == ".json":
        return load_json_trace(trace_path)
    if suffix == ".csv":
        return load_csv_trace(trace_path)
    raise TraceLoadError(f"Unsupported trace format '{suffix}'. Expected .jsonl, .json, or .csv")


def load_json_trace(path: Path) -> ServingTrace:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ServingTrace.model_validate(raw)
    except json.JSONDecodeError as exc:
        raise TraceLoadError(f"Invalid JSON in {path}: {exc}") from exc
    except ValidationError as exc:
        raise TraceLoadError(f"Invalid ServingTrace in {path}: {exc}") from exc


def load_jsonl_trace(path: Path) -> ServingTrace:
    requests: list[TraceRequest] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            requests.append(TraceRequest.model_validate_json(line))
        except ValidationError as exc:
            raise TraceLoadError(f"Invalid TraceRequest at {path}:{line_no}: {exc}") from exc
    if not requests:
        raise TraceLoadError(f"JSONL trace contains no requests: {path}")
    return ServingTrace(
        trace_id=path.stem,
        backend_name="jsonl_trace",
        model_name=requests[0].model_name,
        requests=sorted(requests, key=lambda req: (req.timestamp_ms, req.request_id)),
        metadata={"source_path": str(path)},
    )


def load_csv_trace(path: Path) -> ServingTrace:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise TraceLoadError(f"CSV trace contains no rows: {path}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("request_id"):
            raise TraceLoadError("CSV trace requires request_id column for every row")
        grouped[row["request_id"]].append(row)

    requests: list[TraceRequest] = []
    for request_id, request_rows in grouped.items():
        first = request_rows[0]
        blocks = []
        for idx, row in enumerate(request_rows):
            try:
                semantic_type = SemanticType(row["semantic_type"])
                token_count = int(row["token_count"])
            except KeyError as exc:
                raise TraceLoadError(f"CSV trace missing required column: {exc}") from exc
            except ValueError as exc:
                raise TraceLoadError(f"Invalid block row for request {request_id}: {row}") from exc
            text = row.get("text") or None
            content_hash = row.get("content_hash") or (stable_hash(semantic_type.value, text) if text else None)
            blocks.append(
                TracePromptBlock(
                    block_id=row.get("block_id") or None,
                    content_hash=content_hash,
                    semantic_type=semantic_type,
                    token_count=token_count,
                    text=text,
                    position=int(row.get("position") or idx),
                )
            )
        try:
            requests.append(
                TraceRequest(
                    request_id=request_id,
                    timestamp_ms=int(first.get("timestamp_ms") or 0),
                    session_id=first.get("session_id") or None,
                    tenant_id=first.get("tenant_id") or None,
                    model_name=first.get("model_name") or None,
                    blocks=blocks,
                    observed_ttft_ms=_optional_float(first.get("observed_ttft_ms")),
                    observed_total_latency_ms=_optional_float(first.get("observed_total_latency_ms")),
                    observed_prefix_hit_tokens=_optional_int(first.get("observed_prefix_hit_tokens")),
                    observed_prefill_tokens=_optional_int(first.get("observed_prefill_tokens")),
                )
            )
        except ValidationError as exc:
            raise TraceLoadError(f"Invalid CSV request group {request_id}: {exc}") from exc
    return ServingTrace(
        trace_id=path.stem,
        backend_name="csv_trace",
        model_name=requests[0].model_name,
        requests=sorted(requests, key=lambda req: (req.timestamp_ms, req.request_id)),
        metadata={"source_path": str(path)},
    )


def save_trace_json(path: str | Path, trace: ServingTrace) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(trace.model_dump_json(indent=2), encoding="utf-8")


def save_trace_jsonl(path: str | Path, trace: ServingTrace) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for request in trace.requests:
            handle.write(request.model_dump_json())
            handle.write("\n")


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def _optional_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None

