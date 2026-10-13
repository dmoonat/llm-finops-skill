# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""Reference Example: FastAPI Middleware for Request-Scoped LLM & Agent Cost Tracking.

Automatically initializes a `CostTracker` run per incoming HTTP request,
records task outcome (`status="success"` vs `"error"`), injects `X-FinOps-*`
HTTP response headers, and exports telemetry to JSONL/CSV/OpenTelemetry/BigQuery.
"""

from __future__ import annotations

import uuid
from typing import Any

from adk_finops import CostTracker
from adk_finops.exporters import JSONLExporter

try:
    from fastapi import FastAPI, Request, Response
except ImportError:
    FastAPI = Any  # type: ignore[misc,assignment]
    Request = Any  # type: ignore[misc,assignment]
    Response = Any  # type: ignore[misc,assignment]


def register_finops_middleware(
    app: Any,
    *,
    jsonl_path: str | None = "logs/finops_costs.jsonl",
    export_tags: dict[str, Any] | None = None,
) -> None:
    """Attaches `adk-finops` request tracking middleware to a FastAPI application."""
    exporter = JSONLExporter(jsonl_path) if jsonl_path else None

    @app.middleware("http")
    async def finops_cost_middleware(request: Request, call_next: Any) -> Response:
        run_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        session_id = request.headers.get("X-Session-ID") or run_id
        request.state.finops_run_id = run_id
        request.state.finops_session_id = session_id

        CostTracker.start_turn(turn_id=run_id, session_id=session_id)
        response: Response | None = None
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 200)
            if status_code >= 400:
                CostTracker.record_task_status(
                    session_id=session_id,
                    run_id=run_id,
                    status="error",
                    error=f"HTTP {status_code}",
                )
            else:
                CostTracker.record_task_status(
                    session_id=session_id,
                    run_id=run_id,
                    status="success",
                )
            return response
        except Exception as exc:
            CostTracker.record_task_status(
                session_id=session_id,
                run_id=run_id,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            CostTracker.record_turn_latency(run_id=run_id, session_id=session_id)
            summary = CostTracker.get_summary(run_id=run_id, session_id=session_id, pop=True)
            if summary:
                turn_data = summary.get("turn", summary)
                if response is not None and hasattr(response, "headers"):
                    response.headers["X-FinOps-Turn-Cost-USD"] = f"{turn_data.get('total_cost_usd', 0.0):.6f}"
                    response.headers["X-FinOps-Session-Cost-USD"] = f"{summary.get('total_cost_usd', 0.0):.6f}"
                    response.headers["X-FinOps-Turn-Tokens"] = str(turn_data.get("total_tokens", 0))
                    response.headers["X-FinOps-Savings-USD"] = f"{summary.get('savings_usd', 0.0):.6f}"
                    response.headers["X-FinOps-Latency-Ms"] = f"{turn_data.get('latency_ms', 0.0):.2f}"
                    response.headers["X-FinOps-Throughput-Tok-Per-Sec"] = f"{turn_data.get('output_tokens_per_sec', 0.0):.2f}"
                if exporter is not None:
                    exporter.export_summary(summary, scope="turn", tags=export_tags)
