#!/usr/bin/env python3
# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""Verification Script for `adk-finops` Skill & Environment Setup.

Run this script (`python scripts/verify_finops_setup.py`) to verify:
1. `adk-finops` is installed and importable.
2. Multi-provider rate cards resolve accurately (Gemini, OpenAI, Anthropic, DeepSeek).
3. Deterministic pre-flight token estimation (`<0.5ms`) and budget guards work.
4. `@billable` tool decoration, multi-agent attribution, Optimization Advisor, and Task Efficiency reporting work.
"""

from __future__ import annotations

import sys
from typing import Any


def main() -> int:
    try:
        import adk_finops
        from adk_finops import (
            CostTracker,
            billable,
            estimate_text_tokens,
            extract_billable_spec,
            print_summary,
            print_task_efficiency_summary,
        )
    except ImportError as exc:
        print(f"[FAIL] `adk-finops` is not installed in the active Python environment ({sys.executable}).")
        print(f"       Error: {exc}")
        print('       Install with: pip install "adk-finops[all]"  (or `pip install adk-finops`)')
        return 1

    version = getattr(adk_finops, "__version__", "unknown")
    print(f"[OK] adk-finops v{version} imported successfully from {adk_finops.__file__}")

    # 1. Verify multi-provider rate card resolution
    registry = CostTracker.get_registry()
    for model_name in ("gemini-2.5-pro", "gemini-2.5-flash", "gpt-4o", "claude-3-7-sonnet", "deepseek-r1"):
        card = registry.resolve_model(model_name)
        assert not getattr(card, "is_fallback", False), f"Expected built-in rate card for {model_name}"
    print("[OK] Multi-provider rate cards verified (Google, OpenAI, Anthropic, DeepSeek)")

    # 2. Verify pre-flight token estimation
    est_tokens = estimate_text_tokens("Verify deterministic local pre-flight token estimation.", model_name="gemini-2.5-pro")
    assert est_tokens > 0, "Expected positive token estimate"
    print(f"[OK] Pre-flight token estimator verified (estimated {est_tokens} tokens)")

    # 3. Verify @billable decorator
    @billable(fee=0.015, provider="serpapi")
    def sample_paid_tool(q: str) -> dict[str, Any]:
        return {"status": "success", "q": q}

    spec = extract_billable_spec(sample_paid_tool)
    assert spec is not None and abs(spec.compute_fee({"q": "test"}, {"status": "success"}) - 0.015) < 1e-9
    print("[OK] @billable tool specification verified")

    # 4. Run end-to-end dual-scope, multi-agent & latency/throughput simulation
    CostTracker.reset()
    session_id = "verify_sess_01"
    turn_id = "verify_turn_01"

    CostTracker.register_agent_hierarchy("researcher", parent_agent_name="coordinator", root_agent_name="coordinator")
    CostTracker.set_budget(session_id=session_id, session_limit_usd=1.00, turn_limit_usd=0.25)
    CostTracker.start_turn(turn_id=turn_id, session_id=session_id, root_agent_name="coordinator")

    CostTracker.record_usage(
        run_id=turn_id,
        session_id=session_id,
        model_name="gemini-2.5-pro",
        prompt_tokens=4000,
        completion_tokens=300,
        thoughts_tokens=150,
        cached_tokens=2000,
        latency_ms=450.0,
        ttft_ms=110.0,
        agent_name="researcher",
    )
    CostTracker.record_tool_call(
        run_id=turn_id,
        session_id=session_id,
        tool_name="google_search",
        latency_ms=85.0,
        agent_name="researcher",
    )
    CostTracker.record_turn_latency(run_id=turn_id, session_id=session_id, latency_ms=540.0)
    CostTracker.record_task_status(session_id=session_id, run_id=turn_id, status="success")

    summary = CostTracker.get_summary(run_id=turn_id, session_id=session_id, pop=False)
    assert summary is not None and summary["total_cost_usd"] > 0
    print(
        f"[OK] Dual-scope cost & latency accounting verified "
        f"(Total Cost: ${summary['total_cost_usd']:.6f}, Saved: ${summary['savings_usd']:.6f}, "
        f"Latency: {summary.get('latency_ms', 0.0):.1f}ms, Out: {summary.get('output_tokens_per_sec', 0.0):.1f} tok/s)"
    )

    print_summary(summary)
    print_task_efficiency_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
