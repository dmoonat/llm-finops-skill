# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""Reference Example: Drop-in Adapters for Any Agent Framework & LLM SDK.

Provides reusable functions to wire `adk-finops` (`CostTracker`, `@billable`,
pre-flight budget checks, and task outcome tracking) into:
- Direct LLM SDKs: Google GenAI (`google-genai`), OpenAI (`openai`), Anthropic (`anthropic`), DeepSeek, LiteLLM
- OpenAI Agents SDK (`agents.Runner`)
- PydanticAI (`pydantic_ai.Agent`)
- CrewAI (`crewai.Crew`)
- AutoGen / AG2 (`autogen`)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from adk_finops import (
    BudgetExceededError,
    CostTracker,
    billable,
    estimate_text_tokens,
    extract_billable_spec,
    is_error_tool_result,
)


def guard_prompt_before_llm_call(
    prompt_text: str,
    *,
    model_name: str,
    run_id: str,
    session_id: str | None = None,
    agent_name: str | None = None,
    on_exceeded: str = "halt",
    fallback_model: str = "gemini-2.5-flash",
) -> str:
    """Runs `<0.5ms` local pre-flight token & budget estimation before calling any LLM SDK.

    Returns the effective `model_name` to call (switching to `fallback_model` if `on_exceeded="downgrade"`),
    or raises `BudgetExceededError` if `on_exceeded="halt"` and the budget would be breached.
    """
    est_tokens = estimate_text_tokens(prompt_text, model_name=model_name)
    exceeded, reason, status = CostTracker.check_preflight_budget(
        model_name=model_name,
        estimated_prompt_tokens=est_tokens,
        run_id=run_id,
        session_id=session_id,
        agent_name=agent_name,
    )
    if not exceeded:
        return model_name

    mode = on_exceeded.lower().strip()
    if mode == "downgrade":
        CostTracker.record_preflight_event(
            run_id=run_id,
            session_id=session_id,
            action="downgraded",
            estimated_prompt_tokens=est_tokens,
            avoided_cost_usd=float(status.get("estimated_call_cost_usd", 0.0)),
        )
        return fallback_model
    if mode == "halt":
        CostTracker.record_preflight_event(
            run_id=run_id,
            session_id=session_id,
            action="blocked",
            estimated_prompt_tokens=est_tokens,
            avoided_cost_usd=float(status.get("estimated_call_cost_usd", 0.0)),
        )
        CostTracker.record_task_status(
            session_id=session_id or run_id,
            run_id=run_id,
            status="budget_exceeded",
            error=reason,
        )
        raise BudgetExceededError(
            reason or "Pre-flight budget exceeded",
            current_cost_usd=float(status.get("current_cost_usd", 0.0)),
            budget_limit_usd=float(status.get("budget_limit_usd") or 0.0),
            scope=str(status.get("scope") or "session"),
            session_id=session_id or run_id,
        )
    return model_name


def record_sdk_response(
    response: Any,
    *,
    model_name: str,
    run_id: str,
    session_id: str | None = None,
    agent_name: str | None = None,
    task_name: str | None = None,
    latency_ms: float | None = None,
    ttft_ms: float | None = None,
) -> dict[str, Any]:
    """Universal response recorder for Google GenAI, OpenAI, Anthropic, DeepSeek, and LiteLLM."""
    prompt_tokens = 0
    completion_tokens = 0
    thoughts_tokens = 0
    cached_tokens = 0

    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is not None:
        prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
        thoughts_tokens = int(getattr(usage_meta, "thoughts_token_count", 0) or 0)
        cached_tokens = int(getattr(usage_meta, "cached_content_token_count", 0) or 0)
    elif hasattr(response, "usage") and response.usage is not None:
        u = response.usage
        if hasattr(u, "input_tokens") and not hasattr(u, "prompt_tokens"):
            # Anthropic Messages API
            uncached_in = int(getattr(u, "input_tokens", 0) or 0)
            cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
            cached_tokens = cache_read
            prompt_tokens = uncached_in + cache_read
            completion_tokens = int(getattr(u, "output_tokens", 0) or 0)
        else:
            # OpenAI / DeepSeek / LiteLLM
            prompt_tokens = int(getattr(u, "prompt_tokens", 0) or 0)
            raw_completion = int(getattr(u, "completion_tokens", 0) or 0)
            comp_details = getattr(u, "completion_tokens_details", None)
            if comp_details is not None:
                thoughts_tokens = int(getattr(comp_details, "reasoning_tokens", 0) or 0)
            prompt_details = getattr(u, "prompt_tokens_details", None)
            if prompt_details is not None:
                cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
            elif hasattr(u, "prompt_cache_hit_tokens"):
                cached_tokens = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
            completion_tokens = max(0, raw_completion - thoughts_tokens)

    return CostTracker.record_usage(
        run_id=run_id,
        session_id=session_id,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        thoughts_tokens=thoughts_tokens,
        cached_tokens=cached_tokens,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        agent_name=agent_name,
        task_name=task_name,
    )


def invoke_tracked_tool(
    tool_fn: Callable[..., Any],
    tool_args: dict[str, Any],
    *,
    run_id: str,
    session_id: str | None = None,
    agent_name: str | None = None,
) -> Any:
    """Executes a tool function and automatically applies `@billable` or registered tool fees and latency."""
    import time

    start_ns = time.perf_counter_ns()
    result = tool_fn(**tool_args)
    latency_ms = round(max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000.0), 2)
    spec = extract_billable_spec(tool_fn)
    if spec is not None:
        if spec.charge_on_error or not is_error_tool_result(result):
            fee = spec.compute_fee(tool_args, result)
            CostTracker.record_tool_call(
                run_id=run_id,
                session_id=session_id,
                tool_name=tool_fn.__name__,
                custom_cost_usd=fee,
                latency_ms=latency_ms,
                agent_name=agent_name,
                provider=spec.provider,
            )
    elif not is_error_tool_result(result):
        CostTracker.record_tool_call(
            run_id=run_id,
            session_id=session_id,
            tool_name=tool_fn.__name__,
            latency_ms=latency_ms,
            agent_name=agent_name,
        )
    return result


if __name__ == "__main__":

    @billable(fee=0.02, provider="financial_data")
    def fetch_sec_filing(ticker: str) -> dict[str, Any]:
        return {"status": "success", "ticker": ticker, "filing": "10-K"}

    CostTracker.reset()
    session_id = "multi_sdk_sess_1"
    turn_id = "turn_1"

    CostTracker.set_budget(session_id=session_id, session_limit_usd=0.50)
    CostTracker.start_turn(turn_id=turn_id, session_id=session_id, root_agent_name="orchestrator")

    active_model = guard_prompt_before_llm_call(
        "Analyze AAPL 10-K filing and compute free cash flow margin.",
        model_name="gpt-4o",
        run_id=turn_id,
        session_id=session_id,
        agent_name="financial_analyst",
        on_exceeded="downgrade",
        fallback_model="gpt-4o-mini",
    )

    invoke_tracked_tool(
        fetch_sec_filing,
        {"ticker": "AAPL"},
        run_id=turn_id,
        session_id=session_id,
        agent_name="financial_analyst",
    )

    with CostTracker.track_llm_call(
        turn_id, model_name=active_model, session_id=session_id, agent_name="financial_analyst"
    ) as call:
        call.mark_first_token()
        call.set_usage(
            prompt_tokens=3200,
            completion_tokens=410,
            cached_tokens=1600,
        )

    CostTracker.record_turn_latency(run_id=turn_id, session_id=session_id)
    CostTracker.record_task_status(session_id=session_id, run_id=turn_id, status="success")
    CostTracker.print_summary(run_id=turn_id, session_id=session_id, pop=False)
