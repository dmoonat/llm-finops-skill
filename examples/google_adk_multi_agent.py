# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""Reference Example: Hierarchical Google ADK Multi-Agent System with adk-finops.

Demonstrates:
1. Attaching `FinOpsCostPlugin` to a Google ADK `App`.
2. Hierarchical Root (Parent) Agent -> Sub-Agent cost attribution.
3. Explicit tool billing (`@billable` fixed and dynamic fees) and cloud grounding.
4. Pre-flight (`<0.5ms`) and post-call budget guardrails (`halt`, `downgrade`, `warn`).
5. Task outcome & wasted spend analytics (`record_task_status` + `print_task_efficiency_summary`).
"""

from __future__ import annotations

import asyncio
from typing import Any

from adk_finops import (
    FinOpsCostPlugin,
    billable,
    print_summary,
    print_task_efficiency_summary,
)


# ==============================================================================
# 1. Billable & Free Tools
# ==============================================================================


@billable(fee=0.015, provider="serpapi")
def patent_search(query: str) -> dict[str, Any]:
    """Paid external patent search tool ($0.015 per successful invocation)."""
    return {
        "status": "success",
        "query": query,
        "hits": ["US-2026-0192831", "US-2026-0441120"],
    }


@billable(
    fee_fn=lambda args, res: 0.002 * len(res.get("pages", [])),
    provider="document_ai",
    charge_on_error=False,
)
def ocr_document(gcs_uri: str) -> dict[str, Any]:
    """Dynamic per-page OCR tool ($0.002/page; $0.00 if result is an error)."""
    return {
        "status": "success",
        "gcs_uri": gcs_uri,
        "pages": ["Page 1 text...", "Page 2 text...", "Page 3 text..."],
    }


def format_markdown_report(title: str, body: str) -> dict[str, str]:
    """Free local utility tool ($0.00 tool fee)."""
    return {"status": "success", "markdown": f"# {title}\n\n{body}"}


# ==============================================================================
# 2. ADK Agent Hierarchy & FinOpsCostPlugin Setup
# ==============================================================================


def build_adk_app() -> tuple[Any, FinOpsCostPlugin]:
    """Constructs a hierarchical Google ADK App wired with FinOpsCostPlugin."""
    from google.adk.agents import Agent
    from google.adk.apps import App

    researcher = Agent(
        name="researcher",
        model="gemini-2.5-flash",
        instruction="Use patent_search and ocr_document to gather technical evidence.",
        tools=[patent_search, ocr_document],
    )

    writer = Agent(
        name="writer",
        model="gemini-2.5-pro",
        instruction="Synthesize findings using format_markdown_report.",
        tools=[format_markdown_report],
    )

    supervisor = Agent(
        name="supervisor",
        model="gemini-2.5-pro",
        instruction="Coordinate researcher and writer to answer user requests.",
        sub_agents=[researcher, writer],
    )

    finops_plugin = FinOpsCostPlugin(
        name="finops_cost_tracker",
        default_model="gemini-2.5-pro",
        budget_limit_usd=2.00,
        turn_budget_limit_usd=0.50,
        agent_budgets={
            "researcher": 0.40,
            "writer": 0.80,
        },
        max_prompt_tokens=200_000,
        preflight_budget_guard=True,
        on_budget_exceeded="downgrade",
        fallback_model="gemini-2.5-flash",
        render_terminal_box=True,
        enable_optimization_advisor=True,
        jsonl_path="logs/finops_costs.jsonl",
        export_tags={"env": "dev", "team": "ip_analytics"},
    )

    app = App(
        name="ip_analytics_app",
        root_agent=supervisor,
        plugins=[finops_plugin],
    )
    return app, finops_plugin


# ==============================================================================
# 3. Post-Run Outcome Verification Helper
# ==============================================================================


def is_valid_business_outcome(response_text: str) -> bool:
    """Checks that the agent returned an actionable answer rather than a polite refusal."""
    text = (response_text or "").strip().lower()
    if not text or "cannot help" in text or "unable to" in text:
        return False
    return True


async def run_agent_turn(
    runner: Any,
    finops_plugin: FinOpsCostPlugin,
    *,
    user_id: str,
    session_id: str,
    prompt: str,
) -> str:
    """Runs an ADK turn with semantic task outcome tracking."""
    from google.genai import types as genai_types

    final_text = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=prompt)],
            ),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text += "".join(p.text or "" for p in event.content.parts)

        if not is_valid_business_outcome(final_text):
            finops_plugin.record_task_status(
                session_id=session_id,
                status="failed",
                error="Agent returned 200 OK but failed business outcome validation",
            )
    except Exception as exc:
        finops_plugin.record_task_status(
            session_id=session_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        print_summary(session_id=session_id)
        raise

    return final_text


if __name__ == "__main__":
    # Offline simulation when running without live Vertex AI / Gemini credentials
    from adk_finops import CostTracker

    CostTracker.reset()
    CostTracker.register_agent_hierarchy("researcher", parent_agent_name="supervisor", root_agent_name="supervisor")
    CostTracker.register_agent_hierarchy("writer", parent_agent_name="supervisor", root_agent_name="supervisor")
    CostTracker.set_budget(
        session_id="demo_sess_1",
        session_limit_usd=2.00,
        turn_limit_usd=0.50,
        agent_limits_usd={"researcher": 0.40, "writer": 0.80},
    )
    CostTracker.start_turn(turn_id="turn_1", session_id="demo_sess_1", root_agent_name="supervisor")
    CostTracker.record_usage(
        run_id="turn_1",
        session_id="demo_sess_1",
        model_name="gemini-2.5-pro",
        prompt_tokens=1500,
        completion_tokens=180,
        thoughts_tokens=120,
        latency_ms=680.0,
        ttft_ms=140.0,
        agent_name="supervisor",
    )
    CostTracker.record_usage(
        run_id="turn_1",
        session_id="demo_sess_1",
        model_name="gemini-2.5-flash",
        prompt_tokens=6000,
        completion_tokens=400,
        cached_tokens=4000,
        latency_ms=420.0,
        ttft_ms=85.0,
        agent_name="researcher",
    )
    CostTracker.record_tool_call(
        run_id="turn_1",
        session_id="demo_sess_1",
        tool_name="patent_search",
        custom_cost_usd=0.015,
        latency_ms=95.0,
        agent_name="researcher",
    )
    CostTracker.record_turn_latency(run_id="turn_1", session_id="demo_sess_1", latency_ms=1210.0)
    CostTracker.record_task_status(session_id="demo_sess_1", run_id="turn_1", status="success")
    CostTracker.print_summary(run_id="turn_1", session_id="demo_sess_1", pop=False)
    print_task_efficiency_summary()
