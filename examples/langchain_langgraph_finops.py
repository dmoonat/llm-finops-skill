# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""Reference Example: LangChain & LangGraph Cost Tracking with adk-finops.

Works with any LangChain chat model (`ChatGoogleGenerativeAI`, `ChatVertexAI`,
`ChatOpenAI`, `ChatAnthropic`, `ChatDeepSeek`) and any LangGraph multi-agent workflow.
"""

from __future__ import annotations

import time
from typing import Any

from adk_finops import (
    CostTracker,
    extract_billable_spec,
    is_error_tool_result,
    print_task_efficiency_summary,
)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Fallback shim when langchain-core is not installed."""

    LLMResult = Any  # type: ignore[misc,assignment]


class AdkFinOpsCallbackHandler(BaseCallbackHandler):
    """LangChain / LangGraph callback handler backed by `adk_finops.CostTracker`."""

    def __init__(
        self,
        turn_id: str,
        session_id: str | None = None,
        agent_name: str = "langgraph_agent",
        root_agent_name: str | None = None,
        tools_by_name: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.turn_id = turn_id
        self.session_id = session_id or turn_id
        self.agent_name = agent_name
        self.root_agent_name = root_agent_name or agent_name
        self.tools_by_name = tools_by_name or {}
        self._llm_start_ns: dict[str, int] = {}
        self._llm_ttft_ms: dict[str, float] = {}
        self._tool_start_ns: dict[str, int] = {}

        CostTracker.register_agent_hierarchy(
            agent_name=self.agent_name,
            parent_agent_name=self.root_agent_name if self.agent_name != self.root_agent_name else None,
            root_agent_name=self.root_agent_name,
        )
        CostTracker.start_turn(
            turn_id=self.turn_id,
            session_id=self.session_id,
            root_agent_name=self.root_agent_name,
        )

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any = None, **kwargs: Any
    ) -> None:
        """Records LLM start timestamp to measure call latency and streaming TTFT."""
        key = str(run_id or "default")
        self._llm_start_ns[key] = time.perf_counter_ns()
        self._llm_ttft_ms.pop(key, None)

    def on_llm_new_token(self, token: str, *, run_id: Any = None, **kwargs: Any) -> None:
        """Captures streaming Time-To-First-Token (TTFT) on the first streamed chunk."""
        key = str(run_id or "default")
        if key not in self._llm_ttft_ms and key in self._llm_start_ns:
            self._llm_ttft_ms[key] = round(
                max(0.0, (time.perf_counter_ns() - self._llm_start_ns[key]) / 1_000_000.0), 2
            )

    def on_llm_end(self, response: LLMResult, *, run_id: Any = None, **kwargs: Any) -> None:
        """Extracts token usage, latency, and TTFT across Gemini, OpenAI, Anthropic, and DeepSeek models."""
        key = str(run_id or "default")
        start_ns = self._llm_start_ns.pop(key, None)
        ttft_ms = self._llm_ttft_ms.pop(key, None)
        latency_ms = (
            round(max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000.0), 2)
            if start_ns is not None
            else None
        )

        llm_out = getattr(response, "llm_output", None) or {}
        for generations in getattr(response, "generations", []):
            for gen in generations:
                msg = getattr(gen, "message", None)
                usage_meta = getattr(msg, "usage_metadata", None) or {}
                gen_info = getattr(gen, "generation_info", None) or {}
                token_usage = (
                    gen_info.get("usage_metadata")
                    or gen_info.get("token_usage")
                    or llm_out.get("token_usage")
                    or {}
                )

                model_name = (
                    (getattr(msg, "response_metadata", None) or {}).get("model_name")
                    or (getattr(msg, "response_metadata", None) or {}).get("model")
                    or llm_out.get("model_name")
                    or gen_info.get("model_name")
                    or "gemini-2.5-flash"
                )

                prompt_tokens = int(
                    usage_meta.get("input_tokens")
                    or token_usage.get("prompt_tokens")
                    or token_usage.get("prompt_token_count")
                    or 0
                )
                raw_output = int(
                    usage_meta.get("output_tokens")
                    or token_usage.get("completion_tokens")
                    or token_usage.get("candidates_token_count")
                    or 0
                )

                out_details = (
                    usage_meta.get("output_token_details")
                    or token_usage.get("completion_tokens_details")
                    or {}
                )
                in_details = (
                    usage_meta.get("input_token_details")
                    or token_usage.get("prompt_tokens_details")
                    or {}
                )

                thoughts_tokens = int(
                    out_details.get("reasoning")
                    or out_details.get("reasoning_tokens")
                    or token_usage.get("thoughts_token_count")
                    or 0
                )
                cached_tokens = int(
                    in_details.get("cache_read")
                    or in_details.get("cached_tokens")
                    or token_usage.get("cached_content_token_count")
                    or token_usage.get("prompt_cache_hit_tokens")
                    or 0
                )
                completion_tokens = max(0, raw_output - thoughts_tokens)

                if prompt_tokens or completion_tokens or thoughts_tokens:
                    CostTracker.record_usage(
                        run_id=self.turn_id,
                        session_id=self.session_id,
                        model_name=str(model_name),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        thoughts_tokens=thoughts_tokens,
                        cached_tokens=cached_tokens,
                        latency_ms=latency_ms,
                        ttft_ms=ttft_ms,
                        agent_name=self.agent_name,
                    )

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *, run_id: Any = None, **kwargs: Any
    ) -> None:
        """Records tool start timestamp to measure tool execution latency."""
        self._tool_start_ns[str(run_id or "default")] = time.perf_counter_ns()

    def on_tool_end(
        self, output: Any, *, name: str = "tool", run_id: Any = None, **kwargs: Any
    ) -> None:
        """Records tool calls, execution latency, and evaluates `@billable` specs if attached."""
        start_ns = self._tool_start_ns.pop(str(run_id or "default"), None)
        tool_lat_ms = (
            round(max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000.0), 2)
            if start_ns is not None
            else None
        )
        if is_error_tool_result(output):
            return

        tool_obj = self.tools_by_name.get(name)
        spec = extract_billable_spec(tool_obj) if tool_obj is not None else None
        if spec is not None:
            fee = spec.compute_fee(kwargs.get("inputs") or {}, output)
            CostTracker.record_tool_call(
                run_id=self.turn_id,
                session_id=self.session_id,
                tool_name=name,
                custom_cost_usd=fee,
                latency_ms=tool_lat_ms,
                agent_name=self.agent_name,
                provider=spec.provider,
            )
        else:
            CostTracker.record_tool_call(
                run_id=self.turn_id,
                session_id=self.session_id,
                tool_name=name,
                latency_ms=tool_lat_ms,
                agent_name=self.agent_name,
            )

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Marks the session as errored when a LangChain/LangGraph node raises."""
        CostTracker.record_task_status(
            session_id=self.session_id,
            run_id=self.turn_id,
            status="error",
            error=f"{type(error).__name__}: {error}",
        )


if __name__ == "__main__":
    # Quick demonstration simulating a multi-node LangGraph run across Claude & GPT-4o
    CostTracker.reset()
    handler_planner = AdkFinOpsCallbackHandler(
        turn_id="lg_turn_1",
        session_id="lg_sess_1",
        agent_name="planner_node",
        root_agent_name="langgraph_supervisor",
    )
    CostTracker.record_usage(
        run_id="lg_turn_1",
        session_id="lg_sess_1",
        model_name="claude-3-7-sonnet",
        prompt_tokens=5000,
        completion_tokens=450,
        cached_tokens=3500,
        latency_ms=820.5,
        ttft_ms=145.0,
        agent_name="planner_node",
    )
    CostTracker.record_usage(
        run_id="lg_turn_1",
        session_id="lg_sess_1",
        model_name="gpt-4o-mini",
        prompt_tokens=2200,
        completion_tokens=300,
        latency_ms=410.2,
        ttft_ms=90.0,
        agent_name="executor_node",
    )
    CostTracker.record_turn_latency(run_id="lg_turn_1", session_id="lg_sess_1", latency_ms=1250.0)
    CostTracker.record_task_status(session_id="lg_sess_1", run_id="lg_turn_1", status="success")
    CostTracker.print_summary(run_id="lg_turn_1", session_id="lg_sess_1", pop=False)
    print_task_efficiency_summary()
