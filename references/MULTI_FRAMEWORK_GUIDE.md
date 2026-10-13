# Multi-Framework & Multi-LLM Integration Guide

While `adk-finops` includes a native plugin for Google ADK, its core **`CostTracker`**, **`@billable`**, **`estimate_text_tokens`**, **latency/TTFT context managers** (`track_llm_call`, `track_tool_call`), **budget guards**, **exporters**, and **Near-Live Web Dashboard** have **zero ADK dependency** (`pip install adk-finops`) and work with **any Python agentic framework** and **any LLM provider**.

---

## 1. Universal LLM Response Helper (`record_llm_response`) & Context Managers

### Option A: `CostTracker.track_llm_call` & `CostTracker.track_tool_call` (Auto-Measures Latency & TTFT)

```python
from adk_finops import CostTracker

# track_run() automatically measures end-to-end turn_latency_ms
with CostTracker.track_run("request_123"):
    # 1. Context manager automatically times llm_latency_ms (and streaming ttft_ms)
    with CostTracker.track_llm_call(
        "request_123", model_name="gemini-3.7-flash", agent_name="researcher"
    ) as call:
        # Optional for streaming responses: call `call.mark_first_token()` on first chunk
        # Either pass raw Gemini / OpenAI / Anthropic response:
        #   call.set_response(response)
        # Or set token counts explicitly:
        call.set_usage(
            prompt_tokens=1500,
            completion_tokens=250,
            thoughts_tokens=100,
            cached_tokens=0,
        )

    # 2. Time tool execution with track_tool_call() (or pass latency_ms=... to record_tool_call)
    with CostTracker.track_tool_call(
        "request_123", tool_name="google_search", agent_name="researcher"
    ):
        pass  # Execute external tool / search API
```

### Option B: Universal Extractor Function (`record_llm_response`)

Drop this universal helper into any non-ADK codebase to automatically extract `prompt_tokens`, `completion_tokens`, `thoughts_tokens` (reasoning), and `cached_tokens` across **Google GenAI, OpenAI, Anthropic Claude, DeepSeek, and LiteLLM**:

```python
from typing import Any
from adk_finops import CostTracker


def record_llm_response(
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
    """Universal extractor for Google GenAI, OpenAI, Anthropic, DeepSeek, and LiteLLM responses."""
    prompt_tokens = 0
    completion_tokens = 0
    thoughts_tokens = 0
    cached_tokens = 0

    # 1. Google GenAI / Vertex AI (response.usage_metadata)
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is not None:
        prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
        thoughts_tokens = int(getattr(usage_meta, "thoughts_token_count", 0) or 0)
        cached_tokens = int(getattr(usage_meta, "cached_content_token_count", 0) or 0)

    # 2. OpenAI / DeepSeek / LiteLLM / Anthropic (response.usage)
    elif hasattr(response, "usage") and response.usage is not None:
        u = response.usage
        # Anthropic messages.create response
        if hasattr(u, "input_tokens") and not hasattr(u, "prompt_tokens"):
            uncached_in = int(getattr(u, "input_tokens", 0) or 0)
            cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
            cached_tokens = cache_read
            # adk-finops expects prompt_tokens to be total input tokens (including cached)
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

            # Separate reasoning tokens from completion_tokens so record_usage doesn't double-add them
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
```

---

## 2. LangChain & LangGraph Integration

Use a `BaseCallbackHandler` to automatically intercept all LLM calls, streaming TTFT, tool invocations, latencies, and errors across chains and LangGraph nodes:

```python
import time
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from adk_finops import CostTracker, extract_billable_spec, is_error_tool_result


class AdkFinOpsCallbackHandler(BaseCallbackHandler):
    """Universal LangChain / LangGraph callback handler powered by adk-finops."""

    def __init__(
        self,
        turn_id: str,
        session_id: str | None = None,
        agent_name: str = "langgraph_agent",
        root_agent_name: str | None = None,
    ) -> None:
        super().__init__()
        self.turn_id = turn_id
        self.session_id = session_id or turn_id
        self.agent_name = agent_name
        self.root_agent_name = root_agent_name or agent_name
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

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any = None, **kwargs: Any) -> None:
        key = str(run_id or "default")
        self._llm_start_ns[key] = time.perf_counter_ns()
        self._llm_ttft_ms.pop(key, None)

    def on_llm_new_token(self, token: str, *, run_id: Any = None, **kwargs: Any) -> None:
        key = str(run_id or "default")
        if key not in self._llm_ttft_ms and key in self._llm_start_ns:
            self._llm_ttft_ms[key] = round((time.perf_counter_ns() - self._llm_start_ns[key]) / 1_000_000.0, 2)

    def on_llm_end(self, response: LLMResult, *, run_id: Any = None, **kwargs: Any) -> None:
        key = str(run_id or "default")
        start_ns = self._llm_start_ns.pop(key, None)
        ttft_ms = self._llm_ttft_ms.pop(key, None)
        latency_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 2) if start_ns is not None else None

        llm_out = response.llm_output or {}
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                usage_meta = getattr(msg, "usage_metadata", None) or {}
                gen_info = gen.generation_info or {}
                token_usage = gen_info.get("usage_metadata") or gen_info.get("token_usage") or llm_out.get("token_usage") or {}

                model_name = (
                    (getattr(msg, "response_metadata", {}) or {}).get("model_name")
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
                out_details = usage_meta.get("output_token_details") or token_usage.get("completion_tokens_details") or {}
                in_details = usage_meta.get("input_token_details") or token_usage.get("prompt_tokens_details") or {}

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

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: Any = None, **kwargs: Any) -> None:
        self._tool_start_ns[str(run_id or "default")] = time.perf_counter_ns()

    def on_tool_end(self, output: Any, *, name: str = "tool", run_id: Any = None, **kwargs: Any) -> None:
        start_ns = self._tool_start_ns.pop(str(run_id or "default"), None)
        tool_lat_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 2) if start_ns is not None else None
        if not is_error_tool_result(output):
            CostTracker.record_tool_call(
                run_id=self.turn_id,
                session_id=self.session_id,
                tool_name=name,
                latency_ms=tool_lat_ms,
                agent_name=self.agent_name,
            )

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        CostTracker.record_task_status(
            session_id=self.session_id,
            run_id=self.turn_id,
            status="error",
            error=f"{type(error).__name__}: {error}",
        )
```

---

## 3. OpenAI Agents SDK / Swarm / PydanticAI Integration

### OpenAI Agents SDK (`agents.Runner`)
After `result = await Runner.run(agent, input=...)`, iterate over `result.raw_responses`:

```python
from adk_finops import CostTracker

session_id = "openai_agent_sess_1"
turn_id = "turn_1"
CostTracker.start_turn(turn_id=turn_id, session_id=session_id, root_agent_name=agent.name)

result = await Runner.run(agent, input="Analyze Q3 revenue trends")
for resp in result.raw_responses:
    u = resp.usage
    if u:
        reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        CostTracker.record_usage(
            run_id=turn_id,
            session_id=session_id,
            model_name=str(agent.model or "gpt-4o"),
            prompt_tokens=u.input_tokens,
            completion_tokens=max(0, u.output_tokens - reasoning),
            thoughts_tokens=reasoning,
            cached_tokens=cached,
            agent_name=agent.name,
        )

CostTracker.record_turn_latency(run_id=turn_id, session_id=session_id)
CostTracker.record_task_status(session_id=session_id, run_id=turn_id, status="success")
CostTracker.print_summary(run_id=turn_id, session_id=session_id, pop=False)
```

### PydanticAI (`agent.run`)
Extract token usage from `result.usage()`:

```python
with CostTracker.track_llm_call(
    turn_id, model_name="gemini-2.5-flash", session_id=session_id, agent_name="invoice_extractor"
) as call:
    result = await my_pydantic_agent.run("Extract invoice line items")
    u = result.usage()
    call.set_usage(
        prompt_tokens=u.request_tokens or 0,
        completion_tokens=u.response_tokens or 0,
        cached_tokens=(u.details or {}).get("cached_tokens", 0),
        thoughts_tokens=(u.details or {}).get("reasoning_tokens", 0),
    )
```

---

## 4. CrewAI & AutoGen (AG2) Multi-Agent Integration

### CrewAI
Track CrewAI execution using `CostTracker.track_run` (which automatically records end-to-end `turn_latency_ms` on exit) or per-agent step callbacks:

```python
from adk_finops import CostTracker

session_id = "crew_run_1"
with CostTracker.track_run(session_id):
    crew_output = crew.kickoff()
    token_usage = crew_output.token_usage
    CostTracker.record_usage(
        run_id=session_id,
        session_id=session_id,
        model_name="gpt-4o",
        prompt_tokens=getattr(token_usage, "prompt_tokens", 0),
        completion_tokens=getattr(token_usage, "completion_tokens", 0),
        cached_tokens=getattr(token_usage, "cached_prompt_tokens", 0),
        agent_name="crew_manager",
    )
    CostTracker.record_task_status(session_id=session_id, status="success")

CostTracker.print_summary(session_id=session_id)
```

---

## 5. Decorating Custom Tools with `@billable` in Non-ADK Frameworks

When using `@billable` outside Google ADK, use `extract_billable_spec` and `is_error_tool_result` in your tool execution wrapper to automatically record tool fees and execution latency onto `CostTracker`:

```python
import time
from typing import Any, Callable
from adk_finops import CostTracker, billable, extract_billable_spec, is_error_tool_result


@billable(fee=0.025, provider="credit_bureau")
def check_credit_score(company_id: str) -> dict[str, Any]:
    return {"status": "success", "score": 780}


def execute_tracked_tool(
    tool_fn: Callable[..., Any],
    tool_args: dict[str, Any],
    *,
    run_id: str,
    session_id: str | None = None,
    agent_name: str | None = None,
) -> Any:
    """Executes any tool function and records @billable fees and tool_latency_ms."""
    start_ns = time.perf_counter_ns()
    result = tool_fn(**tool_args)
    latency_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 2)
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
    else:
        CostTracker.record_tool_latency(
            run_id=run_id,
            session_id=session_id,
            agent_name=agent_name,
            tool_name=tool_fn.__name__,
            latency_ms=latency_ms,
        )
    return result
```
