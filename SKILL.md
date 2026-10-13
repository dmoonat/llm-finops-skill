---
name: llm-finops-skill
description: >-
  Universal LLM FinOps token usage, financial cost tracking, tool/grounding fee
  attribution, pre-flight budget guards, end-to-end latency/TTFT/throughput
  telemetry, wasted spend analytics, and observability using the `adk-finops`
  Python package (https://github.com/dmoonat/adk-finops). Use this skill whenever
  the user wants to track LLM token usage or costs, measure LLM/tool latency,
  TTFT, or token throughput (tok/s), add budget circuit breakers or model
  downgrades, bill custom/MCP tools or cloud grounding, attribute spend across
  multi-agent hierarchies, analyze wasted spend on failed loops, or export FinOps
  telemetry (Rich terminal summary, Near-Live Web Dashboard, BigQuery, JSONL,
  CSV, HTTP, OpenTelemetry) in Google ADK or any Python agentic framework
  (LangChain, LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, PydanticAI,
  LlamaIndex, LiteLLM, FastAPI) with any LLM provider (Google Gemini, OpenAI,
  Anthropic Claude, DeepSeek, or custom models).
license: Apache-2.0
metadata:
  package: adk-finops
  repository: https://github.com/dmoonat/adk-finops
  skill_repository: https://github.com/dmoonat/llm-finops-skill
  pypi: https://pypi.org/project/adk-finops/
---

# `llm-finops-skill` — Universal Agent & LLM FinOps Skill (`adk-finops`)

You are pair-programming with a developer to instrument **token usage, financial cost tracking, tool/grounding billing, budget guardrails, end-to-end latency/TTFT/throughput, and FinOps observability** using [`adk-finops`](https://github.com/dmoonat/adk-finops).

`adk-finops` provides:
1. **`FinOpsCostPlugin`**: Native Google Agent Development Kit (ADK) `BasePlugin` with zero-boilerplate lifecycle hooks (`before_run`, `before_model`, `after_model`, `before_tool`, `after_tool`, `on_event`, `after_run`, and error callbacks).
2. **`CostTracker`**: Zero-dependency, thread-safe and async-safe universal cost, token, and latency accounting engine that works with **any Python agentic framework** (LangChain, LangGraph, CrewAI, AutoGen/AG2, OpenAI Agents SDK, PydanticAI, LlamaIndex, LiteLLM, FastAPI) and **any LLM provider** (Google Gemini, OpenAI, Anthropic, DeepSeek, or custom/self-hosted models).

---

## 1. Pair-Programmer Workflow (Step-by-Step)

When asked to add token/cost tracking, budget guards, latency/throughput metrics, or FinOps telemetry to a project, follow this workflow:

### Step 1: Inspect the Project & Detect the Stack
1. **Detect the Python dependency manager** (`uv.lock`, `poetry.lock`, `Pipfile`, `pyproject.toml`, `requirements.txt`) before installing packages.
2. **Identify the agent framework and LLM SDKs** in use:
   - **Google ADK** (`google.adk.agents.Agent`, `App`, `Runner`) $\rightarrow$ Use `FinOpsCostPlugin`.
   - **LangChain / LangGraph** (`langchain_core`, `langgraph`) $\rightarrow$ Use `CostTracker` via a `BaseCallbackHandler`.
   - **Other Agent Frameworks** (CrewAI, AutoGen, OpenAI Agents SDK, PydanticAI, LlamaIndex) or **Direct LLM SDKs** (`google-genai`, `openai`, `anthropic`, `litellm`) $\rightarrow$ Use `CostTracker` (`track_run`, `start_turn`, `track_llm_call`, `track_tool_call`, `record_usage`, `record_tool_call`) and `@billable`.

### Step 2: Install `adk-finops` with the Appropriate Extras
Install using the project's environment (e.g., `uv add`, `poetry add`, or `.venv/bin/pip install`):

| Target Use Case | Install Command |
| :--- | :--- |
| **Core (Zero external dependencies, any framework)** | `pip install adk-finops` |
| **With Rich 24-bit Terminal Summary Box** | `pip install "adk-finops[rich]"` |
| **With Google ADK Plugin** | `pip install "adk-finops[adk]"` |
| **With BigQuery Streaming Exporter** | `pip install "adk-finops[bigquery]"` |
| **Full Enterprise Suite (ADK + Rich + BigQuery)** | `pip install "adk-finops[all]"` |

---

## 2. Quick Integration Patterns by Framework

### Pattern A: Google ADK (`FinOpsCostPlugin`)

Attach `FinOpsCostPlugin` directly to `App(plugins=[...])` or `Runner(plugins=[...])`. Token usage, costs, tool fees, wall-clock latency, streaming TTFT, and token throughput (`tok/s`) are captured automatically:

```python
from google.adk.agents import Agent
from google.adk.apps import App
from adk_finops import FinOpsCostPlugin, billable


@billable(fee=0.015, provider="serpapi")
def patent_search(query: str) -> dict:
    """Paid external search tool ($0.015 per successful call)."""
    return {"status": "success", "hits": 10}


root_agent = Agent(
    name="coordinator_agent",
    model="gemini-2.5-pro",
    instruction="You are a helpful assistant.",
    tools=[patent_search],
)

finops_plugin = FinOpsCostPlugin(
    name="finops_cost_tracker",
    default_model="gemini-2.5-pro",
    budget_limit_usd=2.00,             # Optional session cap
    turn_budget_limit_usd=0.50,        # Optional per-turn cap
    max_prompt_tokens=200_000,         # Optional pre-flight prompt token cap
    preflight_budget_guard=True,       # <0.5ms pre-flight guard before network call
    on_budget_exceeded="halt",         # "halt", "downgrade", or "warn"
    fallback_model="gemini-2.5-flash", # Used when on_budget_exceeded="downgrade"
    render_terminal_box=True,          # Prints Rich FinOps summary box after each turn
    enable_optimization_advisor=True,  # Zero-LLM advisor for caching/thinking/right-sizing
    jsonl_path="logs/finops_costs.jsonl",
)

app = App(
    name="my_agent_app",
    root_agent=root_agent,
    plugins=[finops_plugin],
)
```

> [!WARNING]
> **Never double-register `FinOpsCostPlugin`**: Register it **only** in `App(plugins=[finops_plugin])` or `Runner(plugins=[finops_plugin])`. Do **not** also pass `finops_plugin.after_model_callback` to `Agent(after_model_callback=...)`, or ADK will invoke the callback twice and double-count tokens.

---

### Pattern B: Standalone / Any Agent Framework (`CostTracker`)

Use `CostTracker` to track single-turn runs, multi-turn sessions, latency/TTFT/throughput, and hierarchical multi-agent workflows across **OpenAI, Anthropic, Google GenAI, DeepSeek, LangChain, CrewAI, AutoGen, or PydanticAI**:

```python
from adk_finops import CostTracker, BudgetExceededError

session_id = "sess_101"
turn_id = "turn_1"

# 1. Register optional multi-agent hierarchy & budgets
CostTracker.register_agent_hierarchy(
    agent_name="researcher",
    parent_agent_name="supervisor",
    root_agent_name="supervisor",
)
CostTracker.set_budget(
    session_id=session_id,
    session_limit_usd=1.00,
    turn_limit_usd=0.25,
    agent_limits_usd={"researcher": 0.40},
    max_prompt_tokens=200_000,
)

# 2. Initialize dual-scope tracking (resets turn_id, preserves cumulative session_id)
CostTracker.start_turn(turn_id=turn_id, session_id=session_id, root_agent_name="supervisor")

# 3. (Optional) Pre-flight budget check before calling the LLM (<0.5ms, $0.00 cost if blocked)
#    Note: Calling check_preflight_budget() also starts an internal timer so the subsequent
#    record_usage() automatically calculates elapsed LLM latency if latency_ms is omitted!
exceeded, reason, preflight = CostTracker.check_preflight_budget(
    model_name="gemini-2.5-pro",
    estimated_prompt_tokens=4500,
    run_id=turn_id,
    session_id=session_id,
    agent_name="researcher",
)
if exceeded:
    raise BudgetExceededError(reason, preflight["current_cost_usd"], preflight["budget_limit_usd"])

# 4. Record LLM usage — either via track_llm_call() context manager (auto-times latency & TTFT)
#    or directly via CostTracker.record_usage(...)
with CostTracker.track_llm_call(
    turn_id, model_name="gemini-2.5-pro", session_id=session_id, agent_name="researcher"
) as call:
    # Execute your SDK call here; for streaming, call `call.mark_first_token()` on first chunk
    call.set_usage(
        prompt_tokens=4200,
        completion_tokens=350,
        thoughts_tokens=180,     # Reasoning/thinking tokens (billed at output rate)
        cached_tokens=2000,      # Context-cached tokens (computes gross vs net savings)
    )
    # Or pass the raw Gemini/OpenAI/Anthropic SDK response directly: `call.set_response(response)`

# 5. Record paid tool or grounding calls (track_tool_call auto-measures tool_latency_ms)
with CostTracker.track_tool_call(
    turn_id, tool_name="google_search", session_id=session_id, agent_name="researcher"
):
    pass  # Execute external tool / search API

# 6. Record end-to-end turn latency & task outcome ("success", "failed", "error", "aborted", "budget_exceeded")
CostTracker.record_turn_latency(run_id=turn_id, session_id=session_id)
CostTracker.record_task_status(session_id=session_id, run_id=turn_id, status="success")

# 7. Render terminal box (includes Latency & Throughput banner) and inspect structured summary
CostTracker.print_summary(run_id=turn_id, session_id=session_id, pop=False)
summary = CostTracker.get_summary(run_id=turn_id, session_id=session_id, pop=False)
```

---

## 3. Extracting Token Counts & Latency Across LLM Providers

When integrating `CostTracker.record_usage(...)` with non-ADK SDKs, map provider response metadata accurately so **thinking tokens** and **cached prompt tokens** are properly billed:

| Provider / SDK | `prompt_tokens` | `completion_tokens` | `thoughts_tokens` (Reasoning) | `cached_tokens` (Cache Read) |
| :--- | :--- | :--- | :--- | :--- |
| **Google GenAI (`google-genai`)** | `usage_metadata.prompt_token_count` | `usage_metadata.candidates_token_count` | `usage_metadata.thoughts_token_count` | `usage_metadata.cached_content_token_count` |
| **OpenAI (`openai` / `o1` / `o3` / `gpt-4o`)** | `usage.prompt_tokens` | `usage.completion_tokens - reasoning_tokens` | `usage.completion_tokens_details.reasoning_tokens` | `usage.prompt_tokens_details.cached_tokens` |
| **Anthropic (`anthropic`)** | `usage.input_tokens + cache_read_input_tokens` | `usage.output_tokens` | `0` *(included in `output_tokens`)* | `usage.cache_read_input_tokens` |
| **DeepSeek (`deepseek-r1` / `v3`)** | `usage.prompt_tokens` | `usage.completion_tokens - reasoning_tokens` | `usage.completion_tokens_details.reasoning_tokens` | `usage.prompt_cache_hit_tokens` |
| **LiteLLM (`litellm`)** | `usage.prompt_tokens` | `usage.completion_tokens - reasoning_tokens` | `getattr(usage.completion_tokens_details, "reasoning_tokens", 0)` | `getattr(usage.prompt_tokens_details, "cached_tokens", 0)` |

> [!IMPORTANT]
> **Prompt Token Convention in `adk-finops`**:
> `CostTracker.calculate_call_cost_and_savings` computes `non_cached_prompt = max(0, prompt_tokens - cached_tokens)`. Therefore, `prompt_tokens` passed to `record_usage()` must be the **total input tokens (including cached tokens)**. Note that Anthropic's API reports `usage.input_tokens` as *uncached* input tokens after cache breakpoint; for Anthropic, pass `prompt_tokens = usage.input_tokens + getattr(usage, "cache_read_input_tokens", 0)` and `cached_tokens = getattr(usage, "cache_read_input_tokens", 0)`.

### End-to-End Latency, TTFT & Token Throughput Metrics
`adk-finops` tracks monotonic wall-clock latency (`time.perf_counter_ns()`), streaming **Time-To-First-Token (TTFT)**, tool execution time, and token throughput across **turn**, **session**, **model**, **agent**, and **tool** scopes:

| Metric Key | Scope | Description |
| :--- | :---: | :--- |
| `latency_ms` | Turn / Session / Agent | Overall wall-clock duration in ms (`turn_latency_ms` when available, else `llm_latency_ms + tool_latency_ms`). |
| `llm_latency_ms` | All Scopes | Cumulative time spent waiting on LLM API calls in ms. |
| `avg_llm_latency_ms` / `p95_llm_latency_ms` | All Scopes | Mean and 95th-percentile LLM call latency in ms. |
| `ttft_ms` / `avg_ttft_ms` | All Scopes | Streaming **Time-To-First-Token** in ms. |
| `tool_latency_ms` | All Scopes | Cumulative tool / grounding execution time in ms. |
| `output_tokens_per_sec` | All Scopes | Output generation speed: `(completion_tokens + thoughts_tokens) / (llm_latency_ms / 1000)`. |
| `total_tokens_per_sec` | All Scopes | Total processing throughput: `total_tokens / (llm_latency_ms / 1000)`. |

---

## 4. Critical Best Practices & Guardrails

1. **Instrument Semantic Task Failures (Wasted Spend Analytics)**:
   - Unhandled exceptions (`status="error"`) and budget halts (`status="budget_exceeded"`) are captured automatically.
   - However, runs that return `200 OK` without throwing an exception default to `status="success"`. Always instruct validator/critic nodes to set `ctx.state["failed"] = True` (in ADK) or call `finops_plugin.record_task_status(session_id, status="failed", error="...")` / `CostTracker.record_task_status(...)` when output validation fails.
2. **Use `@billable` for Paid Custom & MCP Tools**:
   - Free local tools and MCP tools incur `$0.00` by default.
   - Decorate paid external APIs with `@billable(fee=0.015)` or `@billable(fee_fn=lambda args, res: ..., charge_on_error=False)`, or pass `tool_rates={"tool_name": 0.05}` to `FinOpsCostPlugin`.
3. **Register Unlisted or Custom Models**:
   - If a model is not in the bundled rate card, `adk-finops` logs a warning, uses a fallback rate (`$0.30 / $2.50 / $0.03` per 1M), and sets `"is_fallback_rate": True`. Register custom models via `CostTracker.register_rate_card("model-name", {"provider": "openai", "input_per_1m": 2.0, "output_per_1m": 8.0, "cached_input_per_1m": 0.5})` or enable `sync_remote_rates=True`.
4. **Avoid Double-Counting in BigQuery & Local JSONL/CSV Queries**:
   - Exporters emit **both** an overall rollup row (`agent_name IS NULL`) and per-agent attributed rows (`agent_name IS NOT NULL`).
   - Query overall totals with `WHERE scope = 'session' AND agent_name IS NULL`.
   - Query per-agent spend with `WHERE scope = 'session' AND agent_name IS NOT NULL`.

---

## 5. Progressive Disclosure Reference Guides & Examples

Read these reference files on-demand when implementing specific features:

### Detailed Reference Guides (`references/`)
- **[`references/ADK_INTEGRATION.md`](references/ADK_INTEGRATION.md)** — Complete Google ADK `FinOpsCostPlugin` guide, lifecycle hooks (including latency/TTFT tracking), multi-agent hierarchy (`root_self`, `sub_agent`, `root_rollup`), `@billable` tools, and ADK Web UI `finops_cost` state delta.
- **[`references/MULTI_FRAMEWORK_GUIDE.md`](references/MULTI_FRAMEWORK_GUIDE.md)** — Integration patterns for LangChain/LangGraph, OpenAI Agents SDK, CrewAI, AutoGen/AG2, PydanticAI, LlamaIndex, LiteLLM, and direct SDKs (`google-genai`, `openai`, `anthropic`, `deepseek`), including `track_llm_call` and `track_tool_call` latency context managers.
- **[`references/BUDGET_GUARDS_AND_ADVISOR.md`](references/BUDGET_GUARDS_AND_ADVISOR.md)** — Pre-flight (`<0.5ms`) & post-call budget guards, custom provider token profiles (`register_token_profile`), enforcement modes (`halt`, `downgrade`, `warn`), Optimization Advisor, and Task Outcome / Wasted Spend analytics.
- **[`references/EXPORTERS_AND_DASHBOARD.md`](references/EXPORTERS_AND_DASHBOARD.md)** — Near-Live Web Dashboard (`adk-finops dashboard`), Unified CLI/API reporting (`adk-finops report` / `generate_finops_report`), BigQuery auto-schema & SQL queries (including latency/throughput columns), JSONL/CSV/DuckDB, HTTP Push (`POST /api/ingest`), and OpenTelemetry (`gen_ai.finops.*`).
- **[`references/RATE_CARDS_AND_PRICING.md`](references/RATE_CARDS_AND_PRICING.md)** — Built-in model & grounding rate cards, `>200K` context tiering, regional (`non_global`) & `2027` standard pricing, enterprise discounts, and live pricing extraction (`adk-finops extract-pricing`).

### Ready-to-Use Code Templates (`examples/`)
- **[`examples/google_adk_multi_agent.py`](examples/google_adk_multi_agent.py)** — Hierarchical Google ADK coordinator + sub-agents with `FinOpsCostPlugin`, `@billable` tools, budget guards, and task efficiency summary.
- **[`examples/langchain_langgraph_finops.py`](examples/langchain_langgraph_finops.py)** — Universal `AdkFinOpsCallbackHandler` for LangChain and LangGraph supporting Gemini, OpenAI, Anthropic, and DeepSeek with LLM & tool latency tracking.
- **[`examples/multi_framework_adapters.py`](examples/multi_framework_adapters.py)** — Reusable helpers for OpenAI/Anthropic/Google GenAI/LiteLLM SDK responses, `track_llm_call` / `track_tool_call` latency timing, CrewAI, AutoGen, and PydanticAI.
- **[`examples/fastapi_finops_middleware.py`](examples/fastapi_finops_middleware.py)** — FastAPI middleware with request-scoped `CostTracker`, end-to-end latency & throughput headers (`X-FinOps-*`), and exporter flushing.

### Verification Script (`scripts/`)
- **[`scripts/verify_finops_setup.py`](scripts/verify_finops_setup.py)** — Run `python scripts/verify_finops_setup.py` to verify `adk-finops` installation, rate card resolution, pre-flight token estimation, latency/throughput tracking, and cost accounting.
