# Google ADK Integration Reference (`FinOpsCostPlugin`)

This guide covers integrating `adk-finops` into **Google Agent Development Kit (ADK)** applications using the native `FinOpsCostPlugin` (`google.adk.plugins.BasePlugin`).

---

## 1. Architecture & Lifecycle Hooks

`FinOpsCostPlugin` hooks directly into the ADK `App` / `Runner` lifecycle without modifying your agent prompts or tool implementations:

| ADK Hook | `FinOpsCostPlugin` Action |
| :--- | :--- |
| **`before_run_callback`** | Initializes turn & session tracking (`CostTracker.start_turn`), records monotonic turn start timestamp, discovers the agent hierarchy (`root_agent`, `parent_agent`, `sub_agents`), and registers configured budgets. |
| **`before_model_callback`** | Records monotonic LLM call start timestamp and runs **Pre-Flight Token & Budget Estimation** in `< 0.5ms`. Short-circuits (`"halt"`), switches `llm_request.model` in-place (`"downgrade"`), or logs a warning (`"warn"`) before the network call is made. |
| **`after_model_callback`** | Captures streaming **Time-To-First-Token (`ttft_ms`)** on the first `partial=True` chunk; on final response, extracts `prompt_token_count`, `candidates_token_count`, `thoughts_token_count`, `cached_content_token_count`, computes `llm_latency_ms` & `output_tokens_per_sec`, enriches the active OpenTelemetry span, and enforces post-call budget guards. |
| **`before_tool_callback`** | Records monotonic tool start timestamp to measure `tool_latency_ms`. |
| **`after_tool_callback`** | Records elapsed `tool_latency_ms` (`CostTracker.record_tool_latency`, even for `$0.00` tools), inspects tool execution for `@billable` specs, registered `tool_rates`, or cloud grounding tools (`google_search`, `vertex_ai_search`, `google_maps_grounding`), and attributes fees to the active agent. |
| **`on_event_callback`** | Injects live dual-scope telemetry into `event.actions.state_delta["finops_cost"]` for real-time visibility in the ADK Web UI. |
| **`after_run_callback`** | Records end-to-end `turn_latency_ms`, evaluates workflow state (`ctx.state["failed"]`), runs the deterministic Optimization Advisor, renders the Rich terminal summary box (including `⚡ Latency & Throughput`), and flushes telemetry to configured exporters (BigQuery, JSONL, CSV, HTTP, OpenTelemetry). |
| **`on_*_error_callback`** | Captures unhandled run, agent, model, or tool exceptions (`status="error"`, `is_failure=True`) and flushes crashed session telemetry immediately. |

---

## 2. Full `FinOpsCostPlugin` Constructor Reference

```python
from pathlib import Path
from typing import Any
from adk_finops import FinOpsCostPlugin

finops_plugin = FinOpsCostPlugin(
    name="finops_cost_tracker",
    default_model="gemini-2.5-flash",        # Fallback model if missing in LlmResponse
    # --- Pricing & Rate Cards ---
    rate_card_path=None,                     # str | Path to custom JSON rate card
    rate_card=None,                          # dict[str, Any] inline rate card overrides
    sync_remote_rates=False,                 # Sync latest default_rates.json (24h local cache)
    remote_rate_card_url=None,               # Optional custom URL for remote rate syncing
    remote_cache_ttl_seconds=86400,          # Cache TTL for remote rates
    discount_percent=None,                   # Enterprise volume discount (e.g., 15.0 for 15% off)
    region=None,                             # Auto-detects GOOGLE_CLOUD_LOCATION -> ADK_FINOPS_REGION -> "global"
    effective_date=None,                     # ISO date ("YYYY-MM-DD") for date-tiered rates (e.g., "2027-01-01")
    tool_rates=None,                         # dict[str, float] per-call fees (e.g., {"bloomberg_mcp": 0.05})
    # --- Budget Guards & Token Profiles ---
    budget_limit_usd=None,                   # Max cumulative USD per session
    turn_budget_limit_usd=None,              # Max USD per single user turn
    agent_budgets=None,                      # Per-agent USD caps (e.g., {"researcher": 0.50})
    max_prompt_tokens=None,                  # Pre-flight ceiling on estimated input tokens (e.g., 200_000)
    preflight_budget_guard=True,             # Estimate tokens & cost in before_model_callback (<0.5ms)
    on_budget_exceeded="halt",               # "halt" ($0 cloud cost block), "downgrade", or "warn"
    fallback_model="gemini-2.5-flash",       # Target model when on_budget_exceeded="downgrade"
    token_profiles=None,                     # Custom provider tokenization profile overrides
    token_profiles_path=None,                # Path to JSON file with custom token profiles
    # --- Terminal UI & Optimization Advisor ---
    render_terminal_box=True,                # Render Rich/Unicode cost box after every turn
    enable_optimization_advisor=True,        # Deterministic advisor (<1ms, $0 LLM cost)
    # --- Exporters & Live Dashboard ---
    bigquery_table=None,                     # "project.dataset.table" (or ADK_FINOPS_BIGQUERY_TABLE)
    bigquery_auto_create_table=True,         # Auto-create partitioned/clustered table & evolve schema
    jsonl_path=None,                         # e.g., "logs/finops_costs.jsonl"
    csv_path=None,                           # e.g., "logs/finops_costs.csv"
    enable_otel=False,                       # Emit gen_ai.finops.* spans & attributes
    otlp_endpoint=None,                      # Optional OTLP collector endpoint
    enable_dashboard=False,                  # Start embedded Near-Live Web Dashboard thread
    dashboard_port=8088,                     # Local port for embedded dashboard
    dashboard_endpoint=None,                 # Push via HTTPExporter to central dashboard (POST /api/ingest)
    dashboard_api_key=None,                  # Pre-shared key (or ADK_FINOPS_DASHBOARD_API_KEY)
    exporters=None,                          # Additional custom BaseExporter instances
    export_scope="session",                  # "session" (default), "turn", or "both"
    export_tags=None,                        # Custom metadata tags (e.g., {"env": "prod", "team": "search"})
)
```

---

## 3. Hierarchical Multi-Agent Cost Attribution

When a root orchestrator delegates to sub-agents (`sub_agents=[...]`), `FinOpsCostPlugin` automatically walks the ADK agent tree and tags each agent in `breakdown_by_agent`:

- **`root_agent_name`**: Top-level orchestrator name (e.g., `"supervisor"`).
- **`parent_agent_name`**: Immediate parent (`None` for root, `"supervisor"` for direct sub-agents).
- **`agent_role`**:
  - `"root_self"`: Direct LLM and tool spend by the root orchestrator itself.
  - `"sub_agent"`: Spend by a delegated specialist sub-agent.
  - `"root_rollup"`: Overall parent-level rollup combining root + all sub-agents.

```python
from google.adk.agents import Agent
from google.adk.apps import App
from adk_finops import FinOpsCostPlugin

researcher = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Gather factual citations.",
)

coder = Agent(
    name="coder",
    model="gemini-2.5-pro",
    instruction="Write production-ready Python solutions.",
)

supervisor = Agent(
    name="supervisor",
    model="gemini-2.5-pro",
    instruction="Coordinate researcher and coder.",
    sub_agents=[researcher, coder],
)

finops_plugin = FinOpsCostPlugin(
    default_model="gemini-2.5-pro",
    budget_limit_usd=2.00,
    agent_budgets={
        "researcher": 0.40,
        "coder": 1.20,
    },
    on_budget_exceeded="downgrade",
    fallback_model="gemini-2.5-flash",
)

app = App(
    name="engineering_team",
    root_agent=supervisor,
    plugins=[finops_plugin],
)
```

---

## 4. Explicit Tool Billing (`@billable`) & Grounding Classification

`adk-finops` separates free local/MCP tools from paid external APIs and cloud grounding:

1. **Built-In Cloud Grounding Detection**:
   - **Google Search / Enterprise Web Grounding** (`google_search`, `GoogleSearchTool`, `EnterpriseWebSearchTool`): **$0.014 / query** ($14.00 / 1k).
   - **Google Maps Grounding** (`google_maps_grounding`): **$0.014 / query** ($14.00 / 1k).
   - **Vertex AI Search / Grounding with Your Data** (`vertex_search`, `vertex_ai_search`, `GroundingTool`, `data_store_id`): **$0.0025 / prompt** ($2.50 / 1k).
2. **`@billable` Decorator for Custom Functions & Tools**:
   - Supports fixed fees (`fee=0.015`) or dynamic result-based fees (`fee_fn=lambda args, res: ...`).
   - By default, `charge_on_error=False` ensures tools returning error dicts (`{"status": "error", ...}` or `{"error": ...}`) incur **`$0.00`**.

```python
from adk_finops import billable


# Fixed fee per successful call
@billable(fee=0.015, provider="serpapi")
def search_patents(query: str) -> dict:
    """Searches external patent database."""
    return {"status": "success", "hits": 5}


# Dynamic per-unit fee based on response payload
@billable(
    fee_fn=lambda args, res: 0.002 * len(res.get("pages", [])),
    provider="document_ai",
    charge_on_error=False,
)
def ocr_pdf(gcs_uri: str) -> dict:
    """Extracts text from PDF pages ($0.002 per page)."""
    return {"status": "success", "pages": ["p1", "p2", "p3"]}
```

---

## 5. Accessing Dual-Scope Telemetry, Latency, TTFT & Throughput (`turn` vs. `session`)

During or after execution, read `session.state["finops_cost"]` or call `CostTracker.get_summary()`:

```python
stored_session = await session_service.get_session(
    app_name=app.name, user_id=user_id, session_id=session_id
)
finops = stored_session.state.get("finops_cost", {})

turn_cost = finops.get("turn", {}).get("total_cost_usd", 0.0)
session_cost = finops.get("session", {}).get("total_cost_usd", 0.0)
agent_breakdown = finops.get("breakdown_by_agent", {})
tool_breakdown = finops.get("breakdown_by_tool", {})
insights = finops.get("optimization_insights", [])

# End-to-end latency, streaming TTFT, and token throughput (measured automatically):
latency_ms = finops.get("latency_ms", 0.0)                  # Wall-clock duration (ms)
llm_latency_ms = finops.get("llm_latency_ms", 0.0)          # Cumulative LLM API latency (ms)
p95_llm_latency_ms = finops.get("p95_llm_latency_ms", 0.0)  # 95th percentile LLM latency (ms)
ttft_ms = finops.get("ttft_ms", 0.0)                        # Streaming Time-To-First-Token (ms)
tool_latency_ms = finops.get("tool_latency_ms", 0.0)        # Cumulative tool execution latency (ms)
out_tps = finops.get("output_tokens_per_sec", 0.0)          # Output generation speed (tok/s)
total_tps = finops.get("total_tokens_per_sec", 0.0)         # Total processing throughput (tok/s)
```
