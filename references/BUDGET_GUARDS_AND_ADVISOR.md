# Budget Guards, Optimization Advisor & Wasted Spend Reference

This reference covers `adk-finops` Pre-Flight and Post-Call Budget Guards, Provider Token Profiles, the Automated FinOps Optimization Advisor, and Task Outcome / Wasted Spend analytics.

---

## 1. Pre-Flight & Post-Call Budget Guards

`adk-finops` enforces budgets at **four scopes**:
1. **`max_prompt_tokens`**: Hard pre-flight ceiling on estimated input prompt tokens per LLM call (e.g., `200_000` to block accidental 200K+ token dumps or `>200K` tier rate doublings at `$0.00` cloud cost).
2. **`agent_budgets`**: Per-sub-agent USD caps (`{"researcher": 0.50, "coder": 1.00}`).
3. **`turn_budget_limit_usd`**: Maximum USD allowed on any single user interaction turn.
4. **`budget_limit_usd`**: Maximum cumulative USD allowed across an entire conversation session.

### Enforcement Modes (`on_budget_exceeded`)

| Mode | Pre-Flight (`before_model_callback`) & Post-Call Behavior |
| :--- | :--- |
| **`"halt"`** *(default)* | Short-circuits `before_model_callback` by returning a safe `LlmResponse` **before** the network request is sent (`$0.00` cloud cost), or halts subsequent calls if breached mid-turn. Marks `status="budget_exceeded"`. |
| **`"downgrade"`** | Mutates `llm_request.model = fallback_model` (e.g., `gemini-2.5-pro` $\rightarrow$ `gemini-2.5-flash`) **in-place before the request leaves the process** so the call executes immediately at the cheaper rate. |
| **`"warn"`** | Logs `[FinOps Pre-Flight Guard] Warning` with projected cost and sets `exceeded: True` in `finops_cost.budget` without interrupting execution. |

---

## 2. Deterministic Pre-Flight Token Estimation (`< 0.5ms`)

Before an LLM call is sent over the network, `estimate_request_tokens(llm_request, model_name=...)` or `estimate_text_tokens(text, model_name=...)` estimates input tokens in `< 0.5ms` using `PROVIDER_TOKEN_PROFILES`:

```python
from adk_finops import (
    CostTracker,
    estimate_request_tokens,
    estimate_text_tokens,
    register_token_profile,
    update_token_profile,
)

# 1. Estimate tokens for raw text or prompt payload
est_tokens = estimate_text_tokens("Your system prompt and user message...", model_name="claude-3-7-sonnet")

# 2. Check pre-flight budget in custom/non-ADK agent loops
exceeded, reason, status = CostTracker.check_preflight_budget(
    model_name="claude-3-7-sonnet",
    estimated_prompt_tokens=est_tokens,
    run_id="turn_1",
    session_id="sess_1",
    agent_name="analyst",
)

# 3. Register a custom provider token profile (e.g., Mistral or Llama)
register_token_profile(
    "mistral",
    {
        "ascii_chars_per_token": 3.8,
        "tool_system_preamble_tokens": 24,
        "per_tool_schema_overhead": 38,
        "image_tokens": 512,
        "pdf_page_tokens": 750,
    },
)

# 4. Override fields on an existing provider profile
update_token_profile("anthropic", {"pdf_page_tokens": 2500})
```

---

## 3. Automated FinOps Optimization Advisor (`advisor.py`)

At the end of every session (`get_summary()` / `after_run_callback`), `generate_optimization_insights()` runs in `< 1ms` with **zero LLM calls** and calculates potential dollar and percentage savings across three rules:

1. **⚡ Context Caching Opportunity**:
   - Triggers when an agent sends $\ge 2,000$ uncached prompt tokens across $\ge 2$ calls.
   - Computes exact savings using `cached_input_per_1m` vs. `input_per_1m` from the active `RateCardRegistry`.
2. **🧠 Context-Aware Thinking Token Alert (`ROPI`)**:
   - Triggers when `thoughts_tokens >= 500` account for $\ge 60\%$ of total output token spend (or `ropi_score >= 0.50`) and exceed the role-appropriate thinking cap:
     - **Tool-using agents (`tool_calls > 0`)**: Recommends capping `thinking_budget=512` (or `thinking_level='LOW'`) rather than disabling thinking, computing savings strictly on excess thinking tokens above `512 * calls`.
     - **Pro / mandatory-reasoning models (`-pro`, `o1`/`o3`/`o4`, `deepseek-r1`, `qwq`) with `0` tool calls**: Recommends capping `thinking_budget=256` (or `thinking_level='LOW'`).
     - **Simple non-tool Flash routers/classifiers (`tool_calls == 0`)**: Recommends setting `thinking_budget=0`.
3. **🎯 Model Right-Sizing**:
   - Triggers when a flagship model (`gemini-3.1-pro-preview`, `gemini-2.5-pro`, `gpt-4o`, `claude-3-5-sonnet`, or user-configured `shadow_candidate_models`) generates short responses ($< 200$ avg output tokens) with `0` tool calls.
   - Computes exact savings from switching to the lighter model tier (`gemini-3.7-flash`, `gemini-2.5-flash`, `gpt-4o-mini`, `claude-3-5-haiku`).

> [!IMPORTANT]
> **Advisor Disclaimer**: Optimization insights are deterministic heuristics and should be treated as **directional hints**. Always validate model changes, caching, or `thinking_budget` adjustments against your evaluation datasets (`eval` benchmarks), accuracy requirements, and latency SLAs.

---

## 4. Task Outcome & Wasted Spend Analytics

Failed agent retry loops often burn 5x–20x more tokens than successful tasks while delivering zero value. `adk-finops` separates **Effective Spend (`is_failure=False`)** from **Wasted Spend (`is_failure=True`)**:

| Outcome `status` | `is_failure` | How It Is Triggered |
| :--- | :---: | :--- |
| **`"success"`** | `False` | Default when a run finishes normally without error flags. |
| **`"budget_exceeded"`** | `True` | Automatically set when `on_budget_exceeded="halt"` trips. |
| **`"error"`** | `True` | Automatically set on unhandled exceptions (`on_*_error_callback` or `CostTracker.track_run`). |
| **`"failed"`** / **`"aborted"`** | `True` | Set automatically when an ADK node sets `ctx.state["failed"] = True`, or explicitly via `finops_plugin.record_task_status(session_id, status="failed", error="...")` / `CostTracker.record_task_status(...)`. |

### Instrumenting Semantic Failures ("200 OK, Zero Business Value")

Always add post-run business validation so polite refusals or invalid outputs are recorded as `"failed"`:

```python
from adk_finops import CostTracker, print_task_efficiency_summary

# After agent finishes returning a response:
if not is_valid_output(agent_response_text):
    # If using FinOpsCostPlugin (also auto-flushes updated status to BigQuery/JSONL/CSV):
    finops_plugin.record_task_status(
        session_id=session_id,
        status="failed",
        error="Output failed schema or business validation",
    )
    # Or if using standalone CostTracker:
    CostTracker.record_task_status(
        session_id=session_id,
        status="failed",
        error="Output failed schema or business validation",
    )

# Print comparative table of Successful Tasks vs. Failed Loops (Wasted Spend)
print_task_efficiency_summary()
metrics = CostTracker.get_task_efficiency_metrics()
```
