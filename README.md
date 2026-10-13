# `llm-finops-skill` — Universal Agent & LLM FinOps Skill

A universal, standard-format **Agent Skill** ([`dmoonat/llm-finops-skill`](https://github.com/dmoonat/llm-finops-skill)) for **[adk-finops](https://github.com/dmoonat/adk-finops)** — the FinOps cost tracking, token attribution, budget enforcement, and telemetry toolkit for AI agents across **Google ADK** and **any LLM / agentic framework**.

This skill follows the open [Agent Skills specification](https://agentskills.io) (`SKILL.md` with YAML frontmatter + progressive-disclosure `references/`, `examples/`, and `scripts/`) so that **any AI pair programmer** (Claude Code, Gemini CLI / Antigravity, Cursor, OpenAI Codex, Windsurf, Cline, Roo Code, GitHub Copilot) can automatically discover and apply `adk-finops` best practices when helping you build or instrument agents.

---

## What This Skill Enables Your AI Pair Programmer To Do

1. **Instrument Google ADK Agents in 1 Line**: Attach `FinOpsCostPlugin` to `App` / `Runner` for automatic lifecycle hooks (`before_run_callback`, `before_model_callback`, `after_model_callback`, `before_tool_callback`, `after_tool_callback`, `on_event_callback`, `after_run_callback`), hierarchical `root_self` / `sub_agent` / `root_rollup` attribution, and dual-scope (`turn` + `session`) state updates (`state["finops_cost"]`).
2. **Track Any Non-ADK Agentic Framework**: Use standalone `CostTracker` adapters for **LangChain / LangGraph**, **CrewAI**, **AutoGen / AG2**, **OpenAI Agents SDK**, **PydanticAI**, **LlamaIndex**, **LiteLLM**, **FastAPI**, or raw **Google GenAI / OpenAI / Anthropic / DeepSeek** SDKs.
3. **Price Built-In Grounding & Custom Tools (`@billable`)**: Automatically price Google Search & Web Grounding (`$14 / 1K`), Google Maps Grounding (`$14 / 1K`), Vertex AI Search (`$2.50 / 1K`), or custom/MCP tools with `@billable(fee=..., fee_fn=..., charge_on_error=False)` and `tool_rates`.
4. **Enforce Sub-Millisecond Pre-Flight & Post-Call Budget Guards**: Block runaway prompts (`max_prompt_tokens`) or budget breaches in `<0.5ms` before network invocation (`preflight_budget_guard=True`), auto-downgrade expensive models (`fallback_model="gemini-2.5-flash"`), or halt turns/sessions exceeding dollar caps.
5. **Measure End-to-End Latency, Streaming TTFT & Token Throughput**: Automatically capture wall-clock turn duration (`latency_ms`), LLM latency (`llm_latency_ms`, `avg_llm_latency_ms`, `p95_llm_latency_ms`), streaming Time-To-First-Token (`ttft_ms`), tool latency (`tool_latency_ms`), and generation speed (`output_tokens_per_sec`, `total_tokens_per_sec`) via `FinOpsCostPlugin` or standalone `CostTracker.track_llm_call()` / `CostTracker.track_tool_call()`.
6. **Measure Task Efficiency & Wasted Spend**: Record task outcomes (`finops_plugin.record_task_status` / `CostTracker.record_task_status`) and surface zero-LLM cost-saving recommendations via the Automated FinOps Optimization Advisor (`advisor.py`).
7. **Stream Telemetry Anywhere**: Export to **BigQuery** (partitioned by `timestamp`, clustered by `[session_id, agent_name, model_name]`), **JSONL / CSV** (queryable via DuckDB), **HTTP Push** (`POST /api/ingest`), **OpenTelemetry** (`gen_ai.finops.*`), or the **Near-Live Web Dashboard** (`adk-finops dashboard`).

---

## Repository Structure

```text
llm-finops-skill/
├── SKILL.md                                 # Entrypoint skill file (YAML frontmatter + core patterns)
├── README.md                                # Installation & usage guide across AI coding tools
├── LICENSE                                  # Apache-2.0 License
├── references/                              # Progressive-disclosure deep-dive documentation
│   ├── ADK_INTEGRATION.md                   # FinOpsCostPlugin API, ADK hooks, latency/TTFT, multi-agent rollups
│   ├── MULTI_FRAMEWORK_GUIDE.md             # LangChain/LangGraph, OpenAI, Anthropic, CrewAI, AutoGen, PydanticAI
│   ├── BUDGET_GUARDS_AND_ADVISOR.md         # <0.5ms pre-flight guards, fallback_model, wasted spend & advisor
│   ├── EXPORTERS_AND_DASHBOARD.md           # Web dashboard, CLI reports, BigQuery SQL, JSONL/CSV, HTTP, OTel
│   └── RATE_CARDS_AND_PRICING.md            # Rate cards, >200K tiers, regional/2027 SKU pricing, enterprise discounts
├── examples/                                # Copy-pasteable reference implementations
│   ├── google_adk_multi_agent.py            # Hierarchical Google ADK pipeline with FinOpsCostPlugin & @billable
│   ├── langchain_langgraph_finops.py        # Universal LangChain / LangGraph callback handler with latency tracking
│   ├── multi_framework_adapters.py          # Drop-in wrappers for OpenAI, Anthropic, GenAI, DeepSeek, LiteLLM
│   └── fastapi_finops_middleware.py         # FastAPI middleware injecting X-FinOps-* headers & exporting metrics
└── scripts/
    └── verify_finops_setup.py               # Self-test script to verify adk-finops installation, latency & pricing math
```

---

## Installation Across AI Pair Programmers

### Option 1: Using the Universal Skills CLI (`npx skills`)

Install directly into your project or global agent skill directory:

```bash
npx skills add dmoonat/llm-finops-skill
```

### Option 2: Claude Code

Copy or symlink into your personal or project `.claude/skills/` directory:

```bash
# Global (available across all projects)
mkdir -p ~/.claude/skills
cp -R /path/to/llm-finops-skill ~/.claude/skills/llm-finops-skill

# Project-scoped
mkdir -p .claude/skills
cp -R /path/to/llm-finops-skill .claude/skills/llm-finops-skill
```

### Option 3: Gemini CLI / Antigravity / Jetski

Copy or symlink into your Gemini skills directory:

```bash
mkdir -p ~/.gemini/config/skills
cp -R /path/to/llm-finops-skill ~/.gemini/config/skills/llm-finops-skill
```

### Option 4: Cursor / Windsurf / Cline / OpenAI Codex / GitHub Copilot

 Most modern agentic IDEs automatically scan `.agents/skills/`, `.claude/skills/`, or `.cursor/rules/`:

```bash
mkdir -p .agents/skills
cp -R /path/to/llm-finops-skill .agents/skills/llm-finops-skill
```

---

## Example Prompts For Your AI Pair Programmer

Once installed, ask your coding assistant natural questions such as:

- *"Add `adk-finops` cost tracking and a $0.50 session budget guard with automatic downgrade to `gemini-2.5-flash` to my Google ADK multi-agent pipeline."*
- *"Wrap my external compliance and search tools with `@billable` so failed API calls aren't charged and per-result costs are tracked."*
- *"Create a LangGraph callback handler using `adk-finops` `CostTracker` that tracks Claude Sonnet 4.6 and GPT-4o token usage, enforces pre-flight token limits, and exports to JSONL."*
- *"Write a BigQuery SQL query over our `adk-finops` telemetry table to calculate per-agent LLM + tool spend without double-counting `root_rollup` rows."*
- *"Add FastAPI middleware that tracks per-request LLM spend with `adk-finops` and returns `X-FinOps-Total-Cost-USD` response headers."*

---

## Verifying Your `adk-finops` Environment

Run the bundled verification script inside your project's Python virtual environment:

```bash
python scripts/verify_finops_setup.py
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
