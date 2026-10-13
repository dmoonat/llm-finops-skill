# Exporters, Near-Live Web Dashboard & CLI Reporting Reference

`adk-finops` supports real-time local and cloud observability through its **Near-Live Web Dashboard**, **Unified CLI & Python Report Generator**, and **5 built-in exporters** (`BigQueryExporter`, `JSONLExporter`, `CSVExporter`, `HTTPExporter`, `OpenTelemetryExporter`).

---

## 1. Near-Live FinOps Web Dashboard (`adk-finops dashboard`)

`adk-finops` includes a FastAPI + Chart.js web dashboard that auto-refreshes every 2 seconds, featuring:
- **5 Cascading Hierarchy Filters**:
  1. `👑 Root (Parent) Agent`
  2. `↳ Sub-Agent Drilldown`
  3. `Session Filter (Scoped)`
  4. `Model Filter`
  5. `Task Outcome` (`✅ Effective Spend Only` vs. `🔥 Wasted Spend Only`)
- **Interactive Root ➔ Sub-Agent Hierarchy Rollup Cards**
- **Per-Tool Cost Attribution Table (`Agent ➔ Tool ➔ #Calls ➔ Cost`)**
- **Hybrid Local + HTTP Push + BigQuery Deduplication** (deduplicates by `(session_id, scope, agent_name)` so local and cloud records never double-count).

### Launching the Dashboard (4 Deployment Modes)

```bash
# Mode 1: Local Developer Mode (scans ./logs and optional BigQuery)
adk-finops dashboard --host 127.0.0.1 --port 8088 --log-dir ./logs

# Mode 2: Central Cloud Warehouse Mode (BigQuery Only)
adk-finops dashboard --bigquery-table my-project.finops.agent_costs --log-dir "" --host 127.0.0.1 --port 8088

# Mode 3: Multi-Project Shared Directory Mode (comma-separated log directories)
adk-finops dashboard --log-dir "/srv/agents/bot_a/logs,/srv/agents/bot_b/logs" --host 127.0.0.1 --port 8088

# Mode 4: Authenticated HTTP Ingest Server (Pre-Shared Key)
export ADK_FINOPS_DASHBOARD_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
adk-finops dashboard --host 127.0.0.1 --port 8088 --log-dir ./central_logs
```

Or embed the dashboard inside your agent process:
```python
finops_plugin = FinOpsCostPlugin(
    enable_dashboard=True,
    dashboard_port=8088,
    jsonl_path="logs/finops_costs.jsonl",
)
```

---

## 2. Unified CLI & Programmatic Reporting (`adk-finops report`)

Generate deduplicated FinOps reports across local `.jsonl`/`.csv` files, BigQuery, or both:

```bash
# 1. Rich Terminal Report (merges & deduplicates local ./logs + BigQuery)
adk-finops report --log-dir ./logs --bigquery-table my-project.finops.agent_costs

# 2. Local Files Only
adk-finops report --log-dir ./logs --no-bigquery

# 3. Export as GitHub PR Markdown table
adk-finops report --log-dir ./logs --format markdown --output finops_report.md

# 4. Filter by Root Agent and Failed Outcome, export as JSON
adk-finops report --log-dir ./logs --root-agent coordinator_agent --status failed --format json --output failed_runs.json
```

Programmatic Python API:
```python
from adk_finops import generate_finops_report

report = generate_finops_report(
    log_dir="logs",
    bigquery_table="my-project.finops.agent_costs",
    include_bigquery=True,
    output_format="markdown",  # "table", "markdown", or "json"
    output_path="finops_summary.md",
    print_report=True,
)
```

---

## 3. Standalone & Plugin Exporters

All exporters can be configured directly on `FinOpsCostPlugin` or used standalone with `CostTracker`:

```python
import os
from adk_finops import CostTracker
from adk_finops.exporters import (
    BigQueryExporter,
    CSVExporter,
    HTTPExporter,
    JSONLExporter,
    OpenTelemetryExporter,
)

exporters = [
    JSONLExporter("logs/finops_costs.jsonl"),
    CSVExporter("logs/finops_costs.csv"),
    OpenTelemetryExporter(),
    # Push over HTTP to a central `adk-finops dashboard` server:
    # HTTPExporter(endpoint="http://127.0.0.1:8088", api_key=os.getenv("ADK_FINOPS_DASHBOARD_API_KEY")),
    # Stream to Google BigQuery (auto-creates partitioned & clustered table):
    # BigQueryExporter(table_id="my-gcp-project.finops.agent_costs"),
]

summary = CostTracker.get_summary(run_id="turn_1", session_id="sess_1", pop=False)
if summary:
    for exp in exporters:
        exp.export_summary(summary, scope="session", tags={"env": "prod", "service": "claims_agent"})
```

### OpenTelemetry Semantic Conventions (`OpenTelemetryExporter`)

When `enable_otel=True` (or `OpenTelemetryExporter` is used), `adk-finops` enriches the active span and emits `gen_ai.finops.<scope>` spans with standard attributes:
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.thoughts_tokens`, `gen_ai.usage.cached_tokens`, `gen_ai.usage.total_tokens`
- `gen_ai.usage.cost_usd`, `gen_ai.usage.llm_cost_usd`, `gen_ai.usage.tool_cost_usd`, `gen_ai.usage.savings_usd`, `gen_ai.usage.tool_calls_count`
- `gen_ai.finops.breakdown_by_tool` (JSON-serialized `Agent -> Tool -> Calls & Cost` map)
- `gen_ai.finops.task_outcome` (`success`, `failed`, `error`, `budget_exceeded`)
- `gen_ai.finops.is_wasted_spend` (`true` / `false`)
- `gen_ai.finops.latency_ms`, `gen_ai.finops.llm_latency_ms`, `gen_ai.finops.tool_latency_ms`, `gen_ai.finops.ttft_ms`
- `gen_ai.finops.output_tokens_per_sec`, `gen_ai.finops.total_tokens_per_sec`

---

## 4. BigQuery Schema & Querying Best Practices (Avoiding Double-Counting)

When `bigquery_table` is specified, `adk-finops` automatically provisions a table partitioned by `DATE(timestamp)` and clustered by `[session_id, agent_name, model_name]`, and automatically evolves the schema (`ensure_table_exists()`) to add newly introduced `NULLABLE` columns (such as `latency_ms`, `ttft_ms`, `output_tokens_per_sec`, `breakdown_by_tool`).

| Key Columns | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `TIMESTAMP` | UTC event timestamp (Day Partition Key) |
| `session_id` / `turn_id` / `scope` | `STRING` | Session ID (Cluster Key #1), Turn ID, and scope (`"turn"` or `"session"`) |
| `agent_name` / `model_name` | `STRING` | Attributed agent (`NULL` on overall rollup row) & model (Cluster Keys #2 & #3) |
| `prompt_tokens`, `completion_tokens`, `thoughts_tokens`, `cached_tokens`, `total_tokens` | `INTEGER` | Token breakdown |
| `llm_cost_usd`, `tool_cost_usd`, `total_cost_usd`, `gross_cost_usd`, `savings_usd`, `savings_pct` | `FLOAT` | Financial cost & context caching ROI metrics |
| `latency_ms`, `llm_latency_ms`, `tool_latency_ms`, `ttft_ms` | `FLOAT` | Wall-clock, LLM, tool, and streaming Time-To-First-Token latency in ms |
| `output_tokens_per_sec`, `total_tokens_per_sec` | `FLOAT` | Token generation and processing throughput (`tok/s`) |
| `breakdown_by_agent`, `breakdown_by_model`, `breakdown_by_tool`, `tags` | `JSON` | Full hierarchical breakdowns and user/outcome tags (`status`, `is_failure`) |

Every export writes **both**:
- **Overall Rollup Row (`agent_name IS NULL`)**: Cumulative session/turn total across all agents.
- **Attributed Agent Rows (`agent_name IS NOT NULL`)**: Per-agent breakdown rows with `root_agent_name`, `parent_agent_name`, and `agent_role`.

### Query 1: Overall Session Totals, Latency & Caching ROI (BigQuery)
```sql
SELECT
  session_id,
  total_cost_usd,
  total_tokens,
  savings_usd,
  latency_ms,
  output_tokens_per_sec
FROM `my-gcp-project.finops.agent_costs`
WHERE scope = 'session'
  AND agent_name IS NULL;
```

### Query 2: Per-Agent Spend & Latency Breakdown (No Double-Counting)
```sql
SELECT
  agent_name,
  COUNT(DISTINCT session_id) AS sessions,
  ROUND(SUM(llm_cost_usd), 4) AS llm_spend_usd,
  ROUND(SUM(tool_cost_usd), 4) AS tool_spend_usd,
  ROUND(SUM(total_cost_usd), 4) AS total_spend_usd,
  ROUND(AVG(latency_ms), 1) AS avg_latency_ms
FROM `my-gcp-project.finops.agent_costs`
WHERE scope = 'session'
  AND agent_name IS NOT NULL
GROUP BY 1
ORDER BY total_spend_usd DESC;
```

### Query 3: Successful vs. Wasted Spend Analysis (DuckDB on Local `.jsonl` or BigQuery)
```sql
-- DuckDB on local JSONL:
SELECT
  status AS task_outcome,
  is_failure AS is_wasted_spend,
  COUNT(*) AS total_runs,
  ROUND(SUM(total_cost_usd), 4) AS total_spend_usd
FROM read_json_auto('logs/finops_costs.jsonl')
WHERE scope = 'session' AND agent_name IS NULL
GROUP BY 1, 2;
```
