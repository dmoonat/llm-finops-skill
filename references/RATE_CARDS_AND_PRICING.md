# Rate Cards, Multi-Provider Pricing & Extraction Reference

`adk-finops` decouples model and tool pricing from application code via `RateCardRegistry` and `src/adk_finops/rates/default_rates.json`.

---

## 1. Built-In Model Rate Cards (USD per 1M Tokens)

| Model | Provider | Input (`<=200k`) | Output (`<=200k`) | Cached (`<=200k`) | Context `>200k` / Regional (`non_global`) / `2027` Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `gemini-3.1-pro-preview` | Google | $2.00 | $12.00 | $0.20 | **`>200k`**: In $4.00, Out $18.00, Cached $0.40 |
| `gemini-3.8-flash-cyber` | Google | $1.50 | $7.50 | $0.15 | **Non-global**: In $1.65, Out $8.25, Cached $0.165 |
| `gemini-3.8-flash` / `3.7` / `3.6` | Google | $0.75 | $3.75 | $0.075 | **Non-global**: $0.825 / $4.125 • **2027 Standard**: $1.50 / $7.50 (Non-global: $1.65 / $8.25) |
| `gemini-3.5-flash` | Google | $1.50 | $9.00 | $0.15 | **Non-global**: In $1.65, Out $9.90, Cached $0.165 |
| `gemini-3.5-flash-lite` | Google | $0.30 | $2.50 | $0.03 | **Non-global**: In $0.33, Out $2.75, Cached $0.033 |
| `gemini-3.1-flash-lite` | Google | $0.25 | $1.50 | $0.025 | **Non-global**: In $0.275, Out $1.65, Cached $0.0275 |
| `gemini-3-flash` / `preview` | Google | $0.50 | $3.00 | $0.05 | — |
| `gemini-3-pro-image` | Google | $2.00 | $12.00 | $0.20 | **`>200k`**: In $4.00, Out $18.00, Cached $0.40 |
| `gemini-3.1-flash-image` | Google | $0.50 | $3.00 | $0.05 | Nano Banana 2 image generation |
| `gemini-3.1-flash-lite-image` | Google | $0.25 | $1.50 | $0.025 | Nano Banana Lite image generation |
| `gemini-2.5-pro` | Google | $1.25 | $10.00 | $0.125 | **`>200k`**: In $2.50, Out $15.00, Cached $0.25 |
| `gemini-2.5-pro-computer-use-preview` | Google | $1.25 | $10.00 | $0.125 | **`>200k`**: In $2.50, Out $15.00, Cached $0.25 |
| `gemini-2.5-flash` | Google | $0.30 | $2.50 | $0.03 | **`>200k`**: In $0.30, Out $2.50, Cached $0.03 |
| `gemini-2.5-flash-lite` | Google | $0.10 | $0.40 | $0.01 | **`>200k`**: In $0.10, Out $0.40, Cached $0.01 |
| `gemini-2.5-flash-image` | Google | $0.30 | $2.50 | $0.03 | — |
| `gemini-2.5-flash-live` | Google | $0.50 | $2.00 | $0.05 | Live API text rates |
| `gemini-2.0-flash` | Google | $0.15 | $0.60 | $0.0375 | — |
| `codemender` | Google | $0.75 | $3.75 | $0.075 | **2027 Standard**: In $1.50, Out $7.50, Cached $0.15 |
| `gpt-4o` | OpenAI | $2.50 | $10.00 | $1.25 | — |
| `gpt-4o-mini` | OpenAI | $0.15 | $0.60 | $0.075 | — |
| `o1` | OpenAI | $15.00 | $60.00 | $7.50 | Reasoning tokens billed as output |
| `o1-mini` / `o3-mini` | OpenAI | $1.10 | $4.40 | $0.55 | Reasoning tokens billed as output |
| `claude-3-7-sonnet` / `3-5-sonnet` | Anthropic | $3.00 | $15.00 | $0.30 | — |
| `claude-3-5-haiku` | Anthropic | $0.80 | $4.00 | $0.08 | — |
| `claude-3-opus` | Anthropic | $15.00 | $75.00 | $1.50 | — |
| `deepseek-v3` | DeepSeek | $0.14 | $0.28 | $0.014 | — |
| `deepseek-r1` | DeepSeek | $0.55 | $2.19 | $0.14 | Reasoning tokens billed as output |

### Built-In Tool & Grounding Rates
- **Google Search / Enterprise Web Grounding**: `$0.014` per query ($14.00 / 1,000 queries; first 5,000 queries/month free)
- **Google Maps Grounding**: `$0.014` per query ($14.00 / 1,000 queries; first 5,000 queries/month free)
- **Vertex AI Search / Grounding with Your Data**: `$0.0025` per prompt ($2.50 / 1,000 prompts)

---

## 2. Customizing Rate Cards, Regions & Discounts

```python
from adk_finops import CostTracker

# 1. Register a new or fine-tuned model
CostTracker.register_rate_card(
    "claude-opus-5",
    {
        "provider": "anthropic",
        "input_per_1m": 5.0,
        "output_per_1m": 25.0,
        "cached_input_per_1m": 0.5,
    },
)

# 2. Register a fixed fee for a custom or MCP tool
CostTracker.register_tool_rate("bloomberg_mcp_search", 0.05)

# 3. Apply Enterprise Negotiated Discounts
CostTracker.set_discount(15.0)                     # 15% discount across all providers
CostTracker.set_discount(20.0, provider="google")  # 20% discount on Google Cloud models

# 4. Configure Regional (`non_global`) & Date-Tiered (`2027`) Pricing
CostTracker.set_region("us-central1")              # Auto-detected from GOOGLE_CLOUD_LOCATION if set
CostTracker.set_effective_date("2027-01-01")       # Optional: simulate 2027 standard rates

# 5. Load from local JSON file or remote corporate endpoint
CostTracker.load_rate_card_file("./custom_rates.json")
# CostTracker.load_rate_card_url("https://internal.example.com/finops/rates.json")
```

---

## 3. Dynamic Remote Rate Syncing & Live Pricing Extraction Framework

Keep rate cards up to date without upgrading the package (`gemini_enterprise_pricing`, `litellm_registry`, and optional `gcp_cloud_billing_catalog_api`):

```bash
# 1. Sync local rate card cache (~/.cache/adk-finops/remote_rates.json, 24h TTL)
adk-finops sync-rates --force

# 2. Dry-run pricing extraction report comparing live upstream sources vs default_rates.json
adk-finops extract-pricing

# 3. Export merged rate card including newly discovered models (gpt-4.1, o3, claude-4, etc.)
adk-finops extract-pricing --include-new-models --output ./custom_rates.json

# 4. Update bundled src/adk_finops/rates/default_rates.json in-place
adk-finops extract-pricing --update-default --include-new-models
```

Or run programmatically in Python:
```python
from adk_finops import CostTracker, extract_and_sync_pricing

# Runtime remote rate card sync (24h local cache)
CostTracker.sync_remote_rate_card(force=True)

# Full multi-source extraction report
report = extract_and_sync_pricing(
    update_default_rates=False,
    output_path="./custom_rates.json",
    include_new_models=True,
    quiet=False,
)
```

---

## 4. Environment Variables Quick Reference

| Environment Variable | Description |
| :--- | :--- |
| `GOOGLE_CLOUD_LOCATION` | Primary region detector (`"global"`, `"us-central1"`, etc.). Non-global regions apply `non_global` pricing. |
| `ADK_FINOPS_REGION` | Fallback pricing region if `GOOGLE_CLOUD_LOCATION` is unset (defaults to `"global"`). |
| `ADK_FINOPS_EFFECTIVE_DATE` | ISO date (`YYYY-MM-DD`) to evaluate date-tiered pricing such as `standard_pricing_2027`. |
| `ADK_FINOPS_RATE_CARD_PATH` | Path to a custom JSON rate card file. |
| `ADK_FINOPS_SYNC_REMOTE_RATES` | Set to `"true"` or `"1"` to sync remote rate cards on startup (cached 24h). |
| `ADK_FINOPS_DISCOUNT_PERCENT` | Global enterprise discount percentage (e.g., `"15.0"`). |
| `ADK_FINOPS_MAX_PROMPT_TOKENS` | Hard pre-flight ceiling on estimated input tokens per LLM call (e.g., `"200000"`). |
| `ADK_FINOPS_PREFLIGHT_GUARD` | Enable/disable `< 0.5ms` pre-flight budget estimation (`"true"` / `"false"`). |
| `ADK_FINOPS_OPTIMIZATION_ADVISOR` | Enable/disable the deterministic Optimization Advisor (`"true"` / `"false"`). |
| `ADK_FINOPS_BIGQUERY_TABLE` | Auto-enable BigQuery streaming (`"project.dataset.table"`). |
| `ADK_FINOPS_JSONL_PATH` | Auto-enable local JSONL export (`"logs/finops_costs.jsonl"`). |
| `ADK_FINOPS_CSV_PATH` | Auto-enable local CSV export (`"logs/finops_costs.csv"`). |
| `ADK_FINOPS_ENABLE_OTEL` | Auto-enable OpenTelemetry span attributes (`"true"` / `"false"`). |
| `ADK_FINOPS_ENABLE_DASHBOARD` | Auto-start the embedded Near-Live Web Dashboard (`"true"` / `"false"`). |
| `ADK_FINOPS_DASHBOARD_API_KEY` | Pre-shared secret protecting `POST /api/ingest` on the dashboard server (`X-FinOps-Key` header). |
| `GOOGLE_CLOUD_BILLING_API_KEY` | Optional GCP API key used by `adk-finops extract-pricing` to query `cloudbilling.googleapis.com`. |
