# Green Agent Documentation: E-commerce Basket Assessment

## Overview

This green agent is an AgentBeats-compatible assessment agent for evaluating e-commerce grocery recommendation systems. It follows the A2A (Agent-to-Agent) protocol and tests white agents on their ability to predict a user's next grocery basket based on purchase history.

### What This Agent Does

The green agent:
1. Hosts assessments for e-commerce recommendation agents
2. Provides task context including user purchase history
3. Collects predictions from white agents (participating agents)
4. Evaluates performance using precision, recall, F1 scores at product, aisle, and department levels
5. Reports metrics back to the AgentBeats platform or directly to users

---

## Architecture

### High-Level Flow

```
User/Platform Request
        ↓
Green Agent (Assessment Orchestrator)
        ↓
    Prepares Task
        ↓
   Sends to White Agent (via A2A)
        ↓
   Collects Response
        ↓
  Evaluates Predictions
        ↓
   Returns Metrics
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **EcomGreenAgentExecutor** | Core assessment logic | `main_A2A.py` lines 242-684 |
| **Data Processing** | Order history & product catalog handling | `split_user_orders()`, lines 38-90 |
| **Evaluation Engine** | Multi-level F1 scoring | `evaluate_basket()`, lines 108-178 |
| **A2A Server** | Protocol-compliant endpoints | `start_green_agent()`, lines 710-831 |
| **Baseline Model** | Heuristic for comparison | `henry_build_prompt()`, lines 180-239 |

---

## Technical Specifications

### 1. Agent Card (TOML Configuration)

**File**: `ecom_green_agent.toml` (or defaults in code)

```toml
name = "ecom_green_agent"
version = "1.0.0"
description = "Ecom grocery basket assessment agent"

[capabilities]
input = ["text"]
output = ["text"]
streaming = false

skills = []
```

### 2. A2A Endpoints

The green agent exposes standard A2A endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Returns agent card |
| `/agent_card` | GET | Explicit agent card endpoint |
| `/send_message` or `/a2a/send_message` | POST | Receives assessment tasks |
| `/healthz` or `/health` | GET | Health check |
| `/reset` | POST | Clears agent state |

### 3. Task Input Format

The green agent accepts natural language task descriptions with the following key elements:

```json
{
  "message": "Assess white agent at http://localhost:9002",
  "context": {
    "mode": "compete",  // or "baseline"
    "user_id": 12345,   // optional: specific user to test
    "sample_size": 10,  // optional: number of users to evaluate
    "white_agent_url": "http://localhost:9002"
  }
}
```

**Supported Modes**:
- `compete`: Tests the white agent directly
- `baseline`: Tests against Henry's heuristic baseline model

---

## Assessment Process

### Step 1: Data Preparation

The green agent splits user order history into:
- **Current Order** (order_number = n): The ground truth basket to predict
- **Previous Orders** (order_number < n): Historical data provided to white agent
- **Previous Order** (order_number = n-1): Most recent order context

```python
split_data = split_user_orders(user_id, df_products, df_orders)
# Returns:
# - current_order_df: Target basket
# - previous_orders_df: Full history
# - n: Current order number
# - days_since_last: Time delta
```

### Step 2: Task Construction

The green agent formats a task message for the white agent:

```python
task_message = f"""
You are a grocery shopping assistant for user {user_id}.
It has been {days_since_last} days since their last order.

Using the user's purchase history, propose the next basket.
Prioritize frequently repeated items and the user's top departments.
Avoid duplicates; keep total items between 3-12.

PURCHASE HISTORY:
{formatted_history}

OUTPUT FORMAT:
<json>
{{
  "predicted_items": {{
    "product_id_1": quantity_1,
    "product_id_2": quantity_2,
    ...
  }}
}}
</json>
"""
```

### Step 3: White Agent Communication

The green agent sends the task via A2A protocol:

```python
POST {white_agent_url}/send_message
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "assessment-{run_id}",
  "method": "send_message",
  "params": {
    "message": {
      "role": "agent",
      "parts": [{"kind": "text", "text": "{task_message}"}]
    }
  }
}
```

### Step 4: Response Parsing

The green agent extracts predictions from the white agent's response:

```python
# Expected format from white agent:
{
  "predicted_items": {
    "12345": 1,  // product_id: quantity
    "67890": 2
  }
}
```

The agent handles both:
- JSON wrapped in `<json>...</json>` tags
- Raw JSON responses

### Step 5: Multi-Level Evaluation

The green agent computes metrics at three levels:

#### Product-Level Metrics
- **Precision**: % of predicted products that are correct
- **Recall**: % of ground truth products that were predicted
- **F1 Score**: Harmonic mean of precision and recall

#### Aisle-Level Metrics
Maps products → aisles and computes F1 on aisle categories

#### Department-Level Metrics
Maps products → departments and computes F1 on department categories

#### Blended F1 Score
Weighted combination (default: 60% products, 20% aisles, 20% departments):

```python
blended_f1 = 0.6 * product_f1 + 0.2 * aisle_f1 + 0.2 * department_f1
```

### Step 6: Results Reporting

The green agent returns formatted results:

```
Assessment Complete ✅ (Mode: WHITE AGENT)

Metrics:
- F1 Score: 0.753
- Precision: 0.820
- Recall: 0.695
- Blended F1: 0.768
- Latency: 1234ms

Product-level: 15 TP, 3 FP, 6 FN
Aisle-level F1: 0.850
Department-level F1: 0.920
```

---

## Data Requirements

### Products Catalog (`ic_products.csv`)

| Column | Type | Description |
|--------|------|-------------|
| product_id | int | Unique product identifier |
| product_name | str | Product name |
| aisle_id | int | Aisle category ID |
| aisle | str | Aisle category name |
| department_id | int | Department category ID |
| department | str | Department category name |

### Orders Dataset (`super_shortened_orders_products_combined.csv`)

| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier |
| order_id | int | Unique order identifier |
| order_number | int | Sequential order number per user |
| product_id | int | Product in the order |
| add_to_cart_order | int | Order items were added to cart |
| reordered | int | 1 if reordered, 0 if first time |
| days_since_prior_order | float | Days since previous order |

---

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# API endpoints
ECOM_API_BASE="https://green-agent-production.up.railway.app"

# White agent configuration
WHITE_AGENT_NAME="baseline-rebuy"

# Data sources
PRODUCTS_CSV="dataset/ic_products.csv"
ORDERS_CSV="dataset/super_shortened_orders_products_combined.csv"

# Features
USE_REAL_API=true
```

### Runtime Configuration

```python
start_green_agent(
    agent_name="ecom_green_agent",
    host="localhost",
    port=9001,
    products_csv="dataset/ic_products.csv",
    orders_csv="dataset/super_shortened_orders_products_combined.csv"
)
```

---

## Running the Green Agent

### Basic Startup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the agent
python main_A2A.py
```

The agent will start on `http://localhost:9001` by default.


### Railway Deployment

The agent is configured for Railway deployment:

```bash
# Set environment variables in Railway dashboard
ECOM_API_BASE=https://green-agent-production.up.railway.app
PRODUCTS_CSV=dataset/ic_products.csv
ORDERS_CSV=dataset/super_shortened_orders_products_combined.csv
```

---

## Baseline Model: Henry's Heuristic

The green agent includes a baseline model for comparison:

### Algorithm

1. **Frequency Analysis**: Count product appearances across all previous orders
2. **Recency Weighting**: Favor items from the most recent order
3. **Department Balance**: Include items from user's top departments
4. **Size Constraints**: Keep basket between 3-12 items

### Prompt Template

```python
def henry_build_prompt(previous_orders_df, days_since_last, user_id):
    # Builds context including:
    # - Top 10 most frequently purchased products
    # - Top 5 departments
    # - Items from last order
    # - Recency information
```

### Usage

```python
# Test baseline instead of white agent
response = green_agent.execute_message(
    message="Assess baseline performance",
    context={"mode": "baseline"}
)
```

---

## Evaluation Metrics Detailed

### Precision-Recall-F1 Calculation

```python
def _prf1(truth: set, pred: set):
    TP = len(truth & pred)          # True Positives
    FP = len(pred - truth)          # False Positives
    FN = len(truth - pred)          # False Negatives
    
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

### Aisle/Department Mapping

Products are mapped to categories using the catalog:

```python
# Predicted products → aisles
pred_map_df = pd.DataFrame({"product_id": predicted_ids}).merge(
    products_catalog[["product_id", "aisle", "department"]],
    on="product_id",
    how="left"
)

pred_aisles = set(pred_map_df["aisle"].dropna())
pred_departments = set(pred_map_df["department"].dropna())
```

Missing mappings are tracked:
- `pred_aisle_missing`: Count of predicted products with no aisle mapping
- `pred_dept_missing`: Count of predicted products with no department mapping

### Interpretation

| Metric | Good Performance | What It Measures |
|--------|------------------|------------------|
| **Product F1** | > 0.6 | Exact product match accuracy |
| **Aisle F1** | > 0.7 | Category-level understanding |
| **Department F1** | > 0.8 | High-level shopping pattern capture |
| **Blended F1** | > 0.65 | Overall recommendation quality |

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `No rows for user_id=X` | User not in dataset | Check user_id exists in orders CSV |
| `Failed to parse white agent response` | Invalid JSON | Ensure white agent returns proper JSON format |
| `KeyError: 'product_id'` | Missing columns | Verify CSV files have required columns |
| Connection timeout | White agent not responding | Check white agent URL and health endpoint |

### Defensive Coding

The green agent includes several safety measures:

```python
# Type coercion for numeric columns
u["order_number"] = pd.to_numeric(u["order_number"], errors="coerce")

# Fallback for missing data
days_since_last = None if dsl_series.empty else float(dsl_series.dropna().iloc[0])

# JSON parsing with multiple formats
if "<json>" in response_text:
    tags = parse_tags(response_text)
    data = json.loads(tags["json"])
else:
    data = json.loads(response_text)
```

---

## Testing

### Manual Testing with cURL

```bash
# 1. Check agent health
curl http://localhost:9001/healthz

# 2. Get agent card
curl http://localhost:9001/agent_card
```

### Integration Testing

```python
# run test_a2a_green.py
```

---

## Performance Considerations

### Optimization Opportunities

1. **Caching**: Cache product catalog lookups
2. **Parallel Evaluation**: Assess multiple users concurrently
3. **Database Integration**: Use database instead of CSV for large datasets
4. **Streaming Results**: Report metrics progressively for long-running assessments

### Current Limitations

- **Single-threaded**: One assessment at a time per agent instance
- **In-memory storage**: Full datasets loaded into RAM
- **Synchronous I/O**: Blocking calls to white agent
- **No persistence**: Metrics not saved to disk

---

## References

### AgentBeats Documentation
- [AgentBeats Platform Guide](https://agentbeats.ai/docs)
- [A2A Protocol Specification](https://github.com/google-deepmind/a2a)

### Related Blog Posts
- "What is AgentBeats and why should you care as an agent developer?" (2025.9)
- "Agentify the Agent Assessment" (2025.10)

### Code Repository
- GitHub: [agentify-example-tau-bench](https://github.com/agentbeats/agentify-example-tau-bench)

---

## Contact & Support

For questions or issues with this green agent:
- Email: sec+agentbeats@berkeley.edu
- GitHub Issues: [Project Issues](https://github.com/agentbeats/agentify-example-tau-bench/issues)

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Maintainer**: AgentBeats Team
