# E-commerce Auto-Shopping Green Agent

A benchmark evaluation agent for testing AI agents' ability to predict grocery shopping behavior and use e-commerce APIs effectively.

## Overview

This **green agent** evaluates how well **white agents** (the agents being tested) can predict what a user will purchase on their next grocery shopping trip based on their purchase history. White agents must use a real e-commerce API to search for products, build a basket, and complete the task.

### Key Features

- **Real-world dataset**: Built on the [Instacart Kaggle dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) with 1,500+ unique users and 30,000+ transactions
- **Production e-commerce API**: Hosted at `https://green-agent-production.up.railway.app/` with search, cart, and checkout functionality
- **Robust evaluation**: Multi-level F1 scoring (products, aisles, departments) with blended metrics
- **AgentBeats/A2A compatible**: Implements the A2A protocol for agent-to-agent communication
- **Multiple evaluation modes**: Single user, baseline comparison, and multi-user benchmarks

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [API Reference](#api-reference)
- [Development](#development)
- [Examples](#examples)

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CS194_Ecom_GreenAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
- `ECOM_API_BASE`: Railway e-commerce API base URL (default: production URL)
- `PRODUCTS_CSV`: Path to products dataset
- `ORDERS_CSV`: Path to orders dataset
- `AGENT_URL`: URL where this green agent will be accessible (set by AgentBeats)
- `HOST`: Host to bind to (default: localhost, use 0.0.0.0 for production)
- `AGENT_PORT`: Port to run on (default: 9001)

## Quick Start

### Running the Green Agent Locally

Start the green agent server directly:

```bash
bash run.sh
# Or directly:
export ROLE=green
python green_main_A2A.py
```

The agent will start on `http://localhost:9001` (or configured host/port).

### Run a local stub white agent (for self-tests)

Fast stub (signals completion only):
```bash
python white_agent_baseline/stub_white_agent.py  # defaults: host=0.0.0.0, port=9002
```

Baseline stub (replays last order to cart via live API):
```bash
WHITE_HOST=localhost WHITE_PORT=9002 \
ECOM_BASE=https://green-agent-production.up.railway.app \
ORDERS_CSV=green_agent/dataset/super_shortened_orders_products_combined.csv \
python white_agent_baseline/stub_white_agent_baseline.py
```

Baseline GPT agent (minimal prompting):
```bash
cd white_agent_baseline && bash run.sh
```

The baseline agents provide different levels of functionality for testing. Set `white_agent_url` in your payload to `http://localhost:9002` when using any baseline agent.

### Running with AgentBeats Controller (Recommended for Platform Integration)

To integrate with the AgentBeats platform and enable easy resets between test runs:

1. **Install AgentBeats controller**:
```bash
pip install earthshaker
```

2. **Create a run.sh**
```bash
export ROLE=green
export CLOUDRUN_HOST=ecom.taosun.net
python green_agent/green_main_A2A.py
```

2. **Start the controller** (it will automatically use `run.sh`):
```bash
PORT=8010 \
HOST=0.0.0.0 \
CLOUDRUN_HOST=ecom.taosun.net \
agentbeats run_ctrl
```

```bash
PORT=8011 \
HOST=0.0.0.0 \
CLOUDRUN_HOST=ecom_white.taosun.net \
agentbeats run_ctrl
```

The controller will:
- Start/stop/restart your agent via API
- Proxy requests to your agent
- Provide a management UI for monitoring
- Handle environment variables (`$HOST`, `$AGENT_PORT`)

3. **Access your agent** through the controller proxy URL (e.g., `http://localhost:8080/.well-known/agent-card.json`)

The `run.sh` script is already configured to work with the controller.

### Testing with Quick Test Script

Run a quick test against a white agent:

```bash
python quick_test.py
```

This sends a test request to evaluate a white agent's performance on a single user.

## Architecture

### System Components

```
┌─────────────────┐
│  Green Agent    │ ← Orchestrates evaluation
│  (This Repo)    │
└────────┬────────┘
         │
         ├──→ ┌─────────────────┐
         │    │ Railway API     │ ← E-commerce environment
         │    │ (Hosted)        │
         │    └─────────────────┘
         │
         └──→ ┌─────────────────┐
              │  White Agent    │ ← Agent being tested
              │  (External)     │
              └─────────────────┘
```

### Data Flow

1. **Green agent** sends user purchase history to **white agent**
2. **White agent** interacts with **Railway e-commerce API**:
   - Searches for products
   - Adds items to cart (using unique `agent_key`)
3. **White agent** signals completion with `##READY_FOR_CHECKOUT##`
4. **Green agent** calls Railway `/checkout` to retrieve final cart
5. **Green agent** compares predicted basket vs. ground truth and returns metrics

## Usage

### Evaluation Modes

#### 1. Single User Assessment (White Agent Mode)

Evaluate a white agent on one user:

```python
import asyncio
import json
from utils.my_a2a import send_message

async def evaluate_single_user():
    response = await send_message(
        "http://localhost:9001",  # Green agent URL
        json.dumps({
            "user_id": 123,
            "mode": "white_agent",
            "white_agent_url": "http://localhost:9002/",
            "environment_base": "https://green-agent-production.up.railway.app",
            "agent_key": "user123"
        })
    )
    print(response)

asyncio.run(evaluate_single_user())
```

#### 2. Baseline Mode

Test baseline policy (repeats previous order):

```python
response = await send_message(
    "http://localhost:9001",
    json.dumps({
        "user_id": 123,
        "use_baseline": True
    })
)
```

#### 3. Benchmark Mode (Multiple Users)

Run comprehensive evaluation across multiple users:

```python
response = await send_message(
    "http://localhost:9001",
    json.dumps({
        "mode": "benchmark",
        "num_users": 100,
        "white_agent_url": "http://localhost:9002/",
        "environment_base": "https://green-agent-production.up.railway.app",
        "use_baseline": False,
        "random_state": 42  # For reproducibility
    })
)
```

### Configuration Format

#### XML-Wrapped Format (AgentBeats Platform)

```xml
<config>
{
  "user_id": 123,
  "mode": "white_agent",
  "white_agent_url": "http://localhost:9002/",
  "agent_key": "user123"
}
</config>
<white_agent_url>http://localhost:9002/</white_agent_url>
```

#### Direct JSON Format

```json
{
  "user_id": 123,
  "mode": "white_agent", 
  "white_agent_url": "http://localhost:9002/",
  "environment_base": "https://green-agent-production.up.railway.app",
  "agent_key": "user123",
  "task_id": "optional_task_id"
}
```

## Evaluation Metrics

### Primary Metric: Blended F1 Score

The evaluation uses a multi-level approach:

```
Blended F1 = 0.6 × Product_F1 + 0.2 × Aisle_F1 + 0.2 × Department_F1
```

This rewards:
- **Exact matches** (same product) → Full credit
- **Similar items** (same aisle/department) → Partial credit

### Metric Breakdown

For each level (products, aisles, departments):
- **Precision**: `TP / (TP + FP)` - How many predicted items were correct?
- **Recall**: `TP / (TP + FN)` - How many actual items were predicted?
- **F1 Score**: `2 × (Precision × Recall) / (Precision + Recall)`

### Example Output

```
Assessment Complete ✅

Metrics:
- F1 Score: 0.308
- Precision: 1.000
- Recall: 0.182
- Blended F1: 0.366
- Latency: 1250ms

Product-level: 2 TP, 0 FP, 9 FN
Aisle-level F1: 0.333
Department-level F1: 0.571
```

## API Reference

### Railway E-commerce API

**Base URL**: `https://green-agent-production.up.railway.app`

All endpoints require an `agent_key` parameter to isolate agent sessions.

#### Search Products

```http
GET /search_products?query=milk&agent_key=your_key
```

**Response**:
```json
{
  "results": [
    {
      "product_id": 196,
      "product_name": "Organic Whole Milk",
      "aisle": "milk",
      "department": "dairy eggs"
    }
  ]
}
```

#### Get Product Details

```http
GET /get_product?product_id=196&agent_key=your_key
```

#### Add to Cart (live API schema)

```http
POST /cart/add
Content-Type: application/json

{
  "agent_key": "your_key",
  "items": [
    {"product_id": 196, "qty": 1},
    {"product_id": 25133, "qty": 2}
  ]
}
```

#### View Cart

```http
GET /cart?agent_key=your_key
```

#### Checkout (Called by Green Agent)

```http
POST /checkout
Content-Type: application/json

{
  "agent_key": "your_key"
}
```

**Response**:
```json
{
  "items": [
    {"product_id": 196, "qty": 1},
    {"product_id": 25133, "qty": 2}
  ],
  "total_items": 3
}
```

### Green Agent A2A Endpoints

- `GET /.well-known/agent-card.json` → Agent card
- `GET /.well-known/agent.json`      → Agent info (if exposed by controller)
- `POST /`                           → JSON-RPC `message/send` entrypoint

Example assessment request:
```http
POST /
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {"kind": "text", "text": "{\"user_id\":1,\"white_agent_url\":\"http://localhost:9002\",\"environment_base\":\"https://green-agent-production.up.railway.app\",\"agent_key\":\"demo-key\",\"use_baseline\":false}"}
      ]
    }
  }
}
```

## Development

### AgentBeats Platform Integration

To run assessments on the AgentBeats platform, you need to deploy your green agent with the AgentBeats controller. This enables:

- **Remote access**: Others can run assessments against your green agent
- **Easy resets**: Multiple test runs without manual restarts
- **Management UI**: Monitor and control your agent
- **Platform discovery**: Your agent becomes discoverable on AgentBeats

#### Integration Steps

**1. Wrap with AgentBeats Controller**

The controller is already configured via `run.sh`. It handles:
- Starting/stopping the agent process
- Proxying HTTP requests
- Exposing service API for state management
- Setting environment variables (`$HOST`, `$AGENT_PORT`)

**2. Deploy Your Agent**

Choose a deployment method:

**Option A: Cloud VM Deployment**
```bash
# On your cloud VM (with public IP)
1. Clone repository
2. Install dependencies: pip install -r requirements.txt
3. Install controller: pip install earthshaker
4. Set up SSL/TLS (e.g., with Nginx)
5. Start controller: agentbeats run_ctrl
```

**Option B: Container Deployment (Recommended)**

Create a `Procfile`:
```
web: agentbeats run_ctrl
```

Build and deploy (example with Google Cloud):
```bash
# Build with Cloud Buildpacks
gcloud builds submit --pack image=gcr.io/PROJECT/green-agent

# Deploy to Cloud Run
gcloud run deploy green-agent \
  --image gcr.io/PROJECT/green-agent \
  --platform managed \
  --allow-unauthenticated
```

Cloud Run automatically provides HTTPS and scales the container.

**3. Publish on AgentBeats**

Once deployed:
1. Visit the AgentBeats platform
2. Submit your controller URL (e.g., `https://your-agent.run.app`)
3. Fill out the agent registration form
4. Your agent is now discoverable for assessments!

#### Controller Management UI

Access the controller's management page at your deployment URL to:
- View agent status (running/stopped)
- Start/restart/stop the agent
- Monitor logs
- Test agent endpoints

Example controller endpoints:
- `GET /` - Management UI
- `GET /.well-known/agent-card.json` - Agent card (proxied)
- `POST /agent/start` - Start agent
- `POST /agent/stop` - Stop agent
- `POST /agent/restart` - Restart agent

### Project Structure

```
CS194_Ecom_GreenAgent/
├── green_agent/             # Green agent (evaluation/orchestration)
│   ├── green_main_A2A.py        # Main green agent server
│   ├── quick_test.py            # Quick testing script
│   ├── run.sh                   # Startup script
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # Environment configuration
│   ├── ecom_green_agent.toml    # Agent card configuration
│   ├── utils/
│   │   ├── my_a2a.py           # A2A communication helpers
│   │   └── __init__.py         # Utility functions (parse_tags)
│   ├── dataset/
│   │   ├── ic_products.csv      # Product catalog (~50k products)
│   │   └── super_shortened_orders_products_combined.csv  # User orders
│   └── docs/
│       ├── a2a_white_agent_interop_demo.md  # A2A white-agent interop walkthrough
│       └── fastAPI_demo_guide.md
├── white_agent/             # Optimized white agent
│   ├── my_white_agent.py        # OpenAI-powered white agent
│   ├── run.sh                   # Startup script
│   ├── requirements.txt         # Python dependencies
│   └── test_my_agent.py         # Test script
└── white_agent_baseline/    # Baseline white agents
    ├── baseline_white_agent.py  # Minimal GPT baseline (no prompting)
    ├── stub_white_agent.py      # Fast stub (signals completion only)
    ├── stub_white_agent_baseline.py  # Replays last order to cart
    └── run.sh                   # Run baseline GPT agent
```

### Key Classes

#### `EcomGreenAgentExecutor`

Main executor class that handles:
- User sampling and data preparation
- Baseline policy (repeat previous order)
- White agent communication via A2A protocol
- Metric calculation and reporting

**Key Methods**:
- `execute()` - Main entry point for assessment requests
- `_run_single_assessment()` - Evaluate one user
- `_run_benchmark()` - Evaluate multiple users
- `_white_agent_policy()` - Coordinate with white agent
- `_baseline_policy()` - Baseline comparison
- `_call_railway_checkout()` - Retrieve final cart from Railway

### Dataset Schema

#### Products CSV
```csv
product_id,product_name,aisle_id,department_id,aisle,department
196,Soda,77,7,soft drinks,beverages
```

#### Orders CSV
```csv
order_id,user_id,order_number,order_dow,order_hour_of_day,days_since_prior_order,product_id,add_to_cart_order,reordered,product_name,aisle_id,department_id,aisle,department
```

## Examples

### Example 1: Prompt Generated for White Agent

```
You are a grocery shopping assistant for user 100.
It has been 12 days since their last order.
Using the user's purchase history below, propose the next basket.

Top departments by repeat count:
- produce (x6)
- deli (x4)
- meat seafood (x3)

Top products by repeat count:
- [27344] Uncured Genoa Salami (x2)
- [30795] Sesame Seaweed Salad (x2)
- [21616] Organic Baby Arugula (x2)

Most recent orders:
- Order #4 (id 2337051): Uncured Genoa Salami, Smoked Salmon, Sesame Seaweed Salad, Banana...

### Instructions:
1. Use /search_products to find items
2. Use /get_product for details
3. Use /add_to_cart to add items (one at a time)
4. Use /cart to verify your selections
5. When you're done adding items, send: "##READY_FOR_CHECKOUT##"

IMPORTANT: Do NOT attempt to checkout. Simply send "##READY_FOR_CHECKOUT##" when ready.
```

### Example 2: White Agent Implementation (Pseudocode)

```python
# White agent receives prompt with user history
user_prompt = green_agent_message

# Parse top products from history
top_products = extract_top_products(user_prompt)

# Search and add to cart
for product_name in top_products:
    results = api.search_products(product_name)
    if results:
        api.add_to_cart(results[0].id, quantity=1)

# Signal completion
send_message_to_green_agent("##READY_FOR_CHECKOUT##")
```

### Example 3: Evaluation Output

```json
{
  "task_id": "user100_task",
  "user_id": 100,
  "metrics": {
    "precision": 1.0,
    "recall": 0.182,
    "f1": 0.308,
    "blended_f1": 0.366,
    "latency_ms": 1250
  },
  "proposed_items": {
    "196": 1,
    "25133": 2
  },
  "mode": "white_agent"
}
```

## Testing

### Unit Testing

```bash
# Test individual components
python -m pytest tests/
```

### Integration Testing

```bash
# Test with baseline agent
python quick_test.py

# Test with your white agent
# 1. Start your white agent on port 9002
# 2. Run green agent
cd green_agent && bash run.sh
# 3. Send evaluation request
cd green_agent && python -c "
import asyncio
import json
from utils.my_a2a import send_message

asyncio.run(send_message(
    'http://localhost:9001',
    json.dumps({'user_id': 1, 'white_agent_url': 'http://localhost:9002/'})
))
"
```


## Contributing

### Team

- **Henry Michaelson** (hmichaelson@berkeley.edu)
- **Tao Sun** (tao_sun@berkeley.edu) 
- **Arlen Kumar** (arlen1788@berkeley.edu) 

### Related Work

Our benchmark builds upon and extends existing e-commerce agent research:

1. **DeepShop**: A Benchmark for Deep Research Shopping Agents ([arxiv](https://arxiv.org/html/2506.02839v1))
2. **AgentRecBench**: Benchmarking LLM Agent-based Personalized Recommender Systems ([arxiv](https://arxiv.org/html/2505.19623v2))
3. **What Is Your AI Agent Buying?** ([arxiv](https://arxiv.org/pdf/2508.02630))

**Novel Contributions**:
- Combines predictive modeling with real-time API tool use
- Multi-level evaluation (products + categories)
- Contamination-resistant (random user sampling per run)
- Production-grade e-commerce environment

## License

[Add your license here]

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{ecom-green-agent-2024,
  title={E-commerce Auto-Shopping Green Agent: A Benchmark for AI Shopping Assistants},
  author={Michaelson, Henry and Sun, Tao and Kumar, Arlen},
  year={2025},
  institution={UC Berkeley}
}
```

## Acknowledgments

- **Dataset**: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)
- **Platform**: [AgentBeats](https://agentbeats.org)
- **Course**: CS194/CS294 Agentic AI, UC Berkeley Fall 2025
