# E-commerce Auto-Shopping Green Agent

A benchmark evaluation agent for testing AI agents' ability to predict grocery shopping behavior and use e-commerce APIs effectively.

> **🎯 For AgentBeats Users:** This is a production-ready green agent that evaluates shopping prediction capabilities. Deploy it to test white agents, or use the reference implementations to build your own. See [AgentBeats Integration](#agentbeats-platform-integration) for details.

## Overview

This **green agent** evaluates how well **white agents** (the agents being tested) can predict what a user will purchase on their next grocery shopping trip based on their purchase history. White agents must use a real e-commerce API to search for products, build a basket, and complete the task.

### What is a Green Agent?

In the AgentBeats framework:
- **Green agents** are evaluation/benchmark agents that test other agents
- **White agents** are the agents being tested/evaluated

This repository provides both a complete green agent implementation and reference white agent implementations.

### Key Features

- **Real-world dataset**: Built on the [Instacart Kaggle dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) with 1,500+ unique users and 30,000+ transactions
- **Production e-commerce API**: Hosted at `https://green-agent-production.up.railway.app/` with search, cart, and checkout functionality
- **Robust evaluation**: Multi-level F1 scoring (products, aisles, departments) with blended metrics
- **AgentBeats/A2A compatible**: Fully implements the A2A protocol for agent-to-agent communication
- **Multiple evaluation modes**: Single user, baseline comparison, and multi-user benchmarks
- **Easy deployment**: Works locally or on cloud platforms (Railway, Google Cloud Run, etc.)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [API Reference](#api-reference)
- [AgentBeats Integration](#agentbeats-platform-integration)
- [Development](#development)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/henrymichaelson/CS194_Ecom_GreenAgent.git
cd CS194_Ecom_GreenAgent
```

2. Install dependencies for green agent:
```bash
cd green_agent
pip install -r requirements.txt
```

Or for white agent:
```bash
cd white_agent
pip install -r requirements.txt
```

3. (Optional) Configure environment variables:

The agents work out-of-the-box with sensible defaults. You can optionally set:

**Green Agent Environment Variables:**
- `ECOM_API_BASE`: Railway e-commerce API base URL (default: `https://green-agent-production.up.railway.app`)
- `PRODUCTS_CSV`: Path to products dataset (default: `dataset/ic_products.csv`)
- `ORDERS_CSV`: Path to orders dataset (default: `dataset/super_shortened_orders_products_combined.csv`)
- `HOST`: Host to bind to (default: `localhost`, use `0.0.0.0` for production)
- `AGENT_PORT`: Port to run on (default: `9001`)
- `AGENT_URL`: URL where this green agent will be accessible (auto-configured by AgentBeats)

**White Agent Environment Variables:**
- `OPENAI_API_KEY`: Required for OpenAI-powered white agents
- `HOST`: Host to bind to (default: `0.0.0.0`)
- `AGENT_PORT`: Port to run on (default: `9002`)

## Quick Start

### Running the Green Agent Locally

The green agent is the evaluation/orchestration server that assesses white agents.

```bash
cd green_agent
bash run.sh
# Or directly:
export ROLE=green
python green_main_A2A.py
```

The agent will start on `http://localhost:9001` (or configured host/port).

**Verify it's running:**
```bash
curl http://localhost:9001/.well-known/agent-card.json | jq
```

### Run a local stub white agent (for self-tests)

Fast stub (signals completion only):
```bash
python white_agent_baseline/stub_white_agent.py  # defaults: host=0.0.0.0, port=9002
```

Baseline stub (replays last order to cart via live API):
```bash
# Simple run (uses default dataset automatically)
python white_agent_baseline/stub_white_agent_baseline.py

# Or with custom settings:
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
cd green_agent
python quick_test.py
```

This sends a test request to evaluate a white agent's performance on a single user. Edit `quick_test.py` to configure the white agent URL and test parameters.

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

## AgentBeats Platform Integration

### For AgentBeats Users

This repository provides a complete e-commerce shopping prediction benchmark that you can:
1. **Use as a green agent**: Deploy the green agent to evaluate your own white agents
2. **Use as a reference**: Study the white agent implementations to build your own
3. **Fork and extend**: Customize the evaluation logic or datasets for your needs

**What you get:**
- A fully functional green agent ready to deploy on AgentBeats
- Multiple white agent implementations (stub, baseline, optimized)
- Production e-commerce API (no setup required)
- Real-world dataset with 1,500+ users
- Comprehensive evaluation metrics

**Quick deployment to AgentBeats:**
```bash
# Install controller
pip install earthshaker

# Navigate to green agent
cd green_agent

# Start with controller
PORT=9001 HOST=0.0.0.0 agentbeats run_ctrl
```

The controller will use the existing `run.sh` script and expose your green agent with proper A2A endpoints.

### For White Agent Developers

To test your white agent against this benchmark:

1. **Implement A2A protocol**: Your agent must respond to JSON-RPC `message/send` requests
2. **Handle the prompt**: Parse user history and shopping context from the green agent
3. **Use the e-commerce API**: Search products, add to cart using the provided `agent_key`
4. **Signal completion**: Send `##READY_FOR_CHECKOUT##` when done adding items
5. **Get your score**: The green agent returns F1, precision, recall, and blended metrics

**Example white agent payload you'll receive:**
```json
{
  "user_id": 123,
  "white_agent_url": "http://your-agent.com/",
  "environment_base": "https://green-agent-production.up.railway.app",
  "agent_key": "unique-session-key",
  "use_baseline": false
}
```

See `white_agent/my_white_agent.py` for a complete reference implementation.

## Development

### Local Development Setup

1. **Set up your environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd green_agent
pip install -r requirements.txt
```

2. **Run tests:**
```bash
cd green_agent
python -m pytest tests/
```

3. **Start development server:**
```bash
cd green_agent
export ROLE=green
python green_main_A2A.py
```

### Cloud Deployment Options

**Option A: Railway (Easiest)**
1. Connect your GitHub repository to Railway
2. Set build command: `cd green_agent && pip install -r requirements.txt`
3. Set start command: `cd green_agent && bash run.sh`
4. Railway automatically provides HTTPS

**Option B: Google Cloud Run**
```bash
# From green_agent directory
gcloud builds submit --pack image=gcr.io/PROJECT/green-agent
gcloud run deploy green-agent \
  --image gcr.io/PROJECT/green-agent \
  --platform managed \
  --allow-unauthenticated
```

**Option C: Docker**
```bash
cd white_agent
docker build -t my-white-agent .
docker run -p 9002:9002 -e OPENAI_API_KEY=$OPENAI_API_KEY my-white-agent
```

### Publishing to AgentBeats

Once deployed with the AgentBeats controller:
1. Visit the AgentBeats platform website
2. Submit your controller URL (e.g., `https://your-green-agent.run.app`)
3. Fill out agent metadata (name, description, skills)
4. Your agent becomes discoverable for evaluations!

### Project Structure

```
CS194_Ecom_GreenAgent/
├── green_agent/                 # Green agent (evaluation/orchestration)
│   ├── green_main_A2A.py        # Main green agent server
│   ├── quick_test.py            # Quick testing script
│   ├── run.sh                   # Startup script
│   ├── requirements.txt         # Python dependencies
│   ├── ecom_green_agent.toml    # Agent card configuration
│   ├── utils/
│   │   ├── my_a2a.py           # A2A communication helpers
│   │   └── __init__.py         # Utility functions (parse_tags)
│   ├── dataset/
│   │   ├── ic_products.csv      # Product catalog (~50k products)
│   │   ├── super_shortened_orders_products_combined.csv  # User orders
│   │   └── tasks.json          # Task definitions
│   ├── tests/
│   │   ├── test_core.py        # Core functionality tests
│   │   ├── test_evaluate_basket.py  # Basket evaluation tests
│   │   └── test_integration.py # Integration tests
│   └── docs/
│       ├── a2a_white_agent_interop_demo.md  # A2A interop walkthrough
│       ├── fastAPI_demo_guide.md            # FastAPI guide
│       └── GREEN_AGENT_DOCUMENTATION.md     # Full documentation
├── white_agent/                # Optimized white agent implementation
│   ├── my_white_agent.py        # OpenAI-powered white agent
│   ├── run.sh                   # Startup script
│   ├── requirements.txt         # Python dependencies
│   ├── test_my_agent.py         # Test script
│   ├── Dockerfile               # Container configuration
│   └── tests/
│       └── test_white_agent_integration.py  # Integration tests
└── white_agent_baseline/       # Baseline white agents for comparison
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

Run the test suite to verify core functionality:

```bash
cd green_agent
python -m pytest tests/ -v
```

Test coverage includes:
- Core evaluation logic (`test_core.py`)
- Basket evaluation metrics (`test_evaluate_basket.py`)
- End-to-end integration (`test_integration.py`)

### Integration Testing

**Option 1: Quick test with provided script**
```bash
cd green_agent
python quick_test.py
```

**Option 2: Full integration test**
```bash
# Terminal 1: Start a white agent
cd white_agent_baseline
python stub_white_agent_baseline.py

# Terminal 2: Start green agent
cd green_agent
bash run.sh

# Terminal 3: Send evaluation request
cd green_agent
python -c "
import asyncio
import json
from utils.my_a2a import send_message

asyncio.run(send_message(
    'http://localhost:9001',
    json.dumps({
        'user_id': 1,
        'white_agent_url': 'http://localhost:9002/',
        'environment_base': 'https://green-agent-production.up.railway.app',
        'agent_key': 'test-key-123'
    })
))
"
```

**Option 3: Test with your own white agent**
1. Ensure your white agent implements A2A protocol
2. Start your agent on a known port (e.g., 9002)
3. Start the green agent: `cd green_agent && bash run.sh`
4. Send a test request with your agent's URL

### Baseline Comparisons

Test different baseline agents to understand performance expectations:

```bash
# Stub agent (instant, minimal baseline)
cd white_agent_baseline
python stub_white_agent.py

# Baseline with cart replay (repeats last order)
python stub_white_agent_baseline.py

# GPT-powered baseline (minimal prompting)
bash run.sh
```


## FAQ

### How do I test my white agent against this benchmark?

1. Deploy your white agent with A2A protocol support
2. Ensure it can parse shopping history and use the e-commerce API
3. Send a request to this green agent with your white agent's URL
4. The green agent will evaluate and return metrics

### What metrics will I get?

You'll receive:
- **F1 Score**: Balance of precision and recall at product level
- **Precision**: How many predicted items were correct
- **Recall**: How many actual items were predicted
- **Blended F1**: Weighted score across products, aisles, departments
- **Latency**: Response time in milliseconds

### Can I use this locally without deployment?

Yes! Follow the [Quick Start](#quick-start) guide. The e-commerce API is already hosted, so you only need to run the agents locally.

### What's the difference between green and white agents?

- **Green agents** (this repo): Evaluation/benchmark agents that test other agents
- **White agents**: The agents being tested/evaluated

### How do I improve my white agent's score?

1. **Better product matching**: Use search effectively to find the right products
2. **Historical patterns**: Pay attention to frequently reordered items
3. **Category understanding**: Items from the same aisle/department earn partial credit
4. **Efficient API use**: Minimize latency while maximizing coverage

## Contributing

### Team

- **Henry Michaelson** (hmichaelson@berkeley.edu)
- **Tao Sun** (tao_sun@berkeley.edu) 
- **Arlen Kumar** (arlen1788@berkeley.edu)

### How to Contribute

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and test thoroughly
4. Run tests: `cd green_agent && python -m pytest tests/`
5. Submit a pull request

**Areas for contribution:**
- New evaluation metrics
- Additional baseline agents
- Performance optimizations
- Documentation improvements
- Dataset expansions

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
