# Baseline White Agents

This directory contains baseline white agent implementations for comparison and testing purposes. These agents provide different levels of functionality to establish performance benchmarks.

## Available Baselines

### 1. Stub Agent (`stub_white_agent.py`)

The simplest baseline - immediately signals completion without adding items to cart.

**Usage:**
```bash
python stub_white_agent.py
```

**Purpose:** Verify A2A protocol communication and green agent evaluation logic.

**Expected Performance:** F1 = 0 (no predictions made)

### 2. Baseline with Cart Replay (`stub_white_agent_baseline.py`)

Replays the user's last order by adding those exact items to the cart via the e-commerce API.

**Usage:**
```bash
# Simple run (uses default dataset)
python stub_white_agent_baseline.py

# With custom settings
WHITE_HOST=localhost \
WHITE_PORT=9002 \
ECOM_BASE=https://green-agent-production.up.railway.app \
ORDERS_CSV=../green_agent/dataset/super_shortened_orders_products_combined.csv \
python stub_white_agent_baseline.py
```

**Purpose:** Establish a reasonable baseline by assuming users reorder the same items.

**Expected Performance:** F1 ~0.3-0.4 (depends on user reorder rate)

### 3. GPT Baseline (`baseline_white_agent.py`)

Uses OpenAI GPT-4 with minimal prompting - no strategic guidance or optimization.

**Usage:**
```bash
export OPENAI_API_KEY="your-key-here"
bash run.sh
```

**Purpose:** Measure GPT-4 performance without specialized prompting or strategy.

**Expected Performance:** F1 ~0.2-0.35 (baseline LLM performance)

## Configuration

### Environment Variables

All baseline agents support:
- `HOST`: Host to bind to (default: `0.0.0.0`)
- `AGENT_PORT` or `WHITE_PORT`: Port to run on (default: `9002`)

Stub with cart replay also supports:
- `ECOM_BASE`: E-commerce API URL
- `ORDERS_CSV`: Path to orders dataset
- `PRODUCTS_CSV`: Path to products catalog

GPT baseline requires:
- `OPENAI_API_KEY`: Your OpenAI API key

## Testing Baselines

1. **Start a baseline agent:**
```bash
# Example: stub with cart replay
python stub_white_agent_baseline.py
```

2. **Start the green agent** (in another terminal):
```bash
cd ../green_agent
bash run.sh
```

3. **Run evaluation:**
```bash
cd ../green_agent
python quick_test.py
```

The green agent will evaluate the baseline and return metrics.

## Comparison Guide

Use these baselines to understand:
- **Stub agent**: Confirms your green agent works correctly
- **Cart replay baseline**: Shows how well "repeat last order" performs
- **GPT baseline**: Establishes vanilla LLM performance
- **Optimized agent** (in `../white_agent/`): Shows impact of prompt engineering and strategy

## Project Structure

```
white_agent_baseline/
├── stub_white_agent.py               # Fast stub (no cart additions)
├── stub_white_agent_baseline.py      # Cart replay baseline
├── baseline_white_agent.py           # Minimal GPT baseline
└── run.sh                            # Run GPT baseline
```

## Notes

- These agents are for testing and comparison only
- The optimized white agent is in the `../white_agent/` directory
- All baselines implement the same A2A protocol as the main white agent

## Contributing

See the main repository README for contribution guidelines.
