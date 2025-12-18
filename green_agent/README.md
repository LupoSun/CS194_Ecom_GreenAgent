# Green Agent

AgentBeats-compatible green assessment agent for e-commerce basket prediction. Communicates over the A2A protocol, evaluates white agents, and reports blended F1 metrics across products, aisles, and departments.

## Quickstart

**Prerequisites:** Python 3.10+ (or Docker)

### Option 1: Local Python

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the green agent:
```bash
bash run.sh
```

Or directly:
```bash
HOST=localhost AGENT_PORT=9001 python green_main_A2A.py
```

### Option 2: Docker (AgentBeats-compliant)

1. Build the image:
```bash
docker build -t ecom-green-agent .
```

2. Run the container:
```bash
docker run -p 9001:9001 ecom-green-agent
```

For publishing to AgentBeats, see `../AGENTBEATS_SETUP.md`.

### Testing

3. Start a white agent to test:
   - **Your agent**: Run your service on port 9002
   - **Stub agent**: `cd ../white_agent_baseline && python stub_white_agent.py`
   - **Optimized agent**: `cd ../white_agent && bash run.sh` (requires `OPENAI_API_KEY`)

4. Run a test:
```bash
python quick_test.py
```

For detailed A2A protocol testing, see `docs/a2a_white_agent_interop_demo.md`.
For AgentBeats platform deployment, see `../AGENTBEATS_SETUP.md`.

## Project Structure

- `green_main_A2A.py` - A2A server, task orchestration, scoring, and checkout
- `quick_test.py` - Quick testing script for white agents
- `dataset/` - Instacart-based CSV files (products, orders)
- `utils/` - A2A communication helpers and utilities
- `tests/` - Unit and integration tests
- `docs/GREEN_AGENT_DOCUMENTATION.md` - Full architecture and API reference
- `docs/a2a_white_agent_interop_demo.md` - Step-by-step interop walkthrough

White agent implementations:
- `../white_agent/` - Optimized OpenAI-powered white agent
- `../white_agent_baseline/` - Stub and baseline agents for testing

## Configuration

Environment variables:
- `HOST` - Host to bind to (default: `localhost`)
- `AGENT_PORT` - Port to run on (default: `9001`)
- `ECOM_API_BASE` - E-commerce API URL (default: Railway production URL)
- `AGENT_KEY` - Session key for API calls (auto-generated if not set)

Configuration file: `ecom_green_agent.toml`

## Testing

Run the full test suite:
```bash
python -m pytest tests/
```

Quick integration test:
```bash
# Terminal 1: Start a white agent
cd ../white_agent_baseline && python stub_white_agent.py

# Terminal 2: Run quick test
python quick_test.py
```

Test different users by changing `user_id` in the test payload.

## Documentation

- **Quick start**: This README
- **Full documentation**: `docs/GREEN_AGENT_DOCUMENTATION.md`
- **A2A interop guide**: `docs/a2a_white_agent_interop_demo.md`
- **FastAPI guide**: `docs/fastAPI_demo_guide.md`

## Support

For issues or questions, provide:
- Steps to reproduce
- A2A payload used
- Logs from both green and white agents
