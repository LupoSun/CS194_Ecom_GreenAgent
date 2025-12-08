# Green Agent Demo

AgentBeats-compatible **green assessment agent** for e-commerce basket prediction. The agent communicates over the A2A protocol, evaluates any compliant white agent, and reports blended F1 metrics across products, aisles, and departments.

## Quickstart
- Python 3.10+, `pip install -r requirements.txt`
- Optional: `python -m venv ../venv && source ../venv/bin/activate`
- Start green agent: `HOST=localhost AGENT_PORT=9001 python green_main_A2A.py`
- Start a white agent:
  - Real white agent: run your service and set `WHITE=http://...`
  - Stub: `cd ../white_agent_baseline && WHITE_HOST=localhost WHITE_PORT=9002 python stub_white_agent.py`
  - Optimized: `cd ../white_agent && bash run.sh` (requires `OPENAI_API_KEY`)
- Smoke test A2A: follow `docs/a2a_white_agent_interop_demo.md` for the curl + JSON-RPC flow.

## What’s Inside
- `green_main_A2A.py` — A2A server, task orchestration, scoring, and checkout wiring.
- `quick_test.py` — Quick testing script for evaluating white agents.
- `dataset/` — Trimmed Instacart-like CSVs used by the evaluator.
- `utils/` — A2A communication helpers and utility functions.
- `tests/` — Unit and integration tests.
- `docs/GREEN_AGENT_DOCUMENTATION.md` — Full architecture, endpoints, and prompt/eval details.
- `docs/a2a_white_agent_interop_demo.md` — Recorded-friendly walkthrough for green↔white interop.

White agent implementations are in sibling directories:
- `../white_agent/` — Optimized OpenAI-powered white agent.
- `../white_agent_baseline/` — Stub and baseline white agents for testing.

## Reuse & Extensibility
- Protocol-first: standard A2A endpoints (`/.well-known/agent-card.json`, `/send_message`, `/reset`) so any A2A-capable white agent can plug in.
- Pure-Python evaluator (`evaluate_basket`) accepts iterable product IDs and optional catalog joins; easy to embed or swap.
- Configurable: `ecom_green_agent.toml` and environment variables (`HOST`, `AGENT_PORT`, `ECOM_API_BASE`, `AGENT_KEY`, `WHITE`) control runtime without code edits.
- Baseline hooks: `use_baseline` flag and `henry_build_prompt` make it trivial to compare against the heuristic baseline.

## Documentation & Clarity
- Start here (README) for entry points.
- Deep dives live in `docs/GREEN_AGENT_DOCUMENTATION.md` (architecture, formats) and `docs/a2a_white_agent_interop_demo.md` (end-to-end demo).
- In-code docstrings focus on the reusable bits: data prep (`split_user_orders`), evaluation (`evaluate_basket`), and prompt building (`henry_build_prompt`).

## Testing Pointers
- Fast local interop: run the stub white agent from `../white_agent_baseline/` and hit the green agent with the sample payload from `docs/a2a_white_agent_interop_demo.md`.
- Use `quick_test.py` for rapid testing iterations.
- Swap `USER_ID` to vary histories; toggle `use_baseline` to sanity-check evaluator consistency.
- Run test suite: `python -m pytest tests/`

## Support
Issues or questions? Open a ticket with steps to reproduce and your A2A payload. Include agent logs from both green and white sides for faster turnaround.
