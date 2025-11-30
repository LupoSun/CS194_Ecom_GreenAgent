# 🎥 A2A Interop Demo: Green Agent ↔ White Agent

Step-by-step, recording-friendly script to prove the green agent speaks A2A and can assess **any** white agent.

---

## 🧰 Prereqs
- `cd green_agent_demo`
- `python -m venv ../venv && source ../venv/bin/activate` (or your env)
- `pip install -r requirements.txt`
- Environment:
  - `export OPENAI_API_KEY=...` (needed for the sample white agent in `agentbeats_src/white_agent`)
  - `export ECOM_API_BASE=http://localhost:8001` (or your checkout/mock base; used by the green agent)
- Suggested port map (match your recording overlays):
  - Green agent: `http://localhost:9001`
  - White agent: `http://localhost:9002`

For the recording, keep three panes: **white agent logs**, **green agent logs**, **commands**.

---

## 1️⃣ Start a white agent (A2A server)
Pick one:
- **Your real white agent**: start it however your team does; set `WHITE` to its base URL.
- **Stub white agent (recommended for local self-test)**: runs instantly and always returns the completion signal.

To run the stub:
```bash
cd green_agent_demo
WHITE_HOST=localhost WHITE_PORT=9002 python stub_white_agent.py
```
You should see “Starting stub white agent...” and the completion signal printed.

- **Baseline stub (replays last order and calls your live API)**: uses `/cart/add` with payload `{"agent_key": "...","items":[{"product_id":...,"qty":...}]}` against your `ECOM_BASE`.
```bash
cd green_agent_demo
WHITE_HOST=localhost WHITE_PORT=9002 \
ECOM_BASE=https://green-agent-production.up.railway.app \
ORDERS_CSV=dataset/super_shortened_orders_products_combined.csv \
python stub_white_agent_baseline.py
```
```bash
cd green_agent_demo
python - <<'PY'
from agentbeats_src.white_agent.agent import start_white_agent
start_white_agent(host="localhost", port=9002)
PY
```
You should see “Starting white agent...” and Uvicorn boot logs.

---

## 2️⃣ Start the green agent (A2A assessment server)
In another terminal:
```bash
cd green_agent_demo
HOST=localhost AGENT_PORT=9001 python main_A2A.py
```
Watch for “AgentBeats-compatible Green Agent...” in the logs.

---

## 3️⃣ Quick card checks (record this)
In the commands pane:
```bash
GREEN=http://localhost:9001
WHITE=http://localhost:9002

curl -s $GREEN/.well-known/agent-card.json | jq '.name, .version'
curl -s $WHITE/.well-known/agent-card.json | jq '.name, .version'
```
If your build exposes extra health endpoints, feel free to show them, but they are not required; the card check is sufficient with this route set.

---

## 4️⃣ Build the A2A request payload (green → white)
Generate a JSON-RPC file that tells the green agent to route the task to the white agent.
```bash
USER_ID=1
AGENT_KEY="a2a-demo-$(date +%s)"   # unique cart namespace
ECOM_BASE="https://green-agent-production.up.railway.app"

python - <<'PY'
import json, os, pathlib
payload = {
  "jsonrpc": "2.0",
  "id": "green-a2a-demo",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "messageId": "demo-msg-1",
      "kind": "message",
      "parts": [{
        "kind": "text",
        "text": json.dumps({
          "user_id": int(os.environ["USER_ID"]),
          "white_agent_url": os.environ["WHITE"],
          "environment_base": os.environ["ECOM_BASE"],
          "agent_key": os.environ["AGENT_KEY"],
          "use_baseline": False
        })
      }]
    }
  }
}
out = pathlib.Path("/tmp/a2a_white_task.json")
out.write_text(json.dumps(payload, indent=2))
print(f"Wrote {out}")
PY
cat /tmp/a2a_white_task.json
```

---

## 5️⃣ Send the task via A2A (record terminal + logs)
```bash
curl -s -X POST $GREEN/ \
  -H "Content-Type: application/json" \
  -d @/tmp/a2a_white_task.json | jq
```

What to capture on screen:
- Green agent log: “Mode=WHITE_AGENT” then “Sending task to white agent...”
- White agent log: should show the task prompt; it will iterate and eventually emit `##READY_FOR_CHECKOUT##`.
- Green agent log: “Completion signal received” then a checkout call.

---

## 6️⃣ Expected response snippet
The `jq` output should include metrics similar to:
```json
{
  "jsonrpc": "2.0",
  "id": "green-a2a-demo",
  "result": {
    "role": "agent",
    "parts": [
      {
        "kind": "text",
        "text": "Assessment Complete ✅\n\nMetrics:\n- F1 Score: 0.70–0.75\n- Precision: ...\n- Recall: ...\n- Blended F1: ...\n- Latency: <2s"
      }
    ]
  }
}
```
Numbers vary by user_id and white-agent behavior; the key proof is that metrics return and no timeout occurs.

---

## 7️⃣ Variations to capture (optional)
- Swap `USER_ID=100825` to show different histories.
- Point `WHITE` to another A2A-compatible agent URL to prove interchangeability.
- Flip `use_baseline` to `true` to show the green agent can self-benchmark when no white agent is provided.

---

## ✅ Success criteria
- Agent cards reachable for both agents.
- White agent receives the prompt and sends `##READY_FOR_CHECKOUT##`.
- Green agent completes evaluation and returns metrics (F1/precision/recall/blended).
- Checkout call succeeds using `ECOM_API_BASE` (non-empty `items` list).
