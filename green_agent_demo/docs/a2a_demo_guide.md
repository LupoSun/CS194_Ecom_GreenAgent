# 🧪 A2A Green Agent Demo (No White Agent Needed)

This demo shows your A2A green agent running in **baseline mode** - where it assesses its own built-in baseline policy instead of an external white agent.

---

## 🔧 Setup

```bash
BASE=http://localhost:9001
USER_ID=1
```

Another test case:
```bash
USER_ID=100825
```

---

## 1️⃣ Check Agent is Running

### Get Agent Card
```bash
curl -s $BASE/agent_card | jq
```

**Expected Output:**
```json
{
  "name": "ecom_green_agent",
  "version": "1.0.0",
  "description": "Ecom grocery basket assessment agent using Henry's evaluator",
  "url": "http://localhost:9001",
  ...
}
```

### Health Check
```bash
curl -s $BASE/healthz | jq
```

**Expected Output:**
```json
{
  "status": "healthy",
  "agent": "ecom_green_agent"
}
```

---

## 2️⃣ Reset Agent State (Clear Previous Runs)

```bash
curl -s -X POST $BASE/reset | jq
```

**Expected Output:**
```json
{
  "status": "reset",
  "message": "Agent state cleared"
}
```

---

## 3️⃣ Send Assessment Task (Baseline Mode)

### Create task configuration
```bash
cat > /tmp/a2a_request.json << 'EOF'
{
  "jsonrpc": "2.0",
  "id": "demo-1",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "messageId": "test-msg-001",
      "kind": "message",
      "parts": [
        {
          "kind": "text",
          "text": "{\"user_id\": 1, \"use_baseline\": true}"
        }
      ]
    }
  }
}
EOF
```

### Send via A2A protocol
```bash
curl -s -X POST http://localhost:9001/ \
  -H "Content-Type: application/json" \
  -d @/tmp/a2a_request.json | jq
```

**Expected Output:**
```json
{
  "jsonrpc": "2.0",
  "id": "demo-1",
  "result": {
    "role": "agent",
    "parts": [
      {
        "kind": "text",
        "text": "Assessment Complete ✅\n\nMetrics:\n- F1 Score: 0.XXX\n- Precision: 0.XXX\n..."
      }
    ],
    ...
  }
}
```

---

## Or Using Python

Create a test script:

```python
# test_a2a_green.py
import asyncio
import sys
sys.path.append('ab_src')  # Adjust to your path

from my_a2a import send_message
import json

async def test_green_agent():
    green_url = "http://localhost:9001"
    
    # Test 1: Baseline mode
    task_config = {
        "user_id": 1,
        "task_id": "test_baseline_user1",
        "use_baseline": True,
        "environment_base": "http://localhost:8001"
    }
    
    print("📤 Sending task to green agent...")
    print(f"Config: {json.dumps(task_config, indent=2)}")
    
    response = await send_message(
        green_url,
        json.dumps(task_config),
        task_id=task_config["task_id"]
    )
    
    print("\n📥 Response from green agent:")
    print(response)
    
    # Extract result text
    if hasattr(response, 'root') and hasattr(response.root, 'result'):
        result = response.root.result
        if hasattr(result, 'parts') and result.parts:
            from a2a.utils import get_text_parts
            text_parts = get_text_parts(result.parts)
            if text_parts:
                print("\n✅ Assessment Result:")
                print(text_parts[0])

if __name__ == "__main__":
    asyncio.run(test_green_agent())
```

Run it:
```bash
python test_a2a_green.py
```

---

## 7️⃣ Expected Results

### For User 1 (Baseline)
```
Assessment Complete ✅

Metrics:
- F1 Score: 0.700-0.750
- Precision: 0.650-0.800
- Recall: 0.700-0.800
- Blended F1: 0.720-0.770
- Latency: <100ms

Product-level: 8 TP, 2 FP, 2 FN
Aisle-level F1: 0.850
Department-level F1: 0.900
```

**Why these numbers?**
- Baseline repeats the (n-1)th order
- People often buy similar items repeatedly
- F1 ~0.7-0.75 is typical for simple repeat strategies

---

## 🎯 What's Happening Under the Hood?

```
1. Launcher sends task → Green Agent
                         │
2. Green Agent:          │
   - Loads user history  │
   - use_baseline=True   │
   - Runs baseline logic │
   - Predicts items      │
   - Evaluates vs truth  │
   - Returns metrics     │
                         ▼
3. Launcher receives metrics
```

---

## 🔄 Next Steps: Testing with a White Agent

When you a white agent is present, change the task config:

```json
{
  "user_id": 1,
  "white_agent_url": "http://localhost:9002",
  "use_baseline": false,
  "environment_base": "http://localhost:8001"
}
```

Then the flow becomes:
```
Launcher → Green Agent → White Agent → Railway API
                ↓                ↓
           Evaluates      Makes prediction
```

---

## ✅ Success Criteria

- ✅ Agent card accessible at `/agent_card`
- ✅ Health check returns `healthy`
- ✅ Reset clears state
- ✅ Baseline assessment returns F1 ~0.7-0.75
- ✅ Metrics include precision, recall, blended F1

---

