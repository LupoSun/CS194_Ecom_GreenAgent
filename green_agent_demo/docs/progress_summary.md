# A2A/AgentBeats Integration Progress Summary


---

## 📋 Summary

Successfully converted the FastAPI-based green agent to be **AgentBeats-compatible** using the **A2A (Agent-to-Agent) protocol**, while preserving all existing evaluation logic from Henry's implementation.

**Status:** ✅ Baseline mode working and tested | 🔜 White agent mode pending for implementation and test?

---

## ✅ Completed Tasks

### 1. A2A Protocol Integration

**What was done:**
- Installed and integrated `a2a-python` library
- Implemented `AgentExecutor` class to wrap evaluation logic
- Replaced FastAPI REST endpoints with A2A message-based communication
- Integrated with `A2AStarletteApplication` framework

**Technical details:**
```python
class EcomGreenAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Parse A2A message
        # Run evaluation
        # Return A2A response
```

### 2. Dual-Mode Architecture

Built green agent that supports **two operational modes**:

#### Mode 1: Baseline Policy (Demo/Testing)
- **Purpose:** Quick testing, demos, establishing baseline metrics
- **How it works:** Self-contained assessment without external agents
- **Strategy:** "Repeat previous order" baseline policy
- **Input example:**
```json
{
  "user_id": 1,
  "use_baseline": true
}
```

#### Mode 2: White Agent Evaluation (Production)
- **Purpose:** Actual agent assessment for AgentBeats platform
- **How it works:** Evaluates external agents via A2A protocol
- **Key feature:** Sends tool manifest with Railway API specification
- **Input example:**
```json
{
  "user_id": 1,
  "use_baseline": false,
  "white_agent_url": "http://localhost:9002",
  "environment_base": "http://localhost:8001"
}
```

### 3. Tool Manifest Implementation

Created comprehensive tool specification for white agents to interact with the Railway ecommerce environment:

**Available Tools:**
- **Search API**: Product search by name/aisle/department
  - Endpoint: `GET /search`
  - Parameters: `q`, `field`, `aisle`, `department`, `limit`, `offset`

- **Cart API**: Add items to shopping cart
  - Endpoint: `POST /cart/add`
  - Body: `{"agent_key": "...", "items": [{"product_id": int, "qty": int}]}`

- **Checkout API**: Finalize purchase
  - Endpoint: `POST /checkout`
  - Body: `{"agent_key": "..."}`

**Session Management:**
- Each assessment gets unique `agent_key` for isolation
- Format: `user_{user_id}_{task_id}`

### 4. AgentBeats-Compatible Endpoints

Added required endpoints for platform compatibility:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Agent card (A2A standard) |
| GET | `/agent_card` | Explicit agent card |
| GET | `/healthz` | Health check |
| POST | `/reset` | Clear agent state |
| POST | `/` | A2A send_message (main execution) |

**Agent Card Example:**
```json
{
  "name": "ecom_green_agent",
  "version": "1.0.0",
  "description": "Ecom green agent - supports baseline and white agent evaluation",
  "url": "http://localhost:9001",
  "capabilities": {
    "input": ["text"],
    "output": ["text"],
    "streaming": false
  }
}
```

### 5. Preserved Existing Evaluation Logic

All Henry's evaluation logic remains **unchanged and identical**:

**Core Functions (unchanged):**
- ✅ `split_user_orders()` - User history data preparation
- ✅ `evaluate_basket()` - Blended F1 calculation
- ✅ `henry_build_prompt()` - Prompt generation for agents
- ✅ `_prf1()` - Precision/Recall/F1 computation

**Metrics (identical):**
- Precision, Recall, F1 for products
- Precision, Recall, F1 for aisles
- Precision, Recall, F1 for departments
- Blended F1: 60% products + 20% aisles + 20% departments

### 6. Testing & Validation

**Test Scripts Created:**
1. `quick_test.py` - Single user baseline test
3. Updated demo guide with step-by-step instructions

**Validation Results:**
```python
# Test: User 1, Baseline Mode
✅ Assessment Complete (Mode: BASELINE)

Metrics:
- F1 Score: 0.700
- Precision: 0.778 (7 TP, 2 FP)
- Recall: 0.636 (7 TP, 4 FN)
- Blended F1: 0.753
- Latency: 2ms

Context Performance:
- Aisle-level F1: 0.778
- Department-level F1: 0.889
```

**Status:** ✅ Matches expected baseline performance (~0.70 F1)


## 📊 Results & Performance

### Baseline Mode Test Results

**Test Case: User 1**
```
Input: {"user_id": 1, "use_baseline": true}

Output:
Assessment Complete ✅ (Mode: BASELINE)

Metrics:
- F1 Score: 0.700
- Precision: 0.778
- Recall: 0.636
- Blended F1: 0.753
- Latency: 2ms

Product-level: 7 TP, 2 FP, 4 FN
Aisle-level F1: 0.778
Department-level F1: 0.889
```

**Interpretation:**
- F1 of 0.70 is **expected and good** for "repeat last order" strategy
- Users do buy similar items repeatedly (hence 7 TP)
- But also try new things (4 FN) and skip some (2 FP)
- Strong department/aisle coverage (0.889, 0.778) shows good category prediction


---

## 🏗️ Architecture Changes

### Last Submission: FastAPI Architecture

```
┌─────────────────────────────────────────┐
│         Client (REST)                   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│    FastAPI Green Agent                  │
│                                         │
│  POST /task                             │
│    ↓                                    │
│  Parse Task                             │
│    ↓                                    │
│  Baseline Policy                        │
│    ↓                                    │
│  Evaluate (Henry's metrics)             │
│    ↓                                    │
│  Return JSON                            │
└─────────────────────────────────────────┘
```

### This Time: A2A Architecture

```
┌─────────────────────────────────────────┐
│         Client (A2A)                    │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│    A2A Green Agent                      │
│                                         │
│  POST / (A2A send_message)              │
│    ↓                                    │
│  Parse A2A Message                      │
│    ↓                                    │
│  Mode Decision                          │
│    ↙          ↘                         │
│  Baseline    White Agent                │
│  Policy      Communication              │
│    │            ↓                       │
│    │      Send Tool Manifest            │
│    │            ↓                       │
│    │      Receive Prediction            │
│    ↓            ↓                       │
│    ← Evaluate (Henry's metrics) ←       │
│    ↓                                    │
│  Format A2A Response                    │
│    ↓                                    │
│  Return A2A Message                     │
└─────────────────────────────────────────┘
```

### White Agent Mode Flow 

```
┌──────────────┐
│   Launcher   │ Starts assessment
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│   Green Agent                            │
│   • Loads user history from dataset      │
│   • Builds prompt                        │
│   • Creates tool manifest                │
│   • Sends to white agent via A2A         │
└──────┬───────────────────────────────────┘
       │ A2A Message with:
       │ - User history prompt
       │ - Tool manifest (search, cart, checkout)
       │ - Session key (agent_key)
       ▼
┌──────────────────────────────────────────┐
│   White Agent                            │
│   • Receives task & tools                │
│   • Uses LLM to make decisions           │
│   • Calls Railway API:                   │
│     - GET /search (explore products)     │
│     - POST /cart/add (build basket)      │
│     - POST /checkout (finalize)          │
│   • Returns prediction JSON              │
└──────┬───────────────────────────────────┘
       │ {"predicted_items": {123: 2, 456: 1}}
       ▼
┌──────────────────────────────────────────┐
│   Green Agent                            │
│   • Receives white agent prediction      │
│   • Evaluates vs ground truth            │
│   • Calculates metrics (Henry's)         │
│   • Returns assessment results           │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   Launcher   │ Gets final metrics
└──────────────┘
```

### What are the different modes of the green agent?
Example:
```
{
    "mode": "benchmark",
    "num_users": 5,
    "use_baseline": true,
    "white_agent_url": "http://localhost:9002"
    # use_baseline=true overrides white_agent_url
}
```
#### Flow Diagram
```
Benchmark Mode Started
        │
        ▼
    Extract config:
    - use_baseline
    - white_agent_url
        │
        ▼
    For each user:
        │
        ▼
    ┌─────────────────────────┐
    │ if use_baseline==True   │──Yes──→ Use Baseline Policy
    │    OR                   │
    │ white_agent_url is None │
    └────────┬────────────────┘
             │ No
             ▼
    Use White Agent Policy
```

---

## 🔄 Implementation Details

### Files Created/Modified

**New Files:**
- `main_A2A.py` - Complete A2A implementation
- `test_a2a_green` - Python test script for the green agent's baseline policy
- `a2a_demo_guide.md` - Documentation
- `ecom_green_agent.toml` - Agent card configuration
- `ab_src` folder - Storing the AgentBeats example files for their tau-Bench green and white agent as well as some helper methods that are also referenced in our green agent implementation (`ab_src.myutil...`).
- `main_A2A_BeforeMeeting.py` - Just a backup

**New but Undone**
- `dummy_white_agent_card.toml` Agent card for the dummy white agent
- `white_repl.py` Dummy white agent
- `launcher.py` The launcher to initiates and coordinates the evaluation process. Once both the green agent and the white agent are ready

**Unchanged Files:**
- `main_FastAPI.py` - Original implementation (kept for reference)
- Dataset files (`ic_products.csv`, `super_shortened_orders_products_combined.csv`)
- `.env` configuration
- `green_agent_card.toml` - The old agent card for the green agent via FastAPI

### Dependencies

see requirements.txt

### Environment Variables

**No changes required:**
- `PRODUCTS_CSV` - Path to products dataset
- `ORDERS_CSV` - Path to orders dataset
- `ECOM_API_BASE` - Railway API base URL

### Configuration

**Agent runs on:**
- Host: `localhost`
- Port: `9001` (configurable)

**Supports:**
- Message-only A2A agents (no task generation yet)
- JSON-based task configuration
- Session-based isolation via `agent_key`

---

### To Dos
1.  Finish the dummy white agent (Optional) and test if the green agent is able to correctly:
  - Set up the environment (Exposing the availale Railway APIs to the white agent via Tool Manifest) 
  - Make tasks for the white agent
  - Parse and evaluate the predictions from the white agent
2. Finish `launcher.py` according to Agentbeat's tau-Bench pattern, so that the green agent is ready for any white agent. 

#### Decision to make
1. Who calls checkout from the Ecomerce platfrom? Currently there is a clear devide between the role of green agent. The white agent is expected to autonomously searh, add, and check out, the green agent only gets the prediction of the white agent as strucured text output. If we want the green agent to control the check out, there is a minimal tweak to be done, but it also makes the green agent more robust.