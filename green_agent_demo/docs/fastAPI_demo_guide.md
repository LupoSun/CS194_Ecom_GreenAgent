# 🧪 Green Agent Demo

---

## 🔧 Setup
Adjust if your server runs on a different port or host.

```bash
BASE=http://localhost:8000
USER=1
```

Another test case
```bash
USER=100825
```
---

## 1️⃣ Check service health
```bash
curl -s $BASE/agent_card | jq
curl -s -X POST $BASE/reset | jq
```

---

## 2️⃣ Preview prompt for the white agent
```bash
curl -s "$BASE/prompt?user_id=$USER" | jq
```

---

## 3️⃣ Build a task from dataset (truth = user's nth order)
```bash
curl -s "$BASE/make_task?user_id=$USER" | jq
```

---

## 4️⃣ Run one task (baseline = repeat last order if no history)
Save and post the generated task JSON.

```bash
curl -s "$BASE/make_task?user_id=$USER" | jq '.task' > /tmp/task.json
cat /tmp/task.json | jq   # peek the task



curl -s -X POST $BASE/task   -H "Content-Type: application/json"   -d @/tmp/task.json | jq
```

---


## 6️⃣ View accumulated runs & summary
```bash
curl -s $BASE/runs | jq
curl -s $BASE/summary | jq
```

---

## 🩺 Optional: Environment health (Railway API)
```bash
curl -s https://green-agent-production.up.railway.app/mock/healthz | jq
```

---

### ✅ Expected Behavior
- `/prompt` → returns shopping prompt text.  
- `/make_task` → generates next-basket task with `ground_truth_items`.  
- `/task` → evaluates baseline repeat-last-order; expect F1 ≈ 0.7–0.75.  
- `/assess_many` → runs multiple agents (baseline vs random) and summarizes.  
- `/runs` & `/summary` → show accumulated evaluation history.
