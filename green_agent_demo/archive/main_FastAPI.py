# main.py — Green Agent (Henry-true): prompt + evaluator + REST endpoints
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Iterable

import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid


# =========================================================
# Load .env + resolve dataset paths
# =========================================================
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

def _abs(p: str | None) -> str | None:
    if not p:
        return None
    return str((ROOT / p).resolve()) if not os.path.isabs(p) else p

ECOM_API_BASE    = os.environ.get("ECOM_API_BASE")
WHITE_AGENT_NAME = os.environ.get("WHITE_AGENT_NAME", "baseline-rebuy")
PRODUCTS_CSV     = _abs(os.environ.get("PRODUCTS_CSV", "dataset/ic_products.csv"))
ORDERS_CSV       = _abs(os.environ.get("ORDERS_CSV",   "dataset/super_shortened_orders_products_combined.csv"))

for path, name in [(PRODUCTS_CSV, "PRODUCTS_CSV"), (ORDERS_CSV, "ORDERS_CSV")]:
    if not path or not os.path.exists(path):
        raise RuntimeError(f"{name} points to a missing file: {path}")
if not ECOM_API_BASE:
    raise RuntimeError("ECOM_API_BASE not set (check .env)")

# ---- toggle between mock and real Railway API via .env ----
USE_REAL_API = os.environ.get("USE_REAL_API", "false").lower() == "true"

def session_key_for(user_id: str, task_id: str | None = None) -> str:
    """
    Real endpoints require 'agent_key' (a session id).
    We'll reuse user_id plus task_id for uniqueness.
    """
    return f"{user_id}__{task_id}" if task_id else user_id



# =========================================================
# Load datasets once at startup
# =========================================================
DF_PRODUCTS = pd.read_csv(PRODUCTS_CSV)
DF_ORDERS   = pd.read_csv(ORDERS_CSV)

# =========================================================
# ------------------- HENRY'S CODE ------------------------
# Prompt builder (history -> instruction) and evaluator.
# presence PR/F1 on products, aisles, departments + blended F1 weighting.
# =========================================================

def split_user_orders(user_id: int, df_products: pd.DataFrame, df_orders: pd.DataFrame):
    """
    Returns dict with:
      - current_order_df   (order_number == n)   + joined product details
      - previous_orders_df (order_number  < n)   + joined product details
      - previous_order_df  (order_number == n-1) + joined product details
      - n, n_order_ids, days_since_last
    """
    u = df_orders.loc[df_orders["user_id"] == user_id].copy()
    if u.empty:
        raise ValueError(f"No rows for user_id={user_id}")

    # Defensive types
    for col in ("order_number", "order_id", "product_id"):
        if col in u.columns:
            u[col] = pd.to_numeric(u[col], errors="coerce")

    dfp = df_products.copy()
    if "product_id" in dfp.columns:
        dfp["product_id"] = pd.to_numeric(dfp["product_id"], errors="coerce")

    n = int(u["order_number"].max())
    cur = u[u["order_number"] == n].copy()
    prev_all = u[u["order_number"] < n].copy()
    prev_1 = u[u["order_number"] == (n-1)].copy() if n > 1 else prev_all.iloc[0:0].copy()

    join_cols = ["product_id","product_name","aisle_id","department_id","aisle","department"]
    prod_small = dfp[[c for c in join_cols if c in dfp.columns]].copy()

    current_order_df   = cur.merge(prod_small, on="product_id", how="left")
    previous_orders_df = prev_all.merge(prod_small, on="product_id", how="left")
    previous_order_df  = prev_1.merge(prod_small, on="product_id", how="left")

    # Nicely ordered
    if "add_to_cart_order" in current_order_df.columns:
        current_order_df   = current_order_df.sort_values(["order_id","add_to_cart_order"], kind="stable")
    if {"order_number","order_id","add_to_cart_order"}.issubset(previous_orders_df.columns):
        previous_orders_df = previous_orders_df.sort_values(["order_number","order_id","add_to_cart_order"], kind="stable")
    if "add_to_cart_order" in previous_order_df.columns:
        previous_order_df  = previous_order_df.sort_values(["order_id","add_to_cart_order"], kind="stable")

    n_order_ids = sorted(current_order_df["order_id"].dropna().unique().tolist()) if "order_id" in current_order_df.columns else []
    dsl_series = current_order_df.get("days_since_prior_order", pd.Series(dtype=float))
    days_since_last = None if dsl_series.empty else (float(dsl_series.dropna().iloc[0]) if not dsl_series.dropna().empty else None)

    return {
        "current_order_df": current_order_df,
        "previous_orders_df": previous_orders_df,
        "previous_order_df": previous_order_df,
        "n": n,
        "n_order_ids": n_order_ids,
        "days_since_last": days_since_last
    }

def _prf1(truth: set, pred: set) -> Dict[str, Any]:
    tp = len(truth & pred)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "support": len(truth),
        "pred_size": len(pred)
    }

def evaluate_basket(
    predicted_product_ids: Iterable[int],
    nth_order_df: pd.DataFrame,
    *,
    products_catalog: Optional[pd.DataFrame] = None,
    weights: Tuple[float, float, float] = (0.6, 0.2, 0.2)  # (products, aisles, departments)
) -> Dict[str, Any]:
    """
    Presence overlap on products/aisles/departments with blended F1.
    """
    w_pid, w_aisle, w_dept = weights
    if abs((w_pid + w_aisle + w_dept) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")

    if "product_id" not in nth_order_df.columns:
        raise ValueError("nth_order_df must contain 'product_id'")

    # Truth sets
    truth_pids = set(pd.to_numeric(nth_order_df["product_id"], errors="coerce").dropna().astype(int).tolist())
    truth_aisles = set(nth_order_df["aisle"].dropna().astype(str).tolist()) if "aisle" in nth_order_df.columns else set()
    truth_depts  = set(nth_order_df["department"].dropna().astype(str).tolist()) if "department" in nth_order_df.columns else set()

    # Pred sets
    pred_pids = set(pd.Series(list(predicted_product_ids), dtype="Int64").dropna().astype(int).tolist())

    # Map predicted → aisles/depts
    missing_map = {"pred_aisle_missing": 0, "pred_dept_missing": 0}
    pred_aisles, pred_depts = set(), set()
    if products_catalog is not None and {"product_id","aisle","department"}.issubset(products_catalog.columns):
        pred_map_df = pd.DataFrame({"product_id": list(pred_pids)}).merge(
            products_catalog[["product_id","aisle","department"]],
            on="product_id", how="left"
        )
        missing_map["pred_aisle_missing"] = int(pred_map_df["aisle"].isna().sum())
        missing_map["pred_dept_missing"]  = int(pred_map_df["department"].isna().sum())
        pred_aisles = set(pred_map_df["aisle"].dropna().astype(str).tolist())
        pred_depts  = set(pred_map_df["department"].dropna().astype(str).tolist())
    else:
        # Fallback: try to infer from nth_order_df if same ids appear there
        pred_map_df = pd.DataFrame({"product_id": list(pred_pids)}).merge(
            nth_order_df[["product_id","aisle","department"]].drop_duplicates(),
            on="product_id", how="left"
        )
        pred_aisles = set(pred_map_df["aisle"].dropna().astype(str).tolist())
        pred_depts  = set(pred_map_df["department"].dropna().astype(str).tolist())

    m_products = _prf1(truth_pids, pred_pids)
    m_aisles   = _prf1(truth_aisles, pred_aisles) if (truth_aisles or pred_aisles) else {**_prf1(set(), set())}
    m_depts    = _prf1(truth_depts,  pred_depts)  if (truth_depts  or pred_depts)  else {**_prf1(set(), set())}

    blended = w_pid * m_products["f1"] + w_aisle * m_aisles["f1"] + w_dept * m_depts["f1"]

    return {
        "products": m_products,
        "aisles": m_aisles,
        "departments": m_depts,
        "blended_f1": blended,
        "meta": {
            "truth_counts": {
                "products": len(truth_pids),
                "aisles": len(truth_aisles),
                "departments": len(truth_depts),
            },
            "pred_counts": {
                "products": len(pred_pids),
                "aisles": len(pred_aisles),
                "departments": len(pred_depts),
            },
            "missing_mappings": missing_map
        }
    }

def henry_build_prompt(previous_orders_df: pd.DataFrame, days_since_last, user_id):
    lines = []
    lines.append(f"You are a grocery shopping assistant for user {user_id}.")
    if days_since_last is not None:
        try:
            lines.append(f"It has been {int(days_since_last)} days since their last order.")
        except Exception:
            pass
    lines.append("Using the user's purchase history below, propose the next basket.")
    lines.append("Prioritize frequently repeated items and the user's top departments.")
    lines.append("Avoid duplicates; keep total items ~3–12.")

    if previous_orders_df is None or len(previous_orders_df) == 0:
        lines.append("\nNo previous orders are available; start with common staples.")
        lines.append(_henry_output_instructions())
        return "\n".join(lines)

    top_prods = (
        previous_orders_df.groupby(["product_id","product_name"], dropna=False)
        .size().reset_index(name="times_bought")
        .sort_values(["times_bought","product_name"], ascending=[False, True])
    )
    depts = (
        previous_orders_df.groupby(["department_id","department"], dropna=False)
        .size().reset_index(name="count")
        .sort_values(["count","department"], ascending=[False, True])
    )

    # Show last few orders compactly
    orders_preview = []
    if "order_number" in previous_orders_df.columns and "order_id" in previous_orders_df.columns:
        recent_nums = sorted(previous_orders_df["order_number"].dropna().unique().tolist())[-3:]
        for onum in recent_nums:
            g = previous_orders_df[previous_orders_df["order_number"] == onum]
            names = g["product_name"].fillna("").astype(str).tolist()
            preview = ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
            oid = g["order_id"].dropna().iloc[0] if not g["order_id"].dropna().empty else "NA"
            orders_preview.append(f"- Order #{int(onum)} (id {oid}): {preview}")

    lines.append("\nTop departments by repeat count:")
    for _, row in depts.head(5).iterrows():
        lines.append(f"- {row['department']} (x{int(row['count'])})")

    lines.append("\nTop products by repeat count:")
    for _, row in top_prods.head(10).iterrows():
        lines.append(f"- [{int(row['product_id'])}] {row['product_name']} (x{int(row['times_bought'])})")

    if orders_preview:
        lines.append("\nMost recent orders:")
        lines.extend(orders_preview)

    lines.append(_henry_output_instructions())
    return "\n".join(lines)

def _henry_output_instructions():
    return (
        "\nReturn ONLY JSON in this exact shape:\n"
        "{\n"
        '  "predicted_items": { "<product_id>": <quantity>, ... },\n'
        '  "notes": "<very brief reasoning, optional>"\n'
        "}\n"
        "Rules:\n"
        "- product_id must exist in the catalog; quantities are positive integers.\n"
        "- Include 3–12 items; bias toward repeated favorites.\n"
        "- Do NOT include duplicate product_ids.\n"
    )

# =========================================================
# In-memory run log + helpers
# =========================================================
RUNS: List[Dict[str, Any]] = []

def _aggregate(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics_list:
        return {"n": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "avg_latency_ms": 0, "avg_blended_f1": 0.0}
    n = len(metrics_list)
    p = sum(float(m["precision"]) for m in metrics_list) / n
    r = sum(float(m["recall"])    for m in metrics_list) / n
    f = sum(float(m["f1"])        for m in metrics_list) / n
    lat = int(sum(int(m.get("latency_ms", 0)) for m in metrics_list) / n)
    bf = sum(float(m.get("blended_f1", 0.0)) for m in metrics_list) / n
    return {"n": n, "precision": p, "recall": r, "f1": f, "avg_latency_ms": lat, "avg_blended_f1": bf}

def to_int_keyed(d: Dict[Any, Any]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k, v in d.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            pass
    return out

# =========================================================
# Baseline participant (no LLM): re-buy history
# =========================================================
class HistoryItem(BaseModel):
    product_id: int
    qty: int

class Task(BaseModel):
    task_id: str
    user_id: str
    history: List[HistoryItem] = []
    goal: Optional[str] = None
    ground_truth_items: Dict[str, int]
    prompt: Optional[str] = None  # optional pre-supplied prompt

class Metrics(BaseModel):
    precision: float
    recall: float
    f1: float
    latency_ms: int

class AssessResponse(BaseModel):
    metrics: Metrics
    trace: Dict[str, Any]

class TasksPayload(BaseModel):
    tasks: List[Task]

def baseline_policy(history: List[HistoryItem], user_id: str) -> Dict[int, int]:
    # If history is provided, use it; otherwise repeat (n-1)th order from dataset.
    if history:
        basket: Dict[int, int] = {}
        for h in history:
            basket[h.product_id] = basket.get(h.product_id, 0) + h.qty
        return basket

    uid = int(str(user_id).replace("user_", ""))
    parts = split_user_orders(uid, DF_PRODUCTS, DF_ORDERS)
    prev_df = parts["previous_order_df"]
    return {int(pid): int(qty) for pid, qty in prev_df["product_id"].value_counts().items()}


# =========================================================
# Environment helpers (Arlen's /mock/* endpoints)
# =========================================================
def env_healthcheck() -> bool:
    for path in ("/healthz", "/mock/healthz"):
        try:
            r = requests.get(f"{ECOM_API_BASE}{path}", timeout=5)
            if r.ok:
                return True
        except Exception:
            continue
    return False

def env_add_to_cart(user_id: str, product_id: int, qty: int) -> Dict[str, Any]:
    r = requests.post(
        f"{ECOM_API_BASE}/mock/cart/add",
        json={"user_id": user_id, "product_id": product_id, "qty": qty},
        timeout=10,
    )
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"/mock/cart/add failed: {r.text}")
    return r.json()

def env_checkout(user_id: str) -> Dict[str, Any]:
    r = requests.post(
        f"{ECOM_API_BASE}/mock/checkout",
        params={"user_id": user_id},
        timeout=15,
    )
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"/mock/checkout failed: {r.text}")
    return r.json()

# =========================================================
# Environment helpers (Arlen's commerce endpoints)
# =========================================================

# add this near your other env helpers
def env_cart_add_real(agent_key: str, items: list[dict]) -> dict:
    r = requests.post(
        f"{ECOM_API_BASE}/cart/add",
        json={"agent_key": agent_key, "items": items},
        timeout=10,
    )
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"/cart/add failed: {r.text}")
    return r.json()

def env_checkout_real(agent_key: str) -> dict:
    r = requests.post(
        f"{ECOM_API_BASE}/checkout",
        json={"agent_key": agent_key},
        timeout=10,
    )
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"/checkout failed: {r.text}")
    return r.json()

def env_get_cart_real(agent_key: str) -> dict:
    r = requests.get(f"{ECOM_API_BASE}/cart", params={"agent_key": agent_key}, timeout=10)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"/cart failed: {r.text}")
    return r.json()

def env_search(query: str, field: str="name", aisle: str|None=None, department: str|None=None,
               limit: int=20, offset: int=0) -> dict:
    r = requests.get(f"{ECOM_API_BASE}/search", params={
        "q": query, "field": field, "aisle": aisle, "department": department,
        "limit": limit, "offset": offset
    }, timeout=10)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"/search failed: {r.text}")
    return r.json()

def apply_basket_and_checkout(
    user_id: str,
    proposed_items: Dict[int, int],
    *,
    task_id: str | None = None
) -> Dict[str, Any]:
    """
    Sends the proposed basket to Railway and checks out.
    Uses /mock/* or /cart/add + /checkout based on USE_REAL_API.
    Returns the checkout_result dict from the environment.
    """
    if USE_REAL_API:
        # real flow: one request with all items, plus agent_key
        items = [{"product_id": int(pid), "qty": int(qty)} for pid, qty in proposed_items.items()]
        agent_key = session_key_for(user_id, task_id)
        if items:
            env_cart_add_real(agent_key, items)
        return env_checkout_real(agent_key)
    else:
        # mock flow: one call per item, then checkout (using user_id)
        for pid, qty in proposed_items.items():
            env_add_to_cart(user_id, int(pid), int(qty))
        return env_checkout(user_id)


# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(title="Ecom-GreenAgent (Henry-true)", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- /agent_card ------------------------------------------------------------
from fastapi import Request
from datetime import datetime

@app.get("/agent_card")
def agent_card(request: Request):
    base = str(request.base_url).rstrip("/")

    return {
        "name": "GreenAgent-Henry",
        "version": "3.0.0",
        "kind": "green-agent",
        "description": (
            "Green agent for AgentBeats: prepares grocery tasks, provides tools/env "
            "to a white agent, and evaluates results (Blended-F1)."
        ),

        # How launchers can talk to you
        "protocols": {
            "a2a": {
                "message_kinds": ["message"],   # simple message-in/message-out
                "reset": True                   # supports POST /reset
            }
        },

        # Stable relative paths this service exposes
        "endpoints": {
            "self":               "/agent_card",
            "health":             "/healthz",
            "reset":              "/reset",
            "a2a_send_message":   "/a2a/send_message",
            # Optional: keep these public if for debugging
            "prepare_run":        "/prepare_run",
            "finalize_run":       "/finalize_run",
        },

        # Absolute URLs
        "urls": {
            "self":             f"{base}/agent_card",
            "health":           f"{base}/healthz",
            "reset":            f"{base}/reset",
            "a2a_send_message": f"{base}/a2a/send_message",
            "prepare_run":      f"{base}/prepare_run",
            "finalize_run":     f"{base}/finalize_run",
        },

        # Quick contract hints (lightweight)
        "io_contracts": {
            "a2a_send_message": {
                "input_schema": {
                    "type": "object",
                    "required": ["white_agent_url", "user_id"],
                    "properties": {
                        "white_agent_url": {"type": "string"},
                        "user_id": {"type": "integer"},
                        "task_id": {"type": "string"},
                    }
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["message"]},
                        "parts": {"type": "array"}
                    }
                }
            }
        },

        # Runtime status
        "status": {
            "ok": True,
            "server_time_utc": datetime.utcnow().isoformat() + "Z"
        },

        # Who we are
        "author": {
            "name": "Henry Miller Michaelson, Tao Sun, Arlen Kumar",
            "org": "UC Berkeley",
            "contact": "hmichaelson@berkeley.edu, tao_sun@berkeley.edu, arlen1788@berkeley.edu"
        },
        "license": "MIT"
    }



@app.post("/reset")
def reset():
    RUNS.clear()
    return {"ok": env_healthcheck(), "cleared_runs": True}


# ---------- Prepare & Wrap up ----------

class PrepareReq(BaseModel):
    user_id: int
    task_id: str | None = None
    max_products: int = 10  # optional knobs for prompt formatting

class PrepareResp(BaseModel):
    task_id: str
    user_id: str
    agent_key: str
    environment_base: str
    use_real_api: bool
    prompt: str
    tool_manifest: dict  # simple schema for white agent

@app.post("/prepare_run", response_model=PrepareResp)
def prepare_run(req: PrepareReq):
    # Build dataset splits + prompt for the *white* agent
    parts = split_user_orders(req.user_id, DF_PRODUCTS, DF_ORDERS)
    prompt_text = henry_build_prompt(parts["previous_orders_df"], parts["days_since_last"], req.user_id)

    # Session the white agent will use with Railway
    t_id = req.task_id or f"user{req.user_id}_n{parts['n']}"
    agent_key = session_key_for(f"user_{req.user_id}", task_id=t_id) if USE_REAL_API else f"user_{req.user_id}"

    tool_manifest = {
        "actions": [
            {"name": "search", "method": "GET", "path": "/search", "params": ["q","field","aisle","department","limit","offset"]},
            {"name": "cart_add", "method": "POST", "path": "/cart/add", "body": {"agent_key":"<string>","items":[{"product_id":"<int>","qty":"<int>"}]}},
            {"name": "cart_get", "method": "GET", "path": "/cart", "params": ["agent_key"]},
            {"name": "checkout", "method": "POST", "path": "/checkout", "body": {"agent_key":"<string>"}}
        ],
        "notes": "Use agent_key for all cart/* and checkout calls."
    }

    return PrepareResp(
        task_id=t_id,
        user_id=f"user_{req.user_id}",
        agent_key=agent_key,
        environment_base=ECOM_API_BASE,
        use_real_api=USE_REAL_API,
        prompt=prompt_text,
        tool_manifest=tool_manifest
    )


class FinalizeReq(BaseModel):
    task_id: str
    user_id: str
    agent_key: str | None = None
    # Optional: let white agent submit what it *thinks* it bought (if it tracked locally)
    predicted_items: Dict[int,int] | None = None

@app.post("/finalize_run")
def finalize_run(req: FinalizeReq):
    """
    If predicted_items not supplied, we fetch the final cart/receipt by agent_key (real API)
    and evaluate against the user's nth order.
    """
    uid_int = int(str(req.user_id).replace("user_", ""))
    parts = split_user_orders(uid_int, DF_PRODUCTS, DF_ORDERS)
    nth_order_df = parts["current_order_df"]

    # Determine predicted items
    proposed: Dict[int,int]
    if req.predicted_items:
        proposed = {int(k): int(v) for k, v in req.predicted_items.items()}
    elif USE_REAL_API and req.agent_key:
        # pull from the environment (cart or receipt) — adjust to your API's shape
        cart = env_get_cart_real(req.agent_key)
        # Expect something like {"items":[{"product_id":..., "qty":...}, ...]}
        items = cart.get("items", [])
        proposed = {}
        for it in items:
            pid = int(it.get("product_id"))
            qty = int(it.get("qty", 1))
            proposed[pid] = proposed.get(pid, 0) + qty
    else:
        raise HTTPException(400, "predicted_items missing and cannot fetch without agent_key in real mode")

    # Evaluate (Henry)
    henry_res = evaluate_basket(
        predicted_product_ids=list(proposed.keys()),
        nth_order_df=nth_order_df,
        products_catalog=DF_PRODUCTS,
        weights=(0.6, 0.2, 0.2)
    )

    metrics = {
        "precision": henry_res["products"]["precision"],
        "recall":    henry_res["products"]["recall"],
        "f1":        henry_res["products"]["f1"],
        "blended_f1": henry_res["blended_f1"]
    }

    # Log a run for bookkeeping
    RUNS.append({
        "task_id": req.task_id,
        "user_id": req.user_id,
        "metrics": {**metrics, "latency_ms": 0},
        "proposed_items": proposed,
        "checkout_total": None,
        "prompt_present": True
    })

    return {"metrics": metrics, "trace": {"proposed_items": proposed, "henry_eval": henry_res}}


# ---------- Prompt & task materialization ----------
@app.get("/prompt")
def preview_prompt(user_id: int = Query(...)):
    parts = split_user_orders(user_id, DF_PRODUCTS, DF_ORDERS)
    prompt_text = henry_build_prompt(parts["previous_orders_df"], parts["days_since_last"], user_id)
    return {"user_id": user_id, "prompt": prompt_text, "n": parts["n"], "n_order_ids": parts["n_order_ids"]}

@app.get("/make_task")
def make_task(user_id: int = Query(..., description="User ID to build task from")):
    parts = split_user_orders(user_id, DF_PRODUCTS, DF_ORDERS)
    cur = parts["current_order_df"]
    if cur.empty:
        raise HTTPException(404, f"No current order found for user_id={user_id}")
    gt = cur["product_id"].value_counts().to_dict()
    prompt_text = henry_build_prompt(parts["previous_orders_df"], parts["days_since_last"], user_id)
    task = {
        "task_id": f"user{user_id}_n{parts['n']}",
        "user_id": f"user_{user_id}",
        "history": [],
        "goal": "Build the next basket based on prior behavior.",
        "ground_truth_items": {str(int(k)): int(v) for k, v in gt.items()},
        "prompt": prompt_text
    }
    return {"task": task, "meta": {"n_order_ids": parts["n_order_ids"], "n": parts["n"]}}

# ---------- Helpers ----------
def _parse_uid(u: str) -> Optional[int]:
    try:
        return int(str(u).replace("user_", ""))
    except Exception:
        return None

# ---------- Assessment endpoints ----------
@app.post("/task", response_model=AssessResponse)
def run_one_assessment(task: Task):
    """Run a single assessment using:
       - Henry prompt (from dataset)
       - Baseline: history if provided, else repeat (n-1)th order
       - Henry evaluator (products/aisles/departments + blended F1)
    """
    if not env_healthcheck():
        raise HTTPException(status_code=503, detail="Environment health check failed")

    # Build Henry prompt & fetch user splits
    prompt_text = task.prompt or None
    uid_int = _parse_uid(task.user_id)
    parts = None
    if uid_int is not None:
        try:
            parts = split_user_orders(uid_int, DF_PRODUCTS, DF_ORDERS)
            if prompt_text is None:
                prompt_text = henry_build_prompt(parts["previous_orders_df"], parts["days_since_last"], task.user_id)
        except Exception:
            parts = None
            prompt_text = prompt_text or None

    # Baseline proposal:
    # - If history provided in payload, use it
    # - Else: repeat the user's (n-1)th order from dataset
    if task.history:
        proposed_items: Dict[int, int] = {}
        for h in task.history:
            proposed_items[h.product_id] = proposed_items.get(h.product_id, 0) + h.qty
    else:
        if parts is None:
            raise HTTPException(status_code=400, detail="No history provided and user not found in dataset for last-order baseline.")
        prev_df = parts["previous_order_df"]
        proposed_items = {int(pid): int(qty) for pid, qty in prev_df["product_id"].value_counts().items()}

    t0 = time.time()


    # Apply to environment (mock vs real) and checkout
    checkout_result = apply_basket_and_checkout(task.user_id, proposed_items, task_id=task.task_id)

    # Henry evaluator: score vs user's actual nth order (current order)
    nth_order_df = parts["current_order_df"] if parts is not None else None
    if nth_order_df is None or nth_order_df.empty:
        # Fall back to payload GT if dataset nth order is unavailable
        # (Convert GT dict -> set of product_ids)
        gt_set = set(int(k) for k in task.ground_truth_items.keys())
        henry_res = {
            "products": _prf1(gt_set, set(proposed_items.keys())),
            "aisles": _prf1(set(), set()),
            "departments": _prf1(set(), set()),
            "blended_f1": _prf1(gt_set, set(proposed_items.keys()))["f1"],
        }
    else:
        henry_res = evaluate_basket(
            predicted_product_ids=list(proposed_items.keys()),
            nth_order_df=nth_order_df,
            products_catalog=DF_PRODUCTS,
            weights=(0.6, 0.2, 0.2),
        )

    metrics_dict = {
        "precision": henry_res["products"]["precision"],
        "recall":    henry_res["products"]["recall"],
        "f1":        henry_res["products"]["f1"],
    }
    latency_ms = int((time.time() - t0) * 1000)
    metrics = Metrics(**metrics_dict, latency_ms=latency_ms)

    trace = {
        "white_agent": WHITE_AGENT_NAME,
        "task": task.model_dump(),
        "proposed_items": proposed_items,
        "prompt_text": prompt_text,
        "henry_eval": {
            "products": henry_res["products"],
            "aisles": henry_res.get("aisles", {}),
            "departments": henry_res.get("departments", {}),
            "blended_f1": henry_res.get("blended_f1", metrics.f1),
        },
        "environment": {"base_url": ECOM_API_BASE, "checkout_result": checkout_result},
    }

    RUNS.append({
        "task_id": task.task_id,
        "user_id": task.user_id,
        "metrics": {**metrics.model_dump(), "blended_f1": henry_res.get("blended_f1", metrics.f1)},
        "proposed_items": {int(k): int(v) for k, v in proposed_items.items()},
        "checkout_total": checkout_result.get("total", None),
        "prompt_present": prompt_text is not None,
    })

    return AssessResponse(metrics=metrics, trace=trace)


@app.post("/assess_many")
def assess_many(payload: TasksPayload):
    """Run multiple tasks using the same baseline + Henry evaluator. Returns per-task results + macro summary."""
    results: List[Dict[str, Any]] = []

    for t in payload.tasks:
        if not env_healthcheck():
            raise HTTPException(status_code=503, detail="Environment health check failed")

        # Per-task prompt & dataset splits
        prompt_text = t.prompt or None
        uid_int = _parse_uid(t.user_id)
        parts = None
        if uid_int is not None:
            try:
                parts = split_user_orders(uid_int, DF_PRODUCTS, DF_ORDERS)
                if prompt_text is None:
                    prompt_text = henry_build_prompt(parts["previous_orders_df"], parts["days_since_last"], t.user_id)
            except Exception:
                parts = None

        # Baseline: use provided history else repeat (n-1)th order
        if t.history:
            proposed_items: Dict[int, int] = {}
            for h in t.history:
                proposed_items[h.product_id] = proposed_items.get(h.product_id, 0) + h.qty
        else:
            if parts is None:
                raise HTTPException(status_code=400, detail=f"No history provided and user {t.user_id} not in dataset.")
            prev_df = parts["previous_order_df"]
            proposed_items = {int(pid): int(qty) for pid, qty in prev_df["product_id"].value_counts().items()}

        t0 = time.time()

        # Apply to environment (mock vs real) and checkout
        checkout_result = apply_basket_and_checkout(task.user_id, proposed_items, task_id=task.task_id)


        # Evaluate vs nth order; fallback to payload GT if necessary
        nth_order_df = parts["current_order_df"] if parts is not None else None
        if nth_order_df is None or nth_order_df.empty:
            gt_set = set(int(k) for k in t.ground_truth_items.keys())
            henry_res = {
                "products": _prf1(gt_set, set(proposed_items.keys())),
                "aisles": _prf1(set(), set()),
                "departments": _prf1(set(), set()),
                "blended_f1": _prf1(gt_set, set(proposed_items.keys()))["f1"],
            }
        else:
            henry_res = evaluate_basket(
                predicted_product_ids=list(proposed_items.keys()),
                nth_order_df=nth_order_df,
                products_catalog=DF_PRODUCTS,
                weights=(0.6, 0.2, 0.2),
            )

        m = {
            "precision": henry_res["products"]["precision"],
            "recall":    henry_res["products"]["recall"],
            "f1":        henry_res["products"]["f1"],
            "latency_ms": int((time.time() - t0) * 1000),
            "blended_f1": henry_res.get("blended_f1", henry_res["products"]["f1"]),
        }

        run_record = {
            "task_id": t.task_id,
            "user_id": t.user_id,
            "metrics": m,
            "proposed_items": {int(k): int(v) for k, v in proposed_items.items()},
            "checkout_total": checkout_result.get("total", None),
            "prompt_present": prompt_text is not None,
            "prompt_text": prompt_text,
        }
        RUNS.append(run_record)
        results.append(run_record)

    summary = _aggregate([r["metrics"] for r in results])
    return {"results": results, "summary": summary}


# ---------- Runs & summary ----------
@app.get("/runs")
def list_runs():
    return {"runs": RUNS, "summary": _aggregate([r["metrics"] for r in RUNS])}

@app.get("/summary")
def summary_only():
    return _aggregate([r["metrics"] for r in RUNS])

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
