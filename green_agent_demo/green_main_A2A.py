"""
AgentBeats-compatible Green Agent for Ecom Assessment
Integrates existing FastAPI logic with A2A protocol
"""

import os
import uvicorn
import tomllib  
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Iterable
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import requests
import numpy as np

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, Message
from a2a.utils import new_agent_text_message, get_text_parts

from utils import my_a2a, parse_tags
from html import unescape

ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

RAILWAY_CHECKOUT_URL = os.environ.get("ECOM_API_BASE", "https://your-railway-api.railway.app/checkout") + "/checkout"
COMPLETION_SIGNAL = "##READY_FOR_CHECKOUT##"  # Signal white agent sends when done

# Existing helper functions
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

    # Map predicted → aisles/depts using products catalog
    if products_catalog is None or not {"product_id","aisle","department"}.issubset(products_catalog.columns):
        raise ValueError("products_catalog must be provided with product_id, aisle, and department columns")
    
    pred_map_df = pd.DataFrame({"product_id": list(pred_pids)}).merge(
        products_catalog[["product_id","aisle","department"]],
        on="product_id", how="left"
    )
    
    missing_map = {
        "pred_aisle_missing": int(pred_map_df["aisle"].isna().sum()),
        "pred_dept_missing": int(pred_map_df["department"].isna().sum())
    }
    
    pred_aisles = set(pred_map_df["aisle"].dropna().astype(str).tolist())
    pred_depts = set(pred_map_df["department"].dropna().astype(str).tolist())

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
        # Changed: Show last 5 orders instead of 3
        recent_nums = sorted(previous_orders_df["order_number"].dropna().unique().tolist())[-5:]
        for onum in recent_nums:
            g = previous_orders_df[previous_orders_df["order_number"] == onum]
            names = g["product_name"].fillna("").astype(str).tolist()
            # Changed: Show up to 15 items per order
            preview = ", ".join(names[:50]) + (" ..." if len(names) > 50 else "")
            oid = g["order_id"].dropna().iloc[0] if not g["order_id"].dropna().empty else "NA"
            orders_preview.append(f"- Order #{int(onum)} (id {oid}): {preview}")

    lines.append("\nTop departments by repeat count:")
    for _, row in depts.head(8).iterrows():  # Increased to 8
        lines.append(f"- {row['department']} (x{int(row['count'])})")

    lines.append("\nTop products by repeat count:")
    for _, row in top_prods.head(20).iterrows():  # Increased to 20
        lines.append(f"- {row['product_name']} (x{int(row['times_bought'])})")

    if orders_preview:
        lines.append("\nMost recent orders:")
        lines.extend(orders_preview)

    lines.append(_henry_output_instructions())
    return "\n".join(lines)

def _henry_output_instructions():
    return f"""
        ### Instructions:
        1. Use /search_products to find items
        2. Use /get_product for details
        3. Use /add_to_cart to add items (one at a time)
        4. Use /cart to verify your selections
        5. When you're done adding items, send the message: "{COMPLETION_SIGNAL}"

        IMPORTANT: Do NOT attempt to checkout. Simply send "{COMPLETION_SIGNAL}" when ready.
        """


class EcomGreenAgentExecutor(AgentExecutor):
    """
    A2A-compatible executor that wraps existing assessment logic
    """
    
    def __init__(self, df_products: pd.DataFrame, df_orders: pd.DataFrame):
        self.df_products = df_products
        self.df_orders = df_orders
        self.runs: List[Dict[str, Any]] = []
    
    def _sample_user_ids(self, n: int, random_state: Optional[int] = None) -> List[int]:
        """
        Sample n random user IDs from orders dataset using numpy RNG.
        Supports reproducibility via random_state.
        
        Args:
            n: Number of users to sample
            random_state: Optional seed for deterministic sampling
            
        Returns:
            List of sampled user IDs
            
        Raises:
            ValueError: If n exceeds available users
        """
        user_ids = self.df_orders["user_id"].unique()
        
        if n > len(user_ids):
            raise ValueError(
                f"Requested sample size {n} exceeds unique user count {len(user_ids)}."
            )
        
        rng = np.random.default_rng(random_state)
        sampled = rng.choice(user_ids, size=n, replace=False)
        
        return sampled.tolist()

    def _build_baseline_prompt(self, task_info: Dict) -> str:
        """Build prompt for baseline evaluation (Henry method)"""
        user_id = task_info["user_id"]
        order_data = split_user_orders(user_id, self.df_products, self.df_orders)
        return henry_build_prompt(
            order_data["previous_orders_df"],
            order_data["days_since_last"],
            user_id
        )
    
    def _build_task_message(self, task_info: Dict) -> str:
        """Build task description for white agent - WITHOUT CHECKOUT TOOL"""
        user_id = task_info["user_id"]
        order_data = split_user_orders(user_id, self.df_products, self.df_orders)
        
        railway_base = task_info.get("railway_base_url", "https://ecom.railway.app")
        agent_key = task_info.get("agent_key", "test_agent")
        
        # CHANGE: Store agent_key for later checkout call
        self.current_agent_key = agent_key
        
        prompt = henry_build_prompt(
            order_data["previous_orders_df"],
            order_data["days_since_last"],
            user_id
        )
        
        # Tool list WITHOUT /checkout
        tools_text = f"""
            ### Available Tools:

            **Base URL**: {railway_base}
            **Your agent_key**: {agent_key}

            1. **GET /search**
            Query params: ?q=<search_term>
            Returns: List of matching products

            2. **GET /insights/product**  
            Query params: ?product_id=<id>
            Returns: Detailed product information

            3. **POST /cart/add**
            Body: {{"items": [{{"product_id": <id>, "qty": <qty>}}], "agent_key": "{agent_key}"}}
            Returns: Confirmation

            4. **GET /cart**
            Query params: ?agent_key={agent_key}
            Returns: Current cart contents

            **IMPORTANT**: Do NOT call /checkout. When you've added all items to the cart, 
            send the message "{COMPLETION_SIGNAL}" to indicate you're ready for evaluation.
            """
        
        return f"""{prompt}

            {tools_text}

            Remember: After adding items, send "{COMPLETION_SIGNAL}" when done.
            """
    
    def _call_railway_checkout(self, agent_key: str) -> List[int]:
        """
        Call Railway /checkout API to get final cart contents
        Returns list of product_ids in the cart
        """
        try:
            print(f"[Green Agent] Calling Railway /checkout for agent_key={agent_key}")
            
            # Call Railway API
            response = requests.post(
                RAILWAY_CHECKOUT_URL,
                json={"agent_key": agent_key},
                timeout=30
            )
            print(f"[Green Agent] Checkout response status: {response.status_code}")
            
            try:
                checkout_data = response.json()
            except Exception as e:
                print(f"[Green Agent] Failed to parse checkout JSON: {response.text}")
                raise e

            print(f"[Green Agent] Checkout response: {json.dumps(checkout_data, indent=2)}")
            
            # Extract product_ids from items
            items = checkout_data.get("items", [])
            product_ids = []
            
            for item in items:
                # Handle both list of dicts and list of IDs if API changes
                if isinstance(item, dict):
                    pid = item.get("product_id")
                    qty = item.get("qty", 1)
                    if pid is not None:
                        product_ids.extend([int(pid)] * int(qty))
                elif isinstance(item, int):
                    product_ids.append(item)
            
            print(f"[Green Agent] Extracted {len(product_ids)} products from checkout")
            return product_ids
            
        except requests.exceptions.RequestException as e:
            print(f"[Green Agent] ERROR calling Railway checkout: {e}")
            raise ValueError(f"Failed to call Railway checkout API: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[Green Agent] ERROR parsing checkout response: {e}")
            raise ValueError(f"Invalid checkout response format: {e}")
    
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Main execution flow - routes to appropriate mode"""
        print("Green agent: Received assessment task...")
        
        try:
            print("Green agent: Received assessment request")
            user_input = context.get_user_input()
            
            # Try to parse as XML-wrapped format first (AgentBeats platform style)
            if "<config>" in user_input or "<white_agent_url>" in user_input:
                print("Green agent: Detected XML-wrapped format")
                tags = parse_tags(user_input)
                
                # Extract white agent URL if provided
                white_agent_url = tags.get("white_agent_url", "").strip()
                
                # Extract config JSON
                config_str = tags.get("config", "{}")
                task_config = json.loads(config_str)
                
                # Merge white_agent_url into config if provided
                if white_agent_url:
                    task_config["white_agent_url"] = white_agent_url
            
            else:
                # Direct JSON format (your current format)
                print("Green agent: Detected direct JSON format")
                task_config = json.loads(user_input)

            # Auto-detect mode if not provided (fixes AgentBeats platform launch)
            if "mode" not in task_config:
                if "user_id" in task_config:
                    # If user_id is present, assume single user mode
                    task_config["mode"] = "white_agent" if task_config.get("white_agent_url") else "baseline"
                else:
                    # If no user_id, assume benchmark mode (Platform default)
                    print("Green agent: No mode/user_id specified, defaulting to BENCHMARK mode")
                    task_config["mode"] = "benchmark"
                    # Set defaults if missing
                    if "num_users" not in task_config:
                        task_config["num_users"] = 5
                    if "environment_base" not in task_config:
                         task_config["environment_base"] = "https://green-agent-production.up.railway.app"

            agent_key = task_config.get("agent_key", f"user{task_config.get('user_id', 0)}")
            self.current_agent_key = agent_key
            print(f"[Green Agent] Using agent_key: {agent_key}")
            
            # ============================================================
            # MODE 1: BENCHMARK (Green agent controls multiple users)
            # ============================================================
            if task_config.get("mode") == "benchmark":
                # Clear previous runs
                self.runs = []
                await self._run_benchmark(task_config, event_queue)
                return
            
            # ============================================================
            # MODE 2 & 3: SINGLE USER (Launcher controls)
            # ============================================================
            user_id = task_config["user_id"]
            task_id = task_config.get("task_id", f"user{user_id}_task")
            white_agent_url = task_config.get("white_agent_url")
            
            # Default use_baseline to False if white_agent_url is present, otherwise True
            default_use_baseline = False if white_agent_url else True
            use_baseline = task_config.get("use_baseline", default_use_baseline)

            env_base_url = task_config.get("environment_base", "http://localhost:8001")
            
            # Determine mode
            if use_baseline or not white_agent_url:
                mode = "baseline"
                print(f"Green agent: Mode=BASELINE for user {user_id}")
            else:
                mode = "white_agent"
                print(f"Green agent: Mode=WHITE_AGENT for user {user_id}")
            
            # Run single assessment
            await self._run_single_assessment(
                user_id, task_id, mode, white_agent_url,
                env_base_url, event_queue
            )
            
        except Exception as e:
            error_msg = f"Assessment failed: {str(e)}"
            print(f"Green agent error: {error_msg}")
            import traceback
            traceback.print_exc()
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_agent_text_message("Assessment cancelled.")
        )
    
    # testing
    

    # ========================================================================
    # SINGLE USER ASSESSMENT
    # ========================================================================
    
    async def _run_single_assessment(
        self, user_id: int, task_id: str, mode: str,
        white_agent_url: Optional[str], env_base_url: str,
        event_queue: EventQueue
    ):
        """Run assessment for a single user"""
        
        # 1. Prepare task data
        parts = split_user_orders(user_id, self.df_products, self.df_orders)
        prompt_text = henry_build_prompt(
            parts["previous_orders_df"],
            parts["days_since_last"],
            user_id
        )
        
        # 2. Get prediction based on mode
        if mode == "baseline":
            predicted_items = self._baseline_policy(parts["previous_order_df"])
        else:
            predicted_items = await self._white_agent_policy(
                white_agent_url, prompt_text, env_base_url,
                user_id, task_id, event_queue
            )
        
        # Get ground truth
        ground_truth_pids = set(pd.to_numeric(parts["current_order_df"]["product_id"], errors="coerce").dropna().astype(int).tolist())
        predicted_pids = set(predicted_items.keys())
        
        print(f"Green agent: Predicted {len(predicted_items)} items, Ground truth {len(ground_truth_pids)} items")
        
        # 3. Evaluate prediction
        t0 = time.time()
        henry_res = evaluate_basket(
            predicted_product_ids=list(predicted_items.keys()),
            nth_order_df=parts["current_order_df"],
            products_catalog=self.df_products,
            weights=(0.6, 0.2, 0.2)
        )
        latency_ms = int((time.time() - t0) * 1000)
        
        # 4. Format and return results
        metrics = {
            "precision": henry_res["products"]["precision"],
            "recall": henry_res["products"]["recall"],
            "f1": henry_res["products"]["f1"],
            "blended_f1": henry_res["blended_f1"],
            "latency_ms": latency_ms
        }
        
        self.runs.append({
            "task_id": task_id,
            "user_id": user_id,
            "metrics": metrics,
            "proposed_items": predicted_items,
            "mode": mode
        })
        
        result_msg = self._format_results(
            metrics, henry_res, mode, 
            predicted_pids=predicted_pids,
            ground_truth_pids=ground_truth_pids
        )
        await event_queue.enqueue_event(new_agent_text_message(result_msg))
        
        print(f"Green agent: Complete. Blended F1={metrics['blended_f1']:.3f} (Product F1={metrics['f1']:.3f}), Mode={mode}")
    
    # ========================================================================
    # BENCHMARK MODE - Multiple users
    # ========================================================================
    
    def _get_user_order_size(self, user_id: int, min_items: int = 10) -> Tuple[bool, int]:
        """
        Check if a user's current order meets the minimum size requirement.
        
        Returns:
            (is_valid, order_size) - True if order >= min_items, False otherwise
        """
        try:
            parts = split_user_orders(user_id, self.df_products, self.df_orders)
            ground_truth_pids = set(pd.to_numeric(
                parts["current_order_df"]["product_id"], 
                errors="coerce"
            ).dropna().astype(int).tolist())
            order_size = len(ground_truth_pids)
            return (order_size >= min_items, order_size)
        except Exception as e:
            print(f"  ⚠️  Error checking user {user_id}: {e}")
            return (False, 0)
    
    def _sample_valid_user(self, all_users: list, used_users: set, min_items: int = 10, max_attempts: int = 50) -> Optional[int]:
        """
        Sample a random user that has an order >= min_items.
        Keep resampling until we find a valid user or hit max_attempts.
        
        Returns:
            user_id if found, None if max_attempts exhausted
        """
        available_users = [u for u in all_users if u not in used_users]
        
        if not available_users:
            return None
        
        for attempt in range(max_attempts):
            user_id = random.choice(available_users)
            is_valid, order_size = self._get_user_order_size(user_id, min_items)
            
            if is_valid:
                print(f"  ✅ Found valid user {user_id} with {order_size} items (attempt {attempt + 1})")
                return user_id
            else:
                print(f"  ⏭️  User {user_id} has only {order_size} items, resampling... (attempt {attempt + 1})")
                # Remove from available pool for this round
                available_users.remove(user_id)
                if not available_users:
                    break
        
        print(f"  ❌ Could not find valid user after {max_attempts} attempts")
        return None

    async def _run_benchmark(self, task_config: dict, event_queue: EventQueue):
        """
        Run batch benchmark - Green agent controls user selection
        """
        print("Green agent: Mode=BENCHMARK")
        
        num_users = task_config.get("num_users", 10)
        white_agent_url = task_config.get("white_agent_url")
        env_base_url = task_config.get("environment_base", "http://localhost:8001")
        min_order_size = task_config.get("min_order_size", 10)  # Configurable threshold
        
        # Default use_baseline to False if white_agent_url is present, otherwise True
        default_use_baseline = False if white_agent_url else True
        use_baseline = task_config.get("use_baseline", default_use_baseline)
        
        # Get all available users
        all_users = self.df_orders["user_id"].unique().tolist()
        used_users = set()  # Track users we've already tested
        
        print(f"Green agent: Will sample {num_users} users with orders >= {min_order_size} items")
        print(f"Green agent: Pool of {len(all_users)} total users available")
        
        await event_queue.enqueue_event(
            new_agent_text_message(f"Starting benchmark with {num_users} users (min {min_order_size} items per order)...")
        )
        
        # Run assessments - sample valid users on-the-fly
        results = []
        skipped_users = []
        
        for test_idx in range(1, num_users + 1):
            print(f"\n{'='*60}")
            print(f"Green agent: [{test_idx}/{num_users}] Sampling new user...")
            
            # Sample a valid user (with order >= min_items)
            user_id = self._sample_valid_user(all_users, used_users, min_order_size)
            
            if user_id is None:
                print(f"  ❌ Could not find any more valid users. Stopping at {len(results)} results.")
                break
            
            used_users.add(user_id)
            print(f"Green agent: Testing User {user_id}")
            
            try:
                # Prepare data (we know this user is valid, but get the data)
                parts = split_user_orders(user_id, self.df_products, self.df_orders)
                
                # Get prediction
                if use_baseline or not white_agent_url:
                    predicted_items = self._baseline_policy(parts["previous_order_df"])
                else:
                    prompt_text = henry_build_prompt(
                        parts["previous_orders_df"],
                        parts["days_since_last"],
                        user_id
                    )
                    predicted_items = await self._white_agent_policy(
                        white_agent_url, prompt_text, env_base_url,
                        user_id, f"benchmark_{user_id}", event_queue
                    )
                
                # Get ground truth and predicted sets
                ground_truth_pids = set(pd.to_numeric(parts["current_order_df"]["product_id"], errors="coerce").dropna().astype(int).tolist())
                predicted_pids = set(predicted_items.keys())
                
                # Evaluate
                henry_res = evaluate_basket(
                    predicted_product_ids=list(predicted_items.keys()),
                    nth_order_df=parts["current_order_df"],
                    products_catalog=self.df_products,
                    weights=(0.6, 0.2, 0.2)
                )
                
                metrics = {
                    "user_id": user_id,
                    "precision": henry_res["products"]["precision"],
                    "recall": henry_res["products"]["recall"],
                    "f1": henry_res["products"]["f1"],
                    "blended_f1": henry_res["blended_f1"]
                }
                
                results.append(metrics)
                
                # Print detailed results for this user (using Blended F1 as primary)
                overlap_pids = predicted_pids & ground_truth_pids
                missed_pids = ground_truth_pids - predicted_pids
                extra_pids = predicted_pids - ground_truth_pids
                
                # Get aisle and department sets for detailed reporting
                truth_aisles = set(parts["current_order_df"]["aisle"].dropna().astype(str).tolist()) if "aisle" in parts["current_order_df"].columns else set()
                truth_depts = set(parts["current_order_df"]["department"].dropna().astype(str).tolist()) if "department" in parts["current_order_df"].columns else set()
                
                pred_aisles = set(henry_res["meta"]["pred_counts"]["aisles"]) if isinstance(henry_res["meta"]["pred_counts"]["aisles"], (list, set)) else set()
                pred_depts = set(henry_res["meta"]["pred_counts"]["departments"]) if isinstance(henry_res["meta"]["pred_counts"]["departments"], (list, set)) else set()
                
                # Need to get actual aisle/dept names from catalog
                if predicted_pids:
                    pred_catalog = self.df_products[self.df_products["product_id"].isin(predicted_pids)]
                    pred_aisles = set(pred_catalog["aisle"].dropna().astype(str).tolist())
                    pred_depts = set(pred_catalog["department"].dropna().astype(str).tolist())
                
                overlap_aisles = truth_aisles & pred_aisles
                missed_aisles = truth_aisles - pred_aisles
                extra_aisles = pred_aisles - truth_aisles
                
                overlap_depts = truth_depts & pred_depts
                missed_depts = truth_depts - pred_depts
                extra_depts = pred_depts - truth_depts
                
                print(f"  📊 Results: Blended F1={metrics['blended_f1']:.3f} | Product F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
                
                # Product-level details
                print(f"\n  🛒 PRODUCT LEVEL:")
                print(f"     Ground Truth: {len(ground_truth_pids)} items - {sorted(list(ground_truth_pids))[:10]}{'...' if len(ground_truth_pids) > 10 else ''}")
                print(f"     Predicted: {len(predicted_pids)} items - {sorted(list(predicted_pids))[:10]}{'...' if len(predicted_pids) > 10 else ''}")
                print(f"     ✅ Correct (TP={len(overlap_pids)}): {sorted(list(overlap_pids))[:8]}{'...' if len(overlap_pids) > 8 else ''}")
                print(f"     ❌ Missed (FN={len(missed_pids)}): {sorted(list(missed_pids))[:8]}{'...' if len(missed_pids) > 8 else ''}")
                print(f"     ⚠️  Extra (FP={len(extra_pids)}): {sorted(list(extra_pids))[:8]}{'...' if len(extra_pids) > 8 else ''}")
                
                # Aisle-level details
                print(f"\n  🏪 AISLE LEVEL (F1={henry_res['aisles']['f1']:.3f}, P={henry_res['aisles']['precision']:.3f}, R={henry_res['aisles']['recall']:.3f}):")
                print(f"     Ground Truth: {len(truth_aisles)} aisles - {sorted(list(truth_aisles))[:10]}{'...' if len(truth_aisles) > 10 else ''}")
                print(f"     Predicted: {len(pred_aisles)} aisles - {sorted(list(pred_aisles))[:10]}{'...' if len(pred_aisles) > 10 else ''}")
                print(f"     ✅ Correct (TP={len(overlap_aisles)}): {sorted(list(overlap_aisles))[:8]}{'...' if len(overlap_aisles) > 8 else ''}")
                print(f"     ❌ Missed (FN={len(missed_aisles)}): {sorted(list(missed_aisles))[:8]}{'...' if len(missed_aisles) > 8 else ''}")
                print(f"     ⚠️  Extra (FP={len(extra_aisles)}): {sorted(list(extra_aisles))[:8]}{'...' if len(extra_aisles) > 8 else ''}")
                
                # Department-level details
                print(f"\n  🏬 DEPARTMENT LEVEL (F1={henry_res['departments']['f1']:.3f}, P={henry_res['departments']['precision']:.3f}, R={henry_res['departments']['recall']:.3f}):")
                print(f"     Ground Truth: {len(truth_depts)} depts - {sorted(list(truth_depts))[:10]}{'...' if len(truth_depts) > 10 else ''}")
                print(f"     Predicted: {len(pred_depts)} depts - {sorted(list(pred_depts))[:10]}{'...' if len(pred_depts) > 10 else ''}")
                print(f"     ✅ Correct (TP={len(overlap_depts)}): {sorted(list(overlap_depts))[:8]}{'...' if len(overlap_depts) > 8 else ''}")
                print(f"     ❌ Missed (FN={len(missed_depts)}): {sorted(list(missed_depts))[:8]}{'...' if len(missed_depts) > 8 else ''}")
                print(f"     ⚠️  Extra (FP={len(extra_depts)}): {sorted(list(extra_depts))[:8]}{'...' if len(extra_depts) > 8 else ''}")
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's an OpenAI API error (rate limit, etc)
                if "OpenAI API Error" in error_msg or "rate_limit" in error_msg.lower() or "White agent failed" in error_msg:
                    print(f"  ⚠️  OpenAI API Error detected: {error_msg[:100]}")
                    print(f"  Pausing 1 second and continuing to next user...")
                    skipped_users.append({"user_id": user_id, "reason": error_msg[:100]})
                    await asyncio.sleep(1)
                else:
                    print(f"  Error: {error_msg[:200]}")
                    skipped_users.append({"user_id": user_id, "reason": error_msg[:100]})
                
                continue
        
        # Aggregate results
        if results:
            avg_blended = sum(r["blended_f1"] for r in results) / len(results)
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_precision = sum(r["precision"] for r in results) / len(results)
            avg_recall = sum(r["recall"] for r in results) / len(results)
            
            mode_label = "BASELINE" if use_baseline else "WHITE AGENT"
            
            summary_msg = f"""
                Benchmark Complete ✅ (Mode: {mode_label})

                Tested {len(results)} users successfully (min 10 items per order)
                Skipped {len(skipped_users)} users (small orders or errors)

                🎯 PRIMARY METRIC - Blended F1: {avg_blended:.3f}
                
                Average Metrics:
                - Blended F1: {avg_blended:.3f} ⭐ (PRIMARY)
                - Product F1: {avg_f1:.3f}
                - Precision: {avg_precision:.3f}
                - Recall: {avg_recall:.3f}

                Per-user results (top 10):
                """
            for r in results[:10]:  # Show first 10
                summary_msg += f"\n  User {r['user_id']}: Blended F1={r['blended_f1']:.3f} (Product F1={r['f1']:.3f})"
            
            if len(results) > 10:
                summary_msg += f"\n  ... and {len(results) - 10} more"
            
            if skipped_users:
                summary_msg += f"\n\nSkipped users:"
                for s in skipped_users[:5]:  # Show first 5 skipped
                    summary_msg += f"\n  User {s['user_id']}: {s['reason']}"
                if len(skipped_users) > 5:
                    summary_msg += f"\n  ... and {len(skipped_users) - 5} more"
            
            # Print to stdout as well so user sees it in terminal
            print(summary_msg)

            await event_queue.enqueue_event(new_agent_text_message(summary_msg))
            print(f"\n🎯 Green agent: Benchmark complete. Avg Blended F1={avg_blended:.3f} (Product F1={avg_f1:.3f})")
        else:
            await event_queue.enqueue_event(
                new_agent_text_message("Benchmark failed: No results collected")
            )
    
    # ========================================================================
    # BASELINE MODE - Repeat previous order
    # ========================================================================
    
    def _baseline_policy(self, previous_order_df: pd.DataFrame) -> Dict[int, int]:
        """Baseline: repeat the previous order"""
        if previous_order_df.empty:
            return {}
        
        basket = {}
        for pid in previous_order_df["product_id"].dropna():
            pid_int = int(pid)
            basket[pid_int] = basket.get(pid_int, 0) + 1
        
        return basket
    
    # ========================================================================
    # WHITE AGENT MODE - Send task to external agent
    # ========================================================================

    async def _white_agent_policy(
        self,
        white_agent_url: str | None,
        prompt_text: str,
        env_base_url: str,
        user_id: int,
        task_id: str,
        event_queue: EventQueue
    ) -> Dict[int, int]:
        """
        Send task to white agent, wait for completion signal,
        then call Railway checkout to get actual cart contents.
        
        Returns: Dict[product_id -> quantity]
        """
        
        # Ensure we have an agent_key
        if not hasattr(self, 'current_agent_key'):
            self.current_agent_key = f"user{user_id}"
        
        # Build task info with agent_key
        # Ensure unique agent_key per task if not provided in task_info, 
        # but respect self.current_agent_key if it's set for single-user mode.
        # For benchmark mode, we want unique keys per user.
        effective_agent_key = self.current_agent_key
        if "benchmark" in task_id:
             effective_agent_key = f"bench_user{user_id}_{int(time.time())}"
             # Update the class member so checkout uses the same key
             self.current_agent_key = effective_agent_key

        task_info = {
            "user_id": user_id,
            "railway_base_url": env_base_url,
            "agent_key": effective_agent_key
        }
        
        # Build task message (this will store agent_key)
        task_message = self._build_task_message(task_info)
        
        print(f"\n[Green Agent] 📝 GENERATED PROMPT for User {user_id}:")
        print("="*60)
        print(task_message)
        print("="*60 + "\n")
        
        print(f"[Green Agent] Sending task to white agent at {white_agent_url}")
        print(f"[Green Agent] Agent key: {self.current_agent_key}")
        
        # Send initial task to white agent
        try:
            resp_msg = await my_a2a.send_message(
                white_agent_url,
                task_message,
            )
        except Exception as e:
            print(f"[Green Agent] Failed to send message to white agent: {e}")
            raise ValueError(f"Could not reach white agent: {e}")
        
        # Interaction loop - wait for completion signal
        max_turns = 20
        turn = 0
        completion_received = False

        while turn < max_turns:
            turn += 1
            
            # Extract parts from response - using the pattern from quick_test.py
            if hasattr(resp_msg, 'root'):
                from a2a.types import SendMessageSuccessResponse, JSONRPCErrorResponse
                
                if isinstance(resp_msg.root, JSONRPCErrorResponse):
                    print(f"[Green Agent] Error from white agent: {resp_msg.root.error.message}")
                    break
                
                if isinstance(resp_msg.root, SendMessageSuccessResponse):
                    result = resp_msg.root.result
                    # result has a 'parts' attribute directly (it's a MessageResult, not a Task)
                    text_parts = get_text_parts(result.parts)
                else:
                    print(f"[Green Agent] Unexpected response type: {type(resp_msg.root)}")
                    break
            else:
                print(f"[Green Agent] Invalid response structure")
                break
            
            if not text_parts:
                print(f"[Green Agent] No text in white agent response (turn {turn})")
                break
            
            response_text = text_parts[0]
            print(f"[Green Agent] Turn {turn}: {response_text[:400]}...")  # Increased length to see more context
            
            # Check for completion signal
            if COMPLETION_SIGNAL in response_text:
                print(f"[Green Agent] ✅ Completion signal received at turn {turn}!")
                # Check if this was after an error - raise exception to skip this user
                if "OpenAI API Error" in response_text or "Error after" in response_text:
                    error_snippet = response_text[:200] if len(response_text) > 200 else response_text
                    print(f"[Green Agent] ⚠️  Warning: White agent completed with errors. Skipping user.")
                    raise ValueError(f"White agent failed: {error_snippet}")
                completion_received = True
                break
            
            # If not complete, acknowledge and let white agent continue
            try:
                # Add a helpful prompt to nudge the agent if it seems stuck
                continue_msg = "Acknowledged. Please continue your task. Remember to send '##READY_FOR_CHECKOUT##' when you are done adding items."
                resp_msg = await my_a2a.send_message(
                    white_agent_url,
                    continue_msg,
                )
            except Exception as e:
                print(f"[Green Agent] Error in turn {turn}: {e}")
                break
            
        # Verify completion signal was received
        if not completion_received:
            error_msg = f"White agent did not send completion signal after {max_turns} turns"
            print(f"[Green Agent] ❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Call Railway checkout to get actual cart contents
        print(f"[Green Agent] Calling Railway checkout API...")
        agent_key = self.current_agent_key
        
        try:
            product_ids = self._call_railway_checkout(agent_key)
        except Exception as e:
            print(f"[Green Agent] ❌ Checkout failed: {e}")
            raise ValueError(f"Railway checkout failed: {e}")
        
        # Handle empty cart
        if not product_ids:
            print(f"[Green Agent] ⚠️ Warning: Checkout returned empty cart")
            print(f"[Green Agent] White agent may not have added any items to cart")
            return {}
        
        # Convert list of product_ids to dict (product_id -> quantity)
        basket = {}
        for pid in product_ids:
            basket[pid] = basket.get(pid, 0) + 1
        
        print(f"[Green Agent] ✅ Final basket: {len(basket)} unique products, {len(product_ids)} total items")
        print(f"[Green Agent] Products: {list(basket.keys())[:10]}...")
        
        return basket
        
    
    def _format_results(self, metrics: Dict, full_eval: Dict, mode: str, 
                        predicted_pids: set = None, ground_truth_pids: set = None) -> str:
        """Format assessment results with ground truth"""
        emoji = "✅" if metrics["blended_f1"] > 0.4 else "⚠️"  # Changed to use blended_f1
        mode_label = "BASELINE" if mode == "baseline" else "WHITE AGENT"
        
        # Calculate overlaps
        overlap_pids = predicted_pids & ground_truth_pids if (predicted_pids and ground_truth_pids) else set()
        missed_pids = ground_truth_pids - predicted_pids if (predicted_pids and ground_truth_pids) else set()
        extra_pids = predicted_pids - ground_truth_pids if (predicted_pids and ground_truth_pids) else set()
        
        result = f"""
                Assessment Complete {emoji} (Mode: {mode_label})

                🎯 PRIMARY METRIC:
                - Blended F1: {metrics['blended_f1']:.3f} ⭐
                
                📊 DETAILED METRICS:
                - Product F1: {metrics['f1']:.3f}
                - Precision: {metrics['precision']:.3f}
                - Recall: {metrics['recall']:.3f}
                - Latency: {metrics['latency_ms']}ms

                📈 DETAILED BREAKDOWN:
                Product-level: {full_eval['products']['tp']} TP, {full_eval['products']['fp']} FP, {full_eval['products']['fn']} FN
                  - Precision: {full_eval['products']['precision']:.3f}
                  - Recall: {full_eval['products']['recall']:.3f}
                  - F1: {full_eval['products']['f1']:.3f}
                
                Aisle-level: {full_eval['aisles']['tp']} TP, {full_eval['aisles']['fp']} FP, {full_eval['aisles']['fn']} FN
                  - Precision: {full_eval['aisles']['precision']:.3f}
                  - Recall: {full_eval['aisles']['recall']:.3f}
                  - F1: {full_eval['aisles']['f1']:.3f}
                
                Department-level: {full_eval['departments']['tp']} TP, {full_eval['departments']['fp']} FP, {full_eval['departments']['fn']} FN
                  - Precision: {full_eval['departments']['precision']:.3f}
                  - Recall: {full_eval['departments']['recall']:.3f}
                  - F1: {full_eval['departments']['f1']:.3f}
                """
        
        # Add ground truth details if available
        if predicted_pids and ground_truth_pids:
            result += f"""
                🎯 GROUND TRUTH vs PREDICTED:
                Ground Truth: {len(ground_truth_pids)} items - {sorted(list(ground_truth_pids))[:15]}{'...' if len(ground_truth_pids) > 15 else ''}
                Predicted: {len(predicted_pids)} items - {sorted(list(predicted_pids))[:15]}{'...' if len(predicted_pids) > 15 else ''}
                
                ✅ Correct (TP): {len(overlap_pids)} items - {sorted(list(overlap_pids))[:10]}{'...' if len(overlap_pids) > 10 else ''}
                ❌ Missed (FN): {len(missed_pids)} items - {sorted(list(missed_pids))[:10]}{'...' if len(missed_pids) > 10 else ''}
                ⚠️ Extra (FP): {len(extra_pids)} items - {sorted(list(extra_pids))[:10]}{'...' if len(extra_pids) > 10 else ''}
                """
        
        return result


def load_agent_card_toml(agent_name: str, script_dir: Path) -> dict:
    """Load agent card from TOML file, or return defaults"""
    toml_path = script_dir / f"{agent_name}.toml"
    
    if toml_path.exists():
        print(f"Loading agent card from {toml_path}")
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    else:
        # Return minimal default configuration matching A2A spec
        print(f"No TOML found at {toml_path}, using defaults")
        return {
            "name": agent_name,
            "version": "1.0.0",
            "description": "Ecom grocery basket assessment agent",
            "capabilities": {
                "input": ["text"],
                "output": ["text"],
                "streaming": False
            },
            "skills": [],
        }


def start_green_agent(
    agent_name: str = "ecom_green_agent",
    host: str = "localhost",
    port: int = 9001,
    products_csv: Optional[str] = None,
    orders_csv: Optional[str] = None
):
    """
    Start the AgentBeats-compatible green agent
    """
    # Resolve paths relative to script location or use environment variables
    import os
    
    if products_csv is None:
        products_csv = os.environ.get("PRODUCTS_CSV")
        if products_csv is None:
            products_csv = str(ROOT / "dataset" / "ic_products.csv")
    
    if orders_csv is None:
        orders_csv = os.environ.get("ORDERS_CSV")
        if orders_csv is None:
            orders_csv = str(ROOT / "dataset" / "super_shortened_orders_products_combined.csv")
    
    # Convert to absolute paths if relative
    if not os.path.isabs(products_csv):
        products_csv = str(ROOT / products_csv)
    if not os.path.isabs(orders_csv):
        orders_csv = str(ROOT / orders_csv)
    
    print(f"Loading datasets...")
    print(f"  Products: {products_csv}")
    print(f"  Orders: {orders_csv}")
    
    if not os.path.exists(products_csv):
        raise FileNotFoundError(f"Products CSV not found: {products_csv}")
    if not os.path.exists(orders_csv):
        raise FileNotFoundError(f"Orders CSV not found: {orders_csv}")
    
    df_products = pd.read_csv(products_csv)
    df_orders = pd.read_csv(orders_csv)
    print(f"Loaded {len(df_products)} products, {len(df_orders)} order records")
    
    # Load agent card
    agent_card_dict = load_agent_card_toml(agent_name, ROOT)
    
    # IMPORTANT: Use AGENT_URL from environment
    agent_url = os.getenv("AGENT_URL")
    if agent_url:
        agent_card_dict["url"] = agent_url
        print(f"AGENT_URL: {agent_url}")
    else:
        # Fallback for local testing
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("AGENT_PORT", "9001"))
        agent_card_dict["url"] = f"http://{host}:{port}"

    agent_card = AgentCard(**agent_card_dict)
    
    # Create executor with your evaluation logic
    executor = EcomGreenAgentExecutor(df_products, df_orders)
    
    # Build A2A app
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore()
    )
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    # Get the underlying Starlette app
    starlette_app = app.build()
    
    # Debug: print all registered routes
    print(f"\nA2A framework routes:")
    for route in starlette_app.routes:
        # Use getattr to avoid direct attribute access which static analyzers may flag
        methods = getattr(route, 'methods', ['*'])
        path = getattr(route, 'path', None)
        if path is not None:
            print(f"  {methods} {path}")
        else:
            # Safely attempt to print a regex path if available, with fallbacks to avoid attribute errors
            path_regex = getattr(route, "path_regex", None)
            if path_regex is not None:
                pattern = getattr(path_regex, "pattern", None)
                if pattern is not None:
                    print(f"  [*] {pattern}")
                else:
                    print(f"  [*] {path_regex}")
            else:
                # Final fallback: print route representation
                print(f"  [*] {route!r}")
    
    
    # # Add AgentBeats-required endpoints
    from starlette.responses import JSONResponse
    from starlette.routing import Route, Mount
    async def status_endpoint(request):
        return JSONResponse({"status": "ok", "agent": "running"})
    
    # Add the status route
    starlette_app.routes.append(
        Route("/status", endpoint=status_endpoint, methods=["GET"])
    )
    
    # async def get_agent_card_root(request):
    #     """Root endpoint returns agent card (A2A standard)"""
    #     return JSONResponse(agent_card_dict)
    
    # async def get_agent_card_explicit(request):
    #     """Explicit /agent_card endpoint"""
    #     return JSONResponse(agent_card_dict)
    
    # async def healthcheck(request):
    #     """Health check endpoint for AgentBeats"""
    #     return JSONResponse({"status": "healthy", "agent": agent_card_dict["name"]})
    
    # async def reset_endpoint(request):
    #     """Reset endpoint - clears agent state"""
    #     # Clear any stored runs/state in the executor
    #     executor.runs.clear()
    #     return JSONResponse({"status": "reset", "message": "Agent state cleared"})
    
    # The A2A app should already have mounted /a2a/* routes
    # If not, we need to check what path it actually uses


    # # Add all standard endpoints for AgentBeats compatibility
    # starlette_app.routes.extend([
    #     Route("/", endpoint=get_agent_card_root, methods=["GET"]),
    #     Route("/agent_card", endpoint=get_agent_card_explicit, methods=["GET"]),
    #     Route("/healthz", endpoint=healthcheck, methods=["GET"]),
    #     Route("/health", endpoint=healthcheck, methods=["GET"]),
    #     Route("/reset", endpoint=reset_endpoint, methods=["POST"]),
    # ])

    # print(f"\nStarting green agent on {host}:{port}...")
    # print(f"\nAgentBeats-compatible endpoints:")
    # print(f"  GET  http://{host}:{port}/           → Agent card")
    # print(f"  GET  http://{host}:{port}/agent_card → Agent card")
    # print(f"  GET  http://{host}:{port}/healthz    → Health check")
    # print(f"  POST http://{host}:{port}/reset      → Reset state")
    # print(f"\nAgent ready for AgentBeats platform! ✅")
    # print(f"\n⚠️  Check the 'A2A framework routes' printed above for exact paths!")

    uvicorn.run(starlette_app, host=host, port=port)


if __name__ == "__main__":
    # Read from environment variables set by AgentBeats controller
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("AGENT_PORT", "9001"))
    role = os.getenv("ROLE", "green") 

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Adjust these paths to the dataset location
    products_csv = os.path.join(script_dir, "dataset", "ic_products.csv")
    orders_csv = os.path.join(script_dir, "dataset", "super_shortened_orders_products_combined.csv")
    
    if role == "green":
        start_green_agent(
            host=host,
            port=port,
            products_csv=products_csv,
            orders_csv=orders_csv
        )
    else:
        raise ValueError(f"Unknown role: {role}. Expected 'green'")