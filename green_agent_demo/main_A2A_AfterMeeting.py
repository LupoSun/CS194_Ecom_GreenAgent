"""
AgentBeats-compatible Green Agent for Ecom Assessment
Integrates existing FastAPI logic with A2A protocol
"""

import os
import uvicorn
import tomllib  
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Iterable
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, Message
from a2a.utils import new_agent_text_message, get_text_parts

from ab_src.my_util import my_a2a, parse_tags
from html import unescape

ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

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


class EcomGreenAgentExecutor(AgentExecutor):
    """
    A2A-compatible executor that wraps your existing assessment logic
    """
    
    def __init__(self, df_products: pd.DataFrame, df_orders: pd.DataFrame):
        self.df_products = df_products
        self.df_orders = df_orders
        self.runs: List[Dict[str, Any]] = []
    
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Main execution flow - routes to appropriate mode"""
        print("Green agent: Received assessment task...")
        
        try:
            # Parse task config
            user_input = context.get_user_input()
            task_config = json.loads(user_input)
            
            # ============================================================
            # MODE 1: BENCHMARK (Green agent controls multiple users)
            # ============================================================
            if task_config.get("mode") == "benchmark":
                await self._run_benchmark(task_config, event_queue)
                return
            
            # ============================================================
            # MODE 2 & 3: SINGLE USER (Launcher controls)
            # ============================================================
            user_id = task_config["user_id"]
            task_id = task_config.get("task_id", f"user{user_id}_task")
            use_baseline = task_config.get("use_baseline", True)
            white_agent_url = task_config.get("white_agent_url")
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
        
        print(f"Green agent: Predicted {len(predicted_items)} items")
        
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
        
        result_msg = self._format_results(metrics, henry_res, mode)
        await event_queue.enqueue_event(new_agent_text_message(result_msg))
        
        print(f"Green agent: Complete. F1={metrics['f1']:.3f}, Mode={mode}")
    
    # ========================================================================
    # BENCHMARK MODE - Multiple users
    # ========================================================================
    
    async def _run_benchmark(self, task_config: dict, event_queue: EventQueue):
        """
        Run batch benchmark - Green agent controls user selection
        """
        print("Green agent: Mode=BENCHMARK")
        
        num_users = task_config.get("num_users", 10)
        white_agent_url = task_config.get("white_agent_url")
        env_base_url = task_config.get("environment_base", "http://localhost:8001")
        use_baseline = task_config.get("use_baseline", True)
        
        # Select users from dataset
        all_users = self.df_orders["user_id"].unique().tolist()
        
        # Use specific users if provided, otherwise random sample
        if "user_ids" in task_config:
            test_users = task_config["user_ids"][:num_users]
        else:
            test_users = random.sample(all_users, min(num_users, len(all_users)))
        
        print(f"Green agent: Testing {len(test_users)} users")
        if len(test_users) <= 5:
            print(f"  Users: {test_users}")
        else:
            print(f"  Users: {test_users[:5]} ... and {len(test_users)-5} more")
        
        await event_queue.enqueue_event(
            new_agent_text_message(f"Starting benchmark with {len(test_users)} users...")
        )
        
        # Run assessments
        results = []
        for idx, user_id in enumerate(test_users, 1):
            print(f"\nGreen agent: [{idx}/{len(test_users)}] User {user_id}")
            
            try:
                # Prepare data
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
                print(f"  F1={metrics['f1']:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Aggregate results
        if results:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_precision = sum(r["precision"] for r in results) / len(results)
            avg_recall = sum(r["recall"] for r in results) / len(results)
            avg_blended = sum(r["blended_f1"] for r in results) / len(results)
            
            mode_label = "BASELINE" if use_baseline else "WHITE AGENT"
            
            summary_msg = f"""
                Benchmark Complete ✅ (Mode: {mode_label})

                Tested {len(results)} users

                Average Metrics:
                - F1 Score: {avg_f1:.3f}
                - Precision: {avg_precision:.3f}
                - Recall: {avg_recall:.3f}
                - Blended F1: {avg_blended:.3f}

                Per-user results:
                """
            for r in results[:10]:  # Show first 10
                summary_msg += f"\n  User {r['user_id']}: F1={r['f1']:.3f}"
            
            if len(results) > 10:
                summary_msg += f"\n  ... and {len(results) - 10} more"
            
            await event_queue.enqueue_event(new_agent_text_message(summary_msg))
            print(f"\nGreen agent: Benchmark complete. Avg F1={avg_f1:.3f}")
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
        self, white_agent_url: str, prompt: str, env_base_url: str,
        user_id: int, task_id: str, event_queue: EventQueue
    ) -> Dict[int, int]:
        """Send task to white agent with tool manifest"""
        
        # Build tool manifest for white agent
        agent_key = f"user_{user_id}_{task_id}"
        tool_manifest = {
            "environment_base": env_base_url,
            "agent_key": agent_key,
            "actions": [
                {
                    "name": "search",
                    "method": "GET",
                    "path": "/search",
                    "description": "Search for products",
                    "params": {"q": "query", "field": "name", "limit": 20}
                },
                {
                    "name": "cart_add",
                    "method": "POST",
                    "path": "/cart/add",
                    "description": "Add items to cart",
                    "body": {
                        "agent_key": agent_key,
                        "items": [{"product_id": 123, "qty": 2}]
                    }
                },
                {
                    "name": "checkout",
                    "method": "POST",
                    "path": "/checkout",
                    "description": "Finalize cart",
                    "body": {"agent_key": agent_key}
                }
            ]
        }
        
        task_message = f"""
            {prompt}

            ENVIRONMENT API:
            Base URL: {env_base_url}
            Your session key: {agent_key}

            Available tools:
            {json.dumps(tool_manifest['actions'], indent=2)}

            INSTRUCTIONS:
            1. Use search to explore products (optional)
            2. Use cart_add to build your basket
            3. Use checkout when done
            4. Return JSON: {{"predicted_items": {{"<product_id>": <qty>, ...}}}}
            """
        
        await event_queue.enqueue_event(
            new_agent_text_message("Requesting prediction from white agent...")
        )
        
        try:
            from ab_src.my_util import my_a2a
            
            white_response = await my_a2a.send_message(
                white_agent_url, task_message, context_id=None
            )
            
            return self._parse_white_agent_response(white_response)
            
        except Exception as e:
            raise Exception(f"White agent communication failed: {e}")
    
    def _parse_white_agent_response(self, response) -> Dict[int, int]:
        """Extract predicted items from white agent response"""
        from a2a.types import SendMessageSuccessResponse
        
        res_root = response.root
        if not isinstance(res_root, SendMessageSuccessResponse):
            raise ValueError("Expected SendMessageSuccessResponse")
        
        res_result = res_root.result
        if not isinstance(res_result, Message):
            raise ValueError("Expected Message result")
        
        text_parts = get_text_parts(res_result.parts)
        if not text_parts:
            raise ValueError("No text in white agent response")
        
        response_text = text_parts[0]
        
        # Parse JSON from response
        try:
            if "<json>" in response_text:
                tags = parse_tags(response_text)
                data = json.loads(tags["json"])
            else:
                data = json.loads(response_text)
            
            predicted = data.get("predicted_items", {})
            return {int(k): int(v) for k, v in predicted.items()}
        except Exception as e:
            raise ValueError(f"Failed to parse white agent response: {e}")
    
    def _format_results(self, metrics: Dict, full_eval: Dict, mode: str) -> str:
        """Format assessment results"""
        emoji = "✅" if metrics["f1"] > 0.5 else "⚠️"
        mode_label = "BASELINE" if mode == "baseline" else "WHITE AGENT"
        
        return f"""
                Assessment Complete {emoji} (Mode: {mode_label})

                Metrics:
                - F1 Score: {metrics['f1']:.3f}
                - Precision: {metrics['precision']:.3f}
                - Recall: {metrics['recall']:.3f}
                - Blended F1: {metrics['blended_f1']:.3f}
                - Latency: {metrics['latency_ms']}ms

                Product-level: {full_eval['products']['tp']} TP, {full_eval['products']['fp']} FP, {full_eval['products']['fn']} FN
                Aisle-level F1: {full_eval['aisles']['f1']:.3f}
                Department-level F1: {full_eval['departments']['f1']:.3f}
                """

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
    products_csv: str = None,
    orders_csv: str = None
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
    
    # Load agent card from TOML or use defaults
    agent_card_dict = load_agent_card_toml(agent_name, ROOT)
    agent_card_dict["url"] = f"http://{host}:{port}"  # Set URL at runtime
    
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
        if hasattr(route, 'path'):
            methods = getattr(route, 'methods', ['*'])
            print(f"  {methods} {route.path}")
        elif hasattr(route, 'path_regex'):
            print(f"  [*] {route.path_regex.pattern}")
    
    # Add AgentBeats-required endpoints
    from starlette.responses import JSONResponse
    from starlette.routing import Route, Mount
    
    async def get_agent_card_root(request):
        """Root endpoint returns agent card (A2A standard)"""
        return JSONResponse(agent_card_dict)
    
    async def get_agent_card_explicit(request):
        """Explicit /agent_card endpoint"""
        return JSONResponse(agent_card_dict)
    
    async def healthcheck(request):
        """Health check endpoint for AgentBeats"""
        return JSONResponse({"status": "healthy", "agent": agent_card_dict["name"]})
    
    async def reset_endpoint(request):
        """Reset endpoint - clears agent state"""
        # Clear any stored runs/state in the executor
        executor.runs.clear()
        return JSONResponse({"status": "reset", "message": "Agent state cleared"})
    
    # The A2A app should already have mounted /a2a/* routes
    # If not, we need to check what path it actually uses

    
    # Add all standard endpoints for AgentBeats compatibility
    starlette_app.routes.extend([
        Route("/", endpoint=get_agent_card_root, methods=["GET"]),
        Route("/agent_card", endpoint=get_agent_card_explicit, methods=["GET"]),
        Route("/healthz", endpoint=healthcheck, methods=["GET"]),
        Route("/health", endpoint=healthcheck, methods=["GET"]),
        Route("/reset", endpoint=reset_endpoint, methods=["POST"]),
    ])
    
    print(f"\nStarting green agent on {host}:{port}...")
    print(f"\nAgentBeats-compatible endpoints:")
    print(f"  GET  http://{host}:{port}/           → Agent card")
    print(f"  GET  http://{host}:{port}/agent_card → Agent card")
    print(f"  GET  http://{host}:{port}/healthz    → Health check")
    print(f"  POST http://{host}:{port}/reset      → Reset state")
    print(f"\nA2A Protocol endpoints (check logs above for actual paths):")
    print(f"  Likely: POST http://{host}:{port}/send_message")
    print(f"  Or: POST http://{host}:{port}/a2a/send_message")
    print(f"\nAgent ready for AgentBeats platform! ✅")
    print(f"\n⚠️  Check the 'A2A framework routes' printed above for exact paths!")
    uvicorn.run(starlette_app, host=host, port=port)


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Adjust these paths to the dataset location
    products_csv = os.path.join(script_dir, "dataset", "ic_products.csv")
    orders_csv = os.path.join(script_dir, "dataset", "super_shortened_orders_products_combined.csv")
    
    start_green_agent(
        products_csv=products_csv,
        orders_csv=orders_csv
    )