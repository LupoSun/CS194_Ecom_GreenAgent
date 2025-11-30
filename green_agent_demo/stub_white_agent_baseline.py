"""
Baseline-style stub white agent for A2A demos.
It reads the user's last order from the local dataset, adds those items to cart
via the provided environment base URL, then emits the completion signal so the
green agent can checkout and evaluate.
"""

import os
import re
import json
from typing import Dict, List, Optional

import pandas as pd
import requests
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

COMPLETION_SIGNAL = "##READY_FOR_CHECKOUT##"


def load_orders(df_path: str) -> pd.DataFrame:
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Orders CSV not found: {df_path}")
    df = pd.read_csv(df_path)
    # Defensive types
    for col in ("user_id", "order_number", "order_id", "product_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def extract_user_id(text: str) -> Optional[int]:
    # Handles "user 123" or "user_id\": 123"
    for pattern in [r"user_id\"?\s*[:=]\s*(\d+)", r"user\s+(\d+)"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def extract_agent_key(text: str) -> Optional[str]:
    # Looks for "agent_key" label in the tool list
    m = re.search(r"agent_key[^A-Za-z0-9_-]*([A-Za-z0-9._-]+)", text)
    return m.group(1) if m else None


def extract_env_base(text: str) -> Optional[str]:
    # Looks for "Base URL: http://..."
    m = re.search(r"Base URL\**:\s*([^\s]+)", text, re.IGNORECASE)
    return m.group(1) if m else None


def last_order_items(df_orders: pd.DataFrame, user_id: int) -> Dict[int, int]:
    u = df_orders.loc[df_orders["user_id"] == user_id].copy()
    if u.empty or "order_number" not in u.columns:
        return {}
    n = int(u["order_number"].max())
    prev = u[u["order_number"] == (n - 1)] if n > 1 else u.iloc[0:0]
    basket: Dict[int, int] = {}
    for pid in prev.get("product_id", pd.Series(dtype=float)).dropna():
        pid_int = int(pid)
        basket[pid_int] = basket.get(pid_int, 0) + 1
    return basket


def add_items_to_cart(base_url: str, agent_key: str, basket: Dict[int, int]) -> None:
    """
    Push last-order items to the live API.
    Expected schema (per /cart/add docs):
    {
      "agent_key": "...",
      "items": [{"product_id": <int>, "qty": <int>}, ...]
    }
    """
    add_url = base_url.rstrip("/") + "/cart/add"
    payload = {
        "agent_key": agent_key,
        "items": [{"product_id": pid, "qty": qty} for pid, qty in basket.items()],
    }
    try:
        resp = requests.post(add_url, json=payload, timeout=15)
        resp.raise_for_status()
        print(f"[Stub White] Added {len(payload['items'])} items via {add_url} -> {resp.status_code}")
    except Exception as e:
        print(f"[Stub White] Failed to add items via {add_url}: {e}")


class BaselineWhiteAgentExecutor(AgentExecutor):
    def __init__(self, df_orders: pd.DataFrame, default_env_base: str):
        self.df_orders = df_orders
        self.default_env_base = default_env_base

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        user_id = extract_user_id(text)
        agent_key = extract_agent_key(text)
        env_base = extract_env_base(text) or self.default_env_base

        if not user_id or not agent_key or not env_base:
            msg = (
                "Missing user_id/agent_key/env_base in task; "
                "cannot add items. Sending completion anyway."
            )
            print(f"[Stub White] {msg}")
            await event_queue.enqueue_event(new_agent_text_message(msg, context_id=context.context_id))
            await event_queue.enqueue_event(new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id))
            return

        basket = last_order_items(self.df_orders, user_id)
        print(f"[Stub White] User {user_id} last-order basket: {basket}")

        if basket:
            add_items_to_cart(env_base, agent_key, basket)
        else:
            print(f"[Stub White] No previous order items found for user {user_id}")

        await event_queue.enqueue_event(
            new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(new_agent_text_message("Cancelled.", context_id=context.context_id))


def build_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="stub-white-baseline",
        name="Stub White Agent (Baseline Cart)",
        description="Adds last-order items to cart then signals completion.",
        tags=["stub", "baseline", "demo"],
        examples=[],
    )
    return AgentCard(
        name="stub_white_agent_baseline",
        description="Disposable white agent that replays last order to cart.",
        url=url,
        version="0.0.2",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


def start_stub_white_agent_baseline(
    host: str = "0.0.0.0",
    port: int = 9002,
    orders_csv: str = "dataset/super_shortened_orders_products_combined.csv",
    env_base_default: str = "",
) -> None:
    df_orders = load_orders(orders_csv)
    env_base = env_base_default or os.environ.get("ECOM_BASE", "http://localhost:8001")
    executor = BaselineWhiteAgentExecutor(df_orders, env_base)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    card = build_agent_card(f"http://{host}:{port}")
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print(f"Starting stub white agent (baseline) on http://{host}:{port}")
    print(f"Using orders: {orders_csv}")
    print(f"Default env base: {env_base}")
    print(f"Completion signal: {COMPLETION_SIGNAL}")
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    host = os.environ.get("WHITE_HOST", "0.0.0.0")
    port = int(os.environ.get("WHITE_PORT", "9002"))
    orders_csv = os.environ.get(
        "ORDERS_CSV",
        os.path.join(os.path.dirname(__file__), "dataset", "super_shortened_orders_products_combined.csv"),
    )
    env_base = os.environ.get("ECOM_BASE", "http://localhost:8001")
    start_stub_white_agent_baseline(
        host=host,
        port=port,
        orders_csv=orders_csv,
        env_base_default=env_base,
    )
