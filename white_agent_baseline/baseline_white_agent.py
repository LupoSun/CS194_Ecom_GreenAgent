"""
Baseline White Agent using OpenAI with MINIMAL prompting.
This is a baseline to establish performance with no strategic guidance.
"""

import os
import re
import json
import asyncio
import uvicorn
import requests
from typing import Dict, Optional, Any

from openai import OpenAI
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

COMPLETION_SIGNAL = "##READY_FOR_CHECKOUT##"

# --- Helper Functions for Parsing Context ---

def extract_value(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None

def extract_context_from_message(text: str) -> Dict[str, Any]:
    """
    Parses the incoming message from the green agent to find:
    - agent_key (required for API calls)
    - environment_base (URL of the shop)
    - user_id (context)
    """
    ctx = {}
    
    # Try JSON parsing first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            ctx["agent_key"] = data.get("agent_key")
            ctx["environment_base"] = data.get("environment_base")
            ctx["user_id"] = data.get("user_id")
    except json.JSONDecodeError:
        pass

    # Fallback to Regex extraction
    if not ctx.get("agent_key"):
        ctx["agent_key"] = extract_value(r"agent_key[^A-Za-z0-9_-]*([A-Za-z0-9._-]+)", text)
    
    if not ctx.get("environment_base"):
        ctx["environment_base"] = extract_value(r"Base URL\**:\s*([^\s]+)", text)
        if not ctx["environment_base"]:
            if "green-agent-production.up.railway.app" in text:
                ctx["environment_base"] = "https://green-agent-production.up.railway.app"

    if not ctx.get("user_id"):
        ctx["user_id"] = extract_value(r"user_id\"?\s*[:=]\s*(\d+)", text)
        if not ctx["user_id"]:
            ctx["user_id"] = extract_value(r"user\s+(\d+)", text)
            
    return ctx

# --- E-commerce API Client ---

class ShopAPI:
    def __init__(self, base_url: str, agent_key: str):
        self.base_url = base_url.rstrip("/")
        self.agent_key = agent_key
        self.session = requests.Session()

    def search_products(self, query: str) -> str:
        """Search for products by name."""
        try:
            url = f"{self.base_url}/search"
            params = {"q": query, "agent_key": self.agent_key}
            print(f"[BaselineWhiteAgent] API Request: GET {url} params={params}")
            resp = self.session.get(url, params=params, timeout=10)
            print(f"[BaselineWhiteAgent] API Response Status: {resp.status_code}")
            resp.raise_for_status()
            
            data = resp.json()
            if isinstance(data, dict):
                results = data.get("items", [])
                if not results and "results" in data:
                    results = data.get("results", [])
            elif isinstance(data, list):
                results = data
            else:
                results = []

            print(f"[BaselineWhiteAgent] API Response: Found {len(results)} items")
            return json.dumps(results[:5])
        except Exception as e:
            print(f"[BaselineWhiteAgent] API Error: {str(e)}")
            return f"Error searching products: {str(e)}"

    def add_to_cart(self, product_id: int, quantity: int = 1) -> str:
        """Add a product to the cart."""
        try:
            url = f"{self.base_url}/cart/add"
            payload = {
                "agent_key": self.agent_key,
                "items": [{"product_id": int(product_id), "qty": int(quantity)}]
            }
            print(f"[BaselineWhiteAgent] API Request: POST {url} json={payload}")
            resp = self.session.post(url, json=payload, timeout=10)
            print(f"[BaselineWhiteAgent] API Response Status: {resp.status_code}")
            resp.raise_for_status()
            print(f"[BaselineWhiteAgent] API Response Body: {resp.text}")
            return f"Successfully added product {product_id} (qty: {quantity}) to cart."
        except Exception as e:
            print(f"[BaselineWhiteAgent] API Error: {str(e)}")
            return f"Error adding to cart: {str(e)}"

    def view_cart(self) -> str:
        """View current cart contents."""
        try:
            url = f"{self.base_url}/cart"
            params = {"agent_key": self.agent_key}
            print(f"[BaselineWhiteAgent] API Request: GET {url} params={params}")
            resp = self.session.get(url, params=params, timeout=10)
            print(f"[BaselineWhiteAgent] API Response Status: {resp.status_code}")
            resp.raise_for_status()
            print(f"[BaselineWhiteAgent] API Response Body: {resp.text}")
            return json.dumps(resp.json())
        except Exception as e:
            print(f"[BaselineWhiteAgent] API Error: {str(e)}")
            return f"Error viewing cart: {str(e)}"

# --- OpenAI Agent Executor ---

class BaselineWhiteAgentExecutor(AgentExecutor):
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_message = context.get_user_input()
        print(f"\n[BaselineWhiteAgent] Received task: {user_message[:100]}...")

        # 1. Parse Context
        ctx = extract_context_from_message(user_message)
        agent_key = ctx.get("agent_key")
        base_url = ctx.get("environment_base")

        if not agent_key or not base_url:
            msg = "Error: Could not extract agent_key or environment_base from instructions."
            print(f"[BaselineWhiteAgent] {msg}")
            await event_queue.enqueue_event(new_agent_text_message(msg, context_id=context.context_id))
            return

        print(f"[BaselineWhiteAgent] Configured with Key: {agent_key}, URL: {base_url}")
        shop = ShopAPI(base_url, agent_key)

        # 2. Define Tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search for products by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Product name or keyword"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_to_cart",
                    "description": "Add a product to the shopping cart.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "integer", "description": "Product ID"},
                            "quantity": {"type": "integer", "description": "Quantity (default 1)"}
                        },
                        "required": ["product_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "view_cart",
                    "description": "View the current cart.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_shopping",
                    "description": "Complete the shopping task.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    }
                }
            }
        ]

        # 3. MINIMAL PROMPT - Just basic instructions
        messages = [
            {
                "role": "system", 
                "content": "You are a shopping assistant. Use the available tools to help with the shopping task."
            },
            {"role": "user", "content": user_message}
        ]

        print("[BaselineWhiteAgent] Starting...")
        
        max_retries = 3
        max_iterations = 50  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call LLM with retry logic
            retry_count = 0
            completion = None
            
            while retry_count < max_retries:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto"
                    )
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    
                    if "rate_limit" in error_str.lower() or "429" in error_str:
                        retry_count += 1
                        wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                        wait_time = float(wait_match.group(1)) if wait_match else (2 ** retry_count)
                        
                        print(f"[BaselineWhiteAgent] Rate limit hit. Waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        
                        if retry_count >= max_retries:
                            err_msg = f"OpenAI API Error after {max_retries} retries: {error_str}"
                            print(f"[BaselineWhiteAgent] {err_msg}")
                            await event_queue.enqueue_event(new_agent_text_message(err_msg, context_id=context.context_id))
                            await event_queue.enqueue_event(new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id))
                            return
                    else:
                        err_msg = f"OpenAI API Error: {error_str}"
                        print(f"[BaselineWhiteAgent] {err_msg}")
                        await event_queue.enqueue_event(new_agent_text_message(err_msg, context_id=context.context_id))
                        await event_queue.enqueue_event(new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id))
                        return
            
            if completion is None:
                print(f"[BaselineWhiteAgent] Failed to get completion")
                await event_queue.enqueue_event(new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id))
                return

            message = completion.choices[0].message
            messages.append(message)

            # If no tool calls, check if done
            if not message.tool_calls:
                print(f"[BaselineWhiteAgent] LLM Message: {message.content}")
                if message.content and "READY_FOR_CHECKOUT" in message.content:
                    break
                # If just talking, continue
                continue

            # Execute Tools
            tool_outputs = []
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                print(f"[BaselineWhiteAgent] Tool Call: {fn_name}({args})")

                if fn_name == "finish_shopping":
                    print("[BaselineWhiteAgent] Task complete.")
                    await event_queue.enqueue_event(
                        new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id)
                    )
                    return

                result = ""
                if fn_name == "search_products":
                    result = shop.search_products(args["query"])
                elif fn_name == "add_to_cart":
                    result = shop.add_to_cart(args["product_id"], args.get("quantity", 1))
                elif fn_name == "view_cart":
                    result = shop.view_cart()
                
                tool_outputs.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
            
            messages.extend(tool_outputs)
        
        # Hit max iterations
        print(f"[BaselineWhiteAgent] Hit max iterations ({max_iterations}), completing...")
        await event_queue.enqueue_event(
            new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def build_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="baseline-white",
        name="Baseline White Agent",
        description="Baseline shopping agent with minimal prompting",
        tags=["baseline", "gpt", "minimal"],
        examples=[],
    )
    return AgentCard(
        name="baseline_white_agent",
        description="Baseline white agent with minimal prompting.",
        url=url,
        version="0.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )

if __name__ == "__main__":
    host = os.environ.get("WHITE_HOST", os.environ.get("HOST", "0.0.0.0"))
    port = int(os.environ.get("WHITE_PORT", os.environ.get("AGENT_PORT", "9002")))
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ WARNING: OPENAI_API_KEY not found in environment variables.")
    else:
        key = os.environ.get("OPENAI_API_KEY")
        print(f"✅ OPENAI_API_KEY found (starts with {key[:7]}...)")
    
    agent_url = os.environ.get("AGENT_URL")
    if not agent_url:
        agent_url = f"http://{host}:{port}"
    
    print(f"Agent URL: {agent_url}")
    
    card = build_agent_card(agent_url)
    executor = BaselineWhiteAgentExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor, 
        task_store=InMemoryTaskStore()
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    
    starlette_app = app.build()
    
    print(f"\nRegistered routes:")
    for route in starlette_app.routes:
        methods = getattr(route, 'methods', ['*'])
        path = getattr(route, 'path', str(route))
        print(f"  {methods} {path}")
    
    print(f"\nStarting Baseline White Agent on {host}:{port}")
    print(f"Agent Card URL: {agent_url}")
    uvicorn.run(starlette_app, host=host, port=port)
