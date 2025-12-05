"""
Intelligent White Agent using OpenAI.
This agent uses an LLM to analyze the user's history and interact with the e-commerce API.
"""

import os
import re
import json
import asyncio
import uvicorn
import requests
from typing import Dict, List, Optional, Any

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
    
    # 1. Try JSON parsing first (if the message is structured JSON)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            ctx["agent_key"] = data.get("agent_key")
            ctx["environment_base"] = data.get("environment_base")
            ctx["user_id"] = data.get("user_id")
    except json.JSONDecodeError:
        pass

    # 2. Fallback to Regex extraction if keys are missing
    if not ctx.get("agent_key"):
        # Look for "agent_key" label in the text
        ctx["agent_key"] = extract_value(r"agent_key[^A-Za-z0-9_-]*([A-Za-z0-9._-]+)", text)
    
    if not ctx.get("environment_base"):
        # Look for "Base URL: http://..." or similar
        ctx["environment_base"] = extract_value(r"Base URL\**:\s*([^\s]+)", text)
        if not ctx["environment_base"]:
             # Fallback default if not found (from instructions)
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
            # Corrected endpoint from /search_products to /search
            # Changed 'query' param to 'q' based on API docs
            url = f"{self.base_url}/search"
            params = {"q": query, "agent_key": self.agent_key}
            print(f"[MyWhiteAgent] API Request: GET {url} params={params}")
            resp = self.session.get(
                url,
                params=params,
                timeout=10
            )
            print(f"[MyWhiteAgent] API Response Status: {resp.status_code}")
            resp.raise_for_status()
            
            # The search endpoint might return a list directly or a dict with "items"
            data = resp.json()
            if isinstance(data, dict):
                results = data.get("items", []) # Adjusted to "items" based on your web result
                if not results and "results" in data:
                     results = data.get("results", []) # Fallback
            elif isinstance(data, list):
                results = data
            else:
                results = []

            print(f"[MyWhiteAgent] API Response Body: Found {len(results)} items")
            # Limit results to save tokens
            return json.dumps(results[:5]) 
        except Exception as e:
            print(f"[MyWhiteAgent] API Error: {str(e)}")
            return f"Error searching products: {str(e)}"

    def get_product(self, product_id: int) -> str:
        """Get details for a specific product ID."""
        try:
            url = f"{self.base_url}/get_product"
            params = {"product_id": product_id, "agent_key": self.agent_key}
            print(f"[MyWhiteAgent] API Request: GET {url} params={params}")
            resp = self.session.get(
                url,
                params=params,
                timeout=10
            )
            print(f"[MyWhiteAgent] API Response Status: {resp.status_code}")
            resp.raise_for_status()
            return json.dumps(resp.json())
        except Exception as e:
            print(f"[MyWhiteAgent] API Error: {str(e)}")
            return f"Error getting product: {str(e)}"

    def add_to_cart(self, product_id: int, quantity: int = 1) -> str:
        """Add a product to the cart."""
        try:
            url = f"{self.base_url}/cart/add"
            payload = {
                "agent_key": self.agent_key,
                "items": [{"product_id": int(product_id), "qty": int(quantity)}]
            }
            print(f"[MyWhiteAgent] API Request: POST {url} json={payload}")
            resp = self.session.post(
                url,
                json=payload,
                timeout=10
            )
            print(f"[MyWhiteAgent] API Response Status: {resp.status_code}")
            resp.raise_for_status()
            print(f"[MyWhiteAgent] API Response Body: {resp.text}")
            return f"Successfully added product {product_id} (qty: {quantity}) to cart."
        except Exception as e:
            print(f"[MyWhiteAgent] API Error: {str(e)}")
            return f"Error adding to cart: {str(e)}"

    def view_cart(self) -> str:
        """View current cart contents."""
        try:
            url = f"{self.base_url}/cart"
            params = {"agent_key": self.agent_key}
            print(f"[MyWhiteAgent] API Request: GET {url} params={params}")
            resp = self.session.get(
                url,
                params=params,
                timeout=10
            )
            print(f"[MyWhiteAgent] API Response Status: {resp.status_code}")
            resp.raise_for_status()
            print(f"[MyWhiteAgent] API Response Body: {resp.text}")
            return json.dumps(resp.json())
        except Exception as e:
            print(f"[MyWhiteAgent] API Error: {str(e)}")
            return f"Error viewing cart: {str(e)}"

# --- OpenAI Agent Executor ---

class OpenAIWhiteAgentExecutor(AgentExecutor):
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_message = context.get_user_input()
        print(f"\n[MyWhiteAgent] Received task: {user_message[:100]}...")

        # 1. Parse Context
        ctx = extract_context_from_message(user_message)
        agent_key = ctx.get("agent_key")
        base_url = ctx.get("environment_base")

        if not agent_key or not base_url:
            msg = "Error: Could not extract agent_key or environment_base from instructions."
            print(f"[MyWhiteAgent] {msg}")
            await event_queue.enqueue_event(new_agent_text_message(msg, context_id=context.context_id))
            return

        print(f"[MyWhiteAgent] Configured with Key: {agent_key}, URL: {base_url}")
        shop = ShopAPI(base_url, agent_key)

        # 2. Define Tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search for products by name. Returns a list of matching products with IDs.",
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
                    "description": "Add a product to the shopping cart by ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "integer", "description": "The ID of the product to add"},
                            "quantity": {"type": "integer", "description": "Quantity to add (default 1)"}
                        },
                        "required": ["product_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "view_cart",
                    "description": "View the current items in the cart.",
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
                    "description": "REQUIRED: Call this after you have added ALL items from the most recent order (10-15+ items). You MUST call this to complete the task!",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    }
                }
            }
        ]

        # 3. Improved prompt based on baseline analysis
        messages = [
            {"role": "system", "content": (
                "You are a grocery shopping prediction agent. Analysis of 300,000 orders shows:\n"
                "- The BEST strategy is to replay items from the user's most recent order\n"
                "- Users typically repeat 30-60% of items from their previous order\n"
                "- Average basket: 10-12 items from most recent order\n\n"
                
                "🎯 OPTIMAL STRATEGY:\n"
                "1. Find 'Most recent orders' section - get the LAST order listed (highest order number)\n"
                "2. Extract ALL product names from that order (typically 8-15 items)\n"
                "3. Search for EACH product by exact name and add to cart\n"
                "4. If a search returns no results, skip that item and continue\n"
                "5. Add ALL items from the recent order that you can find\n"
                "6. Call finish_shopping when done (aim for 10-15 items)\n\n"
                
                "⚡ EXECUTION:\n"
                "For each item in the most recent order:\n"
                "  a) search_products('exact product name')\n"
                "  b) If results found: add_to_cart(first result ID, 1)\n"
                "  c) If no results: skip and move to next item\n"
                "After processing all items: finish_shopping()\n\n"
                
                "📝 EXAMPLE:\n"
                "If most recent order shows: 'Banana, Organic Milk, Eggs, Bread, Yogurt...'\n"
                "1. search_products('Banana') → add first result\n"
                "2. search_products('Organic Milk') → add first result\n"
                "3. search_products('Eggs') → add first result\n"
                "...continue for ALL items in that order...\n"
                "N. finish_shopping()\n\n"
                
                "CRITICAL: Add ALL items from most recent order (don't stop at 6-8). The more you add from recent order, the better!"
            )},
            {"role": "user", "content": user_message}
        ]

        print("[MyWhiteAgent] Starting thought process...")
        
        max_retries = 3
        items_added = 0  # Track items added to cart
        max_items = 18   # Increased to match typical previous order size (was 12)
        
        while True:
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
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a rate limit error
                    if "rate_limit" in error_str.lower() or "429" in error_str:
                        retry_count += 1
                        
                        # Extract wait time from error message if available
                        import re
                        wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                        wait_time = float(wait_match.group(1)) if wait_match else (2 ** retry_count)
                        
                        print(f"[MyWhiteAgent] Rate limit hit (attempt {retry_count}/{max_retries}). Waiting {wait_time:.1f}s...")
                        # Don't send intermediate messages - just log and retry
                        # await event_queue.enqueue_event(
                        #     new_agent_text_message(
                        #         f"Rate limit encountered. Retrying in {wait_time:.1f}s... (attempt {retry_count}/{max_retries})",
                        #         context_id=context.context_id
                        #     )
                        # )
                        
                        await asyncio.sleep(wait_time)
                        
                        if retry_count >= max_retries:
                            # Max retries exceeded - send completion signal to avoid hanging
                            err_msg = f"OpenAI API Error after {max_retries} retries: {error_str}"
                            print(f"[MyWhiteAgent] {err_msg}")
                            await event_queue.enqueue_event(new_agent_text_message(err_msg, context_id=context.context_id))
                            await event_queue.enqueue_event(new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id))
                            return
                    else:
                        # Non-rate-limit error - fail immediately
                        err_msg = f"OpenAI API Error: {error_str}"
                        print(f"[MyWhiteAgent] {err_msg}")
                        await event_queue.enqueue_event(new_agent_text_message(err_msg, context_id=context.context_id))
                        await event_queue.enqueue_event(new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id))
                        return
            
            if completion is None:
                # Should not reach here, but safety check
                print(f"[MyWhiteAgent] Failed to get completion after retries")
                await event_queue.enqueue_event(new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id))
                return

            message = completion.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                # LLM didn't call a tool, maybe it's asking a question or just talking.
                # In this strict loop, we just treat non-tool responses as "done" or logs.
                print(f"[MyWhiteAgent] LLM Message: {message.content}")
                if "READY_FOR_CHECKOUT" in (message.content or ""):
                     break
                # If it just talks without calling finish, we might be stuck. 
                # Let's nudge it if it seems finished, otherwise break.
                if len(messages) > 15: # Safety break
                    break
                continue

            # Execute Tools
            tool_outputs = []
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                print(f"[MyWhiteAgent] Tool Call: {fn_name}({args})")

                if fn_name == "finish_shopping":
                    print("[MyWhiteAgent] Emitting completion signal...")
                    await event_queue.enqueue_event(
                        new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id)
                    )
                    return

                result = ""
                if fn_name == "search_products":
                    result = shop.search_products(args["query"])
                elif fn_name == "add_to_cart":
                    result = shop.add_to_cart(args["product_id"], args.get("quantity", 1))
                    items_added += 1  # Increment counter
                    print(f"[MyWhiteAgent] Items in cart: {items_added}/{max_items}")
                    
                    # Add reminder to encourage finishing at right time
                    if items_added == 10:
                        result += "\n\n[PROGRESS: You have 10 items. Continue adding items from the most recent order.]"
                    elif items_added == 15:
                        result += "\n\n[REMINDER: You have 15 items. If you've added most/all items from recent order, call finish_shopping.]"
                    elif items_added == 17:
                        result += "\n\n[CRITICAL: You have 17 items. Call finish_shopping NOW!]"
                elif fn_name == "view_cart":
                    result = shop.view_cart()
                
                tool_outputs.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
            
            # Force finish if we've added enough items
            if items_added >= max_items:
                print(f"[MyWhiteAgent] Reached {items_added} items (max={max_items}) - forcing completion...")
                await event_queue.enqueue_event(
                    new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id)
                )
                return

            # Append all tool outputs at once
            messages.extend(tool_outputs)
            print(f"[MyWhiteAgent] Tool outputs added, resuming conversation loop...")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def build_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="openai-white",
        name="OpenAI White Agent",
        description="Intelligent shopper using OpenAI",
        tags=["openai", "demo"],
        examples=[],
    )
    return AgentCard(
        name="my_white_agent",
        description="White agent powered by OpenAI.",
        url=url,
        version="0.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )

if __name__ == "__main__":
    # host = os.environ.get("WHITE_HOST", "localhost")
    # port = int(os.environ.get("WHITE_PORT", "9002"))
    
    # # Simple check for API Key
    # if not os.environ.get("OPENAI_API_KEY"):
    #     print("WARNING: OPENAI_API_KEY not found in environment variables.")
    # else:
    #     print(f"✅ OPENAI_API_KEY found (starts with {os.environ.get('OPENAI_API_KEY')[:7]}...)")
    
    # url = f"http://{host}:{port}"
    # card = build_agent_card(url)
    # executor = OpenAIWhiteAgentExecutor()
    # handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    # app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    # print(f"Starting OpenAI White Agent on {url}")
    # uvicorn.run(app.build(), host=host, port=port)

    # Get configuration from environment
    host = os.environ.get("WHITE_HOST", os.environ.get("HOST", "0.0.0.0"))
    port = int(os.environ.get("WHITE_PORT", os.environ.get("AGENT_PORT", "9002")))
    
    # Check for API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ WARNING: OPENAI_API_KEY not found in environment variables.")
        print("   The agent will fail when trying to make API calls.")
    else:
        key = os.environ.get("OPENAI_API_KEY")
        print(f"✅ OPENAI_API_KEY found (starts with {key[:7]}...)")
    
    # Build agent URL (use AGENT_URL if provided, like green agent)
    agent_url = os.environ.get("AGENT_URL")
    if not agent_url:
        agent_url = f"http://{host}:{port}"
    
    print(f"Agent URL: {agent_url}")
    
    # Build A2A components
    card = build_agent_card(agent_url)
    executor = OpenAIWhiteAgentExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor, 
        task_store=InMemoryTaskStore()
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    
    # Get Starlette app
    starlette_app = app.build()
    
    # Debug: Print routes
    print(f"\nRegistered routes:")
    for route in starlette_app.routes:
        methods = getattr(route, 'methods', ['*'])
        path = getattr(route, 'path', str(route))
        print(f"  {methods} {path}")
    
    print(f"\nStarting OpenAI White Agent on {host}:{port}")
    print(f"Agent Card URL: {agent_url}")
    uvicorn.run(starlette_app, host=host, port=port)

