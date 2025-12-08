import asyncio
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.my_a2a import send_message

async def test():
    """
    Quick test with 100 users (for full 100-user test, use test_my_agent.py)
    """
    print("🚀 Quick test: 5-user benchmark (for speed)")
    print("   For full 100-user test, use: python white_agent/test_my_agent.py")
    print()
    
    # Quick benchmark with only 5 users for testing
    response = await send_message(
        "http://localhost:9001",
        json.dumps({
            "mode": "benchmark",
            "num_users": 50,  # Changed from 100 to 5 for quick testing
            "white_agent_url": "http://localhost:9002/",
            "environment_base": "https://green-agent-production.up.railway.app",
            "use_baseline": False,
            "random_state": 42,
            "min_order_size": 10  # Explicit default
        }),
        timeout=1800.0 
    )
    
    print(f"\nResponse type: {type(response)}")
    
    if hasattr(response, 'root'):
        from a2a.types import SendMessageSuccessResponse, JSONRPCErrorResponse
        
        if isinstance(response.root, JSONRPCErrorResponse):
            print(f"❌ Error: {response.root.error.message}")
            return
        
        if isinstance(response.root, SendMessageSuccessResponse):
            result = response.root.result
            
            from a2a.utils import get_text_parts
            text_parts = get_text_parts(result.parts)
            
            if text_parts:
                print("\n" + "="*60)
                print("✅ ASSESSMENT RESULT")
                print("="*60)
                print(text_parts[0])
                print("="*60)
            else:
                print("No text parts in response")
                print(f"Parts: {result.parts}")
        else:
            print(f"Unexpected response type: {type(response.root)}")

try:
    asyncio.run(test())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
