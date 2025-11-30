import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ab_src"))
from agentbeats_src.my_util.my_a2a import send_message

async def test():
    print("Sending request to green agent...")
    
    # Don't pass task_id for message-only agents
    response = await send_message(
        "http://localhost:9001",
        json.dumps({"user_id": 1, "use_baseline": True})
        # No task_id parameter!
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
