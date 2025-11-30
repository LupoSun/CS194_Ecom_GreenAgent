import asyncio
import sys
import json
from pathlib import Path

# Make sure the 'ab_src' library is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from agentbeats_src.my_util.my_a2a import send_message
except ImportError:
    print("Error: Could not import 'ab_src'.")
    print("Make sure this script is in the same directory as 'ab_src'")
    print("or adjust the sys.path.insert() call.")
    sys.exit(1)
    
from a2a.types import SendMessageSuccessResponse, JSONRPCErrorResponse
from a2a.utils import get_text_parts

async def run_benchmark_test(num_users=100, random_state=42):
    print("="*70)
    print("TESTING GREEN AGENT (Correctly, via my_a2a library)")
    print("="*70)
    
    # Task configuration
    task_config = {
        "mode": "benchmark",
        "num_users": num_users,
        "use_baseline": True,
        "random_state": random_state
    }
    
    print(f"Sending benchmark request to http://localhost:9001")
    print(f"Config: {json.dumps(task_config, indent=2)}\n")
    
    try:
        # Pass the task config JSON as the text payload
        # The send_message function builds the full A2A request
        response = await send_message(
            "http://localhost:9001",
            json.dumps(task_config)
        )
        
        print(f"\nResponse type: {type(response)}")
        
        if not hasattr(response, 'root'):
             print(f"❌ Error: Unexpected response object: {response}")
             return

        if isinstance(response.root, JSONRPCErrorResponse):
            print(f"❌ Error from agent: {response.root.error.message}")
            return
        
        if isinstance(response.root, SendMessageSuccessResponse):
            result = response.root.result
            text_parts = get_text_parts(result.parts)
            
            if text_parts:
                print("\n" + "="*60)
                print("✅ ASSESSMENT RESULT")
                print("="*60)
                print(text_parts[0].strip())
                print("="*60)
            else:
                print("No text parts in response")
        else:
            print(f"Unexpected response type: {type(response.root)}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark_test(num_users=100, random_state=42))
    except KeyboardInterrupt:
        print("\nTest cancelled.")