import asyncio
import json
import sys
from pathlib import Path

# Add the green_agent directory to sys.path so we can import utils
sys.path.append(str(Path(__file__).parent.parent / "green_agent"))

from utils.my_a2a import send_message

async def main():
    print("🚀 Full benchmark test for White Agent (OpenAI)...")
    print("   This will take 15-30 minutes depending on API speed")
    print()
    
    # Payload to tell Green Agent to test our local White Agent
    payload = {
        "mode": "benchmark",
        "num_users": 100,  # Full 100-user benchmark
        "white_agent_url": "http://localhost:9002/",
        "environment_base": "https://green-agent-production.up.railway.app",
        "use_baseline": False,
        "random_state": 42,  # For reproducible sampling
        "min_order_size": 10  # Explicit default
    }

    print(f"Configuration: {json.dumps(payload, indent=2)}")
    print("\n⏳ Starting benchmark... (this will take a while)\n")

    try:
        response = await send_message(
            "http://localhost:9001", 
            json.dumps(payload),
            timeout=3600.0  # 60 minutes for 100-user benchmark
        )
        
        # Check if response has 'root' attribute (Pydantic model from a2a)
        if hasattr(response, 'root'):
            result = response.root.result
            from a2a.utils import get_text_parts
            text_parts = get_text_parts(result.parts)
            print("\n✅ ASSESSMENT RESULT:")
            for part in text_parts:
                print(part)
        else:
            print(f"\nResponse: {response}")

    except Exception as e:
        print(f"❌ Error during communication: {e}")
        print("Make sure both Green Agent (port 9001) and White Agent (port 9002) are running!")

if __name__ == "__main__":
    asyncio.run(main())

