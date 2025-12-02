import asyncio
import json
import sys
from pathlib import Path

# Add the current directory to sys.path so we can import utils
sys.path.append(str(Path(__file__).parent))

from utils.my_a2a import send_message

async def main():
    print("🚀 Requesting assessment for My White Agent (OpenAI)...")
    
    # Payload to tell Green Agent to test our local White Agent
    payload = {
        "mode": "benchmark",           # <--- Changed from "white_agent"
        "num_users": 15,              
        "white_agent_url": "http://localhost:9002/",
        "environment_base": "https://green-agent-production.up.railway.app",
        "use_baseline": False,
        "random_state": 42             # Optional: for reproducible sampling
    }

    print(f"Configuration: {json.dumps(payload, indent=2)}")

    try:
        response = await send_message(
            "http://localhost:9001", 
            json.dumps(payload)
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

