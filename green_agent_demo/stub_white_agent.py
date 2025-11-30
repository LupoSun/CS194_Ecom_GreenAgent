"""
Minimal stub white agent for A2A demos.
Returns the completion signal immediately so the green agent can finish evaluation.
"""

import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

COMPLETION_SIGNAL = "##READY_FOR_CHECKOUT##"


class StubWhiteAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Immediately tell the green agent we are done
        await event_queue.enqueue_event(
            new_agent_text_message(COMPLETION_SIGNAL, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_agent_text_message("Cancelled.", context_id=context.context_id)
        )


def build_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="stub-white",
        name="Stub White Agent",
        description="Returns the completion signal immediately for testing.",
        tags=["stub", "demo"],
        examples=[],
    )
    return AgentCard(
        name="stub_white_agent",
        description="Disposable white agent that signals readiness immediately.",
        url=url,
        version="0.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


def start_stub_white_agent(host: str = "0.0.0.0", port: int = 9002) -> None:
    url = f"http://{host}:{port}"
    card = build_agent_card(url)
    executor = StubWhiteAgentExecutor()
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print(f"Starting stub white agent on {url}")
    print(f"Completion signal: {COMPLETION_SIGNAL}")
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    host = os.environ.get("WHITE_HOST", "0.0.0.0")
    port = int(os.environ.get("WHITE_PORT", "9002"))
    start_stub_white_agent(host=host, port=port)
