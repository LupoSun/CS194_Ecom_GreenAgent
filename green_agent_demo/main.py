import uvicorn
import typer
from pydantic_settings import BaseSettings

from main_A2A import start_green_agent


app = typer.Typer(help="Ecom Green Agent entrypoint for AgentBeats controller")


class Settings(BaseSettings):
    # These will be populated by the AgentBeats controller via env vars
    host: str = "127.0.0.1"
    agent_port: int = 9001

    # Optional dataset override via env
    products_csv: str | None = None
    orders_csv: str | None = None

    class Config:
        # Ensure env var names map nicely:
        # HOST -> host, AGENT_PORT -> agent_port, PRODUCTS_CSV / ORDERS_CSV
        env_prefix = ""
        case_sensitive = False


@app.command()
def run():
    """Start the Ecom Green Agent on HOST:AGENT_PORT."""
    settings = Settings()

    asgi_app = start_green_agent(
        host=settings.host,
        port=settings.agent_port,
        products_csv=settings.products_csv,
        orders_csv=settings.orders_csv,
    )

    uvicorn.run(
        asgi_app,
        host=settings.host,
        port=settings.agent_port,
    )


if __name__ == "__main__":
    app()