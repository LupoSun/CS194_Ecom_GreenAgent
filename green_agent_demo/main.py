import os
from typing import Optional

import typer
import uvicorn
from pydantic_settings import BaseSettings

from main_A2A import start_green_agent


app = typer.Typer(help="Ecom Green Agent local runner / utility CLI")


class Settings(BaseSettings):
    """
    Settings for running the green agent locally.

    Values can come from:
      - CLI flags
      - environment variables (HOST, PORT, PRODUCTS_CSV, ORDERS_CSV)
    """
    host: str = "127.0.0.1"
    port: int = 9000

    products_csv: Optional[str] = None
    orders_csv: Optional[str] = None

    class Config:
        env_prefix = ""          # HOST -> host, PORT -> port, etc.
        case_sensitive = False


@app.command()
def run():
    """
    Start the Ecom Green Agent locally on HOST:PORT.
    """
    settings = Settings()

    # If CSV paths aren't provided via env/flags, use the defaults from the repo.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    products_csv = settings.products_csv or os.path.join(script_dir, "dataset", "ic_products.csv")
    orders_csv = settings.orders_csv or os.path.join(
        script_dir, "dataset", "super_shortened_orders_products_combined.csv"
    )

    # start_green_agent can either:
    #   - return an ASGI app (preferred), OR
    #   - call uvicorn.run internally and return None.
    asgi_app = start_green_agent(
        host=settings.host,
        port=settings.port,
        products_csv=products_csv,
        orders_csv=orders_csv,
    )

    # If it returned an ASGI app, run it with uvicorn here.
    if asgi_app is not None:
        uvicorn.run(asgi_app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    app()