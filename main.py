print("Launching SPI-Tool dashboard...")

from spi_tool.cli import cli
from spi_tool.ui import create_app

create_app().servable()

if __name__ == "__main__":
    with cli.make_context("cli", ["dashboard", "--show"]) as ctx:
        cli.invoke(ctx)
