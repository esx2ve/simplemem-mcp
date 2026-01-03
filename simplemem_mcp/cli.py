"""CLI for SimpleMem MCP.

Commands:
    serve   - Run MCP server (stdio mode for Claude Code)
    signup  - Get an API key for SimpleMem Cloud
    config  - Show current configuration
"""

import os
import webbrowser

import typer
from rich.console import Console

from simplemem_mcp import DEFAULT_BACKEND_URL

app = typer.Typer(
    name="simplemem-mcp",
    help="Memory for Claude Code - Lightweight MCP server",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve():
    """Run MCP server (stdio mode for Claude Code)."""
    from simplemem_mcp.server import main as server_main
    server_main()


@app.command()
def signup():
    """Get an API key for SimpleMem Cloud."""
    signup_url = "https://simplemem.io/signup"

    console.print("[bold blue]SimpleMem Cloud Signup[/bold blue]")
    console.print()
    console.print(f"Opening: {signup_url}")
    console.print()

    try:
        webbrowser.open(signup_url)
        console.print("[green]Browser opened![/green]")
    except Exception:
        console.print(f"[yellow]Could not open browser. Visit:[/yellow] {signup_url}")

    console.print()
    console.print("[bold]After signup, configure Claude Code:[/bold]")
    console.print()
    console.print('  claude mcp add simplemem --command "uvx simplemem-mcp serve" \\')
    console.print('    --env SIMPLEMEM_API_KEY=<your-key>')


@app.command()
def config():
    """Show current configuration."""
    backend_url = os.environ.get("SIMPLEMEM_BACKEND_URL", DEFAULT_BACKEND_URL)
    api_key = os.environ.get("SIMPLEMEM_API_KEY")

    console.print("[bold blue]SimpleMem Configuration[/bold blue]")
    console.print()
    console.print(f"[bold]Backend URL:[/bold] {backend_url}")

    if api_key:
        masked = "***" + api_key[-4:] if len(api_key) > 4 else "****"
        console.print(f"[bold]API Key:[/bold] {masked}")
    else:
        console.print("[bold]API Key:[/bold] [yellow]not set[/yellow]")

    console.print()

    # Check connection
    console.print("[dim]Checking backend connection...[/dim]")
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            headers = {}
            if api_key:
                headers["X-API-Key"] = api_key
            resp = client.get(f"{backend_url}/health", headers=headers)
            if resp.status_code == 200:
                console.print("[green]Backend: Connected[/green]")
            else:
                console.print(f"[yellow]Backend: HTTP {resp.status_code}[/yellow]")
    except Exception as e:
        console.print(f"[red]Backend: Connection failed ({e})[/red]")


@app.command()
def version():
    """Show version information."""
    from simplemem_mcp import __version__
    console.print(f"simplemem-mcp version {__version__}")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
