"""CLI for SimpleMem MCP.

Commands:
    serve   - Run MCP server (stdio mode for Claude Code)
    setup   - Interactive setup wizard for Claude Code integration
    doctor  - Diagnose connection and configuration issues
    signup  - Get an API key for SimpleMem Cloud
    config  - Show current configuration
"""

import json
import os
import shutil
import webbrowser
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from simplemem_mcp import DEFAULT_BACKEND_URL

app = typer.Typer(
    name="simplemem-mcp",
    help="Memory for Claude Code - Lightweight MCP server",
    no_args_is_help=True,
)
console = Console()

# Claude Code config path
CLAUDE_CONFIG_PATH = Path.home() / ".claude.json"


def _get_claude_config() -> dict:
    """Load Claude Code config, return empty dict if not found."""
    if CLAUDE_CONFIG_PATH.exists():
        try:
            return json.loads(CLAUDE_CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _backup_claude_config() -> Path | None:
    """Create timestamped backup of Claude config."""
    if not CLAUDE_CONFIG_PATH.exists():
        return None

    backup_dir = Path.home() / ".claude_backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"claude_{timestamp}.json"
    shutil.copy2(CLAUDE_CONFIG_PATH, backup_path)

    return backup_path


def _verify_backend(url: str, api_key: str | None = None) -> tuple[bool, str]:
    """Verify backend connectivity.

    Returns:
        (success, message)
    """
    import httpx

    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {}
            if api_key:
                headers["X-API-Key"] = api_key

            # Check health endpoint
            resp = client.get(f"{url}/health", headers=headers)
            if resp.status_code != 200:
                return False, f"Health check failed: HTTP {resp.status_code}"

            # For cloud backend, verify auth
            if api_key:
                resp = client.get(f"{url}/api/v1/memories/stats", headers=headers)
                if resp.status_code == 401:
                    return False, "Invalid API key"
                if resp.status_code == 403:
                    return False, "API key does not have access"

            return True, "Connection successful"
    except httpx.ConnectError:
        return False, f"Cannot connect to {url}"
    except httpx.TimeoutException:
        return False, f"Connection to {url} timed out"
    except Exception as e:
        return False, f"Connection error: {e}"


def _generate_mcp_config(backend_url: str, api_key: str | None = None) -> dict:
    """Generate MCP server configuration."""
    config = {
        "type": "stdio",
        "command": "uvx",
        "args": ["simplemem-mcp", "serve"],
        "env": {
            "SIMPLEMEM_BACKEND_URL": backend_url,
        }
    }

    if api_key:
        config["env"]["SIMPLEMEM_API_KEY"] = api_key

    return config


@app.command()
def serve():
    """Run MCP server (stdio mode for Claude Code)."""
    from simplemem_mcp.server import main as server_main
    server_main()


@app.command()
def setup(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
    cloud: bool = typer.Option(None, "--cloud/--local", help="Choose cloud or local backend"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key for cloud backend"),
    backend_url: str = typer.Option(None, "--url", help="Custom backend URL"),
):
    """Interactive setup wizard for Claude Code integration.

    Configures SimpleMem as an MCP server in Claude Code with the
    optimal settings for your environment.
    """
    console.print()
    console.print(Panel.fit(
        "[bold blue]SimpleMem Setup Wizard[/bold blue]\n\n"
        "This wizard will configure SimpleMem as a memory provider\n"
        "for Claude Code.",
        border_style="blue"
    ))
    console.print()

    # Step 1: Determine backend type
    if cloud is None:
        console.print("[bold]Step 1:[/bold] Choose your backend\n")
        console.print("  [cyan]Cloud[/cyan] - SimpleMem Cloud (recommended)")
        console.print("    • No server to run")
        console.print("    • Memories persist across machines")
        console.print("    • Requires API key")
        console.print()
        console.print("  [cyan]Self-Hosted[/cyan] - Run your own backend")
        console.print("    • Full control over your data")
        console.print("    • Requires running simplemem-lite server")
        console.print()

        choice = Prompt.ask(
            "Select backend",
            choices=["cloud", "self-hosted"],
            default="cloud"
        )
        use_cloud = choice == "cloud"
    else:
        use_cloud = cloud

    console.print()

    # Step 2: Configure backend URL and credentials
    if use_cloud:
        console.print("[bold]Step 2:[/bold] Configure cloud access\n")

        final_url = backend_url or DEFAULT_BACKEND_URL

        if not api_key:
            # Check env var
            api_key = os.environ.get("SIMPLEMEM_API_KEY")

            if not api_key:
                console.print("You'll need an API key for SimpleMem Cloud.")
                console.print(f"Get one at: [link=https://simplemem.io/signup]https://simplemem.io/signup[/link]")
                console.print()
                api_key = Prompt.ask("Enter your API key")

        # Verify connection
        console.print("\n[dim]Verifying connection...[/dim]")
        success, message = _verify_backend(final_url, api_key)

        if success:
            console.print(f"[green]✓[/green] {message}")
        else:
            console.print(f"[red]✗[/red] {message}")
            if not Confirm.ask("\nContinue anyway?", default=False):
                raise typer.Exit(1)
    else:
        console.print("[bold]Step 2:[/bold] Configure self-hosted backend\n")

        final_url = backend_url or Prompt.ask(
            "Backend URL",
            default="http://localhost:8420"
        )
        api_key = None  # Self-hosted typically doesn't need auth in dev mode

        # Verify connection
        console.print("\n[dim]Verifying connection...[/dim]")
        success, message = _verify_backend(final_url)

        if success:
            console.print(f"[green]✓[/green] {message}")
        else:
            console.print(f"[yellow]![/yellow] {message}")
            console.print()
            console.print("[dim]To start a self-hosted backend:[/dim]")
            console.print("  pip install simplemem-lite")
            console.print("  SIMPLEMEM_MODE=dev simplemem-lite serve")
            console.print()
            if not Confirm.ask("Continue anyway?", default=True):
                raise typer.Exit(1)

    console.print()

    # Step 3: Generate and apply config
    console.print("[bold]Step 3:[/bold] Configure Claude Code\n")

    mcp_config = _generate_mcp_config(final_url, api_key)

    # Load existing config
    existing_config = _get_claude_config()

    # Check for existing simplemem entry
    mcp_servers = existing_config.get("mcpServers", {})
    has_existing = "simplemem" in mcp_servers or "simplemem-lite" in mcp_servers

    if has_existing:
        console.print("[yellow]![/yellow] Found existing SimpleMem configuration")
        old_config = mcp_servers.get("simplemem") or mcp_servers.get("simplemem-lite")

        # Show diff (mask sensitive values)
        def mask_config(cfg: dict) -> dict:
            """Mask sensitive values in config for display."""
            result = cfg.copy()
            if "env" in result:
                masked_env = {}
                for k, v in result["env"].items():
                    if any(s in k.upper() for s in ["KEY", "SECRET", "TOKEN", "PASSWORD"]):
                        masked_env[k] = "***" + v[-4:] if len(v) > 4 else "****"
                    else:
                        masked_env[k] = v
                result["env"] = masked_env
            return result

        console.print("\n[dim]Current:[/dim]")
        console.print(f"  {json.dumps(mask_config(old_config), indent=2)}")
        console.print("\n[dim]New:[/dim]")
        console.print(f"  {json.dumps(mask_config(mcp_config), indent=2)}")
        console.print()

        if not Confirm.ask("Replace existing configuration?", default=True):
            raise typer.Exit(0)

    # Prepare new config
    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}

    # Remove old entries, add new
    existing_config["mcpServers"].pop("simplemem-lite", None)
    existing_config["mcpServers"]["simplemem"] = mcp_config

    if dry_run:
        console.print("[yellow]DRY RUN[/yellow] - Would write:")
        console.print(json.dumps(existing_config, indent=2))
        console.print()
        console.print(f"To: {CLAUDE_CONFIG_PATH}")
        raise typer.Exit(0)

    # Backup existing config
    if CLAUDE_CONFIG_PATH.exists():
        backup_path = _backup_claude_config()
        if backup_path:
            console.print(f"[dim]Backed up to: {backup_path}[/dim]")

    # Write new config
    CLAUDE_CONFIG_PATH.write_text(json.dumps(existing_config, indent=2))
    console.print(f"[green]✓[/green] Updated {CLAUDE_CONFIG_PATH}")

    console.print()
    console.print(Panel.fit(
        "[bold green]Setup Complete![/bold green]\n\n"
        "Restart Claude Code to activate SimpleMem.\n\n"
        "[dim]Tip: Use 'simplemem-mcp doctor' to diagnose issues[/dim]",
        border_style="green"
    ))


@app.command()
def doctor():
    """Diagnose connection and configuration issues.

    Checks:
    - Claude Code configuration
    - Backend connectivity
    - Authentication status
    - Memory statistics
    """
    console.print()
    console.print("[bold blue]SimpleMem Doctor[/bold blue]")
    console.print()

    issues = []

    # Check 1: Claude Code config
    console.print("[bold]1. Claude Code Configuration[/bold]")

    if not CLAUDE_CONFIG_PATH.exists():
        console.print(f"  [red]✗[/red] Config not found: {CLAUDE_CONFIG_PATH}")
        issues.append("Run 'simplemem-mcp setup' to configure Claude Code")
    else:
        config = _get_claude_config()
        mcp_servers = config.get("mcpServers", {})

        simplemem_config = mcp_servers.get("simplemem") or mcp_servers.get("simplemem-lite")

        if not simplemem_config:
            console.print(f"  [red]✗[/red] SimpleMem not configured in {CLAUDE_CONFIG_PATH}")
            issues.append("Run 'simplemem-mcp setup' to add SimpleMem")
        else:
            console.print(f"  [green]✓[/green] SimpleMem configured")

            # Show config summary
            env = simplemem_config.get("env", {})
            backend_url = env.get("SIMPLEMEM_BACKEND_URL", DEFAULT_BACKEND_URL)
            has_key = "SIMPLEMEM_API_KEY" in env

            console.print(f"    Backend: {backend_url}")
            console.print(f"    API Key: {'configured' if has_key else 'not set'}")

    console.print()

    # Check 2: Backend connectivity
    console.print("[bold]2. Backend Connectivity[/bold]")

    # Get config from Claude config or env
    config = _get_claude_config()
    mcp_servers = config.get("mcpServers", {})
    simplemem_config = mcp_servers.get("simplemem") or mcp_servers.get("simplemem-lite") or {}
    env = simplemem_config.get("env", {})

    backend_url = env.get("SIMPLEMEM_BACKEND_URL") or os.environ.get("SIMPLEMEM_BACKEND_URL", DEFAULT_BACKEND_URL)
    api_key = env.get("SIMPLEMEM_API_KEY") or os.environ.get("SIMPLEMEM_API_KEY")

    import httpx

    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {}
            if api_key:
                headers["X-API-Key"] = api_key

            # Health check
            resp = client.get(f"{backend_url}/health", headers=headers)
            if resp.status_code == 200:
                console.print(f"  [green]✓[/green] Backend reachable: {backend_url}")
            else:
                console.print(f"  [red]✗[/red] Backend returned HTTP {resp.status_code}")
                issues.append(f"Backend health check failed (HTTP {resp.status_code})")
    except httpx.ConnectError:
        console.print(f"  [red]✗[/red] Cannot connect to {backend_url}")
        if "localhost" in backend_url or "127.0.0.1" in backend_url:
            issues.append("Local backend not running. Start with: SIMPLEMEM_MODE=dev simplemem-lite serve")
        else:
            issues.append(f"Cannot reach {backend_url}")
    except Exception as e:
        console.print(f"  [red]✗[/red] Connection error: {e}")
        issues.append(str(e))

    console.print()

    # Check 3: Authentication (for cloud)
    console.print("[bold]3. Authentication[/bold]")

    if not api_key:
        if backend_url == DEFAULT_BACKEND_URL:
            console.print("  [yellow]![/yellow] No API key configured")
            console.print("    Cloud backend requires authentication")
            issues.append("Set SIMPLEMEM_API_KEY for cloud backend")
        else:
            console.print("  [dim]ℹ[/dim] No API key (self-hosted mode)")
    else:
        try:
            with httpx.Client(timeout=10.0) as client:
                headers = {"X-API-Key": api_key}
                resp = client.get(f"{backend_url}/api/v1/memories/stats", headers=headers)

                if resp.status_code == 200:
                    console.print("  [green]✓[/green] API key valid")
                elif resp.status_code == 401:
                    console.print("  [red]✗[/red] Invalid API key")
                    issues.append("API key is invalid")
                elif resp.status_code == 403:
                    console.print("  [red]✗[/red] API key lacks permissions")
                    issues.append("API key does not have access")
                else:
                    console.print(f"  [yellow]![/yellow] Auth check returned HTTP {resp.status_code}")
        except Exception as e:
            console.print(f"  [yellow]![/yellow] Could not verify: {e}")

    console.print()

    # Check 4: Memory stats (if connected)
    console.print("[bold]4. Memory Statistics[/bold]")

    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {}
            if api_key:
                headers["X-API-Key"] = api_key

            resp = client.get(f"{backend_url}/api/v1/memories/stats", headers=headers)

            if resp.status_code == 200:
                stats = resp.json()
                total = stats.get("total_memories", 0)
                relations = stats.get("total_relations", 0)

                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column()
                table.add_column(justify="right")

                table.add_row("Total memories", str(total))
                table.add_row("Total relations", str(relations))

                types = stats.get("types_breakdown", {})
                if types:
                    table.add_row("", "")
                    for t, count in types.items():
                        table.add_row(f"  {t}", str(count))

                console.print(table)
            else:
                console.print(f"  [dim]Could not fetch stats (HTTP {resp.status_code})[/dim]")
    except Exception:
        console.print("  [dim]Could not connect to backend[/dim]")

    console.print()

    # Summary
    if issues:
        console.print(Panel(
            "[bold yellow]Issues Found[/bold yellow]\n\n" +
            "\n".join(f"• {issue}" for issue in issues),
            border_style="yellow"
        ))
    else:
        console.print(Panel(
            "[bold green]All checks passed![/bold green]\n\n"
            "SimpleMem is properly configured and connected.",
            border_style="green"
        ))


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
    console.print("[bold]After signup, run:[/bold]")
    console.print()
    console.print("  simplemem-mcp setup --api-key <your-key>")


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


@app.command()
def undo():
    """Restore the previous Claude Code configuration."""
    backup_dir = Path.home() / ".claude_backups"

    if not backup_dir.exists():
        console.print("[yellow]No backups found[/yellow]")
        raise typer.Exit(1)

    # Find most recent backup
    backups = sorted(backup_dir.glob("claude_*.json"), reverse=True)

    if not backups:
        console.print("[yellow]No backups found[/yellow]")
        raise typer.Exit(1)

    latest = backups[0]
    console.print(f"Found backup: {latest}")
    console.print()

    # Show diff
    if CLAUDE_CONFIG_PATH.exists():
        console.print("[dim]Current config:[/dim]")
        console.print(CLAUDE_CONFIG_PATH.read_text()[:500])
        console.print()

    console.print("[dim]Backup config:[/dim]")
    console.print(latest.read_text()[:500])
    console.print()

    if Confirm.ask("Restore this backup?", default=True):
        shutil.copy2(latest, CLAUDE_CONFIG_PATH)
        console.print(f"[green]✓[/green] Restored from {latest}")
    else:
        console.print("Cancelled")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
