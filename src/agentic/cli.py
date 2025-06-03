"""
Agentic CLI - Multi-agent AI development workflows from a single CLI
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agentic import __version__
from agentic.core.project_analyzer import ProjectAnalyzer
from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig
from agentic.utils.logging import setup_logging

console = Console()


def print_banner() -> None:
    """Print the Agentic banner"""
    banner_text = f"""
[bold blue]🤖 Agentic v{__version__}[/bold blue]
[dim]Multi-agent AI development workflows from a single CLI[/dim]
    """
    console.print(Panel(banner_text.strip(), border_style="blue"))


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.version_option(version="1.0.0", prog_name="agentic")
@click.pass_context
def cli(ctx: click.Context, debug: bool):
    """🤖 Agentic - Multi-Agent AI Development Orchestrator"""
    
    # Initialize context
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    
    # Configure logging
    logger = setup_logging(debug=debug)
    ctx.obj['logger'] = logger
    
    if debug:
        logger.debug("Debug mode enabled")


@cli.command()
@click.option("--force", is_flag=True, help="Force re-initialization")
@click.pass_context  
def init(ctx: click.Context, force: bool):
    """Initialize Agentic in the current directory"""
    import asyncio
    from agentic.models.config import AgenticConfig
    from agentic.core.orchestrator import Orchestrator
    
    logger = ctx.obj['logger']
    
    try:
        # Load or create configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        # Run async initialization
        async def async_init():
            console.print("[bold blue]🚀 Initializing Agentic...[/bold blue]")
            
            success = await orchestrator.initialize()
            
            if success:
                console.print(f"[bold green]✅ Agentic initialized successfully![/bold green]")
                console.print(f"[dim]Workspace: {config.workspace_path}[/dim]")
                console.print(f"[dim]Active agents: {orchestrator.agent_count}[/dim]")
                
                # Show agent status
                if orchestrator.agent_count > 0:
                    console.print("\n[bold]Active Agents:[/bold]")
                    status = await orchestrator.get_agent_status()
                    for agent_id, info in status.items():
                        agent_type = info.get('type', 'unknown')
                        agent_name = info.get('name', 'unnamed')
                        focus_areas = info.get('focus_areas', [])
                        console.print(f"  • {agent_name} ({agent_type}) - {', '.join(focus_areas)}")
            else:
                console.print("[bold red]❌ Failed to initialize Agentic[/bold red]")
                console.print("[dim]Check logs for details[/dim]")
                raise click.ClickException("Initialization failed")
        
        asyncio.run(async_init())
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--output", "-o", type=click.Choice(["table", "json", "yaml"]), default="table")
@click.pass_context
def analyze(ctx: click.Context, output: str) -> None:
    """Analyze the current codebase structure"""
    asyncio.run(_analyze_project(output, ctx.obj.get("debug", False)))


@cli.command()
@click.argument("agent_type", type=click.Choice(["backend", "frontend", "testing", "reasoning"]))
@click.option("--model", "-m", default="claude-3-5-sonnet", help="AI model to use")
@click.pass_context
def spawn(ctx: click.Context, agent_type: str, model: str) -> None:
    """Manually spawn a specific agent"""
    asyncio.run(_spawn_agent(agent_type, model, ctx.obj.get("debug", False)))


@cli.command()
@click.option("--format", "output_format", 
              type=click.Choice(["table", "json", "simple"]), 
              default="table", help="Output format")
@click.pass_context
def status(ctx: click.Context, output_format: str):
    """Show status of all agents"""
    import asyncio
    from agentic.models.config import AgenticConfig
    from agentic.core.orchestrator import Orchestrator
    
    logger = ctx.obj['logger']
    
    try:
        # Load configuration  
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        async def async_status():
            # Try to get status (may need initialization)
            try:
                if not orchestrator.is_ready:
                    console.print("[dim]Orchestrator not initialized. Run 'agentic init' first.[/dim]")
                    return
                
                status = await orchestrator.get_agent_status()
                health = await orchestrator.health_check()
                
                if output_format == "json":
                    import json
                    combined_status = {
                        "agents": status,
                        "health": health,
                        "total_agents": len(status)
                    }
                    console.print(json.dumps(combined_status, indent=2))
                    
                elif output_format == "simple":
                    console.print(f"Active agents: {len(status)}")
                    for agent_id, info in status.items():
                        name = info.get('name', 'unnamed')
                        agent_type = info.get('type', 'unknown')
                        status_str = info.get('status', 'unknown')
                        console.print(f"  {name} ({agent_type}): {status_str}")
                        
                else:  # table format
                    from rich.table import Table
                    
                    table = Table(title="🤖 Agent Status")
                    table.add_column("Name", style="cyan")
                    table.add_column("Type", style="magenta")
                    table.add_column("Status", style="green")
                    table.add_column("Focus Areas", style="yellow")
                    table.add_column("Health", style="red")
                    
                    for agent_id, info in status.items():
                        name = info.get('name', 'unnamed')
                        agent_type = info.get('type', 'unknown')
                        status_str = info.get('status', 'unknown')
                        focus_areas = ', '.join(info.get('focus_areas', []))
                        is_healthy = "✅" if health.get(agent_id, False) else "❌"
                        
                        table.add_row(name, agent_type, status_str, focus_areas, is_healthy)
                    
                    console.print(table)
                    
                    # Overall health summary
                    healthy_count = sum(1 for h in health.values() if h)
                    total_count = len(health)
                    console.print(f"\n[dim]Overall health: {healthy_count}/{total_count} healthy[/dim]")
                    
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                console.print(f"[bold red]❌ Error getting status: {e}[/bold red]")
        
        asyncio.run(async_status())
        
    except Exception as e:
        logger.error(f"Status command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--agent", "-a", help="Stop specific agent by name")
@click.pass_context
def stop(ctx: click.Context, agent: Optional[str]) -> None:
    """Stop all agents or a specific agent"""
    asyncio.run(_stop_agents(agent, ctx.obj.get("debug", False)))


@cli.command("exec")
@click.argument("command", nargs=-1, required=True)
@click.option("--agent", help="Route to specific agent")
@click.option("--context", help="Additional context for the command")
@click.pass_context
def exec(ctx: click.Context, command: tuple, agent: str, context: str):
    """Execute a command via AI agents"""
    import asyncio
    import json
    from agentic.models.config import AgenticConfig
    from agentic.core.orchestrator import Orchestrator
    
    logger = ctx.obj['logger']
    
    try:
        # Join command arguments
        command_str = " ".join(command)
        
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        # Parse context if provided
        command_context = {}
        if context:
            try:
                command_context = json.loads(context)
            except json.JSONDecodeError:
                command_context = {"note": context}
        
        if agent:
            command_context["preferred_agent"] = agent
        
        # Execute command
        async def async_exec():
            console.print(f"[bold blue]🤖 Executing:[/bold blue] {command_str}")
            
            # Initialize if needed
            if not orchestrator.is_ready:
                console.print("[dim]Initializing orchestrator...[/dim]")
                await orchestrator.initialize()
            
            # Execute the command
            with console.status("[bold green]Agents working..."):
                result = await orchestrator.execute_command(command_str, command_context)
            
            # Display results
            if result.success:
                console.print(f"\n[bold green]✅ Task completed successfully![/bold green]")
                if result.output:
                    console.print("\n[bold]Output:[/bold]")
                    console.print(Panel(result.output, border_style="green"))
            else:
                console.print(f"\n[bold red]❌ Task failed[/bold red]")
                if result.error:
                    console.print("\n[bold]Error:[/bold]")
                    console.print(Panel(result.error, border_style="red"))
            
            # Show execution info
            console.print(f"\n[dim]Task ID: {result.task_id}[/dim]")
            console.print(f"[dim]Agent: {result.agent_id}[/dim]")
        
        asyncio.run(async_exec())
        
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


# Command implementations

async def _analyze_project(output_format: str, debug: bool) -> None:
    """Analyze current project structure"""
    current_dir = Path.cwd()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("🔍 Analyzing codebase...", total=None)
        
        analyzer = ProjectAnalyzer(current_dir)
        project_structure = await analyzer.analyze()
        
        progress.update(task, description="✅ Analysis complete", completed=True)
    
    if output_format == "table":
        _display_analysis_table(project_structure)
    elif output_format == "json":
        import json
        console.print(json.dumps(project_structure.model_dump(), indent=2, default=str))
    elif output_format == "yaml":
        import yaml
        console.print(yaml.dump(project_structure.model_dump(), default_flow_style=False))


def _display_analysis_table(project_structure) -> None:
    """Display project analysis in table format"""
    console.print(f"\n[blue]📊 Project Analysis: {project_structure.root_path.name}[/blue]")
    
    # Tech Stack Table
    table = Table(title="Technology Stack", show_header=True, header_style="bold green")
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Technologies", style="white")
    
    tech_stack = project_structure.tech_stack
    categories = [
        ("Languages", tech_stack.languages),
        ("Frameworks", tech_stack.frameworks),
        ("Databases", tech_stack.databases),
        ("Testing", tech_stack.testing_frameworks),
        ("Build Tools", tech_stack.build_tools),
    ]
    
    for category, items in categories:
        if items:
            table.add_row(category, ", ".join(items))
    
    console.print(table)
    
    # Directory Structure
    if project_structure.source_directories or project_structure.test_directories:
        console.print(f"\n[blue]📁 Directory Structure:[/blue]")
        structure_table = Table(show_header=True, header_style="bold green")
        structure_table.add_column("Type", style="cyan", width=15)
        structure_table.add_column("Directories", style="white")
        
        if project_structure.source_directories:
            dirs = [str(d.relative_to(project_structure.root_path)) for d in project_structure.source_directories]
            structure_table.add_row("Source", ", ".join(dirs))
        
        if project_structure.test_directories:
            dirs = [str(d.relative_to(project_structure.root_path)) for d in project_structure.test_directories]
            structure_table.add_row("Tests", ", ".join(dirs))
        
        console.print(structure_table)


async def _spawn_agent(agent_type: str, model: str, debug: bool) -> None:
    """Spawn a specific agent manually"""
    console.print(f"[blue]🤖 Spawning {agent_type} agent with model {model}...[/blue]")
    
    # TODO: Implement agent spawning
    console.print(f"[yellow]⚠️  Agent spawning not yet implemented[/yellow]")


async def _stop_agents(agent_name: Optional[str], debug: bool) -> None:
    """Stop agents"""
    if agent_name:
        console.print(f"[blue]🛑 Stopping agent: {agent_name}[/blue]")
    else:
        console.print("[blue]🛑 Stopping all agents[/blue]")
    
    # TODO: Implement agent stopping
    console.print("[dim]No active agents to stop[/dim]")


if __name__ == "__main__":
    cli() 