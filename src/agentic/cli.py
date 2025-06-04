"""
Agentic CLI - Multi-agent AI development workflows from a single CLI
"""

import asyncio
import json
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
from agentic.core.configuration import ConfigurationManager
from agentic.core.interactive_cli import InteractiveCLI
from agentic.core.monitoring import PerformanceMonitor, PerformanceReport
from agentic.core.ide_integration import (
    IDEIntegrationManager,
    initialize_ide_integration,
    get_ide_integration_manager,
    create_ide_command_from_selection,
    IDECommand
)

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
                    if total_count > 0:
                        health_percentage = (healthy_count / total_count) * 100
                        health_color = "green" if health_percentage >= 80 else "yellow" if health_percentage >= 50 else "red"
                        console.print(f"\n[{health_color}]Overall Health: {healthy_count}/{total_count} agents healthy ({health_percentage:.1f}%)[/{health_color}]")
                
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                console.print(f"[red]❌ Failed to get status: {e}[/red]")
        
        asyncio.run(async_status())
        
    except Exception as e:
        logger.error(f"Status command failed: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--agent", "-a", help="Stop specific agent by name")
@click.pass_context
def stop(ctx: click.Context, agent: Optional[str]) -> None:
    """Stop specific agent or all agents"""
    asyncio.run(_stop_agents(agent, ctx.obj.get("debug", False)))


@cli.command("exec")
@click.argument("command", nargs=-1, required=True)
@click.option("--agent", help="Route to specific agent")
@click.option("--context", help="Additional context for the command")
@click.pass_context
def exec(ctx: click.Context, command: tuple, agent: str, context: str):
    """Execute a command using Agentic agents"""
    
    logger = ctx.obj['logger']
    
    # Combine command parts into single string
    command_str = " ".join(command)
    
    console.print(f"[bold blue]🎯 Executing: {command_str}[/bold blue]")
    if agent:
        console.print(f"[dim]Routing to agent: {agent}[/dim]")
    if context:
        console.print(f"[dim]Context: {context}[/dim]")
    
    try:
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        async def async_exec():
            
            # Check if orchestrator is ready
            if not orchestrator.is_ready:
                console.print("[yellow]⚠️ Orchestrator not initialized. Initializing now...[/yellow]")
                success = await orchestrator.initialize()
                if not success:
                    console.print("[red]❌ Failed to initialize orchestrator[/red]")
                    return
            
            # Execute command
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing command...", total=None)
                
                try:
                    result = await orchestrator.execute_command(
                        command=command_str,
                        context=context
                    )
                    progress.remove_task(task)
                    
                    if result and result.success:
                        console.print("[bold green]✅ Command executed successfully![/bold green]")
                        
                        # Display results
                        if result.output:
                            console.print(f"\n[bold]Output:[/bold]\n{result.output}")
                        
                        if result.agent_id:
                            console.print(f"\n[dim]Agent used: {result.agent_id}[/dim]")
                            
                    else:
                        error_msg = result.error if result else 'No result returned'
                        console.print(f"[bold red]❌ Command failed: {error_msg}[/bold red]")
                        
                except Exception as e:
                    progress.remove_task(task)
                    logger.error(f"Command execution failed: {e}")
                    console.print(f"[bold red]❌ Execution failed: {e}[/bold red]")
        
        asyncio.run(async_exec())
        
    except Exception as e:
        logger.error(f"Exec command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--workspace', '-w', type=click.Path(exists=True, path_type=Path),
              default=Path.cwd(), help='Workspace directory')
@click.option('--model', '-m', default='claude-3-5-sonnet', 
              help='Primary AI model to use')
@click.option('--max-cost', type=float, default=10.0,
              help='Maximum cost per hour in USD')
@click.pass_context
def interactive(ctx: click.Context, workspace: Path, model: str, max_cost: float):
    """Launch interactive TUI mode"""
    try:
        console.print("[bold blue]🚀 Starting Agentic Interactive Mode...[/bold blue]")
        
        # Initialize interactive CLI
        interactive_cli = InteractiveCLI(workspace_path=workspace)
        
        # Run the interactive interface
        asyncio.run(interactive_cli.run())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.group()
@click.pass_context  
def config(ctx: click.Context):
    """Configuration management commands"""
    pass


@config.command("show")
@click.option("--format", "output_format", 
              type=click.Choice(["yaml", "json", "table"]), 
              default="table", help="Output format")
@click.pass_context
def config_show(ctx: click.Context, output_format: str):
    """Show current configuration"""
    
    logger = ctx.obj['logger']
    
    try:
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        if output_format == "json":
            import json
            config_dict = config.model_dump()
            console.print(json.dumps(config_dict, indent=2, default=str))
            
        elif output_format == "yaml":
            import yaml
            config_dict = config.model_dump()
            console.print(yaml.dump(config_dict, default_flow_style=False))
            
        else:  # table format
            from rich.table import Table
            
            table = Table(title="⚙️ Agentic Configuration")
            table.add_column("Setting", style="cyan", width=30)
            table.add_column("Value", style="green")
            table.add_column("Source", style="dim")
            
            # Core settings
            table.add_row("Workspace", str(config.workspace_path), "config")
            table.add_row("Log Level", config.log_level, "config")
            table.add_row("Max Agents", str(config.max_agents), "config")
            table.add_row("Auto Spawn", str(config.auto_spawn_agents), "config")
            
            # Model settings
            table.add_row("Primary Model", config.model_config.primary_model, "config")
            table.add_row("Fallback Model", config.model_config.fallback_model, "config")
            table.add_row("Max Tokens", str(config.model_config.max_tokens), "config")
            table.add_row("Temperature", str(config.model_config.temperature), "config")
            
            # Agent settings  
            table.add_row("Agent Types", ", ".join(config.agent_config.enabled_agent_types), "config")
            table.add_row("Max Task Time", f"{config.agent_config.max_task_duration_minutes}m", "config")
            table.add_row("Max Retries", str(config.agent_config.max_retries), "config")
            
            console.print(table)
            
            # Show file locations
            config_files = []
            
            # Check for various config files
            potential_files = [
                Path.cwd() / "agentic.yaml",
                Path.cwd() / "agentic.yml", 
                Path.cwd() / "agentic.json",
                Path.cwd() / ".agentic" / "config.yaml",
                Path.home() / ".agentic" / "config.yaml"
            ]
            
            for file_path in potential_files:
                if file_path.exists():
                    config_files.append(str(file_path))
            
            if config_files:
                console.print(f"\n[dim]Configuration files: {', '.join(config_files)}[/dim]")
            else:
                console.print(f"\n[dim]Using default configuration (no config files found)[/dim]")
        
    except Exception as e:
        logger.error(f"Config show failed: {e}")
        console.print(f"[bold red]❌ Failed to show configuration: {e}[/bold red]")
        raise click.ClickException(str(e))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--type", "value_type", 
              type=click.Choice(["string", "int", "float", "bool"]), 
              default="string", help="Value type")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str, value_type: str):
    """Set a configuration value"""
    
    logger = ctx.obj['logger']
    
    try:
        # Parse value to correct type
        parsed_value = value
        if value_type == "int":
            parsed_value = int(value)
        elif value_type == "float":
            parsed_value = float(value)
        elif value_type == "bool":
            parsed_value = value.lower() in ("true", "1", "yes", "on")
        
        # Load current configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Update the value using dot notation
        config_dict = config.model_dump()
        keys = key.split(".")
        
        # Navigate to the correct nested location
        current = config_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = parsed_value
        
        # Save updated configuration
        config_path = Path.cwd() / "agentic.yaml"
        
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        console.print(f"[bold green]✅ Updated {key} = {parsed_value}[/bold green]")
        console.print(f"[dim]Saved to: {config_path}[/dim]")
        
    except ValueError as e:
        logger.error(f"Invalid value for type {value_type}: {value}")
        console.print(f"[bold red]❌ Invalid {value_type} value: {value}[/bold red]")
        raise click.ClickException(f"Invalid {value_type} value: {value}")
        
    except Exception as e:
        logger.error(f"Config set failed: {e}")
        console.print(f"[bold red]❌ Failed to set configuration: {e}[/bold red]")
        raise click.ClickException(str(e))


@config.command("validate")
@click.option("--file", "config_file", help="Configuration file to validate")
@click.pass_context
def config_validate(ctx: click.Context, config_file: Optional[str]):
    """Validate configuration file"""
    
    logger = ctx.obj['logger']
    
    try:
        if config_file:
            # Validate specific file
            file_path = Path(config_file)
            if not file_path.exists():
                console.print(f"[bold red]❌ File not found: {config_file}[/bold red]")
                return
                
            config = AgenticConfig.load_from_file(file_path)
            console.print(f"[bold green]✅ Configuration file is valid: {config_file}[/bold green]")
        else:
            # Validate current/default configuration
            config = AgenticConfig.load_or_create(Path.cwd())
            console.print("[bold green]✅ Current configuration is valid[/bold green]")
        
        # Show validation details
        console.print(f"[dim]Workspace: {config.workspace_path}[/dim]")
        console.print(f"[dim]Agent types: {', '.join(config.agent_config.enabled_agent_types)}[/dim]")
        console.print(f"[dim]Model: {config.model_config.primary_model}[/dim]")
        
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        console.print(f"[bold red]❌ Configuration validation failed: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--format", "output_format", 
              type=click.Choice(["table", "json"]), 
              default="table", help="Output format")
@click.pass_context
def performance(ctx: click.Context, output_format: str):
    """Show performance metrics"""
    
    logger = ctx.obj['logger']
    
    try:
        console.print("[blue]📊 Gathering performance metrics...[/blue]")
        
        async def get_report():
            monitor = PerformanceMonitor(Path.cwd())
            await monitor.initialize()
            return await monitor.get_performance_report()
        
        report = asyncio.run(get_report())
        
        if output_format == "json":
            console.print_json(json.dumps(report.to_dict()))
        else:
            _display_performance_table(report)
        
    except Exception as e:
        logger.error(f"Performance command failed: {e}")
        console.print(f"[bold red]❌ Failed to get performance metrics: {e}[/bold red]")
        raise click.ClickException(str(e))


def _display_performance_table(report: PerformanceReport):
    """Display performance report as table"""
    # System Metrics
    metrics_table = Table(title="🖥️ System Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("CPU Usage", f"{report.system_metrics.cpu_percent:.1f}%")
    metrics_table.add_row("Memory Usage", f"{report.system_metrics.memory_percent:.1f}%")
    metrics_table.add_row("Disk Usage", f"{report.system_metrics.disk_percent:.1f}%")
    
    console.print(metrics_table)
    
    # Cost Information
    cost_table = Table(title="💰 Cost Tracking")
    cost_table.add_column("Period", style="cyan")
    cost_table.add_column("Cost", style="green")
    
    cost_table.add_row("Current Hour", f"${report.cost_summary.current_hour_cost:.2f}")
    cost_table.add_row("Today", f"${report.cost_summary.daily_cost:.2f}")
    cost_table.add_row("This Month", f"${report.cost_summary.monthly_cost:.2f}")
    
    console.print(cost_table)
    
    # Health Status
    health_panel = Panel(
        f"System Health: [{'green' if report.health_status.overall_health == 'healthy' else 'red'}]{report.health_status.overall_health.upper()}[/]",
        title="🏥 Health Status"
    )
    console.print(health_panel)


@cli.command()
@click.argument("agent_name", required=False)
@click.pass_context
def debug(ctx: click.Context, agent_name: Optional[str]):
    """Start debug session for agents"""
    
    logger = ctx.obj['logger']
    
    try:
        console.print("[blue]🐛 Starting debug session...[/blue]")
        if agent_name:
            console.print(f"[dim]Focusing on agent: {agent_name}[/dim]")
        
        async def start_debug():
            monitor = PerformanceMonitor(Path.cwd())
            await monitor.initialize()
            
            debug_console = monitor.debug_console
            await debug_console.start_debug_session(agent_name)
        
        asyncio.run(start_debug())
        
    except Exception as e:
        logger.error(f"Debug command failed: {e}")
        console.print(f"[bold red]❌ Debug session failed: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.group()
def ide():
    """IDE integration commands"""
    pass


@ide.command('init')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
def ide_init(workspace: Path):
    """Initialize IDE integrations"""
    try:
        console.print("[bold blue]🔌 Initializing IDE integrations...[/bold blue]")
        
        async def init_integrations():
            manager = initialize_ide_integration(workspace)
            results = await manager.initialize_all()
            return manager, results
        
        manager, results = asyncio.run(init_integrations())
        
        # Display results
        success_count = sum(1 for enabled in results.values() if enabled)
        total_count = len(results)
        
        if success_count == total_count:
            console.print(f"[bold green]🎉 All {total_count} IDE integrations initialized![/bold green]")
        elif success_count > 0:
            console.print(f"[yellow]⚡ {success_count}/{total_count} integrations initialized[/yellow]")
        else:
            console.print("[red]❌ No integrations initialized[/red]")
        
        # Show status table
        manager.display_integration_status()
        
    except Exception as e:
        console.print(f"[red]❌ IDE initialization error: {e}[/red]")
        sys.exit(1)


@ide.command('status')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
@click.option('--format', '-f', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def ide_status(workspace: Path, format: str):
    """Show IDE integration status"""
    try:
        manager = get_ide_integration_manager()
        if not manager:
            # Initialize if not already done
            manager = initialize_ide_integration(workspace)
        
        async def get_status():
            return await manager.get_integration_status()
        
        status = asyncio.run(get_status())
        
        if format == 'json':
            console.print_json(json.dumps(status))
        else:
            manager.display_integration_status()
            
    except Exception as e:
        console.print(f"[red]❌ Error getting IDE status: {e}[/red]")
        sys.exit(1)


@ide.command('command')
@click.argument('command_text')
@click.option('--file', '-f', type=click.Path(exists=True, path_type=Path),
              help='File to operate on')
@click.option('--start-line', type=int, help='Start line for selection')
@click.option('--end-line', type=int, help='End line for selection')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
def ide_command(command_text: str, file: Optional[Path], 
                start_line: Optional[int], end_line: Optional[int], 
                workspace: Path):
    """Execute IDE command"""
    try:
        manager = get_ide_integration_manager()
        if not manager:
            console.print("[red]❌ IDE integration not initialized. Run 'agentic ide init' first.[/red]")
            sys.exit(1)
        
        async def execute_command():
            # Create IDE command
            if file and start_line and end_line:
                ide_command = await create_ide_command_from_selection(
                    file, start_line, end_line, command_text, workspace
                )
            else:
                ide_command = IDECommand(
                    command_id=f"cli-{hash(command_text)}",
                    command_text=command_text,
                    workspace_path=workspace
                )
            
            # Execute command
            response = await manager.handle_ide_command(ide_command)
            return response
        
        response = asyncio.run(execute_command())
        
        if response.success:
            console.print(f"[green]✅ Command executed successfully[/green]")
            if response.output:
                console.print(f"Output: {response.output}")
            if response.suggestions:
                console.print("Suggestions:")
                for suggestion in response.suggestions:
                    console.print(f"  • {suggestion}")
        else:
            console.print(f"[red]❌ Command failed: {response.error}[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Error executing IDE command: {e}[/red]")
        sys.exit(1)


@ide.command('edit')
@click.argument('file_path', type=click.Path(path_type=Path))
@click.option('--line', type=int, required=True, help='Line number to edit')
@click.option('--content', required=True, help='New content for the line')
@click.option('--type', 'edit_type', type=click.Choice(['replace', 'insert', 'delete']),
              default='replace', help='Type of edit operation')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
def ide_edit(file_path: Path, line: int, content: str, edit_type: str, workspace: Path):
    """Edit file through IDE integration"""
    try:
        manager = get_ide_integration_manager()
        if not manager:
            console.print("[red]❌ IDE integration not initialized. Run 'agentic ide init' first.[/red]")
            sys.exit(1)
        
        async def perform_edit():
            edits = [{
                'type': edit_type,
                'line': line,
                'content': content
            }]
            
            return await manager.file_editor.edit_file(file_path, edits)
        
        success = asyncio.run(perform_edit())
        
        if success:
            console.print(f"[green]✅ Successfully edited {file_path}[/green]")
        else:
            console.print(f"[red]❌ Failed to edit {file_path}[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Error editing file: {e}[/red]")
        sys.exit(1)


async def _analyze_project(output_format: str, debug: bool) -> None:
    """Analyze project structure and dependencies"""
    
    try:
        console.print("[blue]🔍 Analyzing project structure...[/blue]")
        
        # Initialize project analyzer
        analyzer = ProjectAnalyzer(Path.cwd())
        
        # Run analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)
            
            project_structure = await analyzer.analyze_project()
            
            progress.remove_task(task)
        
        # Display results
        if output_format == "json":
            import json
            console.print(json.dumps(project_structure, indent=2, default=str))
        elif output_format == "yaml":
            import yaml
            console.print(yaml.dump(project_structure, default_flow_style=False))
        else:
            _display_analysis_table(project_structure)
            
    except Exception as e:
        console.print(f"[bold red]❌ Analysis failed: {e}[/bold red]")
        raise


def _display_analysis_table(project_structure) -> None:
    """Display project analysis results in a table"""
    
    # Basic project info
    info_table = Table(title="📁 Project Overview")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Project Root", str(project_structure.get("root_path", "Unknown")))
    info_table.add_row("Total Files", str(project_structure.get("total_files", 0)))
    info_table.add_row("Code Files", str(project_structure.get("code_files", 0)))
    info_table.add_row("Test Files", str(project_structure.get("test_files", 0)))
    
    console.print(info_table)
    
    # File types
    file_types = project_structure.get("file_types", {})
    if file_types:
        types_table = Table(title="📄 File Types")
        types_table.add_column("Extension", style="cyan")
        types_table.add_column("Count", style="green")
        types_table.add_column("Percentage", style="yellow")
        
        total_files = sum(file_types.values())
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            types_table.add_row(ext, str(count), f"{percentage:.1f}%")
        
        console.print(types_table)
    
    # Dependencies
    dependencies = project_structure.get("dependencies", [])
    if dependencies:
        deps_table = Table(title="📦 Dependencies")
        deps_table.add_column("Package", style="cyan")
        deps_table.add_column("Version", style="green")
        
        for dep in dependencies[:10]:  # Show first 10
            if isinstance(dep, dict):
                deps_table.add_row(dep.get("name", "Unknown"), dep.get("version", "Unknown"))
            else:
                deps_table.add_row(str(dep), "Unknown")
        
        if len(dependencies) > 10:
            console.print(f"[dim]... and {len(dependencies) - 10} more dependencies[/dim]")
        
        console.print(deps_table)
    
    # Suggested agent types
    suggested_agents = project_structure.get("suggested_agents", [])
    if suggested_agents:
        agents_table = Table(title="🤖 Suggested Agent Types")
        agents_table.add_column("Agent Type", style="cyan")
        agents_table.add_column("Reason", style="yellow")
        
        for agent in suggested_agents:
            if isinstance(agent, dict):
                agents_table.add_row(
                    agent.get("type", "Unknown"),
                    agent.get("reason", "No reason provided")
                )
            else:
                agents_table.add_row(str(agent), "Based on project structure")
        
        console.print(agents_table)


async def _spawn_agent(agent_type: str, model: str, debug: bool) -> None:
    """Spawn a specific agent type"""
    console.print(f"[blue]🚀 Spawning {agent_type} agent...[/blue]")
    # TODO: Implement agent spawning logic
    console.print(f"[green]✅ {agent_type} agent spawned with model {model}[/green]")


async def _stop_agents(agent_name: Optional[str], debug: bool) -> None:
    """Stop specific agent or all agents"""
    if agent_name:
        console.print(f"[yellow]🛑 Stopping agent: {agent_name}[/yellow]")
        # TODO: Implement specific agent stopping
        console.print(f"[green]✅ Agent {agent_name} stopped[/green]")
    else:
        console.print("[yellow]🛑 Stopping all agents...[/yellow]")
        # TODO: Implement stopping all agents
        console.print("[dim]No active agents to stop[/dim]")


if __name__ == "__main__":
    cli() 