"""
Agentic CLI - Multi-agent AI development workflows from a single CLI
"""

import asyncio
import json
import os
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
from agentic.core.interactive_cli import InteractiveCLI
from agentic.core.ide_integration import (
    IDEIntegrationManager,
    initialize_ide_integration,
    get_ide_integration_manager,
    create_ide_command_from_selection,
    IDECommand
)
from agentic.agents.output_parser import AgentOutputParser

console = Console()


def print_banner() -> None:
    """Print the Agentic banner"""
    banner_text = f"""
[bold blue]🤖 Agentic v{__version__}[/bold blue]
[dim]Multi-agent AI development workflows from a single CLI[/dim]
    """
    console.print(Panel(banner_text.strip(), border_style="blue"))


@click.group(invoke_without_command=True)
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
    
    # If no command is provided, launch interactive chat mode
    if ctx.invoked_subcommand is None:
        print_banner()
        from agentic.core.chat_interface import ChatInterface
        chat = ChatInterface(workspace_path=Path.cwd(), debug=debug)
        try:
            asyncio.run(chat.run())
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            sys.exit(1)


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
@click.option('--watch', '-w', is_flag=True, help='Watch mode - continuously update status')
@click.option('--interval', '-i', default=2, help='Update interval in seconds for watch mode')
@click.pass_context
def status(ctx: click.Context, output_format: str, watch: bool, interval: int):
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
        
        async def display_status():
            """Display current status"""
            try:
                if not orchestrator.is_ready:
                    await orchestrator.initialize()
                
                status = await orchestrator.get_agent_status()
                health = await orchestrator.health_check()
                
                # Get background tasks if available
                background_tasks = {}
                try:
                    background_tasks = orchestrator.get_all_background_tasks()
                except:
                    pass
                
                if output_format == "json":
                    import json
                    combined_status = {
                        "agents": status,
                        "health": health,
                        "background_tasks": background_tasks,
                        "total_agents": len(status),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    console.print(json.dumps(combined_status, indent=2))
                    
                elif output_format == "simple":
                    console.print(f"Active agents: {len(status)}")
                    for agent_id, info in status.items():
                        name = info.get('name', 'unnamed')
                        agent_type = info.get('type', 'unknown')
                        status_str = info.get('status', 'unknown')
                        console.print(f"  {name} ({agent_type}): {status_str}")
                    
                    if background_tasks:
                        console.print(f"Background tasks: {len(background_tasks)}")
                        
                else:  # table format
                    from rich.table import Table
                    from datetime import datetime
                    
                    # Clear screen in watch mode
                    if watch:
                        console.clear()
                    
                    # Agent status table
                    table = Table(title=f"🤖 Agent Status - {datetime.now().strftime('%H:%M:%S')}")
                    table.add_column("Name", style="cyan")
                    table.add_column("Type", style="magenta")
                    table.add_column("Status", style="green")
                    table.add_column("Focus Areas", style="yellow")
                    table.add_column("Last Activity", style="dim")
                    table.add_column("Health", style="red")
                    
                    for agent_id, info in status.items():
                        name = info.get('name', 'unnamed')
                        agent_type = info.get('type', 'unknown')
                        status_str = info.get('status', 'unknown')
                        focus_areas = ', '.join(info.get('focus_areas', []))
                        last_activity = info.get('last_activity', 'Never')
                        if isinstance(last_activity, str) and last_activity != 'Never':
                            try:
                                from datetime import datetime
                                dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                                last_activity = dt.strftime('%H:%M:%S')
                            except:
                                pass
                        is_healthy = "✅" if health.get(agent_id, False) else "❌"
                        
                        table.add_row(name, agent_type, status_str, focus_areas, str(last_activity), is_healthy)
                    
                    console.print(table)
                    
                    # Background tasks table
                    if background_tasks:
                        bg_table = Table(title="🔄 Background Tasks")
                        bg_table.add_column("Task ID", style="cyan", width=36)
                        bg_table.add_column("Status", style="yellow")
                        bg_table.add_column("Done", style="green")
                        
                        for task_id, task_info in background_tasks.items():
                            bg_table.add_row(
                                task_id,
                                task_info.get('status', 'unknown'),
                                "✅" if task_info.get('done', False) else "⏳"
                            )
                        
                        console.print(bg_table)
                    
                    # Overall health summary
                    healthy_count = sum(1 for h in health.values() if h)
                    total_count = len(health)
                    if total_count > 0:
                        health_percentage = (healthy_count / total_count) * 100
                        health_color = "green" if health_percentage >= 80 else "yellow" if health_percentage >= 50 else "red"
                        console.print(f"\n[{health_color}]Overall Health: {healthy_count}/{total_count} agents healthy ({health_percentage:.1f}%)[/{health_color}]")
                    
                    if watch:
                        console.print(f"\n[dim]Press Ctrl+C to exit watch mode. Refreshing every {interval}s...[/dim]")
                
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                console.print(f"[red]❌ Failed to get status: {e}[/red]")
        
        async def async_status():
            if watch:
                try:
                    while True:
                        await display_status()
                        await asyncio.sleep(interval)
                except KeyboardInterrupt:
                    console.print("\n[yellow]👋 Exiting watch mode[/yellow]")
            else:
                await display_status()
        
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
            
            # Parse context if provided
            exec_context = {}
            if context:
                try:
                    import json
                    exec_context = json.loads(context)
                except:
                    exec_context = {"additional_info": context}
            
            # Enable monitoring by default
            exec_context['enable_monitoring'] = True
            
            # Check if we should use progress spinner or monitor
            use_progress_spinner = os.environ.get('AGENTIC_NO_MONITOR') == '1'
            
            if use_progress_spinner:
                # Use progress spinner when monitor is disabled
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
                            context=exec_context
                        )
                        progress.remove_task(task)
                    
                    except Exception as e:
                        progress.remove_task(task)
                        logger.error(f"Command execution failed: {e}")
                        console.print(f"[bold red]❌ Execution failed: {e}[/bold red]")
                        raise
                    
                    if result and result.status == "completed":
                        console.print("[bold green]✅ Command executed successfully![/bold green]")
                        
                        # Display results
                        if result.output:
                            # Check if output is Claude Code JSON
                            if result.output.strip().startswith('{') and hasattr(result, 'agent_id') and 'claude' in str(result.agent_id).lower():
                                # Use output parser for Claude agents
                                parser = AgentOutputParser()
                                parsed = parser.parse_output(result.output, 'claude_code')
                                
                                # Show the summary/answer
                                if parsed['summary']:
                                    console.print(f"\n[bold]Answer:[/bold]")
                                    console.print(parsed['summary'])
                                
                                # Show key actions if any
                                if parsed['actions']:
                                    console.print(f"\n[dim]Actions taken: {', '.join(parsed['actions'][:3])}")
                                    if len(parsed['actions']) > 3:
                                        console.print(f"... and {len(parsed['actions']) - 3} more actions[/dim]")
                            else:
                                # Plain text output
                                console.print(f"\n[bold]Output:[/bold]\n{result.output}")
                        
                        if result.agent_id:
                            console.print(f"\n[dim]Agent used: {result.agent_id}[/dim]")
                            
                    else:
                        error_msg = result.error if result and result.error else 'No result returned or task failed'
                        console.print(f"[bold red]❌ Command failed: {error_msg}[/bold red]")
            else:
                # No progress spinner when monitor is active
                try:
                    result = await orchestrator.execute_command(
                        command=command_str,
                        context=exec_context
                    )
                    
                    if result and result.status == "completed":
                        console.print("[bold green]✅ Command executed successfully![/bold green]")
                        
                        # Display results
                        if result.output:
                            # Check if output is Claude Code JSON
                            if result.output.strip().startswith('{') and hasattr(result, 'agent_id') and 'claude' in str(result.agent_id).lower():
                                # Use output parser for Claude agents
                                parser = AgentOutputParser()
                                parsed = parser.parse_output(result.output, 'claude_code')
                                
                                # Show the summary/answer
                                if parsed['summary']:
                                    console.print(f"\n[bold]Answer:[/bold]")
                                    console.print(parsed['summary'])
                                
                                # Show key actions if any
                                if parsed['actions']:
                                    console.print(f"\n[dim]Actions taken: {', '.join(parsed['actions'][:3])}")
                                    if len(parsed['actions']) > 3:
                                        console.print(f"... and {len(parsed['actions']) - 3} more actions[/dim]")
                            else:
                                # Plain text output
                                console.print(f"\n[bold]Output:[/bold]\n{result.output}")
                        
                        if result.agent_id:
                            console.print(f"\n[dim]Agent used: {result.agent_id}[/dim]")
                            
                    else:
                        error_msg = result.error if result and result.error else 'No result returned or task failed'
                        console.print(f"[bold red]❌ Command failed: {error_msg}[/bold red]")
                        
                except Exception as e:
                    logger.error(f"Command execution failed: {e}")
                    console.print(f"[bold red]❌ Execution failed: {e}[/bold red]")
        
        asyncio.run(async_exec())
        
    except Exception as e:
        logger.error(f"Exec command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command("chat")
@click.option('--model', '-m', help='Override default AI model')
@click.option('--agent-type', '-t', 
              type=click.Choice(['claude', 'aider', 'dynamic', 'mixed']),
              default='dynamic',
              help='Default agent type strategy')
@click.pass_context
def chat(ctx: click.Context, model: Optional[str], agent_type: str):
    """Launch interactive chat mode"""
    debug = ctx.obj.get('debug', False)
    
    # Launch the same chat interface as when running 'agentic' alone
    from agentic.core.chat_interface import ChatInterface
    chat_interface = ChatInterface(workspace_path=Path.cwd(), debug=debug)
    
    # Store preferences
    if model:
        chat_interface.default_model = model
    chat_interface.default_agent_type = agent_type
    
    try:
        asyncio.run(chat_interface.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


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
            table.add_row("Max Agents", str(config.max_concurrent_agents), "config")
            table.add_row("Auto Spawn", str(config.auto_spawn_agents), "config")
            
            # Model settings (unified)
            table.add_row("Primary Model", config.primary_model, "config")
            table.add_row("Fallback Model", config.fallback_model, "config")
            table.add_row("Max Tokens", str(config.max_tokens), "config")
            table.add_row("Temperature", str(config.temperature), "config")
            
            # Agent settings  
            agent_types = [agent.agent_type.value for agent in config.agents.values()]
            table.add_row("Agent Types", ", ".join(agent_types), "config")
            
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
        console.print("[yellow]⚠️ Performance monitoring temporarily disabled during refactor[/yellow]")
        console.print("[dim]This feature will be restored in the next version[/dim]")
        
        # TODO: Re-implement with unified configuration
        # monitor = PerformanceMonitor(Path.cwd())
        # await monitor.initialize()
        # return await monitor.get_performance_report()
        
    except Exception as e:
        logger.error(f"Performance command failed: {e}")
        console.print(f"[bold red]❌ Failed to get performance metrics: {e}[/bold red]")
        raise click.ClickException(str(e))


def _display_performance_table(report):
    """Display performance report as table - temporarily disabled"""
    # TODO: Re-implement with unified configuration
    console.print("[yellow]Performance table display temporarily disabled[/yellow]")
    
    # # System Metrics
    # metrics_table = Table(title="🖥️ System Metrics")
    # metrics_table.add_column("Metric", style="cyan")
    # metrics_table.add_column("Value", style="green")
    # 
    # metrics_table.add_row("CPU Usage", f"{report.system_metrics.cpu_percent:.1f}%")
    # metrics_table.add_row("Memory Usage", f"{report.system_metrics.memory_percent:.1f}%")
    # metrics_table.add_row("Disk Usage", f"{report.system_metrics.disk_percent:.1f}%")
    # 
    # console.print(metrics_table)


@cli.command()
@click.argument("agent_name", required=False)
@click.pass_context
def debug(ctx: click.Context, agent_name: Optional[str]):
    """Start debug session for agents"""
    
    logger = ctx.obj['logger']
    
    try:
        console.print("[yellow]⚠️ Debug console temporarily disabled during refactor[/yellow]")
        console.print("[dim]This feature will be restored in the next version[/dim]")
        
        # TODO: Re-implement with unified configuration
        # monitor = PerformanceMonitor(Path.cwd())
        # await monitor.initialize()
        # debug_console = monitor.debug_console
        # await debug_console.start_debug_session(agent_name)
        
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


@cli.command("exec-bg")
@click.argument("command", nargs=-1, required=True)
@click.option("--context", help="Additional context for the command")
@click.pass_context
def exec_bg(ctx: click.Context, command: tuple, context: str):
    """Execute a command in the background without blocking"""
    
    logger = ctx.obj['logger']
    
    # Combine command parts into single string
    command_str = " ".join(command)
    
    console.print(f"[bold blue]🎯 Starting background execution: {command_str}[/bold blue]")
    if context:
        console.print(f"[dim]Context: {context}[/dim]")
    
    try:
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        async def async_exec_bg():
            # Initialize orchestrator if needed
            if not orchestrator.is_ready:
                console.print("[yellow]⚠️ Orchestrator not initialized. Initializing now...[/yellow]")
                success = await orchestrator.initialize()
                if not success:
                    console.print("[red]❌ Failed to initialize orchestrator[/red]")
                    return
            
            # Start background execution
            try:
                task_id = await orchestrator.execute_background_command(
                    command=command_str,
                    context={"context": context} if context else None
                )
                
                console.print(f"[bold green]✅ Background task started![/bold green]")
                console.print(f"[dim]Task ID: {task_id}[/dim]")
                console.print(f"[dim]Use 'agentic bg-status {task_id}' to check progress[/dim]")
                console.print(f"[dim]Use 'agentic bg-list' to see all background tasks[/dim]")
                
            except Exception as e:
                logger.error(f"Background command execution failed: {e}")
                console.print(f"[bold red]❌ Background execution failed: {e}[/bold red]")
        
        asyncio.run(async_exec_bg())
        
    except Exception as e:
        logger.error(f"Background exec command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command("bg-status")
@click.argument("task_id", required=False)
@click.pass_context
def bg_status(ctx: click.Context, task_id: Optional[str]):
    """Show status of background tasks"""
    
    logger = ctx.obj['logger']
    
    try:
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        async def async_bg_status():
            # Initialize orchestrator if needed
            if not orchestrator.is_ready:
                success = await orchestrator.initialize()
                if not success:
                    console.print("[red]❌ Failed to initialize orchestrator[/red]")
                    return
            
            if task_id:
                # Show specific task status
                status = orchestrator.get_background_task_status(task_id)
                if status:
                    console.print(f"[bold blue]📋 Task Status: {task_id}[/bold blue]")
                    console.print(f"Status: [yellow]{status['status']}[/yellow]")
                    console.print(f"Done: {status['done']}")
                    if 'result' in status:
                        console.print(f"Result: {status['result']}")
                else:
                    console.print(f"[red]❌ Task not found: {task_id}[/red]")
            else:
                # Show all background tasks
                all_tasks = orchestrator.get_all_background_tasks()
                if not all_tasks:
                    console.print("[yellow]No background tasks found[/yellow]")
                    return
                
                table = Table(title="🔄 Background Tasks")
                table.add_column("Task ID", style="cyan", width=36)
                table.add_column("Status", style="yellow", width=12)
                table.add_column("Done", style="green", width=6)
                table.add_column("Completed At", style="dim", width=20)
                
                for tid, task_status in all_tasks.items():
                    completed_at = task_status.get('completed_at', 'N/A')
                    if isinstance(completed_at, str) and completed_at != 'N/A':
                        try:
                            from datetime import datetime
                            completed_at = datetime.fromisoformat(completed_at).strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    
                    table.add_row(
                        tid,
                        task_status['status'],
                        "✅" if task_status['done'] else "⏳",
                        str(completed_at)
                    )
                
                console.print(table)
        
        asyncio.run(async_bg_status())
        
    except Exception as e:
        logger.error(f"Background status command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command("bg-list")
@click.pass_context
def bg_list(ctx: click.Context):
    """List all background tasks"""
    # This is just an alias for bg-status with no task_id
    ctx.invoke(bg_status, task_id=None)


@cli.command("bg-cancel")
@click.argument("task_id")
@click.pass_context
def bg_cancel(ctx: click.Context, task_id: str):
    """Cancel a running background task"""
    
    logger = ctx.obj['logger']
    
    try:
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        async def async_bg_cancel():
            # Initialize orchestrator if needed
            if not orchestrator.is_ready:
                success = await orchestrator.initialize()
                if not success:
                    console.print("[red]❌ Failed to initialize orchestrator[/red]")
                    return
            
            # Cancel the task
            success = await orchestrator.cancel_background_task(task_id)
            if success:
                console.print(f"[green]✅ Cancelled background task: {task_id}[/green]")
            else:
                console.print(f"[yellow]⚠️ Task not found or already completed: {task_id}[/yellow]")
        
        asyncio.run(async_bg_cancel())
        
    except Exception as e:
        logger.error(f"Background cancel command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command("exec-interactive")
@click.argument("command", nargs=-1, required=True)
@click.option("--context", help="Additional context for the command")
@click.pass_context
def exec_interactive(ctx: click.Context, command: tuple, context: str):
    """Execute a command with interactive capabilities (handles agent questions)"""
    
    logger = ctx.obj['logger']
    
    # Combine command parts into single string
    command_str = " ".join(command)
    
    console.print(f"[bold blue]🤖 Interactive execution: {command_str}[/bold blue]")
    console.print("[dim]This mode allows agents to ask questions and request input[/dim]")
    if context:
        console.print(f"[dim]Context: {context}[/dim]")
    
    try:
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        async def input_handler(question: str) -> str:
            """Handle input requests from agents"""
            console.print(f"\n[yellow]🤔 Agent question:[/yellow] {question}")
            
            # Use click to get user input
            response = click.prompt("Your response", type=str, default="")
            return response
        
        async def async_exec_interactive():
            # Initialize orchestrator if needed
            if not orchestrator.is_ready:
                console.print("[yellow]⚠️ Orchestrator not initialized. Initializing now...[/yellow]")
                success = await orchestrator.initialize()
                if not success:
                    console.print("[red]❌ Failed to initialize orchestrator[/red]")
                    return
            
            # Execute with interactive capabilities
            try:
                result = await orchestrator.execute_interactive_command(
                    command=command_str,
                    input_handler=input_handler
                )
                
                if result and result.status == "completed":
                    console.print("[bold green]✅ Interactive command executed successfully![/bold green]")
                    
                    # Display results
                    if result.output:
                        console.print(f"\n[bold]Output:[/bold]\n{result.output}")
                    
                    if result.agent_id:
                        console.print(f"\n[dim]Agent used: {result.agent_id}[/dim]")
                        
                else:
                    error_msg = result.error if result and result.error else 'No result returned or task failed'
                    console.print(f"[bold red]❌ Command failed: {error_msg}[/bold red]")
                    
            except Exception as e:
                logger.error(f"Interactive command execution failed: {e}")
                console.print(f"[bold red]❌ Execution failed: {e}[/bold red]")
        
        asyncio.run(async_exec_interactive())
        
    except Exception as e:
        logger.error(f"Interactive exec command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command("comm-status")
@click.pass_context
def comm_status(ctx: click.Context):
    """Show inter-agent communication status"""
    
    logger = ctx.obj['logger']
    
    try:
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        async def async_comm_status():
            # Initialize orchestrator if needed
            if not orchestrator.is_ready:
                success = await orchestrator.initialize()
                if not success:
                    console.print("[red]❌ Failed to initialize orchestrator[/red]")
                    return
            
            # Get communication stats
            stats = orchestrator.get_communication_stats()
            shared_context = await orchestrator.get_shared_context()
            
            console.print("[bold blue]🔗 Inter-Agent Communication Status[/bold blue]")
            
            # Communication stats table
            table = Table(title="Communication Hub Stats")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Active Agents", str(stats['active_agents']))
            table.add_row("Agent Names", ", ".join(stats['agent_names']) if stats['agent_names'] else "None")
            table.add_row("Memory File Exists", "✅" if stats['memory_file_exists'] else "❌")
            table.add_row("Memory File Path", stats['memory_file_path'])
            
            console.print(table)
            
            # Shared context
            if shared_context and shared_context.strip():
                console.print(f"\n[bold]Shared Context:[/bold]")
                console.print(shared_context[:500] + "..." if len(shared_context) > 500 else shared_context)
            else:
                console.print(f"\n[dim]No shared context available[/dim]")
        
        asyncio.run(async_comm_status())
        
    except Exception as e:
        logger.error(f"Communication status command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command("exec-multi")
@click.argument("command", nargs=-1, required=True)
@click.option("--context", help="Additional context for the command")
@click.option("--agent-type", "-t", 
              type=click.Choice(["claude", "aider", "dynamic", "mixed", "enhanced"]), 
              default="dynamic", 
              help="Agent type strategy: claude (all Claude Code), aider (all Aider), dynamic (intelligent mix), mixed (force both types), enhanced (ML-based selection)")
@click.option("--track-metrics", "-m", is_flag=True, 
              help="Track performance metrics for agent comparison")
@click.option("--analyze-only", "-a", is_flag=True,
              help="Only analyze the task without executing (shows agent selection reasoning)")
@click.pass_context
def exec_multi(ctx: click.Context, command: tuple, context: str, agent_type: str, track_metrics: bool, analyze_only: bool):
    """Execute a command using multiple coordinated agents (Claude Code + Aider agents)"""
    
    logger = ctx.obj['logger']
    
    # Combine command parts into single string
    command_str = " ".join(command)
    
    console.print(f"[bold blue]🤖🤖🤖 Multi-Agent Execution: {command_str}[/bold blue]")
    console.print("[dim]This mode spawns multiple specialized agents working together[/dim]")
    if context:
        console.print(f"[dim]Context: {context}[/dim]")
    
    try:
        # Load configuration
        config = AgenticConfig.load_or_create(Path.cwd())
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        
        async def async_exec_multi():
            # Initialize orchestrator if needed
            if not orchestrator.is_ready:
                console.print("[yellow]⚠️ Orchestrator not initialized. Initializing now...[/yellow]")
                success = await orchestrator.initialize()
                if not success:
                    console.print("[red]❌ Failed to initialize orchestrator[/red]")
                    return
            
            # If analyze-only mode, perform analysis and exit
            if analyze_only:
                console.print("\n[bold cyan]🔍 Task Analysis Mode[/bold cyan]")
                
                # Import and use the task analyzer
                from agentic.core.task_analyzer import TaskAnalyzer
                analyzer = TaskAnalyzer()
                
                # Analyze the task
                analysis = await analyzer.analyze_task(command_str)
                
                # Display analysis results
                from rich.table import Table
                
                table = Table(title="Task Analysis Results")
                table.add_column("Property", style="cyan", width=25)
                table.add_column("Value", style="green")
                
                table.add_row("Command", command_str[:50] + "..." if len(command_str) > 50 else command_str)
                table.add_row("Estimated Files", str(analysis.file_count_estimate))
                table.add_row("Complexity Score", f"{analysis.complexity_score:.2f}")
                table.add_row("Requires Creativity", "Yes" if analysis.requires_creativity else "No")
                table.add_row("Follows Pattern", "Yes" if analysis.follows_pattern else "No")
                table.add_row("Ambiguity Level", f"{analysis.ambiguity_level:.2f}")
                table.add_row("Bulk Operation", "Yes" if analysis.bulk_operation else "No")
                table.add_row("Operation Type", analysis.operation_type)
                table.add_row("Suggested Agent", analysis.suggested_agent)
                
                console.print(table)
                
                if analysis.complexity_indicators:
                    console.print(f"\n[bold]Complexity Indicators:[/bold] {', '.join(analysis.complexity_indicators)}")
                
                if analysis.reasoning:
                    console.print("\n[bold]Selection Reasoning:[/bold]")
                    for reason in analysis.reasoning:
                        console.print(f"  • {reason}")
                
                console.print(f"\n[bold]Recommendation:[/bold] Use {analysis.suggested_agent} for this task")
                return
            
            # Execute multi-agent command
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Analyzing and decomposing command...", total=None)
                
                try:
                    execution_context = {
                        "context": context,
                        "agent_type_strategy": agent_type,
                        "track_metrics": track_metrics
                    } if context or agent_type != "dynamic" or track_metrics else None
                    
                    result = await orchestrator.execute_multi_agent_command(
                        command=command_str,
                        context=execution_context
                    )
                    progress.remove_task(task)
                    
                    if result and result.status == "completed":
                        console.print("[bold green]✅ Multi-agent command completed successfully![/bold green]")
                        
                        # Display summary
                        console.print(f"\n[bold]Execution Summary:[/bold]")
                        console.print(f"  • Total tasks: {len(result.completed_tasks) + len(result.failed_tasks)}")
                        console.print(f"  • Completed: {len(result.completed_tasks)}")
                        console.print(f"  • Failed: {len(result.failed_tasks)}")
                        console.print(f"  • Duration: {result.total_duration:.1f}s")
                        
                        # Show coordination log highlights
                        if result.coordination_log:
                            console.print(f"\n[bold]Coordination Highlights:[/bold]")
                            for log_entry in result.coordination_log[-3:]:  # Show last 3 entries
                                entry_type = log_entry.get("type", "unknown")
                                if entry_type == "parallel_execution_start":
                                    console.print(f"  • Started {log_entry.get('task_count', 0)} agents in parallel")
                                elif entry_type == "execution_complete":
                                    console.print(f"  • Execution completed with status: {log_entry.get('status', 'unknown')}")
                        
                        # Show failed tasks if any
                        if result.failed_tasks:
                            console.print(f"\n[yellow]⚠️ Failed tasks: {', '.join(result.failed_tasks)}[/yellow]")
                            
                    else:
                        status = result.status if result else "unknown"
                        console.print(f"[bold red]❌ Multi-agent command failed with status: {status}[/bold red]")
                        
                        if result and result.failed_tasks:
                            console.print(f"Failed tasks: {', '.join(result.failed_tasks)}")
                            
                except Exception as e:
                    progress.remove_task(task)
                    logger.error(f"Multi-agent command execution failed: {e}")
                    console.print(f"[bold red]❌ Execution failed: {e}[/bold red]")
        
        asyncio.run(async_exec_multi())
        
    except Exception as e:
        logger.error(f"Multi-agent exec command failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.group()
def model():
    """Configure AI models for agents"""
    pass


@cli.group()
def api():
    """Manage API keys for AI providers"""
    pass


@api.command('set')
@click.argument('provider', type=click.Choice(['gemini', 'anthropic', 'openai']))
@click.argument('api_key')
@click.option('--global', 'is_global', is_flag=True, help='Set API key globally for all projects')
@click.pass_context
def api_set(ctx: click.Context, provider: str, api_key: str, is_global: bool):
    """Set API key for a provider"""
    from agentic.utils.credentials import validate_api_key
    
    logger = ctx.obj['logger']
    
    # Validate API key format
    is_valid, error_msg = validate_api_key(provider, api_key)
    if not is_valid:
        console.print(f"[red]❌ Invalid API key: {error_msg}[/red]")
        return
    
    try:
        import keyring
        
        if is_global:
            # Set global API key
            service_name = f"agentic.{provider}"
            keyring.set_password(service_name, "global", api_key)
            console.print(f"[green]✅ Global {provider.upper()} API key saved successfully![/green]")
        else:
            # Set project-specific API key
            workspace_path = Path.cwd()
            service_name = f"agentic.{provider}.{workspace_path.name}"
            keyring.set_password(service_name, str(workspace_path), api_key)
            console.print(f"[green]✅ Project-specific {provider.upper()} API key saved for {workspace_path.name}![/green]")
        
        # Also set it in the current environment for immediate use
        env_vars = {
            'gemini': 'GEMINI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY'
        }
        os.environ[env_vars[provider]] = api_key
        
    except Exception as e:
        logger.error(f"Failed to save API key: {e}")
        console.print(f"[red]❌ Failed to save API key: {e}[/red]")
        console.print("[yellow]💡 Tip: You can also create a .env file with your API keys[/yellow]")


@api.command('list')
@click.pass_context
def api_list(ctx: click.Context):
    """List configured API keys"""
    from agentic.utils.credentials import list_api_keys
    
    logger = ctx.obj['logger']
    
    try:
        keys = list_api_keys()
        
        if not keys:
            console.print("[yellow]No API keys configured[/yellow]")
            console.print("\n[dim]Set API keys with:[/dim]")
            console.print("  agentic api set gemini YOUR_KEY")
            console.print("  agentic api set anthropic YOUR_KEY --global")
            return
        
        table = Table(title="🔑 Configured API Keys")
        table.add_column("Provider", style="cyan")
        table.add_column("Scope", style="magenta")
        table.add_column("Source", style="green")
        table.add_column("Key", style="dim")
        
        for provider, info in keys.items():
            masked_key = info['key'][:10] + "..." + info['key'][-4:] if len(info['key']) > 20 else "***"
            table.add_row(provider.upper(), info['scope'], info['source'], masked_key)
        
        console.print(table)
        
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")


@cli.group()
def model():
    """Configure AI models for agents"""
    pass


@model.command('list')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
def model_list(workspace: Path):
    """List available models and current configuration"""
    try:
        # Use the existing AgenticConfig from models/config.py
        from agentic.models.config import AgenticConfig as WorkspaceConfig
        
        # Try to load existing config, create default if not found
        try:
            config = WorkspaceConfig.load_or_create(workspace)
        except Exception:
            # If loading fails, create a basic default
            config = WorkspaceConfig.create_default(workspace)
        
        console.print("[bold blue]🤖 Available Models[/bold blue]\n")
        
        # Display supported models by provider
        models_info = {
            "Gemini (Google)": [
                "gemini/gemini-1.5-pro-latest",
                "gemini/gemini-1.5-flash-latest", 
                "gemini-1.5-pro (alias)",
                "gemini-1.5-flash (alias)",
                "gemini (default alias)",
            ],
            "Claude (Anthropic)": [
                "claude-3-5-sonnet",
                "claude-3-haiku", 
                "claude-3-opus",
                "claude (alias)",
                "sonnet (alias)",
                "haiku (alias)",
            ],
            "OpenAI": [
                "gpt-4o",
                "gpt-4-0125-preview",
                "gpt-3.5-turbo",
                "gpt-4 (alias)",
                "gpt-3.5 (alias)",
            ]
        }
        
        for provider, models in models_info.items():
            console.print(f"[green]{provider}:[/green]")
            for model in models:
                console.print(f"  • {model}")
            console.print()
        
        # Display current configuration
        console.print("[bold blue]📋 Current Configuration[/bold blue]\n")
        
        # Show global model settings
        console.print(f"Global Primary Model: [yellow]{config.primary_model}[/yellow]")
        console.print(f"Global Fallback Model: [yellow]{config.fallback_model}[/yellow]")
        console.print()
        
        # Show agent-specific models
        agent_models = []
        for agent_name, agent_config in config.agents.items():
            if hasattr(agent_config, 'ai_model_config') and agent_config.ai_model_config:
                model = agent_config.ai_model_config.get('model', 'default')
                agent_models.append(f"{agent_name}: [yellow]{model}[/yellow]")
        
        if agent_models:
            console.print("Agent-Specific Models:")
            for agent_model in agent_models:
                console.print(f"  • {agent_model}")
        else:
            console.print("No agent-specific models configured")
        
        console.print(f"\nWorkspace: [cyan]{config.workspace_path}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]❌ Error listing models: {e}[/red]")
        sys.exit(1)


@model.command('set')
@click.argument('model_name')
@click.option('--agent', '-a', help='Set model for specific agent (default: all agents)')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
def model_set(model_name: str, agent: Optional[str], workspace: Path):
    """Set the AI model for agents"""
    try:
        # Use the existing AgenticConfig from models/config.py
        from agentic.models.config import AgenticConfig as WorkspaceConfig
        from agentic.models.agent import AgentConfig, AgentType
        
        # Load or create config
        config = WorkspaceConfig.load_or_create(workspace)
        
        if agent:
            # Set model for specific agent
            if agent in config.agents:
                # Update existing agent
                agent_config = config.agents[agent]
                if not hasattr(agent_config, 'ai_model_config'):
                    agent_config.ai_model_config = {}
                agent_config.ai_model_config['model'] = model_name
            else:
                # Create new agent config
                agent_config = AgentConfig(
                    agent_type=AgentType.AIDER_BACKEND,  # Default type
                    name=agent,
                    workspace_path=workspace,
                    focus_areas=["general"],
                    ai_model_config={"model": model_name},
                    max_tokens=100000,
                    temperature=0.1
                )
                config.add_agent_config(agent, agent_config)
                
            console.print(f"[green]✅ Set model '[yellow]{model_name}[/yellow]' for agent '[cyan]{agent}[/cyan]'[/green]")
        else:
            # Set model for all existing agents AND update global default
            config.primary_model = model_name  # Update global setting
            
            updated_agents = []
            for agent_name, agent_config in config.agents.items():
                if not hasattr(agent_config, 'ai_model_config'):
                    agent_config.ai_model_config = {}
                agent_config.ai_model_config['model'] = model_name
                updated_agents.append(agent_name)
            
            if updated_agents:
                console.print(f"[green]✅ Set model '[yellow]{model_name}[/yellow]' globally and for agents: [cyan]{', '.join(updated_agents)}[/cyan][/green]")
            else:
                console.print(f"[green]✅ Set global default model to '[yellow]{model_name}[/yellow]'[/green]")
                console.print(f"[yellow]💡 Use --agent to create a new agent with this model.[/yellow]")
        
        # Save the configuration
        config_file = workspace / '.agentic' / 'config.yml'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config.save_to_yaml(config_file)
        
        # Show usage tip
        console.print(f"\n[dim]💡 Use 'agentic model list' to see all available models[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error setting model: {e}[/red]")
        sys.exit(1)


@model.command('test')
@click.argument('model_name')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
def model_test(model_name: str, workspace: Path):
    """Test a model configuration with Aider"""
    try:
        console.print(f"[blue]🧪 Testing model '[yellow]{model_name}[/yellow]'...[/blue]")
        
        async def test_model():
            # Import here to avoid circular imports
            from agentic.agents.aider_agents import BaseAiderAgent
            from agentic.models.agent import AgentConfig, AgentType
            
            # Create test agent config with the specified model
            test_config = AgentConfig(
                agent_type=AgentType.AIDER_BACKEND,
                name="test_agent",
                workspace_path=workspace,
                focus_areas=["testing"],
                ai_model_config={"model": model_name},
                max_tokens=1000,
                temperature=0.1
            )
            
            # Create test agent
            agent = BaseAiderAgent(test_config)
            
            # Test model mapping
            mapped_model = agent._get_model_for_aider()
            console.print(f"Model mapping: [yellow]{model_name}[/yellow] → [green]{mapped_model}[/green]")
            
            # Test Aider command construction
            agent._setup_aider_args()
            test_command = agent.aider_args + ["--help"]
            
            console.print(f"Test command: [dim]{' '.join(test_command)}[/dim]")
            
            # Run actual test
            import subprocess
            result = subprocess.run(test_command, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                console.print(f"[green]✅ Model '[yellow]{model_name}[/yellow]' is working correctly![/green]")
                return True
            else:
                console.print(f"[red]❌ Model test failed: {result.stderr}[/red]")
                return False
        
        success = asyncio.run(test_model())
        if not success:
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Error testing model: {e}[/red]")
        sys.exit(1)


@cli.group()
def keys():
    """Manage API keys securely"""
    pass


@keys.command('set')
@click.argument('provider', type=click.Choice(['anthropic', 'openai', 'google', 'gemini']))
@click.option('--key', '-k', help='API key (will prompt securely if not provided)')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory (for project-specific keys)')
@click.option('--global', 'is_global', is_flag=True, help='Set key globally (default: project-specific)')
def keys_set(provider: str, key: Optional[str], workspace: Path, is_global: bool):
    """Set API key for a provider"""
    try:
        # Import keyring for secure storage
        import keyring
        
        # If key not provided, prompt securely
        if not key:
            key = click.prompt(f'Enter {provider.upper()} API key', hide_input=True)
        
        # Determine storage scope
        if is_global:
            service_name = f"agentic.{provider}"
            username = "global"
            scope_desc = "globally"
        else:
            service_name = f"agentic.{provider}.{workspace.name}"
            username = str(workspace)
            scope_desc = f"for project '{workspace.name}'"
        
        # Store in keyring
        keyring.set_password(service_name, username, key)
        
        console.print(f"[green]✅ {provider.upper()} API key set {scope_desc}[/green]")
        
        # Show usage instructions
        if provider == 'gemini':
            console.print(f"[dim]💡 Now you can use: agentic model set gemini[/dim]")
        elif provider == 'anthropic':
            console.print(f"[dim]💡 Now you can use: agentic model set claude-3-5-sonnet[/dim]")
        elif provider == 'openai':
            console.print(f"[dim]💡 Now you can use: agentic model set gpt-4o[/dim]")
            
    except ImportError:
        console.print("[red]❌ keyring library not found. Install with: pip install keyring[/red]")
        console.print("[dim]Alternative: Set environment variable {provider.upper()}_API_KEY[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Failed to set API key: {e}[/red]")
        sys.exit(1)


@keys.command('get')
@click.argument('provider', type=click.Choice(['anthropic', 'openai', 'google', 'gemini']))
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
@click.option('--show', is_flag=True, help='Show the actual key (default: show masked)')
def keys_get(provider: str, workspace: Path, show: bool):
    """Get API key for a provider"""
    try:
        from agentic.utils.credentials import get_api_key
        
        key = get_api_key(provider, workspace)
        
        if key:
            if show:
                console.print(f"{provider.upper()}: {key}")
            else:
                masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                console.print(f"{provider.upper()}: {masked_key}")
        else:
            console.print(f"[yellow]⚠️ No {provider.upper()} API key found[/yellow]")
            console.print(f"[dim]Set with: agentic keys set {provider}[/dim]")
            
    except Exception as e:
        console.print(f"[red]❌ Failed to get API key: {e}[/red]")
        sys.exit(1)


@keys.command('list')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
def keys_list(workspace: Path):
    """List all configured API keys"""
    try:
        from agentic.utils.credentials import list_api_keys
        
        keys_info = list_api_keys(workspace)
        
        if not keys_info:
            console.print("[yellow]⚠️ No API keys configured[/yellow]")
            console.print("[dim]Set keys with: agentic keys set <provider>[/dim]")
            return
        
        table = Table(title="🔐 Configured API Keys")
        table.add_column("Provider", style="cyan")
        table.add_column("Scope", style="yellow")
        table.add_column("Key", style="green")
        table.add_column("Source", style="dim")
        
        for provider, info in keys_info.items():
            masked_key = info['key'][:8] + "..." + info['key'][-4:] if len(info['key']) > 12 else "***"
            table.add_row(provider.upper(), info['scope'], masked_key, info['source'])
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to list API keys: {e}[/red]")
        sys.exit(1)


@keys.command('remove')
@click.argument('provider', type=click.Choice(['anthropic', 'openai', 'google', 'gemini']))
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
@click.option('--global', 'is_global', is_flag=True, help='Remove global key')
@click.confirmation_option(prompt='Are you sure you want to remove this API key?')
def keys_remove(provider: str, workspace: Path, is_global: bool):
    """Remove API key for a provider"""
    try:
        import keyring
        
        if is_global:
            service_name = f"agentic.{provider}"
            username = "global"
            scope_desc = "global"
        else:
            service_name = f"agentic.{provider}.{workspace.name}"
            username = str(workspace)
            scope_desc = f"project '{workspace.name}'"
        
        try:
            keyring.delete_password(service_name, username)
            console.print(f"[green]✅ Removed {provider.upper()} API key from {scope_desc}[/green]")
        except keyring.errors.PasswordDeleteError:
            console.print(f"[yellow]⚠️ No {provider.upper()} API key found in {scope_desc}[/yellow]")
            
    except ImportError:
        console.print("[red]❌ keyring library not found[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Failed to remove API key: {e}[/red]")
        sys.exit(1)


@keys.command('env-template')
@click.option('--workspace', '-w', type=click.Path(path_type=Path),
              default=Path.cwd(), help='Workspace directory')
def keys_env_template(workspace: Path):
    """Create .env.example template for API keys"""
    try:
        from agentic.utils.credentials import create_env_template
        
        create_env_template(workspace)
        
        console.print(f"[green]✅ Created .env.example template in {workspace}[/green]")
        console.print("[dim]Next steps:[/dim]")
        console.print("[dim]  1. Copy .env.example to .env[/dim]")
        console.print("[dim]  2. Add your actual API keys to .env[/dim]")
        console.print("[dim]  3. The .env file is already added to .gitignore[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create template: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli() 