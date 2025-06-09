"""
Interactive Chat Interface for Agentic
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig
from agentic.utils.logging import LoggerMixin


class ChatInterface(LoggerMixin):
    """Interactive chat interface for Agentic"""
    
    def __init__(self, workspace_path: Path, debug: bool = False):
        super().__init__()
        self.workspace_path = workspace_path
        self.debug = debug
        self.console = Console()
        self.orchestrator = None
        self.session = None
        self.history_file = Path.home() / ".agentic" / "chat_history"
        self.history_file.parent.mkdir(exist_ok=True)
        
        # Default preferences
        self.default_model = None
        self.default_agent_type = 'dynamic'
        
        # Define style for the prompt
        self.style = Style.from_dict({
            'prompt': '#00aa00 bold',
            'continuation': '#888888',
        })
        
        # Command completions
        self.commands = [
            '/help', '/exit', '/quit', '/clear', '/status', '/agents',
            '/spawn', '/exec', '/analyze', '/model', '/debug', '/history'
        ]
    
    async def initialize(self) -> bool:
        """Initialize the chat interface"""
        try:
            # Print welcome banner
            self._print_welcome()
            
            # Load configuration
            self.config = AgenticConfig.load_or_create(self.workspace_path)
            
            # Initialize orchestrator
            self.orchestrator = Orchestrator(self.config)
            
            # Initialize orchestrator if needed
            if not self.orchestrator.is_ready:
                self.console.print("[yellow]Initializing Agentic...[/yellow]")
                success = await self.orchestrator.initialize()
                if not success:
                    self.console.print("[red]❌ Failed to initialize orchestrator[/red]")
                    return False
            
            # Show current status
            await self._show_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chat interface: {e}")
            self.console.print(f"[red]❌ Initialization failed: {e}[/red]")
            return False
    
    def _print_welcome(self):
        """Print welcome message"""
        welcome_text = """
[bold blue]🤖 Agentic Interactive Chat[/bold blue]
[dim]Multi-agent AI development assistant[/dim]

Type your commands naturally or use slash commands:
  • [cyan]/help[/cyan] - Show available commands
  • [cyan]/agents[/cyan] - Show active agents
  • [cyan]/exit[/cyan] - Exit chat mode

[dim]Press Ctrl+C to exit anytime[/dim]
        """
        self.console.print(Panel(welcome_text.strip(), border_style="blue"))
    
    async def run(self):
        """Run the interactive chat loop"""
        if not await self.initialize():
            return
        
        # Create prompt session with history
        session = PromptSession(
            history=FileHistory(str(self.history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.commands, ignore_case=True),
            style=self.style
        )
        
        while True:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt(
                        '\n[agentic] > ',
                        multiline=False
                    )
                )
                
                if not user_input.strip():
                    continue
                
                # Process the input
                await self.process_input(user_input.strip())
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error: {e}[/red]")
                if self.debug:
                    self.console.print_exception()
    
    async def process_input(self, user_input: str) -> None:
        """Process user input"""
        # Check for slash commands
        if user_input.startswith('/'):
            await self._handle_command(user_input)
        else:
            # Execute as natural language command
            await self._execute_natural_command(user_input)
    
    async def _handle_command(self, command: str) -> None:
        """Handle slash commands"""
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()
        args = cmd_parts[1:] if len(cmd_parts) > 1 else []
        
        if cmd in ['/exit', '/quit']:
            raise KeyboardInterrupt
        
        elif cmd == '/help':
            self._show_help()
        
        elif cmd == '/clear':
            self.console.clear()
            self._print_welcome()
        
        elif cmd == '/status':
            await self._show_status()
        
        elif cmd == '/agents':
            await self._show_agents()
        
        elif cmd == '/spawn':
            await self._spawn_agent(args)
        
        elif cmd == '/exec':
            if args:
                await self._execute_natural_command(' '.join(args))
            else:
                self.console.print("[yellow]Usage: /exec <command>[/yellow]")
        
        elif cmd == '/analyze':
            await self._analyze_command(args)
        
        elif cmd == '/model':
            if args:
                await self._set_model(' '.join(args))
            else:
                await self._show_model()
        
        elif cmd == '/debug':
            self.debug = not self.debug
            self.console.print(f"[yellow]Debug mode: {'ON' if self.debug else 'OFF'}[/yellow]")
        
        elif cmd == '/history':
            self._show_history()
        
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("[dim]Type /help for available commands[/dim]")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
[bold]Available Commands:[/bold]

[bold cyan]Natural Language:[/bold cyan]
  Simply type what you want to do:
  • "create a React todo app"
  • "analyze the authentication system"
  • "add tests for the user service"

[bold cyan]Slash Commands:[/bold cyan]
  [cyan]/help[/cyan]      - Show this help message
  [cyan]/exit[/cyan]      - Exit chat mode
  [cyan]/clear[/cyan]     - Clear the screen
  [cyan]/status[/cyan]    - Show orchestrator status
  [cyan]/agents[/cyan]    - List active agents
  [cyan]/spawn[/cyan]     - Spawn a specific agent type
  [cyan]/exec[/cyan]      - Execute a command explicitly
  [cyan]/analyze[/cyan]   - Analyze a command without executing
  [cyan]/model[/cyan]     - Show or set the AI model
  [cyan]/debug[/cyan]     - Toggle debug mode
  [cyan]/history[/cyan]   - Show command history

[bold cyan]Agent Selection:[/bold cyan]
  Prefix commands to force specific agents:
  • [green]@claude[/green] "explain this code"
  • [green]@aider[/green] "refactor the database models"
  • [green]@all[/green] "create a full-stack application"
        """
        self.console.print(Panel(help_text.strip(), title="Help", border_style="cyan"))
    
    async def _show_status(self):
        """Show current orchestrator status"""
        try:
            status = await self.orchestrator.get_agent_status()
            
            self.console.print("\n[bold]Orchestrator Status:[/bold]")
            self.console.print(f"  Workspace: [cyan]{self.workspace_path}[/cyan]")
            self.console.print(f"  Active agents: [green]{len(status)}[/green]")
            
            if status:
                for agent_id, info in status.items():
                    self.console.print(f"  • {info.get('name', 'Unknown')} ({info.get('type', 'Unknown')})")
            else:
                self.console.print("  [dim]No active agents[/dim]")
                
        except Exception as e:
            self.console.print(f"[red]Failed to get status: {e}[/red]")
    
    async def _show_agents(self):
        """Show detailed agent information"""
        try:
            status = await self.orchestrator.get_agent_status()
            
            if not status:
                self.console.print("[yellow]No active agents[/yellow]")
                return
            
            from rich.table import Table
            
            table = Table(title="Active Agents")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Focus Areas", style="yellow")
            
            for agent_id, info in status.items():
                table.add_row(
                    info.get('name', 'Unknown'),
                    info.get('type', 'Unknown'),
                    info.get('status', 'Unknown'),
                    ', '.join(info.get('focus_areas', []))
                )
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Failed to get agents: {e}[/red]")
    
    async def _execute_natural_command(self, command: str):
        """Execute a natural language command"""
        # Check for agent prefix
        agent_type = None
        if command.startswith('@'):
            parts = command.split(maxsplit=1)
            if len(parts) >= 2:
                agent_prefix = parts[0][1:].lower()
                command = parts[1]
                
                if agent_prefix == 'claude':
                    agent_type = 'claude'
                elif agent_prefix == 'aider':
                    agent_type = 'aider'
                elif agent_prefix == 'all':
                    agent_type = 'mixed'
        
        # First, analyze if this is a query vs action
        from agentic.core.query_analyzer import QueryAnalyzer
        query_analyzer = QueryAnalyzer()
        query_analysis = await query_analyzer.analyze_query(command)
        
        # Log the analysis for debugging
        self.logger.info(f"Query analysis: type={query_analysis.query_type}, approach={query_analysis.suggested_approach}")
        
        # For task complexity, still use TaskAnalyzer
        from agentic.core.task_analyzer import TaskAnalyzer
        task_analyzer = TaskAnalyzer()
        task_analysis = await task_analyzer.analyze_task(command)
        
        # Show brief processing message - but not with Live for multi-agent to avoid display conflicts
        is_multi_agent = (query_analysis.suggested_approach in ["coordinated_analysis", "multi_agent_implementation"] or 
                         task_analysis.file_count_estimate > 3 or "all" in command.lower())
        
        # For single-agent question tasks, disable the swarm monitor
        is_question_task = (query_analysis.query_type in ["question", "explanation", "analysis"] and 
                           not is_multi_agent)
        
        # Store for use in finally block
        self._is_question_task = is_question_task
        
        # Show processing indicator for single-agent queries with Live display
        if is_question_task:
            from rich.live import Live
            from rich.spinner import Spinner
            from rich.text import Text
            from rich.table import Table
            from rich.panel import Panel
            import time
            import logging
            
            # Suppress verbose logging during query execution to prevent newline spam
            logging.getLogger('agentic').setLevel(logging.WARNING)
            
            # Create a simple status display for single-agent queries
            self._query_start_time = time.time()
            self._query_status = "Initializing agent..."
            
            def create_query_display():
                """Create the display for query processing"""
                table = Table.grid(padding=1)
                table.add_column(style="cyan", no_wrap=True)
                table.add_column(style="white")
                
                elapsed = int(time.time() - self._query_start_time)
                table.add_row(
                    Spinner("dots", style="cyan"),
                    f"[cyan]{self._query_status}[/cyan] ({elapsed}s)"
                )
                
                return Panel(table, title="🤔 Processing Query", border_style="blue")
            
            self._query_live = Live(create_query_display(), refresh_per_second=2)
            self._query_live.start()
            
            # Update function for status changes
            def update_query_status(new_status: str):
                self._query_status = new_status
                if hasattr(self, '_query_live') and self._query_live:
                    self._query_live.update(create_query_display())
            
            self._update_query_status = update_query_status
        
        if is_multi_agent:
            # Just print a simple message for multi-agent to avoid interfering with swarm monitor
            self.console.print(Panel(
                Text.from_markup(f"[cyan]Processing:[/cyan] {command[:60]}...", justify="left"),
                title="🤖 Agentic",
                border_style="blue"
            ))
            
        try:
            # Execute based on query analysis
            if query_analysis.suggested_approach == "coordinated_analysis":
                # Coordinated analysis for complex queries
                context = {
                    'agent_type_strategy': agent_type or 'dynamic',
                    'coordination_type': 'analysis'
                }
                result = await self.orchestrator.execute_multi_agent_command(command, context)
                
            elif query_analysis.suggested_approach == "multi_agent_implementation" or \
                 task_analysis.file_count_estimate > 3 or "all" in command.lower():
                # Multi-agent execution for implementation
                context = {'agent_type_strategy': agent_type or 'dynamic'}
                result = await self.orchestrator.execute_multi_agent_command(command, context)
                
            else:
                # Single agent execution
                # Use suggested agent if available and no override
                if query_analysis.suggested_agent and not agent_type:
                    # Route to specific agent type
                    context = {
                        'preferred_agent': query_analysis.suggested_agent,
                        'enable_monitoring': False,  # Disable for simple questions
                        'status_updater': self._update_query_status if hasattr(self, '_update_query_status') else None
                    }
                    result = await self.orchestrator.execute_command(command, context)
                else:
                    # Disable full swarm monitoring for simple questions to avoid screen clearing
                    context = {
                        'enable_monitoring': False,
                        'status_updater': self._update_query_status if hasattr(self, '_update_query_status') else None
                    }
                    result = await self.orchestrator.execute_command(command, context)
            
            # Stop live display before showing results
            if hasattr(self, '_query_live') and self._query_live:
                self._query_live.stop()
                self._query_live = None
            
            # Display results
            if result:
                if result.status == "completed":
                    # For ExecutionResult (multi-agent), show formatted summary
                    if hasattr(result, 'task_results'):
                        # Use the execution summary formatter
                        from agentic.core.execution_summary import ExecutionSummaryFormatter
                        formatter = ExecutionSummaryFormatter()
                            
                        # Get agent sessions for context
                        agent_sessions = {}
                        if hasattr(self.orchestrator, 'agent_registry'):
                            for agent_id in set(tr.agent_id for tr in result.task_results.values()):
                                agent = self.orchestrator.agent_registry.get_agent_by_id(agent_id)
                                if agent:
                                    # Handle different agent types
                                    if hasattr(agent, 'agent_config'):
                                        # Claude Code agents have agent_config
                                        agent_sessions[agent_id] = {
                                            'agent_config': {
                                                'name': agent.agent_config.name,
                                                'agent_type': {'value': agent.agent_config.agent_type.value}
                                            }
                                        }
                                    elif hasattr(agent, 'config'):
                                        # Aider agents have config (not agent_config)
                                        agent_sessions[agent_id] = {
                                            'agent_config': {
                                                'name': agent.config.name,
                                                'agent_type': {'value': agent.config.agent_type.value}
                                            }
                                        }
                                    else:
                                        # Fallback for other agent types
                                        agent_name = getattr(agent, 'name', 'Unknown')
                                        agent_type = 'unknown'
                                            
                                        agent_sessions[agent_id] = {
                                            'agent_config': {
                                                'name': agent_name,
                                                'agent_type': {'value': agent_type}
                                            }
                                        }
                            
                        # Display formatted summary
                        formatter.format_execution_summary(result, agent_sessions)
                        
                    # For single TaskResult, show output directly
                    elif hasattr(result, 'output') and result.output:
                        self.console.print("\n[bold green]✅ Task completed successfully![/bold green]")
                            
                        # Check if output is JSON (Claude Code format) 
                        output_stripped = result.output.strip()
                        if (output_stripped.startswith('{') or output_stripped.startswith('[')) and hasattr(result, 'agent_id') and 'claude' in str(result.agent_id).lower():
                            # Use our enhanced output parser for Claude agents
                            from agentic.agents.output_parser import AgentOutputParser
                            parser = AgentOutputParser()
                            parsed = parser.parse_output(result.output, 'claude_code')
                            
                            # Show Claude's thinking process
                            thinking = parsed.get('metadata', {}).get('thinking_process', [])
                            if thinking:
                                self.console.print("\n[bold cyan]Claude's Process:[/bold cyan]")
                                for i, thought in enumerate(thinking[:5]):
                                    self.console.print(f"  {i+1}. {thought}")
                                if len(thinking) > 5:
                                    self.console.print(f"  ... and {len(thinking) - 5} more steps\n")
                            
                            # Show actions taken
                            if parsed['actions']:
                                self.console.print("[bold yellow]Actions:[/bold yellow]")
                                for action in parsed['actions'][:8]:
                                    self.console.print(f"  ✓ {action}")
                                if len(parsed['actions']) > 8:
                                    self.console.print(f"  ... and {len(parsed['actions']) - 8} more actions")
                                self.console.print()
                            
                            # Show the response/summary
                            if parsed['summary']:
                                self.console.print("[bold green]Answer:[/bold green]")
                                # Format the summary nicely
                                if any(marker in parsed['summary'] for marker in ['```', '#', '*', '-']):
                                    self.console.print(Markdown(parsed['summary']))
                                else:
                                    # Wrap long lines for readability
                                    import textwrap
                                    wrapped = textwrap.fill(parsed['summary'], width=80)
                                    self.console.print(wrapped)
                            
                            # Show any errors
                            if parsed['errors']:
                                self.console.print("\n[bold red]Errors:[/bold red]")
                                for error in parsed['errors']:
                                    self.console.print(f"  ❌ {error}")
                            
                            # Show usage stats if available
                            try:
                                import json
                                data = json.loads(result.output)
                                if 'usage' in data:
                                    usage = data['usage']
                                    self.console.print("\n[dim]Statistics:[/dim]")
                                    if 'claude_api_calls' in usage:
                                        self.console.print(f"  [dim]Turns: {usage['claude_api_calls']}[/dim]")
                                    if 'total_tokens' in usage:
                                        self.console.print(f"  [dim]Tokens: {usage['total_tokens']:,}[/dim]")
                                    if 'cache_read_tokens' in usage and usage['cache_read_tokens'] > 0:
                                        cache_pct = (usage['cache_read_tokens'] / usage['total_tokens']) * 100
                                        self.console.print(f"  [dim]Cache Hit: {cache_pct:.1f}%[/dim]")
                            except:
                                pass
                                
                        else:
                            # For non-Claude agents or non-JSON output
                            # Skip if we already processed Claude JSON above
                            if not ((output_stripped.startswith('{') or output_stripped.startswith('[')) and 
                                   hasattr(result, 'agent_id') and 'claude' in str(result.agent_id).lower()):
                                # Process non-Claude JSON or plain text output
                                if output_stripped.startswith('{') or output_stripped.startswith('['):
                                    try:
                                        # Try to pretty-print JSON
                                        import json
                                        data = json.loads(result.output)
                                        self.console.print(json.dumps(data, indent=2))
                                    except:
                                        # Not valid JSON, show as-is
                                        if any(marker in result.output for marker in ['```', '#', '*', '-']):
                                            self.console.print(Markdown(result.output))
                                        else:
                                            self.console.print(result.output)
                                else:
                                    # Plain text output
                                    if any(marker in result.output for marker in ['```', '#', '*', '-']):
                                        self.console.print(Markdown(result.output))
                                    else:
                                        self.console.print(result.output)
                else:
                    # Handle both ExecutionResult and TaskResult errors
                    error_msg = None
                        
                    # For ExecutionResult, extract error from failed tasks
                    if hasattr(result, 'task_results') and result.task_results:
                        for task_id, task_result in result.task_results.items():
                            if task_result.status == "failed" and task_result.error:
                                error_msg = task_result.error
                                break
                        
                    # For TaskResult or other results with direct error field
                    if not error_msg:
                        error_msg = getattr(result, 'error', None)
                        
                    # Final fallback
                    if not error_msg:
                        error_msg = 'Task execution failed'
                        
                    self.console.print(f"\n[bold red]❌ Task failed: {error_msg}[/bold red]")
                        
                    # Show failed task details for multi-agent results
                    if hasattr(result, 'task_results'):
                        for task_id, task_result in result.task_results.items():
                            if task_result.status == "failed":
                                error_detail = task_result.error or "No error details available"
                                self.console.print(f"[red]Task {task_id[:8]}...: {error_detail}[/red]")
            else:
                self.console.print("\n[yellow]No result returned[/yellow]")
                    
        except Exception as e:
            self.console.print(f"\n[red]❌ Error: {e}[/red]")
            if self.debug:
                self.console.print_exception()
        finally:
            # Stop the query live display if it exists
            if hasattr(self, '_query_live') and self._query_live:
                self._query_live.stop()
                self._query_live = None
            
            # Restore logging level if it was changed
            if hasattr(self, '_is_question_task') and self._is_question_task:
                import logging
                logging.getLogger('agentic').setLevel(logging.INFO)
    
    async def _spawn_agent(self, args: list):
        """Spawn a specific agent"""
        if not args:
            self.console.print("[yellow]Usage: /spawn <agent_type>[/yellow]")
            self.console.print("[dim]Available types: claude_code, aider_backend, aider_frontend, aider_testing[/dim]")
            return
        
        agent_type = args[0]
        # Implementation would spawn the agent
        self.console.print(f"[yellow]Spawning {agent_type} agent...[/yellow]")
    
    async def _analyze_command(self, args: list):
        """Analyze a command without executing"""
        if not args:
            self.console.print("[yellow]Usage: /analyze <command>[/yellow]")
            return
        
        command = ' '.join(args)
        
        from agentic.core.task_analyzer import TaskAnalyzer
        analyzer = TaskAnalyzer()
        analysis = await analyzer.analyze_task(command)
        
        # Display analysis
        from rich.table import Table
        
        table = Table(title="Task Analysis")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Estimated Files", str(analysis.file_count_estimate))
        table.add_row("Complexity", f"{analysis.complexity_score:.2f}")
        table.add_row("Requires Creativity", "Yes" if analysis.requires_creativity else "No")
        table.add_row("Ambiguity Level", f"{analysis.ambiguity_level:.2f}")
        table.add_row("Suggested Agent", analysis.suggested_agent)
        
        self.console.print(table)
        
        if analysis.reasoning:
            self.console.print("\n[bold]Reasoning:[/bold]")
            for reason in analysis.reasoning:
                self.console.print(f"  • {reason}")
    
    async def _set_model(self, model: str):
        """Set the AI model"""
        # Implementation would update the model
        self.console.print(f"[yellow]Model set to: {model}[/yellow]")
    
    async def _show_model(self):
        """Show current model configuration"""
        self.console.print(f"[cyan]Primary Model:[/cyan] {self.config.primary_model}")
        self.console.print(f"[cyan]Fallback Model:[/cyan] {self.config.fallback_model}")
    
    def _show_history(self):
        """Show command history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = f.readlines()[-20:]  # Last 20 commands
                
            self.console.print("\n[bold]Recent Commands:[/bold]")
            for i, cmd in enumerate(history, 1):
                self.console.print(f"  {i}. {cmd.strip()}")
        else:
            self.console.print("[yellow]No command history found[/yellow]")