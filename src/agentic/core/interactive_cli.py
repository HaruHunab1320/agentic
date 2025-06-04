"""
Phase 4: Enhanced CLI Experience
Rich interactive CLI with real-time updates, progress tracking, and split-screen views
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.tree import Tree
from rich.syntax import Syntax
import click


@dataclass
class AgentSession:
    """Agent session information for display"""
    agent_id: str
    name: str
    agent_type: str
    status: str = "inactive"  # active, busy, error, inactive
    current_task: Optional[str] = None
    current_tasks: List[str] = field(default_factory=list)
    last_activity: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_status_color(self) -> str:
        """Get color for status display"""
        return {
            "active": "green",
            "busy": "yellow", 
            "error": "red",
            "inactive": "dim"
        }.get(self.status, "white")


@dataclass
class CommandHistoryEntry:
    """Command history entry"""
    command: str
    timestamp: datetime
    agent_target: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    output: Optional[str] = None
    
    
@dataclass
class TaskProgress:
    """Task progress information"""
    task_id: str
    description: str
    total_steps: int
    completed_steps: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    status_message: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100
    
    @property
    def eta(self) -> Optional[timedelta]:
        """Calculate estimated time remaining"""
        if self.completed_steps == 0:
            return None
        
        elapsed = datetime.utcnow() - self.start_time
        time_per_step = elapsed.total_seconds() / self.completed_steps
        remaining_steps = self.total_steps - self.completed_steps
        
        return timedelta(seconds=time_per_step * remaining_steps)


class ProgressTracker:
    """Advanced progress tracking with ETA"""
    
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        self.tasks: Dict[str, TaskProgress] = {}
        self.rich_tasks: Dict[str, Any] = {}
    
    def add_task(self, task_id: str, description: str, total_steps: int = 100) -> str:
        """Add a task to track"""
        task_progress = TaskProgress(
            task_id=task_id,
            description=description,
            total_steps=total_steps
        )
        
        rich_task = self.progress.add_task(description, total=total_steps)
        
        self.tasks[task_id] = task_progress
        self.rich_tasks[task_id] = rich_task
        
        return task_id
    
    def update_task(self, task_id: str, completed_steps: int, status_message: Optional[str] = None):
        """Update task progress"""
        if task_id not in self.tasks:
            return
        
        task_info = self.tasks[task_id]
        task_info.completed_steps = completed_steps
        task_info.status_message = status_message
        
        description = task_info.description
        if status_message:
            description = f"{task_info.description} - {status_message}"
        
        self.progress.update(
            self.rich_tasks[task_id],
            completed=completed_steps,
            description=description
        )
    
    def complete_task(self, task_id: str):
        """Mark task as completed"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            self.update_task(task_id, task_info.total_steps, "Complete")
    
    def remove_task(self, task_id: str):
        """Remove a task from tracking"""
        if task_id in self.tasks:
            self.progress.remove_task(self.rich_tasks[task_id])
            del self.tasks[task_id]
            del self.rich_tasks[task_id]
    
    def get_progress_display(self) -> Panel:
        """Get progress display panel"""
        if not self.tasks:
            return Panel("No active tasks", title="Progress")
        
        return Panel(self.progress, title="Task Progress")


class InteractiveCLI:
    """Rich interactive CLI with real-time updates"""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.active_agents: Dict[str, AgentSession] = {}
        self.progress_tracker = ProgressTracker()
        self.command_history: List[CommandHistoryEntry] = []
        self.current_output: List[str] = []
        self.is_running = False
        
        # Performance tracking
        self.session_start = datetime.utcnow()
        self.commands_executed = 0
        self.errors_encountered = 0
        
    def setup_layout(self):
        """Setup split-screen layout"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=8)
        )
        
        self.layout["main"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="right_panel", ratio=2)
        )
        
        self.layout["left_panel"].split_column(
            Layout(name="agent_status", ratio=1),
            Layout(name="progress", ratio=1)
        )
        
        self.layout["right_panel"].split_column(
            Layout(name="output", ratio=2),
            Layout(name="command_history", ratio=1)
        )
    
    async def run_interactive_session(self):
        """Run interactive TUI session"""
        self.setup_layout()
        self.is_running = True
        
        with Live(self.layout, console=self.console, refresh_per_second=4) as live:
            try:
                while self.is_running:
                    await self.update_display()
                    await asyncio.sleep(0.25)
            except KeyboardInterrupt:
                self.is_running = False
                self.console.print("\n[yellow]Interactive session ended[/yellow]")
    
    async def update_display(self):
        """Update all display components"""
        # Update header
        self.layout["header"].update(self.create_header())
        
        # Update agent status
        self.layout["agent_status"].update(self.create_agent_status_panel())
        
        # Update progress
        self.layout["progress"].update(self.progress_tracker.get_progress_display())
        
        # Update main output
        self.layout["output"].update(self.create_main_output_panel())
        
        # Update command history
        self.layout["command_history"].update(self.create_command_history_panel())
    
    def create_header(self) -> Panel:
        """Create header with session information"""
        session_duration = datetime.utcnow() - self.session_start
        duration_str = str(session_duration).split('.')[0]  # Remove microseconds
        
        header_text = Text()
        header_text.append("🤖 Agentic Interactive CLI", style="bold blue")
        header_text.append(f" | Session: {duration_str}", style="dim")
        header_text.append(f" | Commands: {self.commands_executed}", style="dim")
        header_text.append(f" | Agents: {len(self.active_agents)}", style="dim")
        
        if self.errors_encountered > 0:
            header_text.append(f" | Errors: {self.errors_encountered}", style="red")
        
        return Panel(Align.center(header_text), border_style="blue")
    
    def create_agent_status_panel(self) -> Panel:
        """Create agent status table"""
        if not self.active_agents:
            empty_message = Text("No active agents\nRun 'agentic init' to start", style="dim")
            return Panel(Align.center(empty_message), title="Agent Status")
        
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Agent", style="cyan", width=12)
        table.add_column("Type", width=10)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Current Task", width=20)
        table.add_column("Load", justify="center", width=6)
        
        for agent_id, agent_session in self.active_agents.items():
            status_color = agent_session.get_status_color()
            
            # Calculate load bar
            load_percentage = min(len(agent_session.current_tasks) * 20, 100)  # Assume 5 max tasks
            load_bar = "█" * (load_percentage // 10) + "░" * (10 - load_percentage // 10)
            
            current_task = agent_session.current_task
            if current_task and len(current_task) > 20:
                current_task = current_task[:17] + "..."
            
            table.add_row(
                agent_session.name,
                agent_session.agent_type.replace("_", " ").title(),
                f"[{status_color}]{agent_session.status}[/{status_color}]",
                current_task or "-",
                f"[blue]{load_bar}[/blue]"
            )
        
        return Panel(table, title="Agent Status")
    
    def create_main_output_panel(self) -> Panel:
        """Create main output display"""
        if not self.current_output:
            empty_message = Text("No output yet\nExecute commands to see results here", style="dim")
            return Panel(Align.center(empty_message), title="Execution Output")
        
        # Show last 20 lines of output
        output_lines = self.current_output[-20:]
        output_text = "\n".join(output_lines)
        
        # If output is code, try to syntax highlight
        if any(line.strip().startswith(('def ', 'class ', 'import ', 'from ')) for line in output_lines):
            try:
                syntax = Syntax(output_text, "python", theme="monokai", line_numbers=False)
                return Panel(syntax, title="Execution Output")
            except:
                pass
        
        return Panel(output_text, title="Execution Output")
    
    def create_command_history_panel(self) -> Panel:
        """Create command history display"""
        if not self.command_history:
            empty_message = Text("No commands executed yet", style="dim")
            return Panel(Align.center(empty_message), title="Command History")
        
        # Show last 5 commands
        recent_commands = self.command_history[-5:]
        
        history_content = []
        for i, cmd in enumerate(recent_commands, 1):
            timestamp = cmd.timestamp.strftime("%H:%M:%S")
            status_icon = {
                "pending": "⏳",
                "running": "🔄", 
                "completed": "✅",
                "failed": "❌"
            }.get(cmd.status, "❓")
            
            command_text = cmd.command
            if len(command_text) > 40:
                command_text = command_text[:37] + "..."
            
            line = f"{status_icon} [{timestamp}] {command_text}"
            history_content.append(line)
        
        return Panel("\n".join(history_content), title="Recent Commands")
    
    def add_agent(self, agent_id: str, name: str, agent_type: str):
        """Add an agent to the display"""
        self.active_agents[agent_id] = AgentSession(
            agent_id=agent_id,
            name=name,
            agent_type=agent_type,
            status="active",
            last_activity=datetime.utcnow()
        )
    
    def update_agent_status(self, agent_id: str, status: str, current_task: Optional[str] = None):
        """Update agent status"""
        if agent_id in self.active_agents:
            self.active_agents[agent_id].status = status
            self.active_agents[agent_id].current_task = current_task
            self.active_agents[agent_id].last_activity = datetime.utcnow()
    
    def add_agent_task(self, agent_id: str, task: str):
        """Add a task to an agent"""
        if agent_id in self.active_agents:
            self.active_agents[agent_id].current_tasks.append(task)
    
    def remove_agent_task(self, agent_id: str, task: str):
        """Remove a task from an agent"""
        if agent_id in self.active_agents and task in self.active_agents[agent_id].current_tasks:
            self.active_agents[agent_id].current_tasks.remove(task)
    
    def add_command(self, command: str, agent_target: Optional[str] = None) -> str:
        """Add a command to history"""
        command_id = str(uuid.uuid4())
        
        self.command_history.append(CommandHistoryEntry(
            command=command,
            timestamp=datetime.utcnow(),
            agent_target=agent_target,
            status="pending"
        ))
        
        self.commands_executed += 1
        return command_id
    
    def update_command_status(self, command: str, status: str, output: Optional[str] = None):
        """Update command status"""
        # Find the most recent matching command
        for cmd in reversed(self.command_history):
            if cmd.command == command:
                cmd.status = status
                cmd.output = output
                break
        
        if status == "failed":
            self.errors_encountered += 1
    
    def add_output(self, text: str):
        """Add text to output display"""
        lines = text.split('\n')
        self.current_output.extend(lines)
        
        # Keep output manageable
        if len(self.current_output) > 200:
            self.current_output = self.current_output[-100:]
    
    def clear_output(self):
        """Clear output display"""
        self.current_output.clear()
    
    def create_command_autocomplete(self) -> List[str]:
        """Create command autocompletion suggestions"""
        base_commands = [
            "init", "analyze", "spawn", "status", "stop", "history", 
            "config", "debug", "performance", "help", "interactive"
        ]
        
        # Add recent commands (unique)
        recent_commands = []
        seen = set(base_commands)
        for cmd in reversed(self.command_history):
            if cmd.command not in seen and len(recent_commands) < 10:
                recent_commands.append(cmd.command)
                seen.add(cmd.command)
        
        # Add common patterns
        common_patterns = [
            "fix the bug in",
            "add feature for", 
            "refactor the",
            "debug the issue with",
            "explain how",
            "write tests for",
            "optimize the performance of",
            "review the code in",
            "implement authentication for",
            "add error handling to"
        ]
        
        return base_commands + recent_commands + common_patterns
    
    def stop(self):
        """Stop the interactive session"""
        self.is_running = False


# Command autocompletion setup
def setup_click_completion():
    """Setup click command completion"""
    try:
        import click_completion
        click_completion.init()
    except ImportError:
        pass  # click-completion not available


class CLICommandHandler:
    """Handles CLI commands in interactive mode"""
    
    def __init__(self, interactive_cli: InteractiveCLI):
        self.cli = interactive_cli
        self.console = Console()
    
    async def handle_command(self, command: str) -> bool:
        """Handle a command in interactive mode"""
        command = command.strip()
        
        if not command:
            return True
        
        # Add to history
        self.cli.add_command(command)
        
        # Handle built-in commands
        if command.lower() in ('exit', 'quit', 'q'):
            self.cli.stop()
            return False
        
        if command.lower() == 'clear':
            self.cli.clear_output()
            return True
        
        if command.lower() == 'help':
            self._show_help()
            return True
        
        if command.lower().startswith('status'):
            await self._handle_status_command()
            return True
        
        # For other commands, we would integrate with the main orchestrator
        self.cli.add_output(f"Executing: {command}")
        self.cli.update_command_status(command, "running")
        
        # Simulate command execution
        await asyncio.sleep(1)
        
        self.cli.add_output(f"Command completed: {command}")
        self.cli.update_command_status(command, "completed")
        
        return True
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Interactive CLI Commands:
  
  Agent Management:
    init              Initialize Agentic in current directory
    status            Show agent status
    spawn <type>      Spawn a specific agent type
    stop [agent]      Stop agent(s)
  
  Analysis:
    analyze           Analyze current codebase
    performance       Show performance metrics
    debug <agent>     Debug specific agent
  
  Configuration:
    config show       Show current configuration
    config set <key> <value>  Set configuration value
  
  Execution:
    <natural language>  Execute natural language commands
  
  Control:
    clear             Clear output
    help              Show this help
    exit/quit/q       Exit interactive mode
        """
        
        self.cli.add_output(help_text.strip())
    
    async def _handle_status_command(self):
        """Handle status command"""
        if not self.cli.active_agents:
            self.cli.add_output("No active agents")
            return
        
        status_text = "Agent Status:\n"
        for agent_id, agent in self.cli.active_agents.items():
            status_text += f"  {agent.name} ({agent.agent_type}): {agent.status}\n"
            if agent.current_task:
                status_text += f"    Current task: {agent.current_task}\n"
        
        self.cli.add_output(status_text)


# Global interactive CLI instance
interactive_cli = InteractiveCLI()


# Click integration for interactive mode
@click.command()
@click.pass_context
def interactive(ctx: click.Context):
    """Launch interactive CLI mode"""
    click.echo("🚀 Starting Agentic Interactive CLI...")
    click.echo("Press Ctrl+C to exit")
    
    # Setup completion if available
    setup_click_completion()
    
    # Initialize with mock agents for demo
    interactive_cli.add_agent("agent-1", "Backend Agent", "backend")
    interactive_cli.add_agent("agent-2", "Frontend Agent", "frontend") 
    interactive_cli.add_agent("agent-3", "Test Agent", "testing")
    
    # Start interactive session
    try:
        asyncio.run(interactive_cli.run_interactive_session())
    except KeyboardInterrupt:
        click.echo("\n👋 Interactive session ended") 