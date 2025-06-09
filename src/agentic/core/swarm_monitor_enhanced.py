"""
Enhanced real-time monitoring for agent swarm execution with grid layout
Provides live status updates, task-based progress tracking, and activity streaming
"""

import asyncio
import sys
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns

from agentic.utils.logging import LoggerMixin


class AgentStatus(str, Enum):
    """Agent execution status"""
    INITIALIZING = "initializing"
    SETTING_UP = "setting_up"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    WRITING_FILES = "writing_files"
    TESTING = "testing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    IDLE = "idle"


@dataclass
class TaskInfo:
    """Information about a task"""
    task_id: str
    task_name: str
    description: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "pending"
    
    @property
    def is_complete(self) -> bool:
        return self.status in ["completed", "failed"]
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class AgentMonitorInfo:
    """Enhanced information about a monitored agent"""
    agent_id: str
    agent_name: str
    agent_type: str
    role: str
    status: AgentStatus = AgentStatus.INITIALIZING
    current_task: Optional[str] = None
    current_activity: Optional[str] = None
    
    # Task-based tracking
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    task_queue: List[TaskInfo] = field(default_factory=list)
    active_task: Optional[TaskInfo] = None
    
    # Activity streaming
    activity_buffer: List[str] = field(default_factory=list)  # Last N activities
    last_activity_time: float = field(default_factory=time.time)
    
    # Timing
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    # Files and errors
    files_modified: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def progress(self) -> int:
        """Calculate progress based on task completion"""
        if self.total_tasks == 0:
            # If no tasks assigned yet, use status-based progress
            status_progress = {
                AgentStatus.INITIALIZING: 5,
                AgentStatus.SETTING_UP: 10,
                AgentStatus.ANALYZING: 15,
                AgentStatus.PLANNING: 20,
                AgentStatus.EXECUTING: 50,
                AgentStatus.WRITING_FILES: 70,
                AgentStatus.TESTING: 85,
                AgentStatus.FINALIZING: 95,
                AgentStatus.COMPLETED: 100,
                AgentStatus.FAILED: 100,
                AgentStatus.IDLE: 0
            }
            return status_progress.get(self.status, 0)
        
        # Calculate based on task completion
        if self.total_tasks > 0:
            completion_ratio = (self.completed_tasks + self.failed_tasks) / self.total_tasks
            return int(completion_ratio * 100)
        return 0
    
    @property
    def runtime(self) -> float:
        """Get total runtime in seconds"""
        return time.time() - self.start_time


class SwarmMonitorEnhanced(LoggerMixin):
    """Enhanced real-time monitoring with grid layout for agent swarm execution"""
    
    def __init__(self, use_alternate_screen: bool = True):
        super().__init__()
        self.console = Console()
        self.agents: Dict[str, AgentMonitorInfo] = {}
        self.overall_start_time = time.time()
        self.live_display: Optional[Live] = None
        self._update_task = None
        self.use_alternate_screen = use_alternate_screen
        
        # Task analysis info
        self.task_analysis = {
            "total_files": 0,
            "complexity": 0.0,
            "suggested_agents": []
        }
        
        # Activity buffer size
        self.activity_buffer_size = 3
        
    async def start_monitoring(self):
        """Start the monitoring display"""
        if self.live_display is not None:
            return
            
        # Get terminal size
        import shutil
        terminal_size = shutil.get_terminal_size()
        console_width = max(terminal_size.columns, 150)
        
        self.monitor_console = Console(
            file=sys.stderr,
            force_terminal=True,
            force_interactive=True,
            legacy_windows=False,
            width=console_width,
            _environ={"TERM": "xterm-256color"}
        )
        
        if self.use_alternate_screen:
            sys.stderr.write("\033[?1049h")  # Enter alternate screen
            sys.stderr.write("\033[2J\033[H")  # Clear screen
            sys.stderr.flush()
        
        self.live_display = True
        self._update_task = asyncio.create_task(self._update_loop())
        
    async def stop_monitoring(self):
        """Stop the monitoring display"""
        self.live_display = None
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self.use_alternate_screen:
            sys.stderr.write("\033[?1049l")  # Exit alternate screen
            sys.stderr.flush()
        
        await asyncio.sleep(0.1)
        self._show_final_summary()
    
    async def _update_loop(self):
        """Update display with grid layout using carriage returns"""
        while self.live_display:
            try:
                # Clear screen and move cursor to home position
                sys.stderr.write("\033[2J\033[H")
                sys.stderr.flush()
                
                display = self._create_grid_display()
                
                # Capture output to string first to avoid extra newlines
                from io import StringIO
                string_buffer = StringIO()
                temp_console = Console(file=string_buffer, force_terminal=True, width=self.monitor_console.width)
                temp_console.print(display)
                output = string_buffer.getvalue()
                
                # Write output directly without extra newlines
                sys.stderr.write(output.rstrip())  # Remove trailing newlines
                sys.stderr.flush()
                
                await asyncio.sleep(0.5)  # Update twice per second for smoother activity
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error updating display: {e}")
                await asyncio.sleep(1.0)
    
    def register_agent(self, agent_id: str, agent_name: str, agent_type: str, role: str):
        """Register a new agent for monitoring"""
        agent_info = AgentMonitorInfo(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            role=role
        )
        self.agents[agent_id] = agent_info
        self.logger.info(f"Registered agent for monitoring: {agent_name} ({role})")
    
    def set_agent_tasks(self, agent_id: str, tasks: List[Tuple[str, str]]):
        """Set the list of tasks for an agent (id, description)"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.task_queue = [
                TaskInfo(task_id=tid, task_name=tname, description=tname)
                for tid, tname in tasks
            ]
            agent.total_tasks = len(tasks)
            self.logger.info(f"Assigned {len(tasks)} tasks to {agent.role}")
    
    def start_agent_task(self, agent_id: str, task_id: str, task_name: str):
        """Mark a task as started"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Find task in queue
            for task in agent.task_queue:
                if task.task_id == task_id:
                    task.status = "in_progress"
                    task.start_time = time.time()
                    agent.active_task = task
                    agent.current_task = task_name
                    break
            
            agent.status = AgentStatus.EXECUTING
            agent.last_update = time.time()
    
    def complete_agent_task(self, agent_id: str, task_id: str, success: bool = True):
        """Mark a task as completed"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Find and update task
            for task in agent.task_queue:
                if task.task_id == task_id:
                    task.status = "completed" if success else "failed"
                    task.end_time = time.time()
                    
                    if success:
                        agent.completed_tasks += 1
                    else:
                        agent.failed_tasks += 1
                    
                    # Clear active task if it matches
                    if agent.active_task and agent.active_task.task_id == task_id:
                        agent.active_task = None
                    break
            
            agent.last_update = time.time()
            
            # Update status if all tasks complete
            if agent.completed_tasks + agent.failed_tasks >= agent.total_tasks:
                agent.status = AgentStatus.COMPLETED if agent.failed_tasks == 0 else AgentStatus.FAILED
    
    def update_agent_activity(self, agent_id: str, activity: str):
        """Stream activity updates from agent output"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Add to activity buffer
            agent.activity_buffer.append(activity)
            if len(agent.activity_buffer) > self.activity_buffer_size:
                agent.activity_buffer.pop(0)
            
            agent.current_activity = activity
            agent.last_activity_time = time.time()
            agent.last_update = time.time()
    
    def update_agent_status(self, agent_id: str, status: AgentStatus, current_task: Optional[str] = None):
        """Update agent status"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = status
            if current_task is not None:
                agent.current_task = current_task
            agent.last_update = time.time()
    
    def add_file_modified(self, agent_id: str, file_path: str):
        """Record a file modified by an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].files_modified.append(file_path)
    
    def add_error(self, agent_id: str, error: str):
        """Record an error for an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].errors.append(error)
    
    def update_task_analysis(self, total_files: int, complexity: float, suggested_agents: List[str]):
        """Update the task analysis information"""
        self.task_analysis = {
            "total_files": total_files,
            "complexity": complexity,
            "suggested_agents": suggested_agents
        }
    
    def _create_grid_display(self) -> Group:
        """Create grid layout display matching the reference image"""
        components = []
        
        # Header - minimal
        header_text = Text("Agent Swarm Execution", style="bold white", justify="center")
        components.append(Align.center(header_text))
        components.append(Text())  # Spacer
        
        # Create agent grid
        if self.agents:
            # Calculate grid dimensions
            num_agents = len(self.agents)
            cols = min(3, num_agents)  # Max 3 columns
            rows = math.ceil(num_agents / cols)
            
            # Create panels for each agent
            agent_panels = []
            for agent in self.agents.values():
                panel = self._create_agent_panel_enhanced(agent)
                agent_panels.append(panel)
            
            # Arrange in rows
            for row in range(rows):
                start_idx = row * cols
                end_idx = min(start_idx + cols, num_agents)
                row_panels = agent_panels[start_idx:end_idx]
                
                # Create columns for this row
                if row_panels:
                    columns = Columns(row_panels, equal=True, expand=True)
                    components.append(columns)
                    components.append(Text())  # Spacer between rows
        
        # Add execution summary at bottom
        summary = self._create_execution_summary()
        components.append(summary)
        
        return Group(*components)
    
    def _create_agent_panel_enhanced(self, agent: AgentMonitorInfo) -> Panel:
        """Create enhanced panel for a single agent matching reference style"""
        # Determine panel color based on agent type
        agent_colors = {
            "claude_code": "yellow",
            "aider_frontend": "cyan",
            "aider_backend": "magenta",
            "aider_testing": "green",
            "aider": "blue"
        }
        
        # Get color based on agent type
        panel_color = "yellow"  # Default
        for key, color in agent_colors.items():
            if key in agent.agent_type.lower():
                panel_color = color
                break
        
        # Status with emoji
        status_display = f"⚡ Status: [bold {panel_color}]{agent.status.value}[/bold {panel_color}]"
        
        # Progress bar with percentage
        progress = agent.progress
        filled = int(progress / 5)  # 20 chars total
        empty = 20 - filled
        progress_bar = f"Progress: [{'█' * filled}{'░' * empty}] [bold]{progress}%[/bold]"
        
        # Current task/activity
        content_lines = [
            status_display,
            progress_bar,
            "",  # Spacer
        ]
        
        # Show current activity with emoji
        if agent.current_activity:
            # Truncate activity to fit
            activity = agent.current_activity[:60] + "..." if len(agent.current_activity) > 63 else agent.current_activity
            content_lines.append(f"🚀 Current: [cyan]{activity}[/cyan]")
        elif agent.current_task:
            task = agent.current_task[:60] + "..." if len(agent.current_task) > 63 else agent.current_task
            content_lines.append(f"📌 Current: [cyan]{task}[/cyan]")
        else:
            content_lines.append("📌 Current: [dim]Waiting for tasks...[/dim]")
        
        # Runtime
        runtime_str = self._format_duration(agent.runtime)
        content_lines.append(f"⏱  Runtime: {runtime_str}")
        
        # Task completion info if available
        if agent.total_tasks > 0:
            completion_str = f"{agent.completed_tasks}/{agent.total_tasks} tasks"
            content_lines.append(f"✅ Completed: {completion_str}")
        
        # Create panel title with agent info
        title = f"{agent.role.replace('_', ' ').title()} ({agent.agent_type})"
        
        return Panel(
            "\n".join(content_lines),
            title=title,
            title_align="center",
            border_style=panel_color,
            height=10,  # Fixed height for consistent grid
            padding=(1, 2)
        )
    
    def _create_execution_summary(self) -> Panel:
        """Create execution summary matching reference style"""
        total_runtime = time.time() - self.overall_start_time
        
        # Create summary content
        summary_lines = []
        
        # Total runtime with emoji
        summary_lines.append(f"[bold]📊 Execution Summary[/bold]")
        summary_lines.append(f"Total Runtime: [bold cyan]{self._format_duration(total_runtime)}[/bold cyan]")
        summary_lines.append("")
        
        # Task analysis section
        summary_lines.append("[bold]🔍 Task Analysis[/bold]")
        summary_lines.append(f"  Files: {self.task_analysis['total_files']}")
        summary_lines.append(f"  Complexity: {self.task_analysis['complexity']:.2f}")
        
        if self.task_analysis['suggested_agents']:
            agents_str = ", ".join(self.task_analysis['suggested_agents'])
            summary_lines.append(f"  Suggested: {agents_str}")
        
        # Active agents summary
        active = sum(1 for a in self.agents.values() if a.status == AgentStatus.EXECUTING)
        completed = sum(1 for a in self.agents.values() if a.status == AgentStatus.COMPLETED)
        total = len(self.agents)
        
        summary_lines.append("")
        summary_lines.append(f"[bold]👥 Agents[/bold]: {active} active, {completed} completed, {total} total")
        
        return Panel(
            "\n".join(summary_lines),
            border_style="bright_blue",
            padding=(1, 2)
        )
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def _show_final_summary(self):
        """Show final execution summary"""
        total_runtime = time.time() - self.overall_start_time
        
        # Create detailed summary
        self.console.print("\n[bold]🎯 Final Execution Summary[/bold]\n")
        
        # Agent summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Tasks", justify="center")
        table.add_column("Files", justify="center")
        table.add_column("Time", justify="right")
        
        for agent in self.agents.values():
            status_color = "green" if agent.status == AgentStatus.COMPLETED else "red"
            task_str = f"{agent.completed_tasks}/{agent.total_tasks}" if agent.total_tasks > 0 else "0/0"
            
            table.add_row(
                agent.role.replace('_', ' ').title(),
                f"[{status_color}]{agent.status.value}[/{status_color}]",
                task_str,
                str(len(agent.files_modified)),
                self._format_duration(agent.runtime)
            )
        
        self.console.print(table)
        self.console.print(f"\n[bold green]Total execution time: {self._format_duration(total_runtime)}[/bold green]")