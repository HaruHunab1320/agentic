"""
Real-time monitoring for agent swarm execution
Provides live status updates, progress tracking, and benchmarking
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.table import Table
from rich.text import Text

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
class TaskTimingInfo:
    """Timing information for a task"""
    task_id: str
    task_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"
    
    @property
    def duration(self) -> float:
        """Get task duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def duration_str(self) -> str:
        """Get formatted duration string"""
        duration = self.duration
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"


@dataclass
class AgentMonitorInfo:
    """Information about a monitored agent"""
    agent_id: str
    agent_name: str
    agent_type: str
    role: str
    status: AgentStatus = AgentStatus.INITIALIZING
    current_task: Optional[str] = None
    current_activity: Optional[str] = None  # Real-time activity description
    completed_tasks: List[str] = field(default_factory=list)
    task_timings: Dict[str, TaskTimingInfo] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    recent_activities: List[str] = field(default_factory=list)  # Last 5 activities
    last_update: float = field(default_factory=time.time)
    progress: int = 0  # 0-100
    files_created: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SwarmMonitor(LoggerMixin):
    """Real-time monitoring for agent swarm execution"""
    
    def __init__(self, use_alternate_screen: bool = True):
        super().__init__()
        self.console = Console()
        self.agents: Dict[str, AgentMonitorInfo] = {}
        self.overall_start_time = time.time()
        self.overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        )
        self.live_display: Optional[Live] = None
        self._update_task = None
        self.use_alternate_screen = use_alternate_screen
        
    async def start_monitoring(self):
        """Start the monitoring display"""
        # Only start if not already running
        if self.live_display is not None:
            return
            
        # Create a dedicated console for monitoring that writes to stderr
        # Get terminal size dynamically
        import shutil
        terminal_size = shutil.get_terminal_size()
        console_width = max(terminal_size.columns, 150)  # Min 150 for proper display
        
        self.monitor_console = Console(
            file=sys.stderr,
            force_terminal=True,
            force_interactive=True,
            legacy_windows=False,
            width=console_width,  # Dynamic width based on terminal
            _environ={"TERM": "xterm-256color"}
        )
        
        # Use alternate screen buffer to avoid interfering with main output
        if self.use_alternate_screen:
            sys.stderr.write("\033[?1049h")  # Enter alternate screen
            sys.stderr.write("\033[2J\033[H")  # Clear screen
            sys.stderr.flush()
        else:
            # Just add some spacing when not using alternate screen
            sys.stderr.write("\n" * 3)
            sys.stderr.flush()
        
        # Mark display as active
        self.live_display = True  # Just use a flag instead of Live object
        
        # Start update loop with manual rendering
        self._update_task = asyncio.create_task(self._update_loop())
        
    async def stop_monitoring(self):
        """Stop the monitoring display"""
        # Mark display as inactive
        self.live_display = None
        
        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        # Exit alternate screen buffer
        if self.use_alternate_screen:
            sys.stderr.write("\033[?1049l")  # Exit alternate screen
            sys.stderr.flush()
        else:
            # Just add some spacing when not using alternate screen
            sys.stderr.write("\n" * 2)
            sys.stderr.flush()
        
        # Give terminal time to recover
        await asyncio.sleep(0.1)
        
        # Show final summary using the regular console
        self._show_final_summary()
    
    async def _update_loop(self):
        """Continuously update the display with manual rendering using carriage returns"""
        last_output = ""
        update_count = 0
        
        while self.live_display:
            try:
                # Render the display components
                display = self._create_display()
                
                # Capture output to prevent overflow
                from io import StringIO
                string_buffer = StringIO()
                temp_console = Console(file=string_buffer, force_terminal=True, width=self.monitor_console.width)
                temp_console.print(display)
                output = string_buffer.getvalue()
                
                # Only update if content has changed or it's the first update
                if output != last_output or update_count == 0:
                    # Clear screen and move cursor to home position
                    sys.stderr.write("\033[2J\033[H")
                    sys.stderr.flush()
                    
                    # Write output directly without extra newlines
                    sys.stderr.write(output.rstrip())  # Remove trailing newlines
                    sys.stderr.flush()
                    
                    last_output = output
                    update_count += 1
                
                await asyncio.sleep(2.0)  # Update every 2 seconds for stability
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error updating display: {e}")
                # Continue running even if display fails
                await asyncio.sleep(1.0)
    
    def register_agent(self, agent_id: str, agent_name: str, agent_type: str, role: str):
        """Register a new agent for monitoring"""
        agent_info = AgentMonitorInfo(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            role=role
        )
        # Initialize task start time tracking
        agent_info._current_task_start = None
        self.agents[agent_id] = agent_info
        self.logger.info(f"Registered agent for monitoring: {agent_name} ({role})")
    
    def update_agent_activity(self, agent_id: str, activity: str):
        """Update the current activity of an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            # Truncate long activities for display
            display_activity = activity[:80] + "..." if len(activity) > 80 else activity
            agent.current_activity = display_activity
            agent.last_update = time.time()
            
            # Add to recent activities (keep last 5)
            timestamp = time.strftime('%H:%M:%S')
            # Remove duplicate consecutive activities
            if not agent.recent_activities or agent.recent_activities[-1].split('] ', 1)[1] != display_activity:
                agent.recent_activities.append(f"[{timestamp}] {display_activity}")
                if len(agent.recent_activities) > 5:
                    agent.recent_activities.pop(0)
    
    def update_agent_status(self, agent_id: str, status: AgentStatus, current_task: Optional[str] = None):
        """Update agent status"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = status
            if current_task is not None:
                agent.current_task = current_task
            agent.last_update = time.time()
            
            # Update progress based on status
            # Use minimum progress for each status to avoid jumping backwards
            progress_map = {
                AgentStatus.INITIALIZING: 5,
                AgentStatus.SETTING_UP: 10,
                AgentStatus.ANALYZING: 20,
                AgentStatus.PLANNING: 30,
                AgentStatus.EXECUTING: 40,  # Start at 40% instead of 50%
                AgentStatus.WRITING_FILES: 70,
                AgentStatus.TESTING: 85,
                AgentStatus.FINALIZING: 95,
                AgentStatus.COMPLETED: 100,
                AgentStatus.FAILED: agent.progress  # Keep current progress
            }
            
            # For EXECUTING status, calculate dynamic progress based on time
            if status == AgentStatus.EXECUTING:
                # Use task start time if available, otherwise use agent start time
                start_time = getattr(agent, '_current_task_start', None)
                if start_time is None:
                    # If no task start time yet, stay at previous progress
                    agent.progress = max(agent.progress, progress_map.get(status, 50))
                else:
                    elapsed = time.time() - start_time
                    
                    # Smooth progress calculation:
                    # Start from current progress (not 50%)
                    base_progress = max(agent.progress, progress_map.get(AgentStatus.PLANNING, 30))
                    
                    # 0-30s: base to 70% (quick progress)
                    # 30-120s: 70-85% (slower progress)
                    # 120s+: 85-90% (very slow progress, asymptotic)
                    if elapsed < 30:
                        progress = base_progress + ((70 - base_progress) * (elapsed / 30))
                    elif elapsed < 120:
                        progress = 70 + ((elapsed - 30) / 90) * 15  # 70% to 85%
                    else:
                        progress = 85 + min((elapsed - 120) / 300, 1) * 5  # 85% to 90%
                    
                    # Never go backwards
                    agent.progress = max(agent.progress, int(min(progress, 90)))  # Cap at 90% until completion
            else:
                # Never go backwards in progress
                new_progress = progress_map.get(status, agent.progress)
                agent.progress = max(agent.progress, new_progress)
    
    def start_task(self, agent_id: str, task_id: str, task_name: str):
        """Record task start"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            timing = TaskTimingInfo(task_id=task_id, task_name=task_name)
            agent.task_timings[task_id] = timing
            agent.current_task = task_name
            # Store the task start time for progress calculation
            agent._current_task_start = time.time()
            self.update_agent_status(agent_id, AgentStatus.EXECUTING, task_name)
    
    def complete_task(self, agent_id: str, task_id: str, success: bool = True):
        """Record task completion"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if task_id in agent.task_timings:
                timing = agent.task_timings[task_id]
                timing.end_time = time.time()
                timing.status = "completed" if success else "failed"
                
                if success:
                    agent.completed_tasks.append(timing.task_name)
    
    def update_task_progress(self, agent_id: str, progress: int, status_message: Optional[str] = None):
        """Update task progress for an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.progress = max(0, min(100, progress))  # Clamp between 0-100
            if status_message:
                agent.current_task = status_message
            agent.last_update = time.time()
    
    def add_file_created(self, agent_id: str, file_path: str):
        """Record a file created by an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].files_created.append(file_path)
    
    def add_error(self, agent_id: str, error: str):
        """Record an error for an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].errors.append(error)
            self.update_agent_status(agent_id, AgentStatus.FAILED)
    
    def _create_display(self) -> Group:
        """Create the monitoring display as a Group for stable rendering"""
        components = []
        
        # Compact header
        header_text = Text("🚀 Agent Swarm Execution Monitor", style="bold blue", justify="center")
        header = Panel(header_text, height=3, expand=False)
        components.append(header)
        
        # Agent status table
        if self.agents:
            agent_table = self._create_agent_table()
            components.append(agent_table)
            components.append(Text(""))  # Spacer
        
        # Compact summary
        summary = self._create_compact_summary()
        components.append(summary)
        
        return Group(*components)
    
    def _create_agent_table(self) -> Table:
        """Create a compact table showing all agents' status"""
        table = Table(show_header=True, header_style="bold magenta", box=None, expand=True)
        table.add_column("Agent", style="cyan", width=25, overflow="fold")
        table.add_column("Status", justify="center", width=15, overflow="fold")
        table.add_column("Progress", justify="center", width=30, overflow="fold")
        table.add_column("Current Task", style="blue", width=50, overflow="fold")
        table.add_column("Time", justify="right", width=10, overflow="fold")
        
        for agent in self.agents.values():
            # Status with icon
            status_color = {
                AgentStatus.COMPLETED: "green",
                AgentStatus.FAILED: "red",
                AgentStatus.EXECUTING: "yellow",
                AgentStatus.IDLE: "dim"
            }.get(agent.status, "blue")
            
            status_icon = {
                AgentStatus.COMPLETED: "✅",
                AgentStatus.FAILED: "❌",
                AgentStatus.EXECUTING: "⚡",
                AgentStatus.INITIALIZING: "🔄",
                AgentStatus.SETTING_UP: "🔧",
                AgentStatus.ANALYZING: "🔍",
                AgentStatus.PLANNING: "📋",
                AgentStatus.WRITING_FILES: "✍️",
                AgentStatus.TESTING: "🧪",
                AgentStatus.FINALIZING: "📦"
            }.get(agent.status, "⏳")
            
            status_display = f"{status_icon} [{status_color}]{agent.status.value}[/{status_color}]"
            
            # Progress bar
            progress_bar = self._create_progress_bar(agent.progress)
            
            # Current task (truncated to fit column width)
            current_task = agent.current_task[:47] + "..." if agent.current_task and len(agent.current_task) > 50 else (agent.current_task or "-")
            
            # Current activity - show what agent is actually doing
            if agent.current_activity:
                # Truncate activity as well
                activity = agent.current_activity[:45] + "..." if len(agent.current_activity) > 48 else agent.current_activity
                activity_display = f"{current_task}\n[dim italic]{activity}[/dim italic]"
            else:
                activity_display = current_task
            
            # Runtime
            runtime = self._format_duration(time.time() - agent.start_time)
            
            table.add_row(
                f"{agent.role.title()}\n[dim]{agent.agent_type}[/dim]",
                status_display,
                progress_bar,
                activity_display,
                runtime
            )
        
        return table
    
    def _create_compact_summary(self) -> Table:
        """Create a compact summary table"""
        total_runtime = time.time() - self.overall_start_time
        total_agents = len(self.agents)
        completed_agents = sum(1 for a in self.agents.values() if a.status == AgentStatus.COMPLETED)
        failed_agents = sum(1 for a in self.agents.values() if a.status == AgentStatus.FAILED)
        active_agents = total_agents - completed_agents - failed_agents
        total_tasks = sum(len(a.completed_tasks) for a in self.agents.values())
        total_files = sum(len(a.files_created) for a in self.agents.values())
        
        # Create horizontal summary table
        table = Table(show_header=False, box=None, expand=False)
        table.add_column("", style="cyan")
        table.add_column("", style="bold")
        
        table.add_row("⏱️  Total Time:", self._format_duration(total_runtime))
        table.add_row("👥 Active/Total:", f"{active_agents}/{total_agents}")
        table.add_row("✅ Completed:", f"{completed_agents} agents, {total_tasks} tasks")
        if failed_agents > 0:
            table.add_row("❌ Failed:", f"[red]{failed_agents} agents[/red]")
        table.add_row("📄 Files:", f"{total_files} created")
        
        return Panel(table, title="Summary", border_style="green", height=8)
    
    def _create_agent_panel(self, agent: AgentMonitorInfo) -> Panel:
        """Create a panel for a single agent"""
        # Status indicator
        status_color = {
            AgentStatus.COMPLETED: "green",
            AgentStatus.FAILED: "red",
            AgentStatus.EXECUTING: "yellow",
            AgentStatus.IDLE: "dim"
        }.get(agent.status, "blue")
        
        status_icon = {
            AgentStatus.COMPLETED: "✅",
            AgentStatus.FAILED: "❌",
            AgentStatus.EXECUTING: "⚡",
            AgentStatus.INITIALIZING: "🔄",
            AgentStatus.SETTING_UP: "🔧",
            AgentStatus.ANALYZING: "🔍",
            AgentStatus.PLANNING: "📋",
            AgentStatus.WRITING_FILES: "✍️",
            AgentStatus.TESTING: "🧪",
            AgentStatus.FINALIZING: "📦"
        }.get(agent.status, "⏳")
        
        # Build content
        content = []
        content.append(f"{status_icon} Status: [bold {status_color}]{agent.status.value}[/bold {status_color}]")
        content.append(f"Progress: {self._create_progress_bar(agent.progress)}")
        
        if agent.current_task:
            content.append(f"\n📌 Current: [cyan]{agent.current_task[:40]}...[/cyan]")
        
        # Timing info
        runtime = time.time() - agent.start_time
        content.append(f"⏱️  Runtime: {self._format_duration(runtime)}")
        
        # Completed tasks
        if agent.completed_tasks:
            content.append(f"\n✅ Completed: {len(agent.completed_tasks)} tasks")
            for task in agent.completed_tasks[-3:]:  # Show last 3
                content.append(f"   • {task[:35]}...")
        
        # Files created
        if agent.files_created:
            content.append(f"\n📄 Files: {len(agent.files_created)} created")
        
        # Errors
        if agent.errors:
            content.append(f"\n[red]❌ Errors: {len(agent.errors)}[/red]")
        
        title = f"{agent.role.title()} ({agent.agent_type})"
        return Panel(
            "\n".join(content),
            title=title,
            border_style=status_color
        )
    
    def _create_summary_panel(self) -> Panel:
        """Create overall summary panel"""
        total_runtime = time.time() - self.overall_start_time
        
        # Calculate stats
        total_agents = len(self.agents)
        completed_agents = sum(1 for a in self.agents.values() if a.status == AgentStatus.COMPLETED)
        failed_agents = sum(1 for a in self.agents.values() if a.status == AgentStatus.FAILED)
        total_tasks = sum(len(a.completed_tasks) for a in self.agents.values())
        total_files = sum(len(a.files_created) for a in self.agents.values())
        
        # Build summary table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        
        table.add_row("Total Runtime:", self._format_duration(total_runtime))
        table.add_row("Active Agents:", f"{total_agents - completed_agents - failed_agents}/{total_agents}")
        table.add_row("Completed:", f"{completed_agents} agents")
        if failed_agents > 0:
            table.add_row("Failed:", f"[red]{failed_agents} agents[/red]")
        table.add_row("Tasks Completed:", str(total_tasks))
        table.add_row("Files Created:", str(total_files))
        
        # Average task time
        all_timings = []
        for agent in self.agents.values():
            for timing in agent.task_timings.values():
                if timing.end_time:
                    all_timings.append(timing.duration)
        
        if all_timings:
            avg_time = sum(all_timings) / len(all_timings)
            table.add_row("Avg Task Time:", self._format_duration(avg_time))
        
        return Panel(table, title="📊 Execution Summary", border_style="green")
    
    def _create_progress_bar(self, progress: int) -> str:
        """Create a simple progress bar"""
        # Ensure progress is within bounds
        progress = max(0, min(100, progress))
        filled = int(progress / 5)  # 20 chars total
        empty = 20 - filled
        return f"[{'█' * filled}{'░' * empty}] {progress:3d}%"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _show_final_summary(self):
        """Show final execution summary"""
        total_runtime = time.time() - self.overall_start_time
        
        # Create summary table
        table = Table(title="🎯 Final Execution Summary", show_header=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Tasks", justify="right")
        table.add_column("Files", justify="right")
        table.add_column("Time", justify="right")
        
        for agent in self.agents.values():
            status_color = "green" if agent.status == AgentStatus.COMPLETED else "red"
            table.add_row(
                agent.role.title(),
                f"[{status_color}]{agent.status.value}[/{status_color}]",
                str(len(agent.completed_tasks)),
                str(len(agent.files_created)),
                self._format_duration(time.time() - agent.start_time)
            )
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print(f"\n[bold green]Total execution time: {self._format_duration(total_runtime)}[/bold green]")
        
        # Show task timing breakdown
        if any(agent.task_timings for agent in self.agents.values()):
            self.console.print("\n[bold]Task Timing Breakdown:[/bold]")
            timing_table = Table(show_header=True)
            timing_table.add_column("Task", style="cyan")
            timing_table.add_column("Agent", style="dim")
            timing_table.add_column("Duration", justify="right")
            
            # Collect all timings
            all_timings = []
            for agent in self.agents.values():
                for timing in agent.task_timings.values():
                    if timing.end_time:
                        all_timings.append((timing, agent.role))
            
            # Sort by duration
            all_timings.sort(key=lambda x: x[0].duration, reverse=True)
            
            # Show top 10
            for timing, role in all_timings[:10]:
                timing_table.add_row(
                    timing.task_name[:40] + "...",
                    role,
                    timing.duration_str
                )
            
            self.console.print(timing_table)