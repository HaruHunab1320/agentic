"""
Unified Swarm Monitor - Consolidated implementation with best features from all versions

This implementation combines:
- Rich Live display from swarm_monitor_fixed (stable updates)
- Smart progress calculation from original swarm_monitor
- Adaptive layout from swarm_monitor_simple
- Task queue management from swarm_monitor_enhanced
- Clean terminal handling without cursor manipulation
"""

import asyncio
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text


class AgentStatus(Enum):
    """Agent status states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    SETTING_UP = "setting_up"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


class DisplayMode(Enum):
    """Display modes based on terminal size"""
    MINIMAL = "minimal"    # < 80 cols: Just agent names and status
    STANDARD = "standard"  # 80-120 cols: Names, status, current task
    FULL = "full"         # > 120 cols: Everything including files


@dataclass
class TaskInfo:
    """Information about a task"""
    task_id: str
    description: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool = False
    

@dataclass
class TaskTimingInfo:
    """Detailed timing information for tasks"""
    task_id: str
    agent_id: str
    description: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = False


@dataclass
class AgentInfo:
    """Information about an agent"""
    agent_id: str
    agent_name: str
    agent_type: str
    role: str = "general"  # Role in the swarm
    status: AgentStatus = AgentStatus.INITIALIZING
    current_task: Optional[str] = None
    current_task_id: Optional[str] = None
    activity: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    progress: float = 0.0
    status_start_time: float = field(default_factory=time.time)
    # Enhanced features
    task_queue: List[TaskInfo] = field(default_factory=list)
    recent_activities: List[Tuple[float, str]] = field(default_factory=list)
    

class SwarmMonitorUnified:
    """Unified swarm monitor with best features from all implementations"""
    
    def __init__(self, use_alternate_screen: bool = True, display_mode: Optional[DisplayMode] = None):
        self.console = Console()
        self.use_alternate_screen = use_alternate_screen
        self.agents: Dict[str, AgentInfo] = {}
        self.is_monitoring = False
        self.start_time = time.time()
        self._live: Optional[Live] = None
        self._update_task: Optional[asyncio.Task] = None
        
        # Animation state
        self._spinner_frame = 0
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        
        # Task timing from original
        self.task_timings: List[TaskTimingInfo] = []
        
        # Coordinator status tracking
        self.coordinator_status = "Initializing"
        self.coordinator_activity = "Starting up..."
        self.coordinator_phase = "EXPLORATION"
        self.total_tasks_queued = 0
        self.total_tasks_completed = 0
        self.active_execution_id: Optional[str] = None
        
        # Display configuration
        self.display_mode = display_mode
        self.auto_detect_mode = display_mode is None
        
        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        
        # Activity buffer settings
        self.max_recent_activities = 5
        self.activity_buffer_seconds = 30.0
    
    def _detect_display_mode(self) -> DisplayMode:
        """Auto-detect best display mode based on terminal size"""
        width = shutil.get_terminal_size().columns
        
        if width < 80:
            return DisplayMode.MINIMAL
        elif width < 120:
            return DisplayMode.STANDARD
        else:
            return DisplayMode.FULL
    
    def _get_current_display_mode(self) -> DisplayMode:
        """Get current display mode (auto-detect if needed)"""
        if self.auto_detect_mode:
            return self._detect_display_mode()
        return self.display_mode or DisplayMode.STANDARD
    
    def register_agent(self, agent_id: str, agent_name: str, agent_type: str, role: str = "general"):
        """Register a new agent"""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                agent_name=agent_name,
                agent_type=agent_type,
                role=role
            )
    
    def update_agent_status(self, agent_id: str, status: AgentStatus, message: Optional[str] = None):
        """Update agent status"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = status
            agent.status_start_time = time.time()
            
            # Update progress based on status (from original smart calculation)
            if status == AgentStatus.EXECUTING:
                agent.progress = 30.0  # Start at 30% when executing
            elif status == AgentStatus.FINALIZING:
                agent.progress = 90.0
            elif status == AgentStatus.COMPLETED:
                agent.progress = 100.0
            elif status == AgentStatus.FAILED:
                agent.progress = agent.progress  # Keep current progress
            
            if message:
                agent.activity = message
                # Add to recent activities with timestamp
                agent.recent_activities.append((time.time(), message))
                # Keep only recent activities
                cutoff_time = time.time() - self.activity_buffer_seconds
                agent.recent_activities = [
                    (t, m) for t, m in agent.recent_activities 
                    if t > cutoff_time
                ][-self.max_recent_activities:]
    
    def set_agent_tasks(self, agent_id: str, tasks: List[Tuple[str, str]]):
        """Set the task queue for an agent (from enhanced)"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.task_queue = [
                TaskInfo(task_id=tid, description=desc) 
                for tid, desc in tasks
            ]
    
    def start_task(self, agent_id: str, task_id: str, task_description: str):
        """Start a task (compatible with all versions)"""
        self.start_agent_task(agent_id, task_id, task_description)
    
    def start_agent_task(self, agent_id: str, task_id: str, task_description: str):
        """Start a task for an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.current_task = task_description
            agent.current_task_id = task_id
            agent.status = AgentStatus.EXECUTING
            
            # Update task queue if using enhanced features
            for task in agent.task_queue:
                if task.task_id == task_id:
                    task.started_at = datetime.now()
                    break
            
            # Add to task timings (from original)
            self.task_timings.append(TaskTimingInfo(
                task_id=task_id,
                agent_id=agent_id,
                description=task_description,
                start_time=time.time()
            ))
    
    def complete_task(self, agent_id: str, task_id: str, success: bool = True):
        """Complete a task (compatible with all versions)"""
        self.complete_agent_task(agent_id, task_id, success)
    
    def complete_agent_task(self, agent_id: str, task_id: str, success: bool = True):
        """Complete a task for an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            if success:
                agent.tasks_completed += 1
            else:
                agent.tasks_failed += 1
            
            # Update task queue
            for task in agent.task_queue:
                if task.task_id == task_id:
                    task.completed_at = datetime.now()
                    task.success = success
                    break
            
            # Update task timings
            for timing in self.task_timings:
                if timing.task_id == task_id and timing.agent_id == agent_id:
                    timing.end_time = time.time()
                    timing.duration = timing.end_time - timing.start_time
                    timing.success = success
                    break
            
            agent.current_task = None
    
    def update_agent_activity(self, agent_id: str, activity: str):
        """Update agent's current activity"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.activity = activity
            # Add to recent activities
            agent.recent_activities.append((time.time(), activity))
            # Trim old activities
            cutoff_time = time.time() - self.activity_buffer_seconds
            agent.recent_activities = [
                (t, m) for t, m in agent.recent_activities 
                if t > cutoff_time
            ][-self.max_recent_activities:]
    
    def add_file_created(self, agent_id: str, file_path: str):
        """Add a created file"""
        if agent_id in self.agents:
            self.agents[agent_id].files_created.append(file_path)
    
    def add_file_modified(self, agent_id: str, file_path: str):
        """Add a modified file"""
        if agent_id in self.agents:
            self.agents[agent_id].files_modified.append(file_path)
    
    def add_error(self, agent_id: str, error: str):
        """Add an error for an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].errors.append(error)
    
    def update_coordinator_status(self, status: str, activity: str, phase: Optional[str] = None):
        """Update coordinator status"""
        self.coordinator_status = status
        self.coordinator_activity = activity
        if phase:
            self.coordinator_phase = phase
    
    def update_task_queue_status(self, queued: int, completed: int):
        """Update task queue statistics"""
        self.total_tasks_queued = queued
        self.total_tasks_completed = completed
    
    def set_execution_id(self, execution_id: str):
        """Set the current execution ID"""
        self.active_execution_id = execution_id
    
    def _calculate_dynamic_progress(self, agent: AgentInfo) -> float:
        """Calculate dynamic progress based on status and time (from original)"""
        if agent.status == AgentStatus.EXECUTING:
            # Dynamic progress during execution
            elapsed = time.time() - agent.status_start_time
            
            if elapsed < 10:
                # First 10 seconds: 30% to 50%
                return 30.0 + (elapsed / 10) * 20.0
            elif elapsed < 30:
                # Next 20 seconds: 50% to 70%
                return 50.0 + ((elapsed - 10) / 20) * 20.0
            elif elapsed < 60:
                # Next 30 seconds: 70% to 85%
                return 70.0 + ((elapsed - 30) / 30) * 15.0
            else:
                # After 60 seconds: slowly approach 90%
                return min(85.0 + ((elapsed - 60) / 60) * 5.0, 89.0)
        else:
            return agent.progress
    
    def _create_mini_progress_bar(self, progress: float) -> str:
        """Create a mini progress bar for the table"""
        # Use box drawing characters for a smooth progress bar
        filled = "█"
        partial = "▓"
        empty = "░"
        
        # 10 character wide progress bar
        bar_width = 10
        filled_width = int(progress / 10)  # Each block represents 10%
        
        # Create the bar
        bar = filled * filled_width
        
        # Add partial block if needed
        remainder = (progress % 10) / 10
        if remainder > 0.5 and filled_width < bar_width:
            bar += partial
            filled_width += 1
        
        # Fill the rest with empty blocks
        bar += empty * (bar_width - filled_width)
        
        return f"[{'green' if progress >= 80 else 'yellow' if progress >= 50 else 'blue'}]{bar}[/]"
    
    def _create_display(self) -> Panel:
        """Create the display panel"""
        mode = self._get_current_display_mode()
        
        # Create main layout
        layout = Layout()
        
        # Add coordinator status at the top
        coordinator_panel = self._create_coordinator_panel()
        
        # Create agent table
        table = self._create_agent_table(mode)
        
        if mode == DisplayMode.FULL and len(self.agents) > 0:
            # Full layout with coordinator, agents, and activity
            layout.split_column(
                Layout(coordinator_panel, size=5),
                Layout(name="main", ratio=1)
            )
            layout["main"].split_row(
                Layout(table, ratio=2),
                Layout(self._create_activity_panel(), ratio=1)
            )
        else:
            # Simple layout with coordinator and agents
            layout.split_column(
                Layout(coordinator_panel, size=5),
                Layout(table, ratio=1)
            )
        
        return Panel(layout, title="🐝 Swarm Monitor", border_style="blue")
    
    def _create_coordinator_panel(self) -> Panel:
        """Create coordinator status panel"""
        content = Text()
        
        # Status line with spinner for active states
        status_color = "green" if self.coordinator_status == "Running" else "yellow"
        spinner = self._spinner_frames[self._spinner_frame] if self.coordinator_status == "Running" else ""
        content.append(f"Status: ", style="bold")
        if spinner:
            content.append(f"{spinner} ", style=status_color)
        content.append(f"{self.coordinator_status}\n", style=status_color)
        
        # Activity line with dynamic updates
        content.append(f"Activity: ", style="bold")
        activity_text = self.coordinator_activity
        # Add ellipsis animation for ongoing activities
        if self.coordinator_status == "Running" and not activity_text.endswith("..."):
            dots = "." * ((self._spinner_frame % 4))
            activity_text = f"{self.coordinator_activity}{dots}"
        content.append(f"{activity_text}\n", style="white")
        
        # Phase and tasks
        content.append(f"Phase: ", style="bold")
        content.append(f"{self.coordinator_phase}", style="magenta")
        content.append(f"  |  Tasks: ", style="bold")
        content.append(f"{self.total_tasks_completed}/{self.total_tasks_queued} completed", style="cyan")
        
        # Add a mini progress bar for overall completion
        if self.total_tasks_queued > 0:
            completion_pct = (self.total_tasks_completed / self.total_tasks_queued) * 100
            mini_bar = self._create_mini_progress_bar(completion_pct)
            content.append(f" {mini_bar}")
        
        return Panel(content, title="📊 Coordinator Status", border_style="green")
    
    def _create_agent_table(self, mode: DisplayMode) -> Table:
        """Create agent status table based on display mode"""
        table = Table(show_header=True, header_style="bold magenta", title="🤖 Active Agents")
        
        # Add columns based on mode
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Role", style="blue")
        table.add_column("Status", style="yellow")
        
        if mode in [DisplayMode.STANDARD, DisplayMode.FULL]:
            table.add_column("Current Task", style="white")
            table.add_column("Progress", style="green")
        
        if mode == DisplayMode.FULL:
            table.add_column("Activity", style="dim")
            table.add_column("Stats", style="blue")
        
        # Add rows for each agent
        for agent_id, agent in self.agents.items():
            row = [
                agent.agent_name,
                agent.role.replace('_', ' ').title(),
                self._format_status(agent.status)
            ]
            
            if mode in [DisplayMode.STANDARD, DisplayMode.FULL]:
                row.append(agent.current_task or agent.activity or "-")
                # Calculate dynamic progress
                progress = self._calculate_dynamic_progress(agent)
                # Create visual progress bar for active agents
                if agent.status in [AgentStatus.EXECUTING, AgentStatus.ANALYZING, AgentStatus.FINALIZING]:
                    progress_bar = self._create_mini_progress_bar(progress)
                    row.append(f"{progress_bar} {progress:.0f}%")
                else:
                    row.append(f"{progress:.0f}%")
            
            if mode == DisplayMode.FULL:
                # Show recent activity
                if agent.recent_activities:
                    _, recent_activity = agent.recent_activities[-1]
                    row.append(recent_activity[:50] + "..." if len(recent_activity) > 50 else recent_activity)
                else:
                    row.append("-")
                
                # Stats
                stats = f"✓{agent.tasks_completed} ✗{agent.tasks_failed}"
                if agent.files_created or agent.files_modified:
                    stats += f" 📄{len(agent.files_created) + len(agent.files_modified)}"
                row.append(stats)
            
            table.add_row(*row)
        
        return table
    
    def _create_grid_layout(self, main_table: Table) -> Layout:
        """Create grid layout for full display mode (from enhanced)"""
        layout = Layout()
        
        # Determine grid size based on terminal width
        width = shutil.get_terminal_size().columns
        if width > 200:
            # 3-column layout for very wide terminals
            layout.split_column(
                Layout(name="main"),
                Layout(name="details", size=30)
            )
            layout["main"].split_row(
                Layout(main_table),
                Layout(self._create_activity_panel())
            )
        else:
            # 2-column layout
            layout.split_row(
                Layout(main_table),
                Layout(self._create_activity_panel())
            )
        
        return layout
    
    def _create_activity_panel(self) -> Panel:
        """Create activity stream panel"""
        activities = []
        
        # Collect all recent activities from all agents
        all_activities = []
        for agent in self.agents.values():
            for timestamp, activity in agent.recent_activities:
                all_activities.append((timestamp, agent.agent_name, activity))
        
        # Sort by timestamp and take most recent
        all_activities.sort(key=lambda x: x[0], reverse=True)
        
        # Add a live indicator at the top if there are active agents
        active_agents = sum(1 for agent in self.agents.values() 
                          if agent.status in [AgentStatus.EXECUTING, AgentStatus.ANALYZING, AgentStatus.FINALIZING])
        if active_agents > 0:
            pulse = "●" if self._spinner_frame % 4 < 2 else "○"  # Pulsing dot
            activities.append(f"[bold red]{pulse} LIVE[/bold red] - {active_agents} agent{'s' if active_agents > 1 else ''} working\n")
        
        for timestamp, agent_name, activity in all_activities[:10]:
            time_ago = int(time.time() - timestamp)
            if time_ago < 60:
                time_str = f"{time_ago}s ago"
            else:
                time_str = f"{time_ago // 60}m ago"
            
            # Highlight very recent activities
            if time_ago < 5:
                activities.append(f"[bold yellow]→[/bold yellow] [dim]{time_str}[/dim] [{agent_name}] {activity}")
            else:
                activities.append(f"  [dim]{time_str}[/dim] [{agent_name}] {activity}")
        
        content = "\n".join(activities) if activities else "[dim]No recent activity[/dim]"
        
        # Animate the panel title
        title = "Recent Activity" if self._spinner_frame % 10 < 5 else "Recent Activity "
        return Panel(content, title=title, border_style="green")
    
    def _format_status(self, status: AgentStatus) -> str:
        """Format status with emoji and spinner for active states"""
        # Get current spinner frame for active states
        spinner = self._spinner_frames[self._spinner_frame]
        
        # Define which states should show a spinner
        active_states = {
            AgentStatus.INITIALIZING,
            AgentStatus.SETTING_UP,
            AgentStatus.ANALYZING,
            AgentStatus.EXECUTING,
            AgentStatus.FINALIZING
        }
        
        # Base status map without spinners
        status_base = {
            AgentStatus.INITIALIZING: "🔄 Initializing",
            AgentStatus.IDLE: "💤 Idle",
            AgentStatus.SETTING_UP: "🔧 Setting Up",
            AgentStatus.ANALYZING: "🔍 Analyzing",
            AgentStatus.EXECUTING: "⚡ Executing",
            AgentStatus.FINALIZING: "📝 Finalizing",
            AgentStatus.COMPLETED: "✅ Completed",
            AgentStatus.FAILED: "❌ Failed",
            AgentStatus.WAITING: "⏳ Waiting"
        }
        
        base_text = status_base.get(status, status.value)
        
        # Add spinner for active states
        if status in active_states:
            return f"{spinner} {base_text}"
        else:
            return base_text
    
    async def _update_display(self):
        """Update the display continuously"""
        frame_count = 0
        while self.is_monitoring:
            if self._live:
                # Update spinner frame every few updates for smooth animation
                if frame_count % 2 == 0:  # Update spinner every 0.2 seconds (5 FPS for spinner)
                    self._spinner_frame = (self._spinner_frame + 1) % len(self._spinner_frames)
                
                self._live.update(self._create_display())
            
            frame_count += 1
            await asyncio.sleep(0.1)  # 10 FPS for smooth updates
    
    async def start_monitoring(self):
        """Start the monitoring display"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        
        # Create and start live display
        self._live = Live(
            self._create_display(),
            console=self.console,
            refresh_per_second=10,
            screen=self.use_alternate_screen
        )
        
        self._live.start()
        
        # Start update task
        self._update_task = asyncio.create_task(self._update_display())
    
    async def stop_monitoring(self):
        """Stop the monitoring display"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        # Show final summary
        if self._live:
            self._live.update(self._create_final_summary())
            await asyncio.sleep(2)  # Show summary for 2 seconds
            self._live.stop()
    
    def _create_final_summary(self) -> Panel:
        """Create final execution summary (from original)"""
        duration = time.time() - self.start_time
        
        # Summary statistics
        total_agents = len(self.agents)
        total_tasks = sum(a.tasks_completed + a.tasks_failed for a in self.agents.values())
        successful_tasks = sum(a.tasks_completed for a in self.agents.values())
        failed_tasks = sum(a.tasks_failed for a in self.agents.values())
        total_files = sum(len(a.files_created) + len(a.files_modified) for a in self.agents.values())
        
        summary = Table(show_header=False, box=None)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="yellow")
        
        summary.add_row("Duration", f"{duration:.1f}s")
        summary.add_row("Agents", str(total_agents))
        summary.add_row("Tasks Completed", f"{successful_tasks}/{total_tasks}")
        summary.add_row("Files Created/Modified", str(total_files))
        
        if failed_tasks > 0:
            summary.add_row("Failed Tasks", f"[red]{failed_tasks}[/red]")
        
        # Task timing breakdown (from original)
        if self.task_timings:
            timing_table = Table(title="Task Timing Breakdown", show_header=True)
            timing_table.add_column("Task", style="cyan")
            timing_table.add_column("Agent", style="yellow") 
            timing_table.add_column("Duration", style="green")
            timing_table.add_column("Status", style="blue")
            
            for timing in sorted(self.task_timings, key=lambda t: t.duration or 0, reverse=True)[:10]:
                if timing.duration:
                    timing_table.add_row(
                        timing.description[:40] + "..." if len(timing.description) > 40 else timing.description,
                        self.agents[timing.agent_id].agent_name if timing.agent_id in self.agents else timing.agent_id,
                        f"{timing.duration:.1f}s",
                        "✅" if timing.success else "❌"
                    )
            
            layout = Layout()
            layout.split_column(
                Layout(summary),
                Layout(timing_table)
            )
            return Panel(layout, title="🎉 Execution Complete", border_style="green")
        else:
            return Panel(summary, title="🎉 Execution Complete", border_style="green")
    
    # Compatibility methods for different versions
    def update_task_analysis(self, total_files: int, complexity: float, suggested_agents: List[str]):
        """Update task analysis (compatibility with coordination engine)"""
        # Store for potential display
        self._task_analysis = {
            'total_files': total_files,
            'complexity': complexity,
            'suggested_agents': suggested_agents
        }


# Re-export for compatibility
SwarmMonitor = SwarmMonitorUnified
SwarmMonitorFixed = SwarmMonitorUnified
SwarmMonitorEnhanced = SwarmMonitorUnified
SwarmMonitorSimple = SwarmMonitorUnified