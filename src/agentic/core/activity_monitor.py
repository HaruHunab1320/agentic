"""
Real-time Activity Monitor for Agent Execution

Shows what agents are actually doing, not just their status.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from agentic.utils.logging import LoggerMixin


@dataclass
class AgentActivity:
    """Represents a single activity/action by an agent"""
    timestamp: float
    activity_type: str  # 'command', 'output', 'file_created', 'test_run', etc.
    description: str
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        # Format: [HH:MM:SS] Activity description
        time_str = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        return f"[{time_str}] {self.description}"


@dataclass 
class MonitoredAgent:
    """Agent being monitored with activity history"""
    agent_id: str
    agent_name: str
    agent_type: str
    role: str
    current_activity: str = "Initializing..."
    recent_activities: deque = field(default_factory=lambda: deque(maxlen=10))
    is_waiting_for_input: bool = False
    last_update: float = field(default_factory=time.time)
    
    def add_activity(self, activity: AgentActivity):
        """Add a new activity to the history"""
        self.recent_activities.append(activity)
        self.current_activity = activity.description
        self.last_update = time.time()


class ActivityMonitor(LoggerMixin):
    """Monitors and displays real-time agent activities"""
    
    def __init__(self):
        super().__init__()
        self.agents: Dict[str, MonitoredAgent] = {}
        self.global_activities: deque = deque(maxlen=50)
        self._monitor_task: Optional[asyncio.Task] = None
        self._update_handlers: List[Any] = []
    
    def register_agent(self, agent_id: str, agent_name: str, agent_type: str, role: str):
        """Register an agent for activity monitoring"""
        self.agents[agent_id] = MonitoredAgent(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            role=role
        )
        self.add_activity(agent_id, "registered", f"Agent {agent_name} registered")
    
    def add_activity(self, agent_id: str, activity_type: str, description: str, 
                    details: Optional[Dict[str, Any]] = None):
        """Add an activity for an agent"""
        if agent_id not in self.agents:
            return
        
        activity = AgentActivity(
            timestamp=time.time(),
            activity_type=activity_type,
            description=description,
            details=details
        )
        
        # Add to agent's history
        self.agents[agent_id].add_activity(activity)
        
        # Add to global history
        self.global_activities.append((agent_id, activity))
        
        # Notify handlers
        for handler in self._update_handlers:
            try:
                handler(agent_id, activity)
            except:
                pass
    
    def add_update_handler(self, handler):
        """Add a handler to be called on activity updates"""
        self._update_handlers.append(handler)
    
    def remove_update_handler(self, handler):
        """Remove an update handler"""
        if handler in self._update_handlers:
            self._update_handlers.remove(handler)
    
    # Specific activity helpers
    def record_command_execution(self, agent_id: str, command: str):
        """Record that an agent is executing a command"""
        self.add_activity(agent_id, "command", f"Executing: {command[:100]}...")
    
    def record_test_run(self, agent_id: str, test_framework: str, status: str = "running"):
        """Record test execution"""
        if status == "running":
            self.add_activity(agent_id, "test_run", f"Running {test_framework} tests...")
        elif status == "passed":
            self.add_activity(agent_id, "test_success", f"All {test_framework} tests passed ✓")
        else:
            self.add_activity(agent_id, "test_fail", f"{test_framework} tests failed")
    
    def record_file_operation(self, agent_id: str, operation: str, file_path: str):
        """Record file operations"""
        op_map = {
            "create": "Creating",
            "modify": "Modifying", 
            "delete": "Deleting",
            "read": "Reading"
        }
        action = op_map.get(operation, operation)
        self.add_activity(agent_id, f"file_{operation}", f"{action} {file_path}")
    
    def record_analysis(self, agent_id: str, what: str):
        """Record analysis activities"""
        self.add_activity(agent_id, "analysis", f"Analyzing {what}")
    
    def record_waiting(self, agent_id: str, waiting_for: str):
        """Record that agent is waiting"""
        self.agents[agent_id].is_waiting_for_input = True
        self.add_activity(agent_id, "waiting", f"Waiting for {waiting_for}")
    
    def record_output(self, agent_id: str, output_snippet: str, output_type: str = "info"):
        """Record agent output"""
        # Clean and truncate output
        clean_output = output_snippet.strip().replace('\n', ' ')[:150]
        if clean_output:
            self.add_activity(agent_id, f"output_{output_type}", clean_output)
    
    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary of agent's current state and recent activities"""
        if agent_id not in self.agents:
            return {}
        
        agent = self.agents[agent_id]
        return {
            "agent_id": agent_id,
            "agent_name": agent.agent_name,
            "current_activity": agent.current_activity,
            "is_waiting": agent.is_waiting_for_input,
            "recent_activities": [str(activity) for activity in agent.recent_activities],
            "last_update": agent.last_update,
            "idle_time": time.time() - agent.last_update
        }
    
    def get_global_activity_log(self, limit: int = 20) -> List[str]:
        """Get recent activities across all agents"""
        activities = []
        for agent_id, activity in list(self.global_activities)[-limit:]:
            agent_name = self.agents[agent_id].agent_name if agent_id in self.agents else agent_id
            activities.append(f"{agent_name}: {activity}")
        return activities