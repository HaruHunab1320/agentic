"""
Agent models and configurations
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict

from agentic.models.task import Task, TaskResult


class DiscoveryType(str, Enum):
    """Types of discoveries agents can report"""
    API_READY = "api_ready"  # API endpoint is ready/discovered
    TEST_NEEDED = "test_needed"  # Test coverage needed for code
    BUG_FOUND = "bug_found"  # Bug detected in code
    SECURITY_ISSUE = "security_issue"  # Security vulnerability found
    PERFORMANCE_ISSUE = "performance_issue"  # Performance bottleneck detected
    REFACTOR_OPPORTUNITY = "refactor_opportunity"  # Code can be improved
    DEPENDENCY_UPDATE = "dependency_update"  # Dependency needs update
    DOCUMENTATION_NEEDED = "documentation_needed"  # Documentation missing/outdated
    CONFIG_ISSUE = "config_issue"  # Configuration problem detected
    INTEGRATION_POINT = "integration_point"  # Integration opportunity found
    RESEARCH_FINDING = "research_finding"  # Hypothesis or experiment result identified


class Discovery(BaseModel):
    """A discovery made by an agent during execution"""
    type: DiscoveryType = Field(description="Type of discovery")
    description: str = Field(description="Description of the discovery")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    severity: str = Field(default="info", description="Severity: info, warning, error, critical")
    agent_name: str = Field(description="Name of the agent that made the discovery")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the discovery was made")
    file_path: Optional[str] = Field(default=None, description="File path if discovery is file-specific")
    line_number: Optional[int] = Field(default=None, description="Line number if applicable")
    suggested_action: Optional[str] = Field(default=None, description="Suggested action to take")


class AgentType(str, Enum):
    """Types of available agents"""
    # Analysis and reasoning
    CLAUDE_CODE = "claude_code"
    GEMINI = "gemini"
    CODEX_RESEARCH = "codex_research"
    
    # Domain-based agents (current implementation)
    AIDER_BACKEND = "aider_backend"
    AIDER_FRONTEND = "aider_frontend"
    AIDER_TESTING = "aider_testing"
    AIDER_DEVOPS = "aider_devops"
    
    # Granular specialist agents (enterprise vision)
    PYTHON_EXPERT = "python_expert"
    SECURITY_SPECIALIST = "security_specialist"
    FRONTEND_DEVELOPER = "frontend_developer"
    QUALITY_ASSURANCE = "quality_assurance"
    DEVOPS_ENGINEER = "devops_engineer"
    
    # Extensibility
    CUSTOM = "custom"


class AgentCapability(BaseModel):
    """Agent capabilities and specializations"""
    agent_type: AgentType = Field(description="Type of agent")
    specializations: List[str] = Field(default_factory=list, description="Areas of specialization")
    supported_languages: List[str] = Field(default_factory=list, description="Supported programming languages")
    max_context_tokens: int = Field(default=100000, description="Maximum context tokens")
    concurrent_tasks: int = Field(default=1, description="Maximum concurrent tasks")
    reasoning_capability: bool = Field(default=False, description="Whether agent has advanced reasoning")
    file_editing_capability: bool = Field(default=True, description="Whether agent can edit files")
    code_execution_capability: bool = Field(default=False, description="Whether agent can execute code")
    
    # NEW: Enhanced capabilities
    memory_capability: bool = Field(default=False, description="Whether agent supports memory/context persistence")
    session_persistence: bool = Field(default=False, description="Whether agent supports session persistence")
    interactive_capability: bool = Field(default=False, description="Whether agent can handle interactive scenarios")
    inter_agent_communication: bool = Field(default=False, description="Whether agent supports inter-agent communication")
    git_integration: bool = Field(default=False, description="Whether agent has git integration capabilities")


class AgentConfig(BaseModel):
    """Configuration for an agent instance"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    agent_type: AgentType = Field(description="Type of agent")
    name: str = Field(description="Agent instance name")
    workspace_path: Path = Field(description="Agent workspace directory")
    focus_areas: List[str] = Field(default_factory=list, description="Areas this agent should focus on")
    ai_model_config: Dict = Field(default_factory=dict, description="AI model configuration")
    tool_config: Dict = Field(default_factory=dict, description="Tool-specific configuration")
    max_tokens: int = Field(default=100000, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")


class AgentSession(BaseModel):
    """Active agent session"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session ID")
    agent_config: AgentConfig = Field(description="Agent configuration")
    process_id: Optional[int] = Field(default=None, description="Process ID if applicable")
    status: str = Field(default="inactive", description="Session status: inactive, starting, active, busy, error, stopped")
    current_task: Optional[str] = Field(default=None, description="Current task ID being executed")
    workspace: Path = Field(description="Session workspace path")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")
    error: Optional[str] = Field(default=None, description="Error message if session failed")
    
    def mark_active(self) -> None:
        """Mark session as active"""
        self.status = "active"
        self.last_activity = datetime.utcnow()
    
    def mark_busy(self, task_id: str) -> None:
        """Mark session as busy with a task"""
        self.status = "busy"
        self.current_task = task_id
        self.last_activity = datetime.utcnow()
    
    def mark_idle(self) -> None:
        """Mark session as idle/active but not busy"""
        self.status = "active"
        self.current_task = None
        self.last_activity = datetime.utcnow()
    
    def mark_error(self, error: str) -> None:
        """Mark session as having an error"""
        self.status = "error"
        self.last_activity = datetime.utcnow()
    
    @property
    def is_available(self) -> bool:
        """Check if session is available for new tasks"""
        return self.status == "active" and self.current_task is None


class Agent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session: Optional[AgentSession] = None
        self._monitor = None
        self._monitor_agent_id = None
        self._discoveries: List[Discovery] = []  # Store discoveries during execution
        self._discovery_callback = None  # Callback for reporting discoveries
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the agent session"""
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the agent session"""
    
    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a specific task"""
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if agent is healthy and responsive"""
    
    @abstractmethod
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities"""
    
    async def stream_progress(self, task: Task) -> AsyncGenerator[str, None]:
        """Stream progress updates during task execution"""
        # Default implementation - agents can override
        yield f"Starting task: {task.command}"
        result = await self.execute_task(task)
        yield f"Completed task: {result.status}"
    
    def set_monitor(self, monitor, agent_id: str) -> None:
        """Set monitoring instance for status updates"""
        self._monitor = monitor
        self._monitor_agent_id = agent_id
    
    def report_status(self, status: str, message: Optional[str] = None) -> None:
        """Report status to monitor if available"""
        if self._monitor and self._monitor_agent_id:
            try:
                from agentic.core.swarm_monitor_unified import AgentStatus
                status_enum = AgentStatus(status)
                self._monitor.update_agent_status(self._monitor_agent_id, status_enum, message)
            except Exception:
                pass  # Don't let monitoring errors affect execution
    
    def report_discovery(self, discovery_type: DiscoveryType, description: str, 
                        context: Optional[Dict[str, Any]] = None, severity: str = "info",
                        file_path: Optional[str] = None, line_number: Optional[int] = None,
                        suggested_action: Optional[str] = None) -> None:
        """Report a discovery made during task execution"""
        discovery = Discovery(
            type=discovery_type,
            description=description,
            context=context or {},
            severity=severity,
            agent_name=self.name,
            file_path=file_path,
            line_number=line_number,
            suggested_action=suggested_action
        )
        
        # Store locally
        self._discoveries.append(discovery)
        
        # Report via callback if available
        if self._discovery_callback:
            try:
                self._discovery_callback(discovery)
            except Exception:
                pass  # Don't let callback errors affect execution
    
    def set_discovery_callback(self, callback) -> None:
        """Set callback for reporting discoveries in real-time"""
        self._discovery_callback = callback
    
    def get_discoveries(self) -> List[Discovery]:
        """Get all discoveries made by this agent"""
        return self._discoveries.copy()
    
    def clear_discoveries(self) -> None:
        """Clear stored discoveries"""
        self._discoveries.clear()
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if this agent can handle the given task"""
        capabilities = self.get_capabilities()
        
        # Basic checks
        if task.intent.requires_reasoning and not capabilities.reasoning_capability:
            return False
        
        # Check if task type matches agent specializations
        task_areas = task.intent.affected_areas
        agent_specializations = capabilities.specializations
        
        if task_areas and agent_specializations:
            # Check for overlap between task areas and agent specializations
            return bool(set(task_areas) & set(agent_specializations))
        
        return True  # Default to true if no specific requirements
    
    @property
    def is_running(self) -> bool:
        """Check if agent is currently running"""
        return self.session is not None and self.session.status in ("active", "busy")
    
    @property
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return self.session is not None and self.session.is_available
    
    @property
    def agent_type(self) -> AgentType:
        """Get agent type from config (compatibility property)"""
        return self.config.agent_type
    
    @property
    def focus_areas(self) -> List[str]:
        """Get focus areas from config (compatibility property)"""
        return self.config.focus_areas
    
    @property
    def name(self) -> str:
        """Get agent name from config (compatibility property)"""
        return self.config.name
    
    @property
    def workspace_path(self) -> Path:
        """Get workspace path from config (compatibility property)"""
        return self.config.workspace_path 
