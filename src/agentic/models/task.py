"""
Task execution models
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of development tasks"""
    IMPLEMENT = "implement"
    DEBUG = "debug"
    REFACTOR = "refactor"
    EXPLAIN = "explain"
    TEST = "test"
    DOCUMENT = "document"
    DEPLOY = "deploy"


class TaskIntent(BaseModel):
    """Analyzed intent of user command"""
    task_type: TaskType = Field(description="Primary type of task")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Task complexity from 0.0 to 1.0")
    estimated_duration: int = Field(description="Estimated duration in minutes")
    affected_areas: List[str] = Field(default_factory=list, description="Areas of codebase that may be affected")
    requires_reasoning: bool = Field(default=False, description="Whether task requires deep reasoning")
    requires_coordination: bool = Field(default=False, description="Whether task requires multiple agents")
    file_patterns: List[str] = Field(default_factory=list, description="File patterns likely to be involved")


class ProjectContext(BaseModel):
    """Project context information for language/framework awareness"""
    primary_language: Optional[str] = Field(default=None, description="Primary programming language")
    languages: Set[str] = Field(default_factory=set, description="All detected languages")
    framework: Optional[str] = Field(default=None, description="Primary framework (React, Django, etc.)")
    frameworks: Set[str] = Field(default_factory=set, description="All detected frameworks")
    package_manager: Optional[str] = Field(default=None, description="Package manager (npm, pip, etc.)")
    project_type: Optional[str] = Field(default=None, description="Project type (web, cli, library, etc.)")
    style_guides: Dict[str, str] = Field(default_factory=dict, description="Detected style preferences")
    existing_patterns: Dict[str, List[str]] = Field(default_factory=dict, description="Existing code patterns")


class Task(BaseModel):
    """A task to be executed by agents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    command: str = Field(description="Original user command")
    intent: TaskIntent = Field(description="Analyzed task intent")
    assigned_agents: List[str] = Field(default_factory=list, description="Agent IDs assigned to this task")
    dependencies: List[str] = Field(default_factory=list, description="Task IDs this task depends on")
    status: str = Field(default="pending", description="Task status: pending, running, completed, failed")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Task creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Task completion timestamp")
    result: Optional[Dict] = Field(default=None, description="Task execution result")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    
    # Language and project context
    target_language: Optional[str] = Field(default=None, description="Target programming language")
    target_framework: Optional[str] = Field(default=None, description="Target framework")
    project_context: Optional[ProjectContext] = Field(default=None, description="Project context information")
    
    # File context for better awareness
    relevant_files: List[str] = Field(default_factory=list, description="Files relevant to this task")
    modified_files: List[str] = Field(default_factory=list, description="Files modified by this task")
    
    # Additional fields for multi-agent coordination
    agent_type_hint: Optional[str] = Field(default=None, description="Hint for which agent type should handle this task")
    coordination_context: Dict = Field(default_factory=dict, description="Context for multi-agent coordination")
    
    # Compatibility properties for existing code
    @property
    def task_type(self) -> TaskType:
        """Get task type from intent"""
        return self.intent.task_type
    
    @property
    def complexity_score(self) -> float:
        """Get complexity score from intent"""
        return self.intent.complexity_score
    
    @property
    def estimated_duration(self) -> int:
        """Get estimated duration from intent"""
        return self.intent.estimated_duration
    
    @property
    def affected_areas(self) -> List[str]:
        """Get affected areas from intent"""
        return self.intent.affected_areas
    
    @property
    def requires_reasoning(self) -> bool:
        """Get requires reasoning from intent"""
        return self.intent.requires_reasoning
    
    @property
    def requires_coordination(self) -> bool:
        """Get requires coordination from intent"""
        return self.intent.requires_coordination
    
    @property
    def assigned_agent_id(self) -> Optional[str]:
        """Get primary assigned agent ID for compatibility"""
        return self.assigned_agents[0] if self.assigned_agents else None
    
    @classmethod
    def from_intent(cls, intent: TaskIntent, command: str) -> Task:
        """Create a Task from a TaskIntent and command"""
        return cls(
            command=command,
            intent=intent
        )
    
    def mark_started(self) -> None:
        """Mark task as started"""
        self.status = "running"
        self.started_at = datetime.utcnow()
    
    def mark_completed(self, result: Dict) -> None:
        """Mark task as completed with result"""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.result = result
    
    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error"""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error = error
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed (success or failure)"""
        return self.status in ("completed", "failed")


class TaskResult(BaseModel):
    """Result of task execution"""
    task_id: str = Field(description="ID of the executed task")
    agent_id: str = Field(description="ID of the agent that executed the task")
    status: str = Field(description="Execution status: completed, failed, partial")
    output: str = Field(default="", description="Output or result from task execution")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    files_modified: List[Path] = Field(default_factory=list, description="Files that were modified")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    tokens_used: int = Field(default=0, description="Number of tokens used")
    cost: float = Field(default=0.0, description="Cost in USD")
    metadata: Dict = Field(default_factory=dict, description="Additional execution metadata")
    
    class Config:
        # Allow Path objects to be serialized
        arbitrary_types_allowed = True
    
    @property
    def success(self) -> bool:
        """Check if task execution was successful (compatibility)"""
        return self.status == "completed" and self.error is None
    
    @property
    def is_success(self) -> bool:
        """Check if task execution was successful"""
        return self.status == "completed" and self.error is None


class ExecutionPlan(BaseModel):
    """Plan for executing multiple tasks with coordination"""
    id: str = Field(description="Unique plan identifier")
    tasks: List[Task] = Field(default_factory=list, description="Tasks to be executed")
    parallel_groups: Optional[List[List[str]]] = Field(default=None, description="Parallel execution groups by task ID")
    estimated_duration: int = Field(default=0, description="Estimated total duration in minutes")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Task dependencies map")
    
    def add_task(self, task: Task, dependencies: Optional[List[str]] = None) -> None:
        """Add a task to the execution plan"""
        if dependencies:
            task.dependencies = dependencies
            self.dependencies[task.id] = dependencies
        self.tasks.append(task)
        self._recalculate_parallel_groups()
    
    def _recalculate_parallel_groups(self) -> None:
        """Recalculate parallel execution groups based on task dependencies"""
        # Simple topological sort for task dependencies
        task_map = {task.id: task for task in self.tasks}
        visited = set()
        parallel_groups = []
        
        def get_ready_tasks() -> List[str]:
            """Get tasks that have no unmet dependencies"""
            ready = []
            for task in self.tasks:
                if task.id not in visited:
                    # Check if all dependencies are satisfied
                    deps_met = all(dep_id in visited for dep_id in task.dependencies)
                    if deps_met:
                        ready.append(task.id)
            return ready
        
        while len(visited) < len(self.tasks):
            ready_tasks = get_ready_tasks()
            if not ready_tasks:
                # Circular dependency or orphaned tasks
                remaining = [task.id for task in self.tasks if task.id not in visited]
                parallel_groups.append(remaining)
                visited.update(remaining)
                break
            
            parallel_groups.append(ready_tasks)
            visited.update(ready_tasks)
        
        self.parallel_groups = parallel_groups
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_dependent_tasks(self, task_id: str) -> List[Task]:
        """Get tasks that depend on the given task"""
        dependent_tasks = []
        for task in self.tasks:
            if task_id in task.dependencies:
                dependent_tasks.append(task)
        return dependent_tasks
    
    def validate_dependencies(self) -> List[str]:
        """Validate that all task dependencies exist"""
        errors = []
        task_ids = {task.id for task in self.tasks}
        
        for task in self.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    errors.append(f"Task {task.id} depends on non-existent task {dep_id}")
        
        return errors 