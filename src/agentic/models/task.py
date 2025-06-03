"""
Task execution models
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of development tasks"""
    IMPLEMENT = "implement"
    DEBUG = "debug"
    REFACTOR = "refactor"
    EXPLAIN = "explain"
    TEST = "test"
    DOCUMENT = "document"


class TaskIntent(BaseModel):
    """Analyzed intent of user command"""
    task_type: TaskType = Field(description="Primary type of task")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Task complexity from 0.0 to 1.0")
    estimated_duration: int = Field(description="Estimated duration in minutes")
    affected_areas: List[str] = Field(default_factory=list, description="Areas of codebase that may be affected")
    requires_reasoning: bool = Field(default=False, description="Whether task requires deep reasoning")
    requires_coordination: bool = Field(default=False, description="Whether task requires multiple agents")
    file_patterns: List[str] = Field(default_factory=list, description="File patterns likely to be involved")


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
    def is_success(self) -> bool:
        """Check if task execution was successful"""
        return self.status == "completed" and self.error is None


class ExecutionPlan(BaseModel):
    """Plan for executing a command across multiple agents"""
    command: str = Field(description="Original command to execute")
    tasks: List[Task] = Field(default_factory=list, description="Tasks to be executed")
    execution_order: List[List[str]] = Field(default_factory=list, description="Parallel execution groups by task ID")
    estimated_duration: int = Field(default=0, description="Estimated total duration in minutes")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    
    def add_task(self, task: Task, dependencies: Optional[List[str]] = None) -> None:
        """Add a task to the execution plan"""
        if dependencies:
            task.dependencies = dependencies
        self.tasks.append(task)
        self._recalculate_execution_order()
    
    def _recalculate_execution_order(self) -> None:
        """Recalculate execution order based on task dependencies"""
        # Simple topological sort for task dependencies
        task_map = {task.id: task for task in self.tasks}
        visited = set()
        execution_groups = []
        
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
                execution_groups.append(remaining)
                visited.update(remaining)
                break
            
            execution_groups.append(ready_tasks)
            visited.update(ready_tasks)
        
        self.execution_order = execution_groups 