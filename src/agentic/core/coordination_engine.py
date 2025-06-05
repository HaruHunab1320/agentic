"""
Coordination Engine for managing multi-agent task execution
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agentic.core.agent_registry import AgentRegistry
from agentic.core.shared_memory import SharedMemory
from agentic.models.agent import AgentSession
from agentic.models.task import Task, TaskResult
from agentic.utils.logging import LoggerMixin


@dataclass
class ConflictDetection:
    """Represents a conflict between tasks"""
    conflict_type: str  # "file_conflict", "dependency_conflict", "resource_conflict"
    affected_files: List[Path]
    conflicting_agents: List[str]
    severity: str  # "low", "medium", "high"
    auto_resolvable: bool
    resolution_strategy: Optional[str] = None


@dataclass
class ExecutionContext:
    """Context for a multi-agent execution"""
    execution_id: str
    tasks: List[Task]
    status: str  # "running", "completed", "failed"
    started_at: datetime
    completed_tasks: List[str]
    failed_tasks: List[str]
    active_tasks: Dict[str, str]  # task_id -> agent_id
    
    @property
    def total_duration(self) -> float:
        """Get total execution duration in seconds"""
        if self.status == "running":
            return (datetime.utcnow() - self.started_at).total_seconds()
        # For completed/failed, we'd need an end time
        return 0.0


@dataclass
class ExecutionResult:
    """Result of a multi-agent execution"""
    execution_id: str
    status: str
    completed_tasks: List[str]
    failed_tasks: List[str]
    total_duration: float
    task_results: Dict[str, TaskResult]
    coordination_log: List[Dict[str, Any]]


class CoordinationEngine(LoggerMixin):
    """Coordinates multiple agents working together"""
    
    def __init__(self, agent_registry: AgentRegistry, shared_memory: SharedMemory):
        super().__init__()
        self.agent_registry = agent_registry
        self.shared_memory = shared_memory
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.background_tasks: Dict[str, asyncio.Task] = {}  # For background task execution
    
    async def execute_coordinated_tasks(self, tasks: List[Task], 
                                       coordination_plan: Optional[List[List[str]]] = None) -> ExecutionResult:
        """Execute multiple tasks with coordination"""
        execution_id = str(uuid.uuid4())
        
        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            tasks=tasks,
            status="running",
            started_at=datetime.utcnow(),
            completed_tasks=[],
            failed_tasks=[],
            active_tasks={}
        )
        
        self.active_executions[execution_id] = context
        coordination_log = []
        task_results = {}
        
        try:
            self.logger.info(f"Starting coordinated execution {execution_id} with {len(tasks)} tasks")
            
            # Register all tasks in shared memory
            for task in tasks:
                await self.shared_memory.register_task(task)
            
            # Detect conflicts before execution
            conflicts = await self._detect_conflicts(tasks)
            if conflicts:
                coordination_log.append({
                    "timestamp": datetime.utcnow(),
                    "type": "conflict_detection",
                    "conflicts": len(conflicts),
                    "details": [self._conflict_to_dict(c) for c in conflicts]
                })
                
                # Resolve conflicts
                resolved_plan = await self._resolve_conflicts(conflicts, tasks, coordination_plan)
                coordination_plan = resolved_plan
            
            # Execute tasks according to plan
            if coordination_plan:
                # Execute in planned parallel groups
                for group_index, parallel_group in enumerate(coordination_plan):
                    group_tasks = [t for t in tasks if t.id in parallel_group]
                    
                    coordination_log.append({
                        "timestamp": datetime.utcnow(),
                        "type": "parallel_group_start",
                        "group_index": group_index,
                        "task_count": len(group_tasks)
                    })
                    
                    group_results = await self._execute_parallel_group(group_tasks, context)
                    task_results.update(group_results)
                    
                    # Update context
                    for task in group_tasks:
                        if task.id in group_results and group_results[task.id].success:
                            context.completed_tasks.append(task.id)
                        else:
                            context.failed_tasks.append(task.id)
            else:
                # Execute all tasks in parallel (no coordination needed)
                coordination_log.append({
                    "timestamp": datetime.utcnow(),
                    "type": "parallel_execution_start",
                    "task_count": len(tasks)
                })
                
                all_results = await self._execute_parallel_group(tasks, context)
                task_results.update(all_results)
                
                for task in tasks:
                    if task.id in all_results and all_results[task.id].success:
                        context.completed_tasks.append(task.id)
                    else:
                        context.failed_tasks.append(task.id)
            
            # Determine overall status
            if context.failed_tasks:
                context.status = "failed"
            else:
                context.status = "completed"
            
            coordination_log.append({
                "timestamp": datetime.utcnow(),
                "type": "execution_complete",
                "status": context.status,
                "completed": len(context.completed_tasks),
                "failed": len(context.failed_tasks)
            })
            
        except Exception as e:
            self.logger.error(f"Coordinated execution {execution_id} failed: {e}")
            context.status = "failed"
            coordination_log.append({
                "timestamp": datetime.utcnow(),
                "type": "execution_error",
                "error": str(e)
            })
            
            # Attempt rollback
            await self._rollback_execution(context, task_results)
            
        finally:
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
        
        # Complete all tasks in shared memory
        for task in tasks:
            if task.id in task_results:
                await self.shared_memory.complete_task(task.id, task_results[task.id])
        
        return ExecutionResult(
            execution_id=execution_id,
            status=context.status,
            completed_tasks=context.completed_tasks,
            failed_tasks=context.failed_tasks,
            total_duration=context.total_duration,
            task_results=task_results,
            coordination_log=coordination_log
        )
    
    async def _execute_parallel_group(self, tasks: List[Task], context: ExecutionContext) -> Dict[str, TaskResult]:
        """Execute a group of tasks in parallel"""
        self.logger.info(f"Executing parallel group with {len(tasks)} tasks")
        
        # Assign tasks to available agents
        task_agent_pairs = []
        for task in tasks:
            agent = await self._find_best_agent_for_task(task)
            if agent:
                task_agent_pairs.append((task, agent))
                context.active_tasks[task.id] = agent.id
            else:
                self.logger.error(f"No available agent found for task {task.id}")
        
        # Execute tasks in parallel
        async def execute_single_task(task: Task, agent: AgentSession) -> TaskResult:
            try:
                # Get the actual agent instance
                agent_instance = self.agent_registry.get_agent_by_id(agent.id)
                if not agent_instance:
                    raise RuntimeError(f"Agent {agent.id} not found")
                
                # Execute task
                result = await agent_instance.execute_task(task)
                
                # Record in shared memory
                if result.success:
                    files_modified = []  # TODO: Extract from result
                    self.shared_memory.add_recent_change(
                        f"Task completed: {task.command[:50]}...",
                        files_modified,
                        agent.id
                    )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Task {task.id} failed on agent {agent.id}: {e}")
                return TaskResult(
                    task_id=task.id,
                    agent_id=agent.id,
                    success=False,
                    output="",
                    error=str(e)
                )
            finally:
                # Remove from active tasks
                if task.id in context.active_tasks:
                    del context.active_tasks[task.id]
        
        # Run all tasks in parallel
        results = await asyncio.gather(
            *[execute_single_task(task, agent) for task, agent in task_agent_pairs],
            return_exceptions=True
        )
        
        # Process results
        task_results = {}
        for (task, agent), result in zip(task_agent_pairs, results):
            if isinstance(result, Exception):
                task_results[task.id] = TaskResult(
                    task_id=task.id,
                    agent_id=agent.id,
                    success=False,
                    output="",
                    error=str(result)
                )
            else:
                task_results[task.id] = result
        
        return task_results
    
    async def _find_best_agent_for_task(self, task: Task) -> Optional[AgentSession]:
        """Find the best available agent for a task"""
        # Get available agents
        available_agents = self.agent_registry.get_available_agents()
        if not available_agents:
            return None
        
        # Use command router logic for agent selection
        # For now, simple assignment based on task areas
        for area in task.affected_areas:
            area_agents = [a for a in available_agents if area in a.agent_config.focus_areas]
            if area_agents:
                # Return least busy agent
                return min(area_agents, key=lambda a: len(getattr(a, 'current_tasks', [])))
        
        # Fallback to any available agent
        return available_agents[0]
    
    async def _detect_conflicts(self, tasks: List[Task]) -> List[ConflictDetection]:
        """Detect potential conflicts between tasks"""
        conflicts = []
        
        # File-based conflict detection
        file_task_map: Dict[Path, List[Task]] = {}
        
        for task in tasks:
            estimated_files = await self._estimate_affected_files(task)
            for file_path in estimated_files:
                if file_path not in file_task_map:
                    file_task_map[file_path] = []
                file_task_map[file_path].append(task)
        
        # Find files with multiple tasks
        for file_path, tasks_affecting_file in file_task_map.items():
            if len(tasks_affecting_file) > 1:
                conflict = ConflictDetection(
                    conflict_type="file_conflict",
                    affected_files=[file_path],
                    conflicting_agents=[task.assigned_agent_id for task in tasks_affecting_file if task.assigned_agent_id],
                    severity=self._assess_conflict_severity(tasks_affecting_file),
                    auto_resolvable=self._is_auto_resolvable(tasks_affecting_file)
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _estimate_affected_files(self, task: Task) -> List[Path]:
        """Estimate which files a task might affect"""
        # This is a simplified estimation
        # In a real implementation, this could use AI to predict file changes
        
        affected_files = []
        
        # Use project structure to find relevant files
        project_structure = self.shared_memory.get_project_structure()
        if not project_structure:
            return affected_files
        
        # Simple heuristics based on task areas
        if "frontend" in task.affected_areas:
            # Look for frontend files
            for source_dir in project_structure.source_directories:
                if any(name in str(source_dir).lower() for name in ["src", "components", "pages"]):
                    affected_files.extend(list(source_dir.glob("**/*.tsx")))
                    affected_files.extend(list(source_dir.glob("**/*.jsx")))
                    affected_files.extend(list(source_dir.glob("**/*.css")))
        
        if "backend" in task.affected_areas:
            # Look for backend files
            for source_dir in project_structure.source_directories:
                affected_files.extend(list(source_dir.glob("**/*.py")))
                affected_files.extend(list(source_dir.glob("**/*.js")))
                affected_files.extend(list(source_dir.glob("**/*.ts")))
        
        # Limit to reasonable number
        return affected_files[:20]
    
    def _assess_conflict_severity(self, tasks: List[Task]) -> str:
        """Assess the severity of a conflict"""
        # Simple heuristic
        if len(tasks) > 3:
            return "high"
        elif any(task.complexity_score > 0.8 for task in tasks):
            return "high"
        elif len(tasks) > 2:
            return "medium"
        else:
            return "low"
    
    def _is_auto_resolvable(self, tasks: List[Task]) -> bool:
        """Check if conflict can be automatically resolved"""
        # For now, assume read-only tasks don't conflict
        read_only_keywords = ["explain", "analyze", "describe", "show", "list"]
        
        for task in tasks:
            command_lower = task.command.lower()
            if not any(keyword in command_lower for keyword in read_only_keywords):
                return False
        
        return True
    
    async def _resolve_conflicts(self, conflicts: List[ConflictDetection], 
                                tasks: List[Task], 
                                original_plan: Optional[List[List[str]]]) -> List[List[str]]:
        """Resolve conflicts by creating a new execution plan"""
        self.logger.info(f"Resolving {len(conflicts)} conflicts")
        
        # Simple conflict resolution: serialize conflicting tasks
        serialized_tasks = set()
        parallel_groups = []
        
        # First, identify tasks that must be serialized
        for conflict in conflicts:
            if not conflict.auto_resolvable:
                conflicting_task_ids = [
                    task.id for task in tasks 
                    if any(str(f) in [str(af) for af in conflict.affected_files] 
                          for f in await self._estimate_affected_files(task))
                ]
                serialized_tasks.update(conflicting_task_ids)
        
        # Create execution plan
        non_conflicting_tasks = [task.id for task in tasks if task.id not in serialized_tasks]
        
        # Non-conflicting tasks can run in parallel
        if non_conflicting_tasks:
            parallel_groups.append(non_conflicting_tasks)
        
        # Conflicting tasks run one at a time
        for task_id in serialized_tasks:
            parallel_groups.append([task_id])
        
        return parallel_groups
    
    def _conflict_to_dict(self, conflict: ConflictDetection) -> Dict[str, Any]:
        """Convert conflict to dictionary for logging"""
        return {
            "type": conflict.conflict_type,
            "files": [str(f) for f in conflict.affected_files],
            "agents": conflict.conflicting_agents,
            "severity": conflict.severity,
            "auto_resolvable": conflict.auto_resolvable
        }
    
    async def _rollback_execution(self, context: ExecutionContext, task_results: Dict[str, TaskResult]) -> None:
        """Attempt to rollback changes from failed execution"""
        self.logger.warning(f"Attempting rollback for execution {context.execution_id}")
        
        # For now, just log the rollback attempt
        # In a real implementation, this could:
        # - Revert git changes
        # - Restore file backups
        # - Undo database changes
        
        rollback_tasks = []
        for task_id in context.completed_tasks:
            if task_id in task_results:
                result = task_results[task_id]
                self.logger.info(f"Would rollback task {task_id} (agent: {result.agent_id})")
                # TODO: Implement actual rollback logic
        
        if rollback_tasks:
            self.logger.info(f"Rollback would affect {len(rollback_tasks)} completed tasks")
    
    def get_active_executions(self) -> Dict[str, ExecutionContext]:
        """Get currently active executions"""
        return self.active_executions.copy()
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific execution"""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            return {
                "execution_id": execution_id,
                "status": context.status,
                "tasks_total": len(context.tasks),
                "tasks_completed": len(context.completed_tasks),
                "tasks_failed": len(context.failed_tasks),
                "tasks_active": len(context.active_tasks),
                "duration": context.total_duration,
                "started_at": context.started_at
            }
        return None
    
    async def execute_background_task(self, command: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute a task in the background without blocking the CLI"""
        from agentic.core.intent_classifier import IntentClassifier
        from agentic.models.task import Task
        
        # Create a task for the command
        intent_classifier = IntentClassifier()
        intent = await intent_classifier.analyze_intent(command)
        task = Task.from_intent(intent, command)
        
        # Create background task
        background_task = asyncio.create_task(
            self._execute_background_task_internal(task, context or {})
        )
        
        # Store the background task
        self.background_tasks[task.id] = background_task
        
        self.logger.info(f"Started background task {task.id}: {command[:50]}...")
        return task.id
    
    async def _execute_background_task_internal(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        """Internal method to execute a background task"""
        try:
            # Register task in shared memory
            await self.shared_memory.register_task(task)
            
            # Find or spawn an appropriate agent
            routing_decision = await self._route_task_to_agent(task)
            if not routing_decision:
                raise RuntimeError("No suitable agent found for background task")
            
            agent_instance = self.agent_registry.get_agent_by_id(routing_decision.id)
            if not agent_instance:
                raise RuntimeError("Agent instance not found")
            
            # Mark agent as busy
            routing_decision.mark_busy(task.id)
            
            # Execute the task with no timeout
            self.logger.info(f"Background task {task.id} starting execution...")
            result = await agent_instance.execute_task(task)
            
            # Mark task as complete
            await self.shared_memory.complete_task(task.id, result)
            
            # Mark agent as idle
            routing_decision.mark_idle()
            
            self.logger.info(f"Background task {task.id} completed with status: {result.status}")
            return result
            
        except Exception as e:
            self.logger.error(f"Background task {task.id} failed: {e}")
            error_result = TaskResult(
                task_id=task.id,
                status="failed",
                error=str(e),
                agent_id="background"
            )
            await self.shared_memory.complete_task(task.id, error_result)
            return error_result
        finally:
            # Clean up the background task reference
            if task.id in self.background_tasks:
                del self.background_tasks[task.id]
    
    async def _route_task_to_agent(self, task: Task) -> Optional[AgentSession]:
        """Route a task to an available agent"""
        from agentic.core.command_router import CommandRouter
        
        # Use the command router to find the best agent
        command_router = CommandRouter(self.agent_registry)
        routing_decision = await command_router.route_command(task.command, {})
        
        if routing_decision.primary_agent:
            return routing_decision.primary_agent
        
        # If no agent available, try to spawn one
        try:
            # Try to determine agent type from task
            from agentic.models.agent import AgentConfig, AgentType
            
            # Simple heuristic - could be improved
            if any(keyword in task.command.lower() for keyword in ['react', 'frontend', 'ui', 'component']):
                agent_type = AgentType.AIDER_FRONTEND
                focus_areas = ["frontend", "ui", "components"]
            elif any(keyword in task.command.lower() for keyword in ['test', 'testing', 'unittest']):
                agent_type = AgentType.AIDER_TESTING
                focus_areas = ["testing", "qa"]
            else:
                agent_type = AgentType.AIDER_BACKEND
                focus_areas = ["backend", "api", "database"]
            
            config = AgentConfig(
                agent_type=agent_type,
                name=f"bg_{agent_type.value}",
                workspace_path=self.agent_registry.workspace_path,
                focus_areas=focus_areas,
                ai_model_config={"model": "gemini-2.5-pro"}
            )
            
            session = await self.agent_registry.get_or_spawn_agent(config)
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to spawn agent for background task: {e}")
            return None
    
    def get_background_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a background task"""
        if task_id in self.background_tasks:
            bg_task = self.background_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "running" if not bg_task.done() else "completed",
                "done": bg_task.done(),
                "cancelled": bg_task.cancelled() if hasattr(bg_task, 'cancelled') else False
            }
        
        # Check shared memory for completed tasks
        task_progress = self.shared_memory.get_task_progress(task_id)
        if task_progress:
            return {
                "task_id": task_id,
                "status": task_progress.get("status", "unknown"),
                "done": task_progress.get("status") in ["completed", "failed"],
                "result": task_progress.get("result")
            }
        
        return None
    
    def get_all_background_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all background tasks"""
        statuses = {}
        
        # Active background tasks
        for task_id in self.background_tasks:
            status = self.get_background_task_status(task_id)
            if status:
                statuses[task_id] = status
        
        # Recent completed tasks from shared memory
        try:
            task_progress = self.shared_memory.get_all_task_progress()
            for task_id, progress in task_progress.items():
                if task_id not in statuses:  # Don't overwrite active tasks
                    statuses[task_id] = {
                        "task_id": task_id,
                        "status": progress.get("status", "unknown"),
                        "done": True,
                        "completed_at": progress.get("completed_at")
                    }
        except Exception:
            pass  # Shared memory might not have this method yet
        
        return statuses
    
    async def cancel_background_task(self, task_id: str) -> bool:
        """Cancel a running background task"""
        if task_id in self.background_tasks:
            bg_task = self.background_tasks[task_id]
            if not bg_task.done():
                bg_task.cancel()
                self.logger.info(f"Cancelled background task {task_id}")
                return True
        return False 