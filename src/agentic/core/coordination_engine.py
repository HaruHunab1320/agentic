"""
Coordination Engine for managing multi-agent task execution
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import os

from agentic.core.agent_registry import AgentRegistry
from agentic.core.shared_memory import SharedMemory
try:
    from agentic.core.swarm_monitor_enhanced import SwarmMonitorEnhanced as SwarmMonitor, AgentStatus
except ImportError:
    # Fall back to basic monitor if enhanced not available
    from agentic.core.swarm_monitor import SwarmMonitor, AgentStatus
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
    verification_status: Optional[str] = None  # "passed", "failed_after_retries", or None if not run


class CoordinationEngine(LoggerMixin):
    """Coordinates multiple agents working together"""
    
    def __init__(self, agent_registry: AgentRegistry, shared_memory: SharedMemory):
        super().__init__()
        self.agent_registry = agent_registry
        self.shared_memory = shared_memory
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.background_tasks: Dict[str, asyncio.Task] = {}  # For background task execution
        # Check if we're in an environment that might conflict with alternate screen
        # (e.g., when running tests or in CI)
        use_alternate_screen = not (
            os.environ.get('CI') or 
            os.environ.get('GITHUB_ACTIONS') or
            os.environ.get('AGENTIC_NO_ALTERNATE_SCREEN')
        )
        self.swarm_monitor = SwarmMonitor(use_alternate_screen=use_alternate_screen)  # Real-time monitoring
    
    async def execute_coordinated_tasks(self, tasks: List[Task], 
                                       coordination_plan: Optional[List[List[str]]] = None,
                                       execution_mode: Optional['ExecutionMode'] = None) -> ExecutionResult:
        """Execute multiple tasks with coordination"""
        execution_id = str(uuid.uuid4())
        
        # Start monitoring if we have any tasks
        should_monitor = len(tasks) > 0
        original_root_level = None
        original_agentic_level = None
        
        if should_monitor:
            await self.swarm_monitor.start_monitoring()
            
            # Update task analysis if using enhanced monitor
            if hasattr(self.swarm_monitor, 'update_task_analysis'):
                # Get project structure for analysis
                project_structure = self.shared_memory.get_project_structure()
                if project_structure:
                    total_files = sum(len(list(d.rglob('*'))) for d in project_structure.source_directories)
                    complexity = sum(task.complexity_score for task in tasks) / len(tasks) if tasks else 0.0
                    
                    # Determine suggested agents based on tasks
                    suggested_agents = set()
                    for task in tasks:
                        if hasattr(task, 'agent_type_hint'):
                            suggested_agents.add(task.agent_type_hint)
                        elif task.affected_areas:
                            for area in task.affected_areas:
                                if 'frontend' in area:
                                    suggested_agents.add('aider_frontend')
                                elif 'backend' in area:
                                    suggested_agents.add('aider_backend')
                                elif 'test' in area:
                                    suggested_agents.add('aider_testing')
                                else:
                                    suggested_agents.add('claude_code')
                    
                    self.swarm_monitor.update_task_analysis(
                        total_files=total_files,
                        complexity=complexity,
                        suggested_agents=list(suggested_agents)
                    )
            
            # Suppress ALL logs during monitoring for cleaner display
            import logging
            
            # Get root logger and all agentic loggers
            root_logger = logging.getLogger()
            agentic_logger = logging.getLogger('agentic')
            
            # Store original levels
            original_root_level = root_logger.level
            original_agentic_level = agentic_logger.level
            
            # Set to ERROR to suppress most logs
            root_logger.setLevel(logging.ERROR)
            agentic_logger.setLevel(logging.ERROR)
        
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
            error_details = traceback.format_exc()
            self.logger.error(f"Coordinated execution {execution_id} failed: {e}")
            self.logger.error(f"Full traceback:\n{error_details}")
            context.status = "failed"
            coordination_log.append({
                "timestamp": datetime.utcnow(),
                "type": "execution_error",
                "error": str(e),
                "traceback": error_details
            })
            
            # Create error results for all tasks that don't have results yet
            for task in tasks:
                if task.id not in task_results:
                    task_results[task.id] = TaskResult(
                        task_id=task.id,
                        agent_id="orchestrator",
                        status="failed",
                        output="",
                        error=f"Execution failed: {str(e)}"
                    )
                    context.failed_tasks.append(task.id)
            
            # Attempt rollback
            await self._rollback_execution(context, task_results)
            
        finally:
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            # Stop monitoring and restore logging if it was started
            if should_monitor:
                await self.swarm_monitor.stop_monitoring()
                # Restore logging levels
                if 'original_root_level' in locals():
                    import logging
                    logging.getLogger().setLevel(original_root_level)
                    logging.getLogger('agentic').setLevel(original_agentic_level)
        
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
            # Log task details for debugging
            role = getattr(task, 'coordination_context', {}).get('role', 'general')
            agent_type_hint = getattr(task, 'agent_type_hint', 'none')
            self.logger.info(f"[ROUTING] Task {task.id} - Role: {role}, Type hint: {agent_type_hint}")
            
            agent = await self._find_best_agent_for_task(task)
            if agent:
                task_agent_pairs.append((task, agent))
                context.active_tasks[task.id] = agent.id
                self.logger.info(f"[ROUTING] Assigned task {task.id} to agent {agent.agent_config.name} ({agent.agent_config.agent_type.value})")
                
                # Determine a meaningful role based on agent type and task
                if "test" in task.command.lower():
                    display_role = "test_runner"
                elif "claude" in agent.agent_config.agent_type.value.lower():
                    display_role = "analyzer"
                elif "frontend" in agent.agent_config.agent_type.value.lower():
                    display_role = "frontend_dev"
                elif "backend" in agent.agent_config.agent_type.value.lower():
                    display_role = "backend_dev"
                else:
                    display_role = role if role != "general" else agent.agent_config.agent_type.value.replace("_", " ").title()
                
                # Register agent with monitor
                self.swarm_monitor.register_agent(
                    agent_id=agent.id,
                    agent_name=agent.agent_config.name,
                    agent_type=agent.agent_config.agent_type.value,
                    role=display_role
                )
                self.swarm_monitor.update_agent_status(agent.id, AgentStatus.INITIALIZING)
                
                # Set task list for enhanced monitor if available
                if hasattr(self.swarm_monitor, 'set_agent_tasks'):
                    # Get all tasks assigned to this agent
                    # Note: We only have one task per agent in single agent execution
                    agent_tasks = [(task.id, task.command[:60])]
                    self.swarm_monitor.set_agent_tasks(agent.id, agent_tasks)
            else:
                self.logger.error(f"[ROUTING] No available agent found for task {task.id}")
                # Add task to failed results
                context.failed_tasks.append(task.id)
                # Store a proper error result for this task
                task_agent_pairs.append((task, None))  # Add with None agent to handle error properly
        
        # Check if we have any tasks to execute
        if not task_agent_pairs:
            self.logger.error("No tasks could be assigned to agents")
            return {}
        
        # Execute tasks in parallel
        async def execute_single_task(task: Task, agent: Optional[AgentSession]) -> TaskResult:
            try:
                # Handle case where no agent could be assigned
                if agent is None:
                    self.logger.error(f"No agent available to execute task {task.id}")
                    return TaskResult(
                        task_id=task.id,
                        agent_id="none",
                        status="failed",
                        output="",
                        error="No agent available to execute this task. Agent spawn failed."
                    )
                
                # Get the actual agent instance
                agent_instance = self.agent_registry.get_agent_by_id(agent.id)
                if not agent_instance:
                    # Log more debugging info
                    self.logger.error(f"Agent {agent.id} not found in registry")
                    self.logger.error(f"Active agents: {list(self.agent_registry.agents.keys())}")
                    self.logger.error(f"Agent session status: {agent.status}")
                    raise RuntimeError(f"Agent {agent.id} not found in registry")
                
                # Update monitor - starting task
                self.swarm_monitor.update_agent_status(agent.id, AgentStatus.SETTING_UP)
                
                # Use enhanced monitor methods if available
                if hasattr(self.swarm_monitor, 'start_agent_task'):
                    self.swarm_monitor.start_agent_task(agent.id, task.id, task.command[:50] + "...")
                else:
                    self.swarm_monitor.start_task(agent.id, task.id, task.command[:50] + "...")
                
                # Execute task
                result = await agent_instance.execute_task(task)
                
                # Check success based on status field (not success attribute)
                is_success = result.status == "completed"
                
                # Log the result for debugging
                role = getattr(task, 'coordination_context', {}).get('role', 'general')
                self.logger.info(f"Task {task.id} ({role}) completed with status: {result.status}")
                
                # Update monitor - task completed
                if hasattr(self.swarm_monitor, 'complete_agent_task'):
                    self.swarm_monitor.complete_agent_task(agent.id, task.id, is_success)
                else:
                    self.swarm_monitor.complete_task(agent.id, task.id, is_success)
                
                if is_success:
                    self.swarm_monitor.update_agent_status(agent.id, AgentStatus.COMPLETED)
                    self.logger.info(f"Agent {agent.id} ({role}) marked as COMPLETED")
                    
                    # Track files created
                    if hasattr(result, 'files_modified') and result.files_modified:
                        for file_path in result.files_modified:
                            if hasattr(self.swarm_monitor, 'add_file_modified'):
                                self.swarm_monitor.add_file_modified(agent.id, str(file_path))
                            else:
                                self.swarm_monitor.add_file_created(agent.id, str(file_path))
                    
                    # Record in shared memory
                    files_modified = result.files_modified if hasattr(result, 'files_modified') else []
                    self.shared_memory.add_recent_change(
                        f"Task completed: {task.command[:50]}...",
                        files_modified,
                        agent.id
                    )
                else:
                    # Use a more descriptive error message
                    error_msg = result.error
                    if not error_msg:
                        # Try to extract error from output if available
                        if result.output:
                            # Take first line or first 100 chars of output as error
                            error_msg = result.output.split('\n')[0][:100]
                            if len(error_msg) == 100:
                                error_msg += "..."
                        else:
                            error_msg = f"Task failed with status: {result.status}"
                    self.swarm_monitor.add_error(agent.id, error_msg)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Task {task.id} failed on agent {agent.id}: {e}")
                self.logger.error(f"Task execution traceback:\n{traceback.format_exc()}")
                return TaskResult(
                    task_id=task.id,
                    agent_id=agent.id,
                    status="failed",  # Set status field properly
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
                    status="failed",  # Set status field properly
                    output="",
                    error=str(result)
                )
            else:
                task_results[task.id] = result
        
        return task_results
    
    async def _find_best_agent_for_task(self, task: Task) -> Optional[AgentSession]:
        """Find the best available agent for a task, spawning one if needed"""
        # For multi-agent coordination, always spawn fresh agents to ensure proper separation
        # This prevents tasks from being incorrectly assigned to the same agent
        
        # Skip reusing existing agents for multi-agent tasks
        # Each role should get its own dedicated agent instance
        
        # No agents available, try to spawn one based on task hint or areas
        try:
            from agentic.models.agent import AgentConfig, AgentType
            
            # Use agent_type_hint if available
            if hasattr(task, 'agent_type_hint') and task.agent_type_hint:
                self.logger.info(f"Using agent_type_hint: {task.agent_type_hint}")
                if task.agent_type_hint == 'claude_code':
                    agent_type = AgentType.CLAUDE_CODE
                elif task.agent_type_hint == 'aider_frontend':
                    agent_type = AgentType.AIDER_FRONTEND
                elif task.agent_type_hint == 'aider_backend':
                    agent_type = AgentType.AIDER_BACKEND
                elif task.agent_type_hint == 'aider_testing':
                    agent_type = AgentType.AIDER_TESTING
                else:
                    # Default to Claude Code for unknown types
                    self.logger.warning(f"Unknown agent_type_hint: {task.agent_type_hint}, defaulting to Claude Code")
                    agent_type = AgentType.CLAUDE_CODE
            else:
                # Fall back to area-based detection
                if any(keyword in task.affected_areas for keyword in ['frontend', 'ui', 'components']):
                    agent_type = AgentType.AIDER_FRONTEND
                elif any(keyword in task.affected_areas for keyword in ['testing', 'qa']):
                    agent_type = AgentType.AIDER_TESTING
                elif any(keyword in task.affected_areas for keyword in ['analysis', 'architecture', 'design']):
                    agent_type = AgentType.CLAUDE_CODE
                else:
                    agent_type = AgentType.AIDER_BACKEND
            
            # Create agent config with unique name per role
            # Get role from coordination context if available
            role = getattr(task, 'coordination_context', {}).get('role', 'general')
            
            # Configure model based on agent type
            if agent_type in [AgentType.AIDER_FRONTEND, AgentType.AIDER_BACKEND, AgentType.AIDER_TESTING]:
                model_config = {"model": "gemini/gemini-2.0-flash-exp"}  # Aider format
            else:
                model_config = {"model": "sonnet"}  # Claude Code default
            
            config = AgentConfig(
                agent_type=agent_type,
                name=f"{role}_{agent_type.value.lower()}_{task.id[:8]}",  # Unique name per task
                workspace_path=self.agent_registry.workspace_path,
                focus_areas=task.affected_areas if task.affected_areas else ["general"],
                ai_model_config=model_config
            )
            
            # Spawn agent
            session = await self.agent_registry.get_or_spawn_agent(config)
            
            # Check if agent actually started successfully
            if session.status != "active":
                self.logger.error(f"Agent spawn failed for {agent_type.value} - status: {session.status}")
                return None
            
            self.logger.info(f"Spawned {agent_type.value} agent '{config.name}' for task: {task.id}")
            
            # Set monitor reference in agent for status updates
            agent_instance = self.agent_registry.get_agent_by_id(session.id)
            if agent_instance and hasattr(agent_instance, 'set_monitor'):
                agent_instance.set_monitor(self.swarm_monitor, session.id)
            else:
                self.logger.warning(f"Could not set monitor for agent {session.id}")
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to spawn agent for task {task.id}: {e}")
            self.logger.error(f"Spawn error traceback:\n{traceback.format_exc()}")
            return None
    
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
    
    async def execute_multi_agent_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Decompose a single complex command into multiple parallel agent tasks"""
        try:
            self.logger.info(f"Decomposing multi-agent command: {command[:50]}...")
            
            # Analyze the command to determine what agents are needed
            agent_tasks = await self._decompose_command_into_agent_tasks(command, context or {})
            
            if len(agent_tasks) <= 1:
                self.logger.info("Command doesn't require multi-agent coordination, using single agent")
                # Fall back to single agent execution
                return await self._execute_single_agent_fallback(command, context or {})
            
            self.logger.info(f"Command decomposed into {len(agent_tasks)} parallel agent tasks")
            
            # Create tasks for each agent
            tasks = []
            for agent_task in agent_tasks:
                from agentic.models.task import Task
                from agentic.core.intent_classifier import IntentClassifier
                
                # Create intent for this specific agent task
                intent_classifier = IntentClassifier()
                intent = await intent_classifier.analyze_intent(agent_task['command'])
                
                # Override intent with agent-specific info (TaskIntent doesn't have agent_type field)
                intent.affected_areas = agent_task['focus_areas']
                
                # Create task
                task = Task.from_intent(intent, agent_task['command'])
                task.agent_type_hint = agent_task['agent_type']  # Hint for routing
                task.coordination_context = agent_task.get('coordination_context', {})
                
                # Debug: Log task creation
                role = agent_task.get('coordination_context', {}).get('role', 'unknown')
                self.logger.info(f"Created task for {role}: agent_type={agent_task['agent_type']}, task_id={task.id}")
                tasks.append(task)
            
            # Start monitoring display
            await self.swarm_monitor.start_monitoring()
            
            # Suppress ALL logs during monitoring for cleaner display
            import logging
            
            # Get root logger and all agentic loggers
            root_logger = logging.getLogger()
            agentic_logger = logging.getLogger('agentic')
            
            # Store original levels
            original_root_level = root_logger.level
            original_agentic_level = agentic_logger.level
            
            # Set to ERROR to suppress most logs
            root_logger.setLevel(logging.ERROR)
            agentic_logger.setLevel(logging.ERROR)
            
            try:
                # Execute all tasks in parallel with coordination
                result = await self.execute_coordinated_tasks(tasks)
                
                # Send inter-agent messages for coordination
                await self._send_coordination_messages(agent_tasks, result)
                
                # If initial execution succeeded, run verification phase
                if result.status == "completed" and context.get('verify', True):
                    self.logger.info("Starting verification phase...")
                    
                    # Run verification and fixes
                    verification_passed = await self._run_verification_loop(result)
                    
                    # Update result with verification status
                    result.verification_status = "passed" if verification_passed else "failed_after_retries"
                    
                    if not verification_passed:
                        self.logger.warning("Verification failed after maximum iterations")
                
                return result
            finally:
                # Restore logging levels
                if 'original_root_level' in locals():
                    logging.getLogger().setLevel(original_root_level)
                    logging.getLogger('agentic').setLevel(original_agentic_level)
                else:
                    # Fallback if variables weren't set
                    logging.getLogger().setLevel(logging.INFO)
                
                # Stop monitoring and show final summary
                await self.swarm_monitor.stop_monitoring()
            
        except Exception as e:
            self.logger.error(f"Multi-agent command execution failed: {e}")
            raise
    
    async def _decompose_command_into_agent_tasks(self, command: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Intelligently decompose a command into specific agent tasks"""
        command_lower = command.lower()
        agent_tasks = []
        
        # Get agent type strategy from context
        agent_type_strategy = context.get('agent_type_strategy', 'dynamic')
        
        # Check if this is a coordinated analysis request
        if context.get('coordination_type') == 'analysis':
            return await self._decompose_analysis_query(command, context)
        
        # Try LLM-based decomposition first
        if context.get('use_llm_decomposition', True):
            llm_tasks = await self._decompose_with_llm(command, context)
            if llm_tasks:
                return llm_tasks
        
        # Define patterns that suggest multi-agent coordination
        multi_component_patterns = [
            'full stack', 'complete application', 'complete app', 'complete todo', 'complete web',
            'end-to-end', 'frontend and backend', 'react frontend', 'fastapi backend',
            'with tests', 'including tests', 'comprehensive tests', 'and documentation', 'with deployment',
            'microservices', 'multiple services', 'system', 'platform'
        ]
        
        # Check if this is a multi-component request
        is_multi_component = any(pattern in command_lower for pattern in multi_component_patterns)
        
        if not is_multi_component:
            return []  # Single agent will handle
        
        # Task 1: Architecture Analysis
        if any(keyword in command_lower for keyword in ['complete', 'system', 'platform', 'architecture']):
            architect_agent_type = self._get_agent_type_for_role('architect', agent_type_strategy)
            agent_tasks.append({
                'agent_type': architect_agent_type,
                'focus_areas': ['analysis', 'architecture', 'design'],
                'command': f'Create a complete technical specification and project structure for: {command}. Write actual files including README.md, project structure documentation, and API specification files. Focus on component design, data flow, and API contracts.',
                'coordination_context': {
                    'role': 'architect',
                    'dependencies': [],
                    'provides': ['technical_spec', 'api_contracts', 'project_structure']
                }
            })
        
        # Task 2: Backend Development
        # Also include backend for complete systems/applications
        if any(keyword in command_lower for keyword in ['backend', 'api', 'server', 'database', 'authentication', 'fastapi', 'django', 'flask']) or \
           (any(keyword in command_lower for keyword in ['system', 'application', 'platform']) and 'automated' in command_lower):
            backend_agent_type = self._get_agent_type_for_role('backend_developer', agent_type_strategy)
            agent_tasks.append({
                'agent_type': backend_agent_type,
                'focus_areas': ['backend', 'api', 'database', 'authentication'],
                'command': f'Create complete backend/API code for: {command}. Use the language/framework specified in the request (e.g., TypeScript/Node.js, Python/FastAPI, etc.). Create working, runnable backend code with proper file structure.',
                'coordination_context': {
                    'role': 'backend_developer',
                    'dependencies': ['technical_spec'],
                    'provides': ['api_endpoints', 'database_schema', 'backend_services']
                }
            })
        
        # Task 3: Frontend Development
        # Also include frontend for complete systems that need monitoring/dashboards
        if any(keyword in command_lower for keyword in ['frontend', 'ui', 'react', 'vue', 'angular', 'dashboard', 'interface', 'components']) or \
           (any(keyword in command_lower for keyword in ['system', 'application', 'platform']) and any(keyword in command_lower for keyword in ['manage', 'monitor', 'automated'])):
            frontend_agent_type = self._get_agent_type_for_role('frontend_developer', agent_type_strategy)
            self.logger.info(f"Frontend decomposition: strategy={agent_type_strategy}, role='frontend_developer', mapped_type={frontend_agent_type}")
            agent_tasks.append({
                'agent_type': frontend_agent_type,
                'focus_areas': ['frontend', 'ui', 'components', 'styling'],
                'command': f'Create complete frontend/UI code for: {command}. Write actual React components, pages, styling, and API integration code. Create working, runnable frontend code with proper component structure.',
                'coordination_context': {
                    'role': 'frontend_developer',
                    'dependencies': ['api_contracts', 'technical_spec'],
                    'provides': ['ui_components', 'user_interface', 'client_integration']
                }
            })
        
        # Task 4: Testing & QA
        if any(keyword in command_lower for keyword in ['test', 'testing', 'qa', 'quality']) or 'complete' in command_lower:
            testing_agent_type = self._get_agent_type_for_role('qa_engineer', agent_type_strategy)
            agent_tasks.append({
                'agent_type': testing_agent_type,
                'focus_areas': ['testing', 'qa', 'automation'],
                'command': f'Create comprehensive tests for: {command}. Implement unit tests, integration tests, and end-to-end testing. Test all components and their interactions.',
                'coordination_context': {
                    'role': 'qa_engineer',
                    'dependencies': ['backend_services', 'ui_components'],
                    'provides': ['test_suite', 'quality_assurance', 'test_automation']
                }
            })
        
        # Task 5: DevOps/Deployment (if deployment mentioned)
        if any(keyword in command_lower for keyword in ['deploy', 'docker', 'kubernetes', 'aws', 'production', 'ci/cd']):
            devops_agent_type = self._get_agent_type_for_role('devops_engineer', agent_type_strategy)
            agent_tasks.append({
                'agent_type': devops_agent_type,
                'focus_areas': ['devops', 'deployment', 'infrastructure'],
                'command': f'Create deployment configuration for: {command}. Set up Docker, CI/CD, and production deployment scripts.',
                'coordination_context': {
                    'role': 'devops_engineer',
                    'dependencies': ['backend_services', 'ui_components', 'test_suite'],
                    'provides': ['deployment_config', 'infrastructure', 'ci_cd_pipeline']
                }
            })
        
        self.logger.info(f"Decomposed command into {len(agent_tasks)} agent tasks: {[task['coordination_context']['role'] for task in agent_tasks]}")
        return agent_tasks
    
    async def _decompose_with_llm(self, command: str, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Use an LLM to intelligently decompose a complex command into agent tasks"""
        try:
            # For now, we'll use a more intelligent keyword-based approach
            # In a full implementation, this would call an LLM API
            
            # Analyze the command to understand what's needed
            is_system = any(word in command.lower() for word in ['system', 'application', 'platform', 'solution'])
            is_automated = any(word in command.lower() for word in ['automated', 'automatic', 'self-learning', 'autonomous'])
            needs_data = any(word in command.lower() for word in ['data', 'database', 'storage', 'analytics'])
            needs_monitoring = any(word in command.lower() for word in ['monitor', 'manage', 'dashboard', 'visualize'])
            needs_ml = any(word in command.lower() for word in ['learning', 'ml', 'ai', 'predict', 'strategy'])
            is_production = any(word in command.lower() for word in ['production', 'deploy', 'scale'])
            
            # For a production-ready automated system, we need all components
            if is_system and is_automated:
                agent_type_strategy = context.get('agent_type_strategy', 'dynamic')
                tasks = []
                
                # 1. Architecture & Design
                architect_type = self._get_agent_type_for_role('architect', agent_type_strategy)
                tasks.append({
                    'agent_type': architect_type,
                    'focus_areas': ['analysis', 'architecture', 'design', 'system-design'],
                    'command': f'Design the complete architecture for: {command}. Write architecture documentation files including: README.md with system overview, ARCHITECTURE.md with technical design, API_SPEC.md with API contracts, and DATA_MODEL.md with database schemas. Focus on clear, actionable specifications that other developers can implement from.',
                    'coordination_context': {
                        'role': 'architect',
                        'dependencies': [],
                        'provides': ['technical_spec', 'api_contracts', 'system_design']
                    }
                })
                
                # 2. Backend/Core Logic - Split for better distribution
                backend_type = self._get_agent_type_for_role('backend_developer', agent_type_strategy)
                
                # Core Backend Services
                tasks.append({
                    'agent_type': backend_type,
                    'focus_areas': ['backend', 'api', 'core-services'],
                    'command': f'Implement the core backend services and APIs for: {command}. Use TypeScript/Node.js as specified. Focus on API endpoints, service layer, and business logic. Create RESTful APIs and service interfaces.',
                    'coordination_context': {
                        'role': 'backend_developer',
                        'sub_role': 'api_developer',
                        'dependencies': ['technical_spec'],
                        'provides': ['api_endpoints', 'core_services', 'service_interfaces']
                    }
                })
                
                # Data Layer & Integration
                tasks.append({
                    'agent_type': backend_type,
                    'focus_areas': ['backend', 'data', 'integration'],
                    'command': f'Implement the data layer and external integrations for: {command}. Create data models, database schemas, and integrate with blockchain/Meteora APIs. Handle data persistence and external service connections.',
                    'coordination_context': {
                        'role': 'backend_developer',
                        'sub_role': 'data_engineer',
                        'dependencies': ['technical_spec'],
                        'provides': ['data_models', 'integrations', 'blockchain_connectors']
                    }
                })
                
                # 3. Frontend/Dashboard - Needed for monitoring
                if needs_monitoring or is_automated:
                    frontend_type = self._get_agent_type_for_role('frontend_developer', agent_type_strategy)
                    tasks.append({
                        'agent_type': frontend_type,
                        'focus_areas': ['frontend', 'dashboard', 'monitoring', 'visualization'],
                        'command': f'Build the monitoring dashboard and UI for: {command}. Create real-time dashboards, control panels, and data visualization.',
                        'coordination_context': {
                            'role': 'frontend_developer',
                            'dependencies': ['api_contracts', 'technical_spec'],
                            'provides': ['user_interface', 'dashboards', 'monitoring_ui']
                        }
                    })
                
                # 4. ML/Strategy Engine - For self-learning systems
                # Break down complex ML tasks into subtasks for better distribution
                if needs_ml or 'strategy' in command.lower():
                    # Always use the mapping for ML roles
                    ml_type = self._get_agent_type_for_role('ml_engineer', agent_type_strategy)
                    
                    # ML Strategy Designer
                    tasks.append({
                        'agent_type': ml_type,
                        'focus_areas': ['machine-learning', 'strategy', 'architecture'],
                        'command': f'Design the ML architecture and strategy algorithms for: {command}. Focus on the learning system design, model selection, and optimization strategies. Create architecture documentation and model specifications.',
                        'coordination_context': {
                            'role': 'ml_engineer',
                            'sub_role': 'ml_architect',
                            'dependencies': ['core_services', 'data_models'],
                            'provides': ['ml_architecture', 'strategy_design', 'model_specs']
                        }
                    })
                    
                    # ML Implementation Engineer
                    ml_impl_type = self._get_agent_type_for_role('ml_implementer', agent_type_strategy)
                    tasks.append({
                        'agent_type': ml_impl_type,
                        'focus_areas': ['machine-learning', 'implementation', 'algorithms'],
                        'command': f'Implement the ML models and learning algorithms for: {command}. Create the actual code for data processing, model training, prediction, and strategy execution.',
                        'coordination_context': {
                            'role': 'ml_implementer',
                            'sub_role': 'ml_implementation', 
                            'dependencies': ['ml_architecture', 'core_services'],
                            'provides': ['ml_models', 'learning_system', 'prediction_engine']
                        }
                    })
                
                # 5. Testing - Always needed
                testing_type = self._get_agent_type_for_role('qa_engineer', agent_type_strategy)
                tasks.append({
                    'agent_type': testing_type,
                    'focus_areas': ['testing', 'qa', 'validation'],
                    'command': f'Create comprehensive tests for: {command}. Implement unit tests, integration tests, and system validation.',
                    'coordination_context': {
                        'role': 'qa_engineer',
                        'dependencies': ['core_services', 'user_interface'],
                        'provides': ['test_suite', 'quality_assurance']
                    }
                })
                
                # 6. DevOps - For production systems
                if is_production:
                    devops_type = self._get_agent_type_for_role('devops_engineer', agent_type_strategy)
                    tasks.append({
                        'agent_type': devops_type,
                        'focus_areas': ['devops', 'deployment', 'infrastructure'],
                        'command': f'Set up production deployment for: {command}. Create Docker configs, CI/CD pipelines, and monitoring.',
                        'coordination_context': {
                            'role': 'devops_engineer',
                            'dependencies': ['core_services', 'test_suite'],
                            'provides': ['deployment_config', 'infrastructure']
                        }
                    })
                
                return tasks
            
            # Fall back to None to use keyword-based approach
            return None
            
        except Exception as e:
            self.logger.warning(f"LLM decomposition failed: {e}, falling back to keyword approach")
            return None

    def _get_agent_type_for_role(self, role: str, strategy: str) -> str:
        """Determine the optimal agent type for a role based on strategy"""
        
        # Agent type mappings by strategy
        if strategy == "claude":
            # All Claude Code agents
            return "claude_code"
        
        elif strategy == "aider":
            # All Aider agents, specialized by role
            if role == "architect":
                return "aider_backend"  # Use backend for architecture
            elif role == "frontend_developer":
                return "aider_frontend"
            elif role == "qa_engineer":
                return "aider_testing"
            else:
                return "aider_backend"  # Default to backend
        
        elif strategy == "mixed":
            # Force both types - alternate between them
            mixed_mapping = {
                "architect": "claude_code",
                "backend_developer": "aider_backend", 
                "frontend_developer": "aider_frontend",  # Fixed: use Aider for frontend file creation
                "qa_engineer": "aider_testing",  # Aider for multi-file test creation
                "devops_engineer": "aider_backend"  # Fixed: use Aider for config files
            }
            return mixed_mapping.get(role, "claude_code")
        
        elif strategy == "enhanced":
            # Use TaskAnalyzer for intelligent selection
            # For now, return dynamic mapping until we integrate TaskAnalyzer
            # TODO: Integrate TaskAnalyzer here
            enhanced_mapping = {
                "architect": "claude_code",        # Claude for complex design
                "backend_developer": "aider_backend",  # Aider for multi-file
                "frontend_developer": "aider_frontend",  # Aider for components
                "qa_engineer": "aider_testing",    # Aider for multi-file test creation
                "devops_engineer": "aider_backend"    # Aider for configs
            }
            return enhanced_mapping.get(role, "claude_code")
        
        else:  # dynamic (default)
            # Intelligent selection based on role strengths
            dynamic_mapping = {
                "architect": "claude_code",        # Claude excels at analysis
                "backend_developer": "aider_backend",  # Aider good for multi-file backend
                "frontend_developer": "aider_frontend",  # Aider good for component creation
                "qa_engineer": "aider_testing",    # Use Aider Testing for multi-file test creation
                "devops_engineer": "aider_backend",    # Aider handles multi-file configs
                "ml_engineer": "aider_backend",    # Aider for ML implementation (faster)
                "ml_implementer": "aider_backend", # Aider for ML implementation (multi-file)
                "api_developer": "aider_backend",  # Aider for API implementation
                "data_engineer": "aider_backend",   # Aider for data layer implementation
                # Analysis roles
                "architecture_analyst": "claude_code",  # Claude best for high-level analysis
                "code_analyst": "claude_code",      # Claude great at code review
                "implementation_analyst": "claude_code", # Claude for detailed analysis
                "frontend_analyst": "aider_frontend",   # Aider knows frontend patterns
                "backend_analyst": "aider_backend",     # Aider knows backend patterns
                "security_analyst": "claude_code",      # Claude good at security
                "performance_analyst": "claude_code"    # Claude spots patterns
            }
            return dynamic_mapping.get(role, "claude_code")
    
    async def _decompose_analysis_query(self, command: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a codebase analysis query into coordinated agent tasks"""
        command_lower = command.lower()
        agent_tasks = []
        agent_type_strategy = context.get('agent_type_strategy', 'dynamic')
        
        # Determine what aspects need analysis
        needs_architecture = any(word in command_lower for word in ['architecture', 'design', 'structure', 'flow'])
        needs_implementation = any(word in command_lower for word in ['implementation', 'code', 'logic', 'algorithm'])
        needs_frontend = any(word in command_lower for word in ['frontend', 'ui', 'component', 'react', 'view'])
        needs_backend = any(word in command_lower for word in ['backend', 'api', 'server', 'database'])
        needs_security = any(word in command_lower for word in ['security', 'vulnerability', 'safe', 'auth'])
        needs_performance = any(word in command_lower for word in ['performance', 'speed', 'optimization', 'efficient'])
        
        # If no specific aspect mentioned, do a comprehensive analysis
        if not any([needs_architecture, needs_implementation, needs_frontend, needs_backend, needs_security, needs_performance]):
            needs_architecture = True
            needs_implementation = True
        
        # Task 1: High-level architecture analysis (Claude is best at this)
        if needs_architecture:
            agent_tasks.append({
                'agent_type': 'claude_code',  # Always use Claude for architecture analysis
                'focus_areas': ['architecture', 'design', 'analysis'],
                'command': f'Analyze the architecture and design for: {command}. Focus on high-level structure, design patterns, component interactions, and architectural decisions.',
                'coordination_context': {
                    'role': 'architecture_analyst',
                    'dependencies': [],
                    'provides': ['architecture_analysis', 'design_insights']
                }
            })
        
        # Task 2: Implementation analysis
        if needs_implementation:
            # For implementation details, use appropriate specialized agent
            impl_agent = self._get_agent_type_for_role('code_analyst', agent_type_strategy)
            agent_tasks.append({
                'agent_type': impl_agent,
                'focus_areas': ['implementation', 'code_quality', 'best_practices'],
                'command': f'Analyze the implementation details for: {command}. Review code quality, patterns, potential issues, and suggest improvements.',
                'coordination_context': {
                    'role': 'implementation_analyst',
                    'dependencies': [],
                    'provides': ['implementation_analysis', 'code_review']
                }
            })
        
        # Task 3: Frontend-specific analysis
        if needs_frontend:
            frontend_agent = self._get_agent_type_for_role('frontend_analyst', agent_type_strategy)
            agent_tasks.append({
                'agent_type': frontend_agent,
                'focus_areas': ['frontend', 'ui', 'user_experience'],
                'command': f'Analyze the frontend/UI aspects for: {command}. Review component structure, state management, UI patterns, and user experience.',
                'coordination_context': {
                    'role': 'frontend_analyst',
                    'dependencies': [],
                    'provides': ['frontend_analysis', 'ui_review']
                }
            })
        
        # Task 4: Backend-specific analysis
        if needs_backend:
            backend_agent = self._get_agent_type_for_role('backend_analyst', agent_type_strategy)
            agent_tasks.append({
                'agent_type': backend_agent,
                'focus_areas': ['backend', 'api', 'data'],
                'command': f'Analyze the backend/API aspects for: {command}. Review API design, data models, service architecture, and scalability.',
                'coordination_context': {
                    'role': 'backend_analyst',
                    'dependencies': [],
                    'provides': ['backend_analysis', 'api_review']
                }
            })
        
        # Task 5: Security analysis
        if needs_security:
            agent_tasks.append({
                'agent_type': 'claude_code',  # Claude is good at security analysis
                'focus_areas': ['security', 'vulnerabilities', 'best_practices'],
                'command': f'Analyze security aspects for: {command}. Identify potential vulnerabilities, authentication issues, and security best practices.',
                'coordination_context': {
                    'role': 'security_analyst',
                    'dependencies': [],
                    'provides': ['security_analysis', 'vulnerability_report']
                }
            })
        
        # Task 6: Performance analysis
        if needs_performance:
            agent_tasks.append({
                'agent_type': 'claude_code',  # Claude can spot performance patterns
                'focus_areas': ['performance', 'optimization', 'efficiency'],
                'command': f'Analyze performance aspects for: {command}. Identify bottlenecks, inefficiencies, and optimization opportunities.',
                'coordination_context': {
                    'role': 'performance_analyst',
                    'dependencies': [],
                    'provides': ['performance_analysis', 'optimization_suggestions']
                }
            })
        
        self.logger.info(f"Decomposed analysis query into {len(agent_tasks)} coordinated analysis tasks")
        return agent_tasks
    
    async def _send_coordination_messages(self, agent_tasks: List[Dict[str, Any]], result: ExecutionResult) -> None:
        """Send coordination messages between agents"""
        try:
            # This would integrate with the inter-agent communication system
            for i, task in enumerate(agent_tasks):
                role = task['coordination_context']['role']
                dependencies = task['coordination_context']['dependencies']
                provides = task['coordination_context']['provides']
                
                coordination_msg = f"Agent {role} completed. Provides: {', '.join(provides)}. Dependencies met: {', '.join(dependencies)}"
                
                # In a real implementation, this would send via the communication hub
                self.logger.info(f"Coordination message: {coordination_msg}")
                
        except Exception as e:
            self.logger.warning(f"Failed to send coordination messages: {e}")
    
    async def _run_verification_loop(self, initial_result: ExecutionResult) -> bool:
        """Run verification and iterative fixes"""
        from agentic.core.verification_coordinator import VerificationCoordinator
        from rich.console import Console
        
        console = Console()
        verifier = VerificationCoordinator(self.agent_registry.workspace_path)
        
        max_iterations = 3
        iteration = 0
        verification_passed = False
        
        while not verification_passed and iteration < max_iterations:
            console.print(f"\n[yellow]🧪 Verification iteration {iteration + 1}/{max_iterations}[/yellow]")
            
            # Run verification
            verification_result = await verifier.verify_system()
            
            if verification_result.success:
                verification_passed = True
                console.print("[green]✅ All verifications passed![/green]")
                console.print(f"  • Tests: {sum(r.passed_tests for r in verification_result.test_results.values())} passed")
                console.print(f"  • System health: All checks passed")
                break
            else:
                console.print(f"[yellow]⚠️ Verification failed:[/yellow]")
                console.print(f"  • Test failures: {verification_result.total_failures}")
                console.print(f"  • Health issues: {sum(1 for v in verification_result.system_health.values() if not v)}")
                
                # Generate fix tasks
                fix_tasks = await verifier.analyze_failures(verification_result)
                
                if not fix_tasks:
                    console.print("[red]❌ No fix tasks generated, cannot proceed[/red]")
                    break
                
                console.print(f"\n[cyan]🔧 Executing {len(fix_tasks)} fix tasks...[/cyan]")
                
                # Execute fix tasks
                fix_result = await self.execute_coordinated_tasks(fix_tasks)
                
                if fix_result.status != "completed":
                    console.print("[red]❌ Fix tasks failed[/red]")
                    break
                
                # Brief pause before re-verification
                await asyncio.sleep(2)
            
            iteration += 1
        
        if not verification_passed:
            console.print(f"\n[red]❌ Verification failed after {max_iterations} iterations[/red]")
        
        return verification_passed
    
    async def _execute_single_agent_fallback(self, command: str, context: Dict[str, Any]) -> ExecutionResult:
        """Fallback to single agent execution with natural flow"""
        from agentic.core.intent_classifier import IntentClassifier
        from agentic.models.task import Task, TaskType
        from agentic.core.autonomous_execution import AutonomousExecutor, ExecutionMode
        
        intent_classifier = IntentClassifier()
        intent = await intent_classifier.analyze_intent(command)
        
        # Create task with original command - trust the agent to understand
        task = Task.from_intent(intent, command)
        
        # Determine best agent for the task
        agent_type_strategy = context.get('agent_type_strategy', 'dynamic')
        
        # For test tasks, prefer Claude Code for its natural iteration
        if intent.task_type == TaskType.TEST:
            if agent_type_strategy == 'dynamic':
                task.agent_type_hint = 'claude_code'  # Claude is better at iterative testing
                self.logger.info("Routing test task to Claude Code for natural iteration")
            elif agent_type_strategy == 'claude':
                task.agent_type_hint = 'claude_code'
            else:
                task.agent_type_hint = 'aider_backend'
        else:
            # Respect strategy for other tasks
            if agent_type_strategy == 'claude':
                task.agent_type_hint = 'claude_code'
            elif agent_type_strategy == 'aider':
                task.agent_type_hint = 'aider_backend'
            # For 'dynamic', let the system decide
        
        # Use autonomous execution for natural agent flow
        if context.get('autonomous', True):
            autonomous_executor = AutonomousExecutor(self.agent_registry)
            
            # Set user input handler if available
            if hasattr(self, 'user_input_handler'):
                autonomous_executor.set_user_input_handler(self.user_input_handler)
            
            # Determine execution mode
            mode = autonomous_executor.determine_execution_mode(command)
            self.logger.info(f"Using {mode.value} execution mode")
            
            # Execute with natural flow
            return await self.execute_coordinated_tasks([task], execution_mode=mode)
        else:
            # Traditional execution
            return await self.execute_coordinated_tasks([task]) 