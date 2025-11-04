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
from typing import Any, Dict, List, Optional

import os

from agentic.core.agent_registry import AgentRegistry
from agentic.core.shared_memory import SharedMemory
from agentic.core.project_indexer import ProjectIndexer
from agentic.core.change_tracker import ChangeTracker
from agentic.core.swarm_transaction import SwarmTransactionManager
from agentic.core.state_persistence import StatePersistenceManager, StateType
from agentic.core.error_recovery import ErrorRecoveryManager
from agentic.core.result_validation import ResultValidationManager, ValidationSuite
# Use unified monitor with all best features
from agentic.core.swarm_monitor_unified import SwarmMonitorUnified as SwarmMonitor, AgentStatus
from agentic.core.intelligent_coordinator import (
    IntelligentCoordinator, 
    AgentDiscovery as IntelligentDiscovery,
    DiscoveryType as IntelligentDiscoveryType
)
from agentic.models.agent import AgentSession, AgentType, AgentConfig, Discovery, DiscoveryType
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
    ended_at: Optional[datetime] = None
    
    @property
    def total_duration(self) -> float:
        """Get total execution duration in seconds"""
        if self.status == "running":
            return (datetime.utcnow() - self.started_at).total_seconds()
        elif self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
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
    
    def __init__(self, agent_registry: AgentRegistry, shared_memory: SharedMemory, workspace_path: Optional[Path] = None, enable_safety: bool = False):
        super().__init__()
        self.agent_registry = agent_registry
        self.shared_memory = shared_memory
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.background_tasks: Dict[str, asyncio.Task] = {}  # For background task execution
        self.enable_safety = enable_safety  # Flag to enable safety features
        
        # Initialize project indexer
        self.workspace_path = workspace_path or Path.cwd()
        self.project_indexer = ProjectIndexer(self.workspace_path)
        self.project_context = None
        
        # Initialize safety systems
        self.change_tracker = ChangeTracker(self.workspace_path)
        self.transaction_manager = SwarmTransactionManager(self.change_tracker)
        self.state_persistence = StatePersistenceManager(self.workspace_path / ".agentic")
        self.error_recovery = ErrorRecoveryManager()
        self.result_validator = ResultValidationManager(self.workspace_path)
        # Check if we're in an environment that might conflict with alternate screen
        # (e.g., when running tests or in CI)
        use_alternate_screen = not (
            os.environ.get('CI') or 
            os.environ.get('GITHUB_ACTIONS') or
            os.environ.get('AGENTIC_NO_ALTERNATE_SCREEN')
        )
        self.swarm_monitor = SwarmMonitor(use_alternate_screen=use_alternate_screen)  # Real-time monitoring
        
        # Initialize intelligent coordinator with optional verification
        # Enable verification if we have safety features enabled
        enable_verification = self.enable_safety and self.workspace_path.exists()
        self.intelligent_coordinator = IntelligentCoordinator(
            agent_registry, 
            shared_memory,
            workspace_path=self.workspace_path if enable_verification else None,
            enable_verification=enable_verification
        )
        self.logger.info(f"Using IntelligentCoordinator {'with' if enable_verification else 'without'} verification")
        
        # Initialize discovery tracking
        self.discoveries: List[Discovery] = []
        self._discovery_handlers: Dict[DiscoveryType, List[callable]] = {}
    
    async def execute_coordinated_tasks(self, tasks: List[Task], 
                                       coordination_plan: Optional[List[List[str]]] = None,
                                       execution_mode: Optional['ExecutionMode'] = None) -> ExecutionResult:
        """Execute multiple tasks with coordination"""
        # If safety is enabled, use the safe execution path
        if self.enable_safety:
            return await self._execute_coordinated_tasks_safe(tasks, coordination_plan, execution_mode)
        
        # Otherwise, use the standard execution path
        return await self._execute_coordinated_tasks_standard(tasks, coordination_plan, execution_mode)
    
    async def _execute_coordinated_tasks_safe(self, tasks: List[Task], 
                                            coordination_plan: Optional[List[List[str]]] = None,
                                            execution_mode: Optional['ExecutionMode'] = None) -> ExecutionResult:
        """Execute multiple tasks with full safety guarantees"""
        execution_id = None
        transaction = None
        
        try:
            # Create execution ID
            execution_id = str(uuid.uuid4())
            
            # Start state persistence for this execution
            await self.state_persistence.save_state(
                StateType.EXECUTION_CONTEXT,
                execution_id,
                {
                    'tasks': [task.model_dump() for task in tasks],
                    'coordination_plan': coordination_plan,
                    'status': 'starting'
                }
            )
            
            # Start automatic checkpointing
            await self.state_persistence.start_auto_checkpoint(execution_id)
            
            # Prepare agents for transaction
            agent_infos = []
            for task in tasks:
                agent_type = getattr(task, 'agent_type_hint', None)
                # If no agent type hint, determine based on task
                if not agent_type:
                    # Default to claude_code for general tasks
                    agent_type = 'claude_code'
                agent_infos.append({
                    'agent_id': f"{agent_type}_{task.id}",
                    'agent_type': agent_type,
                    'dependencies': getattr(task, 'dependencies', []),
                    'provides': getattr(task, 'provides', [])
                })
            
            # Begin swarm transaction
            transaction = await self.transaction_manager.begin_transaction(
                description=f"Coordinated execution of {len(tasks)} tasks",
                agents=agent_infos,
                rollback_on_failure=True
            )
            
            # Execute with transaction wrapper
            async def execute_with_safety():
                # Call standard execution method
                result = await self._execute_coordinated_tasks_standard(tasks, coordination_plan, execution_mode)
                
                # Validate results if execution succeeded
                if result.status == "completed":
                    validation_suite = await self._validate_execution_results(result, tasks)
                    if validation_suite.has_errors:
                        raise RuntimeError(f"Validation failed: {validation_suite.failed_checks} errors found")
                
                return result
            
            # Execute with error recovery
            result = await self.error_recovery.execute_with_retry(
                operation=execute_with_safety,
                operation_name=f"execution_{execution_id}",
                agent_id="coordinator"
            )
            
            # Mark all agents as complete in transaction
            for task in tasks:
                agent_id = f"{getattr(task, 'agent_type_hint', 'unknown')}_{task.id}"
                if task.id in result.task_results and result.task_results[task.id].success:
                    await self.transaction_manager.mark_agent_complete(
                        transaction.id,
                        agent_id,
                        outputs={'result': result.task_results[task.id].output}
                    )
                else:
                    error_msg = result.task_results.get(task.id, TaskResult(task_id=task.id, agent_id="unknown", status="failed")).error
                    await self.transaction_manager.mark_agent_failed(
                        transaction.id,
                        agent_id,
                        error=error_msg or "Unknown error"
                    )
            
            # Save final state
            await self.state_persistence.save_state(
                StateType.EXECUTION_CONTEXT,
                execution_id,
                {
                    'status': 'completed',
                    'result': result.model_dump() if hasattr(result, 'model_dump') else str(result)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Safe execution failed: {e}")
            
            # Save error state
            if execution_id:
                await self.state_persistence.save_state(
                    StateType.EXECUTION_CONTEXT,
                    execution_id,
                    {
                        'status': 'failed',
                        'error': str(e)
                    }
                )
            
            # Transaction will auto-rollback on exception
            raise
            
        finally:
            # Stop auto checkpointing
            if execution_id:
                self.state_persistence.stop_auto_checkpoint(execution_id)
    
    async def _execute_coordinated_tasks_standard(self, tasks: List[Task], 
                                                 coordination_plan: Optional[List[List[str]]] = None,
                                                 execution_mode: Optional['ExecutionMode'] = None) -> ExecutionResult:
        """Standard execution without safety features (original implementation)"""
        execution_id = str(uuid.uuid4())
        
        # Start monitoring if we have any tasks
        should_monitor = len(tasks) > 0
        original_root_level = None
        original_agentic_level = None
        
        if should_monitor:
            # Update task analysis BEFORE starting monitor
            if hasattr(self.swarm_monitor, 'update_task_analysis'):
                # Get project structure for analysis
                project_structure = self.shared_memory.get_project_structure()
                if project_structure:
                    total_files = sum(len(list(d.rglob('*'))) for d in project_structure.source_directories)
                    complexity = sum(task.complexity_score for task in tasks) / len(tasks) if tasks else 0.0
                    
                    # Determine suggested agents based on tasks
                    suggested_agents = set()
                    for task in tasks:
                        if hasattr(task, 'agent_type_hint') and task.agent_type_hint:
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
            
            # Now start monitoring after task analysis is set
            await self.swarm_monitor.start_monitoring()
            
            # Update coordinator status
            self.swarm_monitor.update_coordinator_status(
                "Running",
                f"Coordinating {len(tasks)} tasks",
                "INITIALIZATION"
            )
            self.swarm_monitor.set_execution_id(execution_id)
            
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
            
            # Update task queue status
            self.swarm_monitor.update_task_queue_status(len(tasks), 0)
            
            # Detect conflicts before execution
            self.swarm_monitor.update_coordinator_status(
                "Running",
                "Detecting task conflicts...",
                "PLANNING"
            )
            
            # Add timeout to prevent hanging on conflict detection
            try:
                conflicts = await asyncio.wait_for(
                    self._detect_conflicts(tasks),
                    timeout=5.0  # 5 second timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Conflict detection timed out, proceeding without conflict resolution")
                conflicts = []
            if conflicts:
                coordination_log.append({
                    "timestamp": datetime.utcnow(),
                    "type": "conflict_detection",
                    "conflicts": len(conflicts),
                    "details": [self._conflict_to_dict(c) for c in conflicts]
                })
                
                # Resolve conflicts
                self.swarm_monitor.update_coordinator_status(
                    "Running",
                    f"Resolving {len(conflicts)} conflicts...",
                    "PLANNING"
                )
                resolved_plan = await self._resolve_conflicts(conflicts, tasks, coordination_plan)
                coordination_plan = resolved_plan
            else:
                self.swarm_monitor.update_coordinator_status(
                    "Running",
                    "No conflicts detected, proceeding with execution",
                    "EXECUTION"
                )
            
            # Update status to execution phase
            self.swarm_monitor.update_coordinator_status(
                "Running",
                "Starting task execution",
                "EXECUTION"
            )
            
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
                    
                    # Log which tasks are in this group
                    group_roles = [t.coordination_context.get('role', 'unknown') for t in group_tasks if hasattr(t, 'coordination_context')]
                    self.logger.info(f"[PARALLEL GROUP {group_index + 1}/{len(coordination_plan)}] Executing tasks: {group_roles}")
                    
                    self.swarm_monitor.update_coordinator_status(
                        "Running",
                        f"Executing parallel group {group_index + 1}/{len(coordination_plan)}...",
                        "EXECUTION"
                    )
                    
                    group_results = await self._execute_parallel_group(group_tasks, context)
                    task_results.update(group_results)
                    
                    # Update context
                    for task in group_tasks:
                        if task.id in group_results and group_results[task.id].success:
                            context.completed_tasks.append(task.id)
                        else:
                            context.failed_tasks.append(task.id)
                    
                    # Update task queue status
                    self.swarm_monitor.update_task_queue_status(
                        len(tasks),
                        len(context.completed_tasks)
                    )
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
                
                # Update task queue status
                self.swarm_monitor.update_task_queue_status(
                    len(tasks),
                    len(context.completed_tasks)
                )
            
            # Determine overall status
            # Set end time for duration calculation
            context.ended_at = datetime.utcnow()
            
            if context.failed_tasks:
                context.status = "failed"
                self.swarm_monitor.update_coordinator_status(
                    "Failed",
                    f"{len(context.failed_tasks)} tasks failed",
                    "COMPLETE"
                )
            else:
                context.status = "completed"
                self.swarm_monitor.update_coordinator_status(
                    "Completed",
                    f"Successfully executed {len(context.completed_tasks)} tasks",
                    "COMPLETE"
                )
            
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
            context.ended_at = datetime.utcnow()
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
        # If safety is enabled and this is called directly, use safe version
        if self.enable_safety:
            return await self._execute_parallel_group_safe(tasks, context)
        
        return await self._execute_parallel_group_standard(tasks, context)
    
    async def _execute_parallel_group_safe(self, tasks: List[Task], context: ExecutionContext) -> Dict[str, TaskResult]:
        """Execute a group of tasks in parallel with safety"""
        # Create barrier for synchronization if needed
        if len(tasks) > 1 and hasattr(context, 'transaction_id'):
            barrier = await self.transaction_manager.create_barrier(
                context.transaction_id,
                "group_execution",
                required_agents=[f"{t.agent_type_hint or 'unknown'}_{t.id}" for t in tasks]
            )
        
        # Execute all tasks in parallel
        results = await asyncio.gather(
            *[self._execute_single_task_safe(task, context) for task in tasks],
            return_exceptions=True
        )
        
        # Convert to dict format
        task_results = {}
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task_results[task.id] = TaskResult(
                    task_id=task.id,
                    agent_id=task.agent_type_hint or "unknown",
                    status="failed",
                    output="",
                    error=str(result)
                )
            else:
                task_results[task.id] = result
        
        return task_results
    
    async def _execute_single_task_safe(self, task: Task, context: ExecutionContext) -> TaskResult:
        """Execute a single task with change tracking"""
        # Create changeset for this task
        changeset_id = self.change_tracker.begin_changeset(
            description=f"Task {task.id}: {task.command[:50]}...",
            agent_id=task.agent_type_hint or "unknown"
        )
        
        try:
            # Track task progress
            await self.state_persistence.save_state(
                StateType.TASK_PROGRESS,
                f"{context.execution_id}:{task.id}",
                {
                    'status': 'running',
                    'changeset_id': changeset_id
                }
            )
            
            # Execute with error recovery
            async def execute_task():
                # Get agent and execute
                agent_session = await self._find_best_agent_for_task(task)
                if not agent_session:
                    raise RuntimeError(f"No agent available for task {task.id}")
                
                agent_id = agent_session.id
                
                # Get the actual agent instance
                agent_instance = self.agent_registry.get_agent_by_id(agent_id)
                if not agent_instance:
                    raise RuntimeError(f"Failed to get agent instance for {agent_id}")
                
                # Register agent with monitor (was missing in safe execution)
                role = getattr(task, 'coordination_context', {}).get('role', 'general')
                
                # Use the actual role from coordination context first
                if role and role != 'general':
                    display_role = role.replace('_', ' ').title()
                # Only fall back to command/type detection if no specific role
                elif "test" in task.command.lower() and role == 'general':
                    display_role = "test_runner"
                elif "claude" in agent_session.agent_config.agent_type.value.lower():
                    display_role = "analyzer"
                elif "frontend" in agent_session.agent_config.agent_type.value.lower():
                    display_role = "frontend_dev"
                elif "backend" in agent_session.agent_config.agent_type.value.lower():
                    display_role = "backend_dev"
                else:
                    display_role = agent_session.agent_config.agent_type.value.replace("_", " ").title()
                
                # Register agent with monitor
                self.swarm_monitor.register_agent(
                    agent_id=agent_id,
                    agent_name=agent_session.agent_config.name,
                    agent_type=agent_session.agent_config.agent_type.value,
                    role=display_role
                )
                
                # Register changeset with transaction if active
                if hasattr(context, 'transaction_id'):
                    await self.transaction_manager.register_agent_changeset(
                        context.transaction_id,
                        agent_id,
                        changeset_id
                    )
                
                # Set discovery callback
                agent_instance.set_discovery_callback(lambda d: self._handle_discovery(d, agent_id))
                
                # Update monitor
                self.swarm_monitor.update_agent_status(agent_id, AgentStatus.SETTING_UP)
                
                # Execute task
                result = await agent_instance.execute_task(task)
                
                # Track any file changes
                if hasattr(result, 'files_modified'):
                    for file_path in result.files_modified:
                        # Read new content
                        try:
                            with open(file_path, 'r') as f:
                                new_content = f.read()
                            
                            # Track the change
                            self.change_tracker.track_file_change(
                                changeset_id=changeset_id,
                                file_path=file_path,
                                new_content=new_content,
                                agent_id=agent_id,
                                task_id=task.id
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to track change for {file_path}: {e}")
                
                return result
            
            # Execute with retry
            result = await self.error_recovery.execute_with_retry(
                operation=execute_task,
                operation_name=f"task_{task.id}",
                agent_id=task.agent_type_hint
            )
            
            # Commit changeset on success
            self.change_tracker.commit_changeset(changeset_id)
            
            # Save completed state
            await self.state_persistence.save_state(
                StateType.TASK_PROGRESS,
                f"{context.execution_id}:{task.id}",
                {
                    'status': 'completed',
                    'changeset_id': changeset_id,
                    'result': result.model_dump() if hasattr(result, 'model_dump') else str(result)
                }
            )
            
            return result
            
        except Exception as e:
            # Rollback changeset on failure
            try:
                self.change_tracker.rollback_changeset(changeset_id)
            except Exception as rollback_error:
                self.logger.error(f"Failed to rollback changeset {changeset_id}: {rollback_error}")
            
            # Save error state
            await self.state_persistence.save_state(
                StateType.TASK_PROGRESS,
                f"{context.execution_id}:{task.id}",
                {
                    'status': 'failed',
                    'changeset_id': changeset_id,
                    'error': str(e)
                }
            )
            
            # Return error result
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_type_hint or "unknown",
                status="failed",
                output="",
                error=str(e),
                execution_time=0
            )
    
    async def _execute_parallel_group_standard(self, tasks: List[Task], context: ExecutionContext) -> Dict[str, TaskResult]:
        """Standard execution of parallel tasks (original implementation)"""
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
                
                # Use the actual role from coordination context first
                if role and role != 'general':
                    display_role = role.replace('_', ' ').title()
                    self.logger.info(f"[ROLE DEBUG] Task {task.id} - Using coordination role: {role} -> {display_role}")
                # Only fall back to command/type detection if no specific role
                elif "test" in task.command.lower() and role == 'general':
                    display_role = "test_runner"
                elif "claude" in agent.agent_config.agent_type.value.lower():
                    display_role = "analyzer"
                elif "frontend" in agent.agent_config.agent_type.value.lower():
                    display_role = "frontend_dev"
                elif "backend" in agent.agent_config.agent_type.value.lower():
                    display_role = "backend_dev"
                else:
                    display_role = agent.agent_config.agent_type.value.replace("_", " ").title()
                
                self.logger.info(f"[ROLE DEBUG] Final display role for {task.id}: {display_role}")
                
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
                
                # Set discovery callback
                agent_instance.set_discovery_callback(lambda d: self._handle_discovery(d, agent.id))
                
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
                elif task.agent_type_hint == 'gemini':
                    agent_type = AgentType.GEMINI
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
                model_config = {"model": "gemini/gemini-2.5-pro-preview-06-05"}  # Latest Gemini 2.5 Pro Preview
            elif agent_type == AgentType.GEMINI:
                model_config = {"model": "gemini-2.5-pro"}  # Gemini model for chief architect
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
                
                # Fallback for Gemini to Claude Code
                if agent_type == AgentType.GEMINI:
                    self.logger.warning("Gemini agent failed to start, falling back to Claude Code for architect role")
                    config.agent_type = AgentType.CLAUDE_CODE
                    config.name = config.name.replace("gemini", "claude_code")
                    session = await self.agent_registry.get_or_spawn_agent(config)
                    
                    if session.status == "active":
                        self.logger.info("Successfully spawned Claude Code as fallback architect")
                        agent_type = AgentType.CLAUDE_CODE
                    else:
                        return None
                else:
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
        
        # Enhanced conflict detection that works for both new and existing projects
        project_structure = self.shared_memory.get_project_structure()
        is_new_project = not project_structure or not project_structure.source_directories
        
        # Check if any source directories actually exist and have files
        has_existing_files = False
        if not is_new_project:
            for source_dir in project_structure.source_directories:
                if source_dir.exists() and any(source_dir.iterdir()):
                    has_existing_files = True
                    break
        
        # For new projects, check for file creation conflicts
        if is_new_project or not has_existing_files:
            self.logger.info("Checking for new file creation conflicts")
            conflicts.extend(await self._detect_new_file_conflicts(tasks))
        
        # For existing projects, check modification conflicts
        if has_existing_files:
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
    
    async def _detect_new_file_conflicts(self, tasks: List[Task]) -> List[ConflictDetection]:
        """Detect conflicts when multiple agents might create the same new files"""
        conflicts = []
        
        # Analyze commands to predict what files agents might create
        predicted_files_map: Dict[str, List[Task]] = {}
        
        for task in tasks:
            predicted_files = self._predict_files_to_create(task)
            for file_name in predicted_files:
                if file_name not in predicted_files_map:
                    predicted_files_map[file_name] = []
                predicted_files_map[file_name].append(task)
        
        # Find files that multiple agents might create
        for file_name, creating_tasks in predicted_files_map.items():
            if len(creating_tasks) > 1:
                # Get roles of conflicting agents
                roles = []
                for task in creating_tasks:
                    role = task.coordination_context.get('role', 'unknown') if hasattr(task, 'coordination_context') else 'unknown'
                    roles.append(role)
                
                conflict = ConflictDetection(
                    conflict_type="new_file_conflict",
                    affected_files=[Path(file_name)],
                    conflicting_agents=roles,
                    severity="high",  # New file conflicts are always high severity
                    auto_resolvable=False,  # Requires manual coordination
                    resolution_strategy=f"Assign {file_name} creation to one agent only"
                )
                conflicts.append(conflict)
                self.logger.warning(f"Detected new file conflict: {file_name} might be created by {roles}")
        
        return conflicts
    
    def _predict_files_to_create(self, task: Task) -> List[str]:
        """Predict which files a task might create based on its command and role"""
        predicted_files = []
        command_lower = task.command.lower()
        
        # Get role from task context
        role = 'unknown'
        if hasattr(task, 'coordination_context') and task.coordination_context:
            role = task.coordination_context.get('role', 'unknown')
        
        # Common project files that multiple agents might try to create
        if any(keyword in command_lower for keyword in ['package.json', 'dependencies', 'npm', 'project']):
            predicted_files.append('package.json')
        
        if any(keyword in command_lower for keyword in ['typescript', 'tsconfig', 'types']):
            predicted_files.append('tsconfig.json')
        
        if any(keyword in command_lower for keyword in ['docker', 'container', 'deployment']):
            predicted_files.extend(['Dockerfile', 'docker-compose.yml'])
        
        if any(keyword in command_lower for keyword in ['test', 'jest', 'vitest', 'testing']):
            predicted_files.extend(['jest.config.js', 'vitest.config.ts'])
        
        if any(keyword in command_lower for keyword in ['eslint', 'lint', 'prettier']):
            predicted_files.extend(['.eslintrc.js', '.prettierrc.json'])
        
        if any(keyword in command_lower for keyword in ['git', 'ignore']):
            predicted_files.extend(['.gitignore'])
        
        if any(keyword in command_lower for keyword in ['readme', 'documentation']):
            predicted_files.append('README.md')
        
        # Role-specific predictions
        if role == 'architect':
            predicted_files.extend(['README.md', 'package.json', 'tsconfig.json'])
        elif role == 'backend_developer':
            predicted_files.extend(['package.json', 'tsconfig.json', '.env.example'])
        elif role == 'frontend_developer':
            predicted_files.extend(['package.json', 'tsconfig.json', 'vite.config.ts', 'index.html'])
        elif role == 'qa_engineer':
            predicted_files.extend(['jest.config.js', 'package.json'])
        elif role == 'devops_engineer':
            predicted_files.extend(['Dockerfile', 'docker-compose.yml', '.dockerignore'])
        
        return list(set(predicted_files))  # Remove duplicates
    
    def _detect_project_type(self, command_lower: str) -> str:
        """Detect the type of project from the command"""
        # Bot/Trading systems
        if any(keyword in command_lower for keyword in ['bot', 'sniper', 'trading', 'automated trading']):
            return 'trading_bot'
        
        # Full stack web applications
        if any(keyword in command_lower for keyword in ['full stack', 'fullstack', 'web app', 'web application']):
            return 'fullstack_web'
        
        # API-only projects
        if 'api' in command_lower and not any(word in command_lower for word in ['frontend', 'ui', 'interface']):
            return 'api_only'
        
        # Frontend-only projects
        if any(keyword in command_lower for keyword in ['frontend', 'ui', 'dashboard']) and \
           not any(keyword in command_lower for keyword in ['backend', 'api', 'server']):
            return 'frontend_only'
        
        # Microservices
        if 'microservice' in command_lower:
            return 'microservices'
        
        # CLI tool
        if any(keyword in command_lower for keyword in ['cli', 'command line', 'terminal']):
            return 'cli_tool'
        
        # Default to monolithic for complete systems
        if any(keyword in command_lower for keyword in ['complete', 'system', 'platform', 'production']):
            return 'monolithic'
        
        return 'unknown'
    
    def _get_structure_guidance(self, project_type: str) -> str:
        """Get specific structure guidance based on project type"""
        structure_map = {
            'trading_bot': "Based on the project type, create these directories:",
            
            'fullstack_web': "Based on the project type, create these directories:",
            
            'api_only': "Based on the project type, create these directories:",
            
            'frontend_only': "Based on the project type, create these directories:",
            
            'microservices': "Based on the project type, create these directories:",
            
            'cli_tool': "Based on the project type, create these directories:",
            
            'monolithic': "Based on the project type, create these directories:"
        }
        
        return structure_map.get(project_type, "Based on the project requirements, create appropriate directories:")
    
    def _get_finalization_agent(self, project_type: str, agent_type_strategy: str) -> str:
        """Determine which agent should handle project finalization"""
        # For most projects, backend developer is best for running setup commands
        # For frontend-only, use frontend developer
        if project_type == 'frontend_only':
            return self._get_agent_type_for_role('frontend_developer', agent_type_strategy)
        else:
            # Backend developer typically handles server setup and dependencies
            return self._get_agent_type_for_role('backend_developer', agent_type_strategy)
    
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
        # Only look for files that actually exist to avoid hanging on glob operations
        if "frontend" in task.affected_areas:
            # Look for frontend files
            for source_dir in project_structure.source_directories:
                if source_dir.exists() and any(name in str(source_dir).lower() for name in ["src", "components", "pages"]):
                    try:
                        # Use iterdir to check if directory has content first
                        if any(source_dir.iterdir()):
                            affected_files.extend(list(source_dir.glob("**/*.tsx"))[:5])
                            affected_files.extend(list(source_dir.glob("**/*.jsx"))[:5])
                            affected_files.extend(list(source_dir.glob("**/*.css"))[:5])
                    except (OSError, PermissionError):
                        pass
        
        if "backend" in task.affected_areas:
            # Look for backend files
            for source_dir in project_structure.source_directories:
                if source_dir.exists():
                    try:
                        # Use iterdir to check if directory has content first
                        if any(source_dir.iterdir()):
                            affected_files.extend(list(source_dir.glob("**/*.py"))[:5])
                            affected_files.extend(list(source_dir.glob("**/*.js"))[:5])
                            affected_files.extend(list(source_dir.glob("**/*.ts"))[:5])
                    except (OSError, PermissionError):
                        pass
        
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
        
        # Enhanced conflict resolution for new file conflicts
        task_dict = {task.id: task for task in tasks}
        dependency_graph = {}  # task_id -> set of tasks that must run before it
        
        # First, handle new file conflicts by establishing dependencies
        for conflict in conflicts:
            if conflict.conflict_type == "new_file_conflict":
                # For new file conflicts, determine which agent should create the file
                file_name = str(conflict.affected_files[0]) if conflict.affected_files else "unknown"
                
                # Find tasks that want to create this file
                conflicting_task_ids = []
                for task in tasks:
                    predicted_files = self._predict_files_to_create(task)
                    if file_name in predicted_files:
                        conflicting_task_ids.append(task.id)
                
                if len(conflicting_task_ids) > 1:
                    # Assign file creation based on role priority
                    priority_task_id = self._determine_file_creation_priority(
                        conflicting_task_ids, task_dict, file_name
                    )
                    
                    # Make other tasks depend on the priority task
                    for task_id in conflicting_task_ids:
                        if task_id != priority_task_id:
                            if task_id not in dependency_graph:
                                dependency_graph[task_id] = set()
                            dependency_graph[task_id].add(priority_task_id)
                            
                    self.logger.info(f"Resolved {file_name} conflict: {priority_task_id} will create it first")
        
        # Handle other conflict types
        serialized_tasks = set()
        for conflict in conflicts:
            if conflict.conflict_type != "new_file_conflict" and not conflict.auto_resolvable:
                conflicting_task_ids = [
                    task.id for task in tasks 
                    if any(str(f) in [str(af) for af in conflict.affected_files] 
                          for f in await self._estimate_affected_files(task))
                ]
                serialized_tasks.update(conflicting_task_ids)
        
        # Build execution plan respecting dependencies
        return self._build_execution_plan_with_dependencies(tasks, dependency_graph, serialized_tasks)
    
    def _determine_file_creation_priority(self, task_ids: List[str], task_dict: Dict[str, Task], file_name: str) -> str:
        """Determine which task should create a file based on role priority"""
        # Priority order for file creation
        file_priority_map = {
            'package.json': ['architect', 'backend_developer', 'frontend_developer', 'qa_engineer', 'devops_engineer'],
            'tsconfig.json': ['architect', 'backend_developer', 'frontend_developer'],
            'README.md': ['architect', 'backend_developer', 'frontend_developer'],
            'Dockerfile': ['devops_engineer', 'backend_developer'],
            'docker-compose.yml': ['devops_engineer', 'architect'],
            '.gitignore': ['architect', 'backend_developer'],
            'jest.config.js': ['qa_engineer', 'backend_developer', 'frontend_developer'],
            'vite.config.ts': ['frontend_developer'],
            'index.html': ['frontend_developer'],
            '.env.example': ['backend_developer', 'architect'],
        }
        
        # Get priority order for this file
        priority_order = file_priority_map.get(file_name, ['architect', 'backend_developer', 'frontend_developer'])
        
        # Find the task with highest priority role
        for priority_role in priority_order:
            for task_id in task_ids:
                task = task_dict.get(task_id)
                if task and hasattr(task, 'coordination_context'):
                    role = task.coordination_context.get('role', '')
                    if role == priority_role:
                        return task_id
        
        # Default to first task if no priority match
        return task_ids[0]
    
    def _build_execution_plan_with_dependencies(self, tasks: List[Task], 
                                               dependency_graph: Dict[str, Set[str]], 
                                               serialized_tasks: Set[str]) -> List[List[str]]:
        """Build execution plan respecting dependencies"""
        parallel_groups = []
        executed = set()
        task_ids = [task.id for task in tasks]
        
        # Topological sort with grouping
        while len(executed) < len(task_ids):
            # Find tasks that can be executed now
            ready_tasks = []
            for task_id in task_ids:
                if task_id not in executed:
                    # Check if all dependencies are satisfied
                    dependencies = dependency_graph.get(task_id, set())
                    if dependencies.issubset(executed):
                        ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Circular dependency or error - just add remaining tasks
                remaining = [tid for tid in task_ids if tid not in executed]
                if remaining:
                    parallel_groups.append(remaining)
                break
            
            # Group ready tasks that can run in parallel
            parallel_group = []
            serial_group = []
            
            for task_id in ready_tasks:
                if task_id in serialized_tasks:
                    serial_group.append(task_id)
                else:
                    parallel_group.append(task_id)
            
            # Add parallel tasks as one group
            if parallel_group:
                parallel_groups.append(parallel_group)
                executed.update(parallel_group)
            
            # Add serialized tasks as individual groups
            for task_id in serial_group:
                parallel_groups.append([task_id])
                executed.add(task_id)
        
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
    
    def _handle_discovery(self, discovery: Discovery, agent_id: str) -> None:
        """Handle a discovery reported by an agent"""
        # Store the discovery
        self.discoveries.append(discovery)
        
        # Log the discovery
        self.logger.info(f"Discovery from {agent_id}: {discovery.type.value} - {discovery.description}")
        
        # Convert to intelligent coordinator discovery format
        intelligent_discovery = IntelligentDiscovery(
            agent_id=agent_id,
            discovery_type=self._map_discovery_type(discovery.type),
            severity=discovery.severity,
            context=discovery.context,
            suggestions=[discovery.suggested_action] if discovery.suggested_action else [],
            affected_files=[discovery.file_path] if discovery.file_path else [],
            timestamp=discovery.timestamp
        )
        
        # Report to intelligent coordinator for dynamic task generation
        asyncio.create_task(self._process_discovery_for_tasks(intelligent_discovery))
        
        # Call registered handlers for this discovery type
        if discovery.type in self._discovery_handlers:
            for handler in self._discovery_handlers[discovery.type]:
                try:
                    handler(discovery, agent_id)
                except Exception as e:
                    self.logger.error(f"Error in discovery handler: {e}")
        
        # Update shared memory with discovery
        asyncio.create_task(self._update_shared_memory_with_discovery(discovery, agent_id))
    
    async def _update_shared_memory_with_discovery(self, discovery: Discovery, agent_id: str) -> None:
        """Update shared memory with discovery information"""
        try:
            discovery_data = {
                "type": discovery.type.value,
                "description": discovery.description,
                "agent": agent_id,
                "timestamp": discovery.timestamp.isoformat(),
                "severity": discovery.severity,
                "context": discovery.context
            }
            
            # Add to shared memory
            await self.shared_memory.update_context(
                f"discovery_{discovery.type.value}_{agent_id}",
                discovery_data
            )
            
            # Also maintain a list of all discoveries
            all_discoveries = await self.shared_memory.get_context("all_discoveries") or []
            all_discoveries.append(discovery_data)
            await self.shared_memory.update_context("all_discoveries", all_discoveries)
            
        except Exception as e:
            self.logger.error(f"Failed to update shared memory with discovery: {e}")
    
    def register_discovery_handler(self, discovery_type: DiscoveryType, handler: callable) -> None:
        """Register a handler for a specific discovery type"""
        if discovery_type not in self._discovery_handlers:
            self._discovery_handlers[discovery_type] = []
        self._discovery_handlers[discovery_type].append(handler)
    
    def get_discoveries(self, discovery_type: Optional[DiscoveryType] = None,
                       agent_id: Optional[str] = None,
                       severity: Optional[str] = None) -> List[Discovery]:
        """Get discoveries with optional filtering"""
        discoveries = self.discoveries.copy()
        
        if discovery_type:
            discoveries = [d for d in discoveries if d.type == discovery_type]
        
        if agent_id:
            discoveries = [d for d in discoveries if d.agent_name == agent_id]
        
        if severity:
            discoveries = [d for d in discoveries if d.severity == severity]
        
        return discoveries
    
    async def _validate_execution_results(self, result: ExecutionResult, tasks: List[Task]) -> ValidationSuite:
        """Validate the results of execution"""
        combined_suite = ValidationSuite(suite_name="Execution validation")
        
        # Collect all modified files
        modified_files = set()
        for task_id, task_result in result.task_results.items():
            if hasattr(task_result, 'files_modified'):
                modified_files.update(task_result.files_modified)
        
        if modified_files:
            # Validate all modified files
            file_suite = await self.result_validator.validate_files(list(modified_files))
            combined_suite.results.extend(file_suite.results)
            combined_suite.total_checks += file_suite.total_checks
            combined_suite.passed_checks += file_suite.passed_checks
            combined_suite.failed_checks += file_suite.failed_checks
            combined_suite.warnings += file_suite.warnings
            
            # Run build validation if appropriate
            build_suite = await self.result_validator.validate_build()
            if build_suite.results:
                combined_suite.results.extend(build_suite.results)
                combined_suite.total_checks += build_suite.total_checks
                combined_suite.passed_checks += build_suite.passed_checks
                combined_suite.failed_checks += build_suite.failed_checks
                combined_suite.warnings += build_suite.warnings
        
        return combined_suite
    
    async def recover_from_crash(self, execution_id: str) -> Optional[ExecutionResult]:
        """Recover execution from a crash using persisted state"""
        try:
            # Load execution state
            exec_state = await self.state_persistence.load_state(
                StateType.EXECUTION_CONTEXT,
                execution_id
            )
            
            if not exec_state:
                self.logger.error(f"No state found for execution {execution_id}")
                return None
            
            state_data = exec_state.state_data
            
            # If already completed, return the result
            if state_data.get('status') == 'completed':
                return ExecutionResult(**state_data['result'])
            
            # Recreate tasks using Pydantic's model validation for proper deserialization
            from agentic.models.task import Task
            tasks = [Task.model_validate(task_data) for task_data in state_data.get('tasks', [])]
            
            # Find incomplete tasks
            incomplete_tasks = []
            for task in tasks:
                task_state = await self.state_persistence.load_state(
                    StateType.TASK_PROGRESS,
                    f"{execution_id}:{task.id}"
                )
                
                if not task_state or task_state.state_data.get('status') != 'completed':
                    incomplete_tasks.append(task)
            
            if incomplete_tasks:
                self.logger.info(f"Resuming execution {execution_id} with {len(incomplete_tasks)} incomplete tasks")
                
                # Resume execution with remaining tasks
                return await self.execute_coordinated_tasks(
                    incomplete_tasks,
                    coordination_plan=state_data.get('coordination_plan')
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to recover execution {execution_id}: {e}")
            return None
    
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
    
    async def execute_multi_agent_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute a command using multiple coordinated agents"""
        self.logger.info(f"Decomposing multi-agent command: {command[:50]}...")
        
        try:
            # Use internal decomposition method
            tasks = await self._decompose_command_into_agent_tasks(command, context)
            if tasks:
                self.logger.info(f"Command decomposed into {len(tasks)} agent tasks")
                return await self.execute_coordinated_tasks(tasks)
            else:
                self.logger.warning("No tasks generated from command decomposition")
                return ExecutionResult(
                    execution_id="none",
                    status="failed",
                    completed_tasks=[],
                    failed_tasks=[],
                    total_duration=0.0,
                    task_results={},
                    coordination_log=[]
                )
        except Exception as e:
            self.logger.error(f"Failed to decompose command: {e}")
            return ExecutionResult(
                execution_id="none",
                status="failed",
                completed_tasks=[],
                failed_tasks=[],
                total_duration=0.0,
                task_results={},
                coordination_log=[{"error": str(e)}]
            )
    
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
            
            # Index project if not already done
            if not self.project_context:
                self.logger.info("Indexing project for context awareness...")
                await self.project_indexer.index_project()
                self.project_context = self.project_indexer.get_project_context()
            
            # Check if this is a follow-up command that needs previous context
            enriched_command = await self._enrich_command_with_context(command, context or {})
            
            # Analyze the command to determine what agents are needed
            agent_tasks = await self._decompose_command_into_agent_tasks(enriched_command, context or {})
            
            self.logger.info(f"Decomposition returned {len(agent_tasks)} tasks")
            for i, task in enumerate(agent_tasks):
                role = task.get('coordination_context', {}).get('role', 'unknown')
                self.logger.info(f"  Task {i+1}: {role} - {task['agent_type']}")
            
            if len(agent_tasks) <= 1:
                # Special case: production bots should always use multi-agent
                if 'bot' in command.lower() and 'production' in command.lower():
                    self.logger.info("Production bot detected - forcing multi-agent execution")
                    # Create a minimal multi-agent setup
                    agent_tasks = [
                        {
                            'agent_type': 'claude_code',
                            'focus_areas': ['architecture', 'design'],
                            'command': f'Design the architecture for: {enriched_command}',
                            'coordination_context': {
                                'role': 'architect',
                                'dependencies': [],
                                'provides': ['technical_spec']
                            }
                        },
                        {
                            'agent_type': 'aider_backend',
                            'focus_areas': ['backend', 'implementation'],
                            'command': f'Implement the backend for: {enriched_command}',
                            'coordination_context': {
                                'role': 'backend_developer',
                                'dependencies': ['technical_spec'],
                                'provides': ['backend_services']
                            }
                        }
                    ]
                else:
                    self.logger.info("Command doesn't require multi-agent coordination, using single agent")
                    # Fall back to single agent execution
                    return await self._execute_single_agent_fallback(enriched_command, context or {})
            
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
                # Ensure agent_type is not None
                agent_type = agent_task.get('agent_type')
                if not agent_type:
                    self.logger.error(f"Agent type is None for task: {agent_task}")
                    agent_type = 'aider_backend'  # Default fallback
                task.agent_type_hint = agent_type  # Hint for routing
                task.coordination_context = agent_task.get('coordination_context', {})
                
                # Add project context to task
                task.project_context = self.project_context
                
                # Get language from context (from user selection) or project
                if context and 'target_language' in context:
                    task.target_language = context['target_language']
                    task.target_framework = context.get('target_framework', None)
                elif self.project_context:
                    task.target_language = self.project_context.primary_language
                    task.target_framework = self.project_context.framework
                else:
                    # Default to TypeScript if not specified
                    task.target_language = 'typescript'
                
                self.logger.info(f"Task language context: {task.target_language}")
                
                # Debug: Log task creation
                role = agent_task.get('coordination_context', {}).get('role', 'unknown')
                self.logger.info(f"Created task for {role}: agent_type={agent_task['agent_type']}, task_id={task.id}")
                tasks.append(task)
            
            # Monitoring is already started in execute_coordinated_tasks, don't start again
            
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
                if result.status == "completed" and context and context.get('verify', True):
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
        
        # Add project context to context dict for downstream use
        if self.project_context:
            context['project_language'] = self.project_context.primary_language
            context['project_framework'] = self.project_context.framework
            context['project_type'] = self.project_context.project_type
        
        # Get agent type strategy from context
        agent_type_strategy = context.get('agent_type_strategy', 'dynamic')
        
        # Detect project type for better coordination
        project_type = self._detect_project_type(command_lower)
        self.logger.info(f"Detected project type: {project_type}")
        
        # Check if forced multi-agent
        if context.get('force_multi_agent', False):
            self.logger.info("Forced multi-agent execution requested")
            # Continue to decomposition below
        
        # Check if this is a coordinated analysis request
        elif context.get('coordination_type') == 'analysis':
            return await self._decompose_analysis_query(command, context)
        
        # Try LLM-based decomposition first
        if context.get('use_llm_decomposition', True):
            llm_tasks = await self._decompose_with_llm(command, context)
            if llm_tasks:
                return llm_tasks
        
        # Add language context to commands if specified
        language_context = ""
        if context.get('target_language'):
            language_context = f"Use {context['target_language']}"
            if context.get('target_framework'):
                language_context += f" with {context['target_framework']}"
            language_context += ". "
        
        # Define patterns that suggest multi-agent coordination
        multi_component_patterns = [
            'full stack', 'complete application', 'complete app', 'complete todo', 'complete web',
            'end-to-end', 'frontend and backend', 'react frontend', 'fastapi backend',
            'with tests', 'including tests', 'comprehensive tests', 'and documentation', 'with deployment',
            'microservices', 'multiple services', 'system', 'platform',
            # Additional patterns for complex systems
            'real-time', 'real time', 'collaborative', 'websocket', 'with database',
            'production-ready', 'production ready', 'with authentication', 'docker', 'redis',
            'with frontend', 'with backend', 'multiple components',
            'build a', 'create a', 'implement a',
            # Complex project indicators
            'editor', 'dashboard', 'application with', 'platform with',
            'e-commerce', 'game', 'chat', 'social',
            # Bot-specific patterns
            'bot', 'sniper bot', 'trading bot', 'automation',
            # Comprehensive project patterns
            'complete with', 'production', 'enterprise', 'scalable'
        ]
        
        # Also check if this is an implementation follow-up that needs multi-agent
        implementation_follow_ups = [
            'implement', 'build', 'create', 'develop', 'construct',
            'the missing logic', 'complete the', 'finish the'
        ]
        
        is_implementation_followup = any(pattern in command_lower for pattern in implementation_follow_ups)
        
        # Check if command contains context from previous analysis
        has_analysis_context = command.startswith("Based on the following analysis:")
        
        # Check if this is a multi-component request OR an implementation follow-up with context
        is_multi_component = any(pattern in command_lower for pattern in multi_component_patterns) or \
                           (is_implementation_followup and has_analysis_context) or \
                           context.get('force_multi_agent', False)
        
        # Additional check for complex bot/system requests
        if not is_multi_component:
            # Check for compound requirements (e.g., "bot complete with tests")
            has_bot = 'bot' in command_lower
            has_tests = any(word in command_lower for word in ['test', 'tests', 'testing'])
            has_production = 'production' in command_lower
            
            if (has_bot and has_tests) or (has_bot and has_production):
                is_multi_component = True
                self.logger.info(f"Detected complex bot request requiring multi-agent: bot={has_bot}, tests={has_tests}, production={has_production}")
        
        if not is_multi_component:
            self.logger.info(f"Command classified as single-agent. Patterns checked: {len(multi_component_patterns)}")
            return []  # Single agent will handle
        
        self.logger.info(f"Multi-agent execution confirmed for: {command[:50]}...")
        self.logger.info(f"[DECOMPOSE DEBUG] Command keywords found - checking for architect task")
        
        # Extract the analysis context if present
        analysis_context = ""
        base_command = command
        if "Based on the following analysis:" in command:
            parts = command.split("\n\nNow ")
            if len(parts) == 2:
                analysis_context = parts[0]  # Contains "Based on the following analysis:" + the analysis
                base_command = parts[1]  # The actual command after "Now"
        
        # Special handling for implementation follow-ups with context
        if is_implementation_followup and has_analysis_context:
            self.logger.info("Detected implementation follow-up with analysis context - spawning comprehensive agent team")
            
            # Always spawn core agents for implementation based on analysis
            # Backend agent
            backend_type = self._get_agent_type_for_role('backend_developer', agent_type_strategy)
            backend_cmd = f'{analysis_context}\n\n{language_context}Implement the backend/API components described in the analysis. {base_command}'
            agent_tasks.append({
                'agent_type': backend_type,
                'focus_areas': ['backend', 'api', 'database'],
                'command': backend_cmd,
                'coordination_context': {
                    'role': 'backend_developer',
                    'dependencies': [],
                    'provides': ['api_endpoints', 'backend_services']
                }
            })
            
            # Frontend agent (if UI is mentioned in analysis)
            if any(word in command_lower for word in ['ui', 'interface', 'dashboard', 'frontend', 'display', 'view']):
                frontend_type = self._get_agent_type_for_role('frontend_developer', agent_type_strategy)
                frontend_cmd = f'{analysis_context}\n\n{language_context}Implement the frontend/UI components described in the analysis. {base_command}'
                agent_tasks.append({
                    'agent_type': frontend_type,
                    'focus_areas': ['frontend', 'ui', 'components'],
                    'command': frontend_cmd,
                    'coordination_context': {
                        'role': 'frontend_developer',
                        'dependencies': ['api_endpoints'],
                        'provides': ['ui_components']
                    }
                })
            
            # Testing agent
            testing_type = self._get_agent_type_for_role('qa_engineer', agent_type_strategy)
            testing_cmd = f'{analysis_context}\n\n{language_context}Implement comprehensive tests for the components described in the analysis. {base_command}'
            agent_tasks.append({
                'agent_type': testing_type,
                'focus_areas': ['testing', 'qa'],
                'command': testing_cmd,
                'coordination_context': {
                    'role': 'qa_engineer',
                    'dependencies': ['backend_services', 'ui_components'],
                    'provides': ['test_suite']
                }
            })
            
            return agent_tasks
        
        # Task 1: Architecture Analysis
        # Include architecture for production systems, bots, and complete applications
        # Also include for any multi-component app (frontend + backend)
        includes_frontend = any(keyword in command_lower for keyword in ['frontend', 'ui', 'react', 'vue', 'angular'])
        includes_backend = any(keyword in command_lower for keyword in ['backend', 'api', 'server', 'node', 'express'])
        is_multi_component_app = includes_frontend and includes_backend
        
        if any(keyword in command_lower for keyword in ['complete', 'system', 'platform', 'architecture', 'production', 'bot', 'build', 'create', 'app', 'application']) or is_multi_component_app:
            self.logger.info("[ARCHITECT DEBUG] Creating architect task")
            architect_agent_type = self._get_agent_type_for_role('architect', agent_type_strategy)
            self.logger.info(f"[ARCHITECT DEBUG] Architect agent type: {architect_agent_type} (strategy: {agent_type_strategy})")
            # Include analysis context in the command if available
            task_command = f'{analysis_context}\n\n' if analysis_context else ''
            # Customize architect command based on project type
            structure_guidance = self._get_structure_guidance(project_type)
            task_command += f'{language_context}Design the architecture and create the project structure for: {base_command}. \n\n📁 DIRECTORY STRUCTURE TASK:\n\n1. CREATE DIRECTORIES by writing .gitkeep files:\n   {structure_guidance}\n   \n   Use the Write tool with these EXACT paths:\n   • backend/.gitkeep (empty content)\n   • frontend/.gitkeep (empty content) \n   • shared/.gitkeep (empty content)\n   • config/.gitkeep (empty content)\n   • docs/.gitkeep (empty content)\n   \n   ⚠️ IMPORTANT: The file path must be exactly "backend/.gitkeep" - not "backend/ directory" or any other variation.\n\n2. CREATE ROOT FILES with these EXACT filenames:\n   • README.md - Project overview\n   • package.json - Node.js workspace config (valid JSON)\n   • .gitignore - Git ignore patterns\n   • tsconfig.json - TypeScript config\n   \n3. CREATE DOCUMENTATION with these EXACT filenames:\n   • ARCHITECTURE.md - System design\n   • API_SPEC.md - API specifications  \n   • DATA_MODEL.md - Data schemas\n\n⛔ DO NOT CREATE:\n   • Files named after commands (like "npm run build" or "http-server public")\n   • Files with special characters in names (like "└── README")\n   • package.js (only create package.json)\n   \nFocus on creating a clean, professional project structure.'
            
            architect_task = {
                'agent_type': architect_agent_type,
                'focus_areas': ['analysis', 'architecture', 'design'],
                'command': task_command,
                'coordination_context': {
                    'role': 'architect',
                    'dependencies': [],
                    'provides': ['technical_spec', 'api_contracts', 'project_structure']
                }
            }
            agent_tasks.append(architect_task)
            self.logger.info(f"[ARCHITECT DEBUG] Added architect task with role: {architect_task['coordination_context']['role']}")
        
        # Task 2: Backend Development
        # Also include backend for complete systems/applications/bots
        if any(keyword in command_lower for keyword in ['backend', 'api', 'server', 'database', 'authentication', 'fastapi', 'django', 'flask', 'bot', 'sniper', 'trading']) or \
           (any(keyword in command_lower for keyword in ['system', 'application', 'platform']) and 'automated' in command_lower) or \
           (is_implementation_followup and has_analysis_context):  # Include for follow-up implementations
            backend_agent_type = self._get_agent_type_for_role('backend_developer', agent_type_strategy)
            # Include analysis context in the command if available
            task_command = f'{analysis_context}\n\n' if analysis_context else ''
            task_command += f'{language_context}Create complete backend/API code for: {base_command}. Work within the backend/ directory if it exists, otherwise create your own structure. Create a separate package.json for the backend, install dependencies, and implement all backend services, APIs, and database logic. Ensure the backend is runnable and properly structured.'
            
            agent_tasks.append({
                'agent_type': backend_agent_type,
                'focus_areas': ['backend', 'api', 'database', 'authentication'],
                'command': task_command,
                'coordination_context': {
                    'role': 'backend_developer',
                    'dependencies': ['technical_spec'],
                    'provides': ['api_endpoints', 'database_schema', 'backend_services']
                }
            })
        
        # Task 3: Frontend Development
        # Also include frontend for complete systems that need monitoring/dashboards
        # Bots typically need dashboards for monitoring
        if any(keyword in command_lower for keyword in ['frontend', 'ui', 'react', 'vue', 'angular', 'dashboard', 'interface', 'components']) or \
           (any(keyword in command_lower for keyword in ['system', 'application', 'platform', 'bot']) and any(keyword in command_lower for keyword in ['manage', 'monitor', 'automated', 'production'])):
            frontend_agent_type = self._get_agent_type_for_role('frontend_developer', agent_type_strategy)
            self.logger.info(f"Frontend decomposition: strategy={agent_type_strategy}, role='frontend_developer', mapped_type={frontend_agent_type}")
            agent_tasks.append({
                'agent_type': frontend_agent_type,
                'focus_areas': ['frontend', 'ui', 'components', 'styling'],
                'command': f'{language_context}Create complete frontend/UI code for: {command}. Work within the frontend/ directory if it exists, otherwise create your own structure. Create a separate package.json for the frontend, set up the build system (Vite/Webpack), install dependencies, and implement all UI components, pages, styling, and API integration. Ensure the frontend is runnable and properly structured.',
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
                'command': f'{language_context}Create comprehensive tests for: {command}. Add tests to the appropriate directories (backend/tests/, frontend/tests/, or e2e/). Set up test configurations, implement unit tests, integration tests, and end-to-end tests. After creating tests, run npm install if needed and execute the test suites to ensure they pass.',
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
                'command': f'{language_context}Create deployment configuration for: {command}. Set up Docker, CI/CD, and production deployment scripts.',
                'coordination_context': {
                    'role': 'devops_engineer',
                    'dependencies': ['backend_services', 'ui_components', 'test_suite'],
                    'provides': ['deployment_config', 'infrastructure', 'ci_cd_pipeline']
                }
            })
        
        # Task 6: Finalization/Setup (always add for multi-agent projects)
        if len(agent_tasks) > 1:  # Only for multi-agent execution
            # Determine which agent should handle finalization based on project type
            finalization_agent_type = self._get_finalization_agent(project_type, agent_type_strategy)
            
            # Build dependencies list based on what agents were actually spawned
            finalizer_dependencies = []
            for task in agent_tasks:
                if 'provides' in task['coordination_context']:
                    # Add main outputs from each agent
                    provides = task['coordination_context']['provides']
                    if provides:
                        finalizer_dependencies.append(provides[0])  # Take first/main output
            
            agent_tasks.append({
                'agent_type': finalization_agent_type,
                'focus_areas': ['setup', 'configuration', 'finalization'],
                'command': f'{language_context}Finalize the project setup. Run necessary installation commands (npm install in appropriate directories), verify all configurations are correct, ensure all services can start properly, and create a quick start guide in the README. Make sure the entire system is ready to run.',
                'coordination_context': {
                    'role': 'finalizer',
                    'dependencies': finalizer_dependencies,  # Dynamic dependencies based on actual agents
                    'provides': ['project_ready']
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
            # All Claude Code agents except architect
            if role == "architect":
                return "gemini"  # Gemini as chief architect
            return "claude_code"
        
        elif strategy == "aider":
            # All Aider agents, specialized by role
            if role == "architect":
                return "gemini"  # Gemini as chief architect
            elif role == "frontend_developer":
                return "aider_frontend"
            elif role == "qa_engineer":
                return "aider_testing"
            else:
                return "aider_backend"  # Default to backend
        
        elif strategy == "mixed":
            # Force both types - alternate between them
            mixed_mapping = {
                "architect": "gemini",  # Gemini as chief architect
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
                "architect": "gemini",        # Gemini for system-wide architecture
                "backend_developer": "aider_backend",  # Aider for multi-file
                "frontend_developer": "aider_frontend",  # Aider for components
                "qa_engineer": "aider_testing",    # Aider for multi-file test creation
                "devops_engineer": "aider_backend"    # Aider for configs
            }
            return enhanced_mapping.get(role, "claude_code")
        
        else:  # dynamic (default)
            # Intelligent selection based on role strengths
            dynamic_mapping = {
                "architect": "gemini",        # Gemini excels at system-wide analysis
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
        from agentic.core.autonomous_execution import AutonomousExecutor
        
        # Note: command may already be enriched if coming from execute_multi_agent_command
        # Check if we need to enrich it (only if not already enriched)
        if not command.startswith("Based on the following analysis:"):
            enriched_command = await self._enrich_command_with_context(command, context)
        else:
            enriched_command = command
        
        intent_classifier = IntentClassifier()
        intent = await intent_classifier.analyze_intent(enriched_command)
        
        # Create task with enriched command that includes context
        task = Task.from_intent(intent, enriched_command)
        
        # Determine best agent for the task
        agent_type_strategy = context.get('agent_type_strategy', 'dynamic')
        
        # Check if this is a follow-up command - if so, prefer Claude Code for continuity
        is_follow_up = any(indicator in command.lower() for indicator in [
            'now implement', 'great!', 'can you now', 'based on that', 'the missing logic'
        ])
        
        if is_follow_up and agent_type_strategy == 'dynamic':
            # For follow-ups, prefer Claude Code to maintain context continuity
            task.agent_type_hint = 'claude_code'
            self.logger.info("Routing follow-up command to Claude Code for context continuity")
        elif intent.task_type == TaskType.TEST:
            # For test tasks, prefer Claude Code for its natural iteration
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
            else:  # For 'dynamic' or any other strategy
                # Default to aider_backend for implementation tasks
                task.agent_type_hint = 'aider_backend'
        
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
    
    async def _enrich_command_with_context(self, command: str, context: Dict[str, Any]) -> str:
        """Enrich a command with previous context if it's a follow-up"""
        # Check if this is a follow-up command
        follow_up_indicators = [
            'now implement', 'great!', 'perfect!', 'thanks!', 'can you now',
            'please implement', 'go ahead', 'proceed with', 'let\'s implement',
            'based on that', 'using this analysis', 'with this information',
            'the missing logic', 'complete the', 'implement it', 'build that'
        ]
        
        command_lower = command.lower()
        is_follow_up = any(indicator in command_lower for indicator in follow_up_indicators)
        
        if not is_follow_up:
            return command
        
        self.logger.info("Detected follow-up command, searching for previous context...")
        found_context = False
        
        # Try to get previous analysis from session manager
        session_manager = context.get('session_manager')
        if session_manager:
            self.logger.debug("Checking session manager for recent analysis")
            # Get recent analysis from session
            recent_analysis = self._get_recent_analysis_from_session(session_manager)
            if recent_analysis:
                # Prepend the analysis to the command
                enriched = f"Based on the following analysis:\n\n{recent_analysis}\n\nNow {command}"
                self.logger.info(f"Enriched command with previous analysis from session (length: {len(recent_analysis)})")
                return enriched
            else:
                self.logger.debug("No recent analysis found in session")
        
        # Try shared memory for recent task results
        self.logger.debug("Checking shared memory for recent task results")
        recent_tasks = self.shared_memory.get_all_task_progress()
        
        # Sort by completion time to get most recent first
        completed_tasks = []
        for task_id, progress in recent_tasks.items():
            if progress.get('status') == 'completed' and progress.get('completed_at'):
                completed_tasks.append((task_id, progress))
        
        completed_tasks.sort(key=lambda x: x[1].get('completed_at', datetime.min), reverse=True)
        
        # Check recent completed tasks
        for task_id, progress in completed_tasks[:5]:  # Check last 5 tasks
            result = progress.get('result')
            if result and hasattr(result, 'output') and result.output:
                # More flexible detection of analysis content
                output_lower = result.output.lower()
                analysis_keywords = [
                    'analysis', 'implementation', 'missing', 'need to', 'would need',
                    'should', 'could', 'require', 'suggest', 'recommend', 'approach'
                ]
                
                if any(word in output_lower for word in analysis_keywords) and len(result.output) > 200:
                    # Check if this was a recent task (within last 10 minutes)
                    completed_at = progress.get('completed_at')
                    if completed_at and (datetime.utcnow() - completed_at).total_seconds() < 600:
                        # Use this analysis as context
                        enriched = f"Based on the following analysis:\n\n{result.output[:2000]}\n\nNow {command}"
                        self.logger.info(f"Enriched command with analysis from task {task_id} (length: {len(result.output)})")
                        found_context = True
                        return enriched
        
        if not found_context:
            self.logger.warning("No previous analysis found for follow-up command - agent will start fresh")
        
        return command
    
    def _get_recent_analysis_from_session(self, session_manager) -> Optional[str]:
        """Extract recent analysis from session history"""
        try:
            # Get current session entries
            if hasattr(session_manager, 'current_session') and session_manager.current_session:
                entries = session_manager.current_session.entries
                # Look for recent analysis responses (last 5 entries)
                for entry in reversed(entries[-5:]):
                    response_lower = entry.response.lower()
                    # Check if this was an analysis response
                    if any(word in response_lower for word in ['analysis', 'need to implement', 'missing logic', 'would need']):
                        return entry.response
            
            # Try last session if no current session
            last_session = session_manager.get_last_session()
            if last_session and last_session.entries:
                for entry in reversed(last_session.entries[-5:]):
                    response_lower = entry.response.lower()
                    if any(word in response_lower for word in ['analysis', 'need to implement', 'missing logic', 'would need']):
                        return entry.response
        except Exception as e:
            self.logger.warning(f"Failed to get analysis from session: {e}")
        
        return None
    
    def _map_discovery_type(self, discovery_type: DiscoveryType) -> IntelligentDiscoveryType:
        """Map agent discovery type to intelligent coordinator discovery type"""
        mapping = {
            DiscoveryType.API_READY: IntelligentDiscoveryType.API_READY,
            DiscoveryType.TEST_NEEDED: IntelligentDiscoveryType.DOCUMENTATION_NEEDED,  # Map to closest equivalent
            DiscoveryType.BUG_FOUND: IntelligentDiscoveryType.BUG_FOUND,
            DiscoveryType.SECURITY_ISSUE: IntelligentDiscoveryType.SECURITY_ISSUE,
            DiscoveryType.PERFORMANCE_ISSUE: IntelligentDiscoveryType.PERFORMANCE_ISSUE,
            DiscoveryType.REFACTOR_OPPORTUNITY: IntelligentDiscoveryType.CODE_SMELL,
            DiscoveryType.DEPENDENCY_UPDATE: IntelligentDiscoveryType.DEPENDENCY_MISSING,
            DiscoveryType.DOCUMENTATION_NEEDED: IntelligentDiscoveryType.DOCUMENTATION_NEEDED,
            DiscoveryType.CONFIG_ISSUE: IntelligentDiscoveryType.DEPENDENCY_MISSING,
            DiscoveryType.INTEGRATION_POINT: IntelligentDiscoveryType.INTEGRATION_READY,
        }
        return mapping.get(discovery_type, IntelligentDiscoveryType.CODE_SMELL)
    
    async def _process_discovery_for_tasks(self, discovery: IntelligentDiscovery) -> None:
        """Process a discovery to generate new tasks dynamically"""
        try:
            # Report discovery to intelligent coordinator
            await self.intelligent_coordinator.report_discovery(discovery)
            
            # Get any newly generated tasks from the coordinator
            # The intelligent coordinator's feedback processor will have created tasks
            # We need to check if there are ready tasks to execute
            ready_tasks = self.intelligent_coordinator.dependency_graph.get_ready_tasks()
            
            if ready_tasks:
                self.logger.info(f"Discovery generated {len(ready_tasks)} new tasks")
                
                # Get the actual task objects
                new_tasks = []
                for task_id in ready_tasks:
                    if task_id in self.intelligent_coordinator.active_tasks:
                        new_tasks.append(self.intelligent_coordinator.active_tasks[task_id])
                
                if new_tasks:
                    # Add new tasks to the current execution context if one exists
                    for execution_id, context in self.active_executions.items():
                        if context.status == "running":
                            # Add tasks to the current execution
                            context.tasks.extend(new_tasks)
                            self.logger.info(f"Added {len(new_tasks)} dynamically generated tasks to execution {execution_id}")
                            
                            # Execute the new tasks
                            asyncio.create_task(self._execute_dynamic_tasks(new_tasks, context))
                            break
                    else:
                        # No active execution, create a background execution
                        self.logger.info(f"Creating background execution for {len(new_tasks)} dynamically generated tasks")
                        asyncio.create_task(self._execute_background_dynamic_tasks(new_tasks))
                        
        except Exception as e:
            self.logger.error(f"Failed to process discovery for tasks: {e}")
            self.logger.error(f"Discovery processing traceback:\n{traceback.format_exc()}")
    
    async def _execute_dynamic_tasks(self, tasks: List[Task], context: ExecutionContext) -> None:
        """Execute dynamically generated tasks within an existing execution context"""
        try:
            # Register tasks in shared memory
            for task in tasks:
                await self.shared_memory.register_task(task)
            
            # Execute tasks
            task_results = await self._execute_parallel_group(tasks, context)
            
            # Update context with results
            for task in tasks:
                if task.id in task_results and task_results[task.id].success:
                    context.completed_tasks.append(task.id)
                else:
                    context.failed_tasks.append(task.id)
                    
        except Exception as e:
            self.logger.error(f"Failed to execute dynamic tasks: {e}")
    
    async def _execute_background_dynamic_tasks(self, tasks: List[Task]) -> None:
        """Execute dynamically generated tasks as background tasks"""
        try:
            # Create a simple execution context
            execution_id = str(uuid.uuid4())
            context = ExecutionContext(
                execution_id=execution_id,
                tasks=tasks,
                status="running",
                started_at=datetime.utcnow(),
                completed_tasks=[],
                failed_tasks=[],
                active_tasks={}
            )
            
            # Execute with coordination
            result = await self.execute_coordinated_tasks(tasks)
            
            self.logger.info(f"Background dynamic task execution completed: {result.status}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute background dynamic tasks: {e}") 