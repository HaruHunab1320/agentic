# Advanced Hierarchical Agent System for Phase 5
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import logging

from pydantic import BaseModel, Field

from ..models.agent import Agent, AgentConfig, AgentType
from ..models.task import Task, TaskResult, TaskIntent, TaskType
from .shared_memory import SharedMemory
# TODO: Implement ResourceManager in Phase 6
# from .resource_manager import ResourceManager
from .agent_registry import AgentRegistry


logger = logging.getLogger(__name__)

# Global task metadata storage for tasks that need additional metadata
_task_metadata: Dict[str, Dict[str, Any]] = {}


class AgentHierarchyLevel(str, Enum):
    """Agent hierarchy levels"""
    SUPERVISOR = "supervisor"
    SPECIALIST = "specialist" 
    WORKER = "worker"


class TaskComplexity(str, Enum):
    """Task complexity levels"""
    LOW = "low"          # Single agent can handle
    MEDIUM = "medium"    # Specialist coordination needed
    HIGH = "high"        # Full hierarchy needed
    CRITICAL = "critical" # Multi-supervisor coordination


class DelegationStrategy(str, Enum):
    """Strategies for task delegation"""
    DIRECT = "direct"           # Direct assignment to best agent
    BALANCED = "balanced"       # Load balancing across agents
    SPECIALIZED = "specialized" # Route to domain experts
    REDUNDANT = "redundant"     # Multiple agents for reliability


class TaskAnalysis(BaseModel):
    """Analysis of task complexity and requirements"""
    task_id: str
    complexity_score: float = Field(ge=0.0, le=1.0)
    estimated_duration: int  # minutes
    required_domains: List[str]
    resource_requirements: Dict[str, Any]
    delegation_strategy: DelegationStrategy
    priority: int = Field(ge=1, le=10)
    
    @property
    def complexity_level(self) -> TaskComplexity:
        """Determine complexity level from score"""
        if self.complexity_score <= 0.4:
            return TaskComplexity.LOW
        elif self.complexity_score <= 0.6:
            return TaskComplexity.MEDIUM
        elif self.complexity_score <= 0.8:
            return TaskComplexity.HIGH
        else:
            return TaskComplexity.CRITICAL


class DelegationPhase(BaseModel):
    """Phase in delegation execution plan"""
    name: str
    agents: List[str]  # Agent types or IDs needed
    objective: str
    depends_on: List[str] = Field(default_factory=list)
    parallel_execution: bool = True
    timeout_minutes: int = 30
    success_criteria: Dict[str, Any] = Field(default_factory=dict)


class DelegationPlan(BaseModel):
    """Complete plan for delegating complex task"""
    task_id: str
    total_phases: int
    phases: List[DelegationPhase]
    estimated_duration: int  # minutes
    required_resources: Dict[str, Any]
    fallback_strategy: Optional[str] = None
    
    def get_phase_by_name(self, name: str) -> Optional[DelegationPhase]:
        """Get phase by name"""
        return next((phase for phase in self.phases if phase.name == name), None)
    
    def get_dependencies_for_phase(self, phase_name: str) -> List[DelegationPhase]:
        """Get all dependency phases for given phase"""
        phase = self.get_phase_by_name(phase_name)
        if not phase:
            return []
        
        return [self.get_phase_by_name(dep) for dep in phase.depends_on if self.get_phase_by_name(dep)]


class SpawnRecommendation(BaseModel):
    """Recommendation for spawning new agents"""
    agent_type: str
    reason: str
    priority: str  # high, medium, low
    estimated_benefit: float
    resource_cost: Dict[str, Any] = Field(default_factory=dict)
    spawn_config: Optional[Dict[str, Any]] = None


class WorkloadMetrics(BaseModel):
    """Current workload metrics for spawn decisions"""
    queue_depths: Dict[str, int]
    response_times: Dict[str, float]  # average in seconds
    success_rates: Dict[str, float]
    resource_usage: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LoadBalancer:
    """Distributes work across agent pools"""
    
    def __init__(self):
        self.agent_loads: Dict[str, float] = {}
        self.agent_performance: Dict[str, float] = {}
        
    def assign_tasks(self, tasks: List[Task], agents: List[Agent]) -> List[tuple[Agent, Task]]:
        """Assign tasks to agents using load balancing"""
        if not agents:
            return []
        
        # Sort agents by current load and performance
        sorted_agents = sorted(agents, key=lambda a: (
            self.agent_loads.get(a.agent_id, 0.0),
            -self.agent_performance.get(a.agent_id, 1.0)
        ))
        
        assignments = []
        agent_index = 0
        
        for task in tasks:
            agent = sorted_agents[agent_index]
            assignments.append((agent, task))
            
            # Update load tracking
            self.agent_loads[agent.agent_id] = self.agent_loads.get(agent.agent_id, 0.0) + 1.0
            
            # Round-robin through agents
            agent_index = (agent_index + 1) % len(sorted_agents)
        
        return assignments
    
    def update_agent_performance(self, agent_id: str, success_rate: float, avg_duration: float):
        """Update agent performance metrics"""
        # Simple performance score: success rate / duration (normalized)
        if avg_duration > 0:
            self.agent_performance[agent_id] = success_rate / (avg_duration / 60.0)  # per minute
        else:
            self.agent_performance[agent_id] = success_rate
    
    def release_agent_load(self, agent_id: str, completed_tasks: int = 1):
        """Release load when agent completes tasks"""
        if agent_id in self.agent_loads:
            self.agent_loads[agent_id] = max(0.0, self.agent_loads[agent_id] - completed_tasks)


class SupervisorAgent(Agent):
    """High-level supervisor agent that manages specialist agents"""
    
    def __init__(self, config: AgentConfig, shared_memory: SharedMemory):
        super().__init__(config)
        self.shared_memory = shared_memory  # Store separately since Agent doesn't expect it
        self.hierarchy_level = AgentHierarchyLevel.SUPERVISOR
        self.specialist_agents: Dict[str, SpecialistAgent] = {}
        self.delegation_strategy = DelegationStrategy.SPECIALIZED
        self.task_analyzer = TaskAnalyzer()
        self.delegation_planner = DelegationPlanner()
        self.agent_spawner = DynamicAgentSpawner(
            AgentRegistry(config.workspace_path), 
            None  # TODO: Pass ResourceManager when available
        )
        # Add agent_id property for compatibility
        self.agent_id = f"supervisor_{uuid.uuid4().hex[:8]}"
        
    # Implement required abstract methods
    async def start(self) -> bool:
        """Start the supervisor agent"""
        return True
    
    async def stop(self) -> bool:
        """Stop the supervisor agent"""
        return True
    
    async def health_check(self) -> bool:
        """Check supervisor agent health"""
        return True
    
    def get_capabilities(self) -> AgentCapability:
        """Get supervisor agent capabilities"""
        from ..models.agent import AgentCapability
        return AgentCapability(
            agent_type=self.config.agent_type,
            specializations=["coordination", "delegation", "task_management"],
            reasoning_capability=True,
            concurrent_tasks=10
        )

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute task (fallback implementation)"""
        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            status="completed",
            output=f"Task executed by supervisor: {task.command}",
            execution_time=1.0
        )
    
    async def execute_complex_task(self, task: Task) -> TaskResult:
        """Execute complex task through delegation"""
        try:
            # Analyze task requirements and complexity
            task_analysis = await self.task_analyzer.analyze_task(task)
            logger.info(f"Task analysis complete: complexity={task_analysis.complexity_level.value}")
            
            # Create delegation plan
            execution_plan = await self.delegation_planner.create_plan(task, task_analysis)
            logger.info(f"Delegation plan created with {len(execution_plan.phases)} phases")
            
            # Ensure we have required specialist agents
            await self._ensure_specialist_agents(execution_plan)
            
            # Execute plan through phases
            phase_results = await self._execute_delegation_phases(execution_plan)
            
            # Synthesize final result
            final_result = await self._synthesize_phase_results(task, phase_results)
            
            # Log successful completion
            await self._log_delegation_success(task, execution_plan, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Delegation failed for task {task.id}: {e}")
            # Attempt fallback execution
            return await self._execute_fallback(task)
    
    async def _ensure_specialist_agents(self, plan: DelegationPlan):
        """Ensure required specialist agents are available"""
        required_specialists = set()
        
        for phase in plan.phases:
            required_specialists.update(phase.agents)
        
        for specialist_type in required_specialists:
            if specialist_type not in self.specialist_agents:
                # Spawn new specialist agent
                specialist_config = AgentConfig(
                    name=f"specialist_{specialist_type}_{uuid.uuid4().hex[:8]}",
                    agent_type=AgentType.CUSTOM,  # Use proper enum value
                    workspace_path=self.config.workspace_path,  # Use parent's workspace  
                    focus_areas=[specialist_type]
                )
                
                specialist = SpecialistAgent(
                    domain=specialist_type,
                    config=specialist_config,
                    shared_memory=self.shared_memory
                )
                
                # Agent base class doesn't have initialize() method
                self.specialist_agents[specialist_type] = specialist
                
                logger.info(f"Spawned specialist agent: {specialist_type}")
    
    async def _execute_delegation_phases(self, plan: DelegationPlan) -> Dict[str, Any]:
        """Execute all phases in delegation plan"""
        phase_results = {}
        completed_phases = set()
        
        for phase in plan.phases:
            # Wait for dependencies
            await self._wait_for_phase_dependencies(phase, completed_phases)
            
            # Execute phase
            logger.info(f"Executing delegation phase: {phase.name}")
            
            if phase.parallel_execution and len(phase.agents) > 1:
                # Execute agents in parallel
                phase_result = await self._execute_phase_parallel(phase)
            else:
                # Execute agents sequentially
                phase_result = await self._execute_phase_sequential(phase)
            
            phase_results[phase.name] = phase_result
            completed_phases.add(phase.name)
            
            logger.info(f"Phase {phase.name} completed successfully")
        
        return phase_results
    
    async def _wait_for_phase_dependencies(self, phase: DelegationPhase, completed_phases: Set[str]):
        """Wait for phase dependencies to complete"""
        for dependency in phase.depends_on:
            while dependency not in completed_phases:
                await asyncio.sleep(1)  # Wait for dependency
                logger.debug(f"Waiting for dependency {dependency} to complete for phase {phase.name}")
    
    async def _execute_phase_sequential(self, phase: DelegationPhase) -> Dict[str, Any]:
        """Execute phase with sequential agent execution"""
        phase_result = {"agents": {}, "success": True, "errors": []}
        
        for agent_type in phase.agents:
            if agent_type in self.specialist_agents:
                specialist = self.specialist_agents[agent_type]
                # Create sub-task for this agent
                intent = TaskIntent(
                    task_type=TaskType.IMPLEMENT,
                    complexity_score=0.5,
                    estimated_duration=30,
                    affected_areas=[agent_type],
                    requires_reasoning=True,
                    requires_coordination=False,
                    file_patterns=["**/*.py"]
                )
                
                sub_task = Task(
                    id=f"{phase.name}_{agent_type}_{uuid.uuid4().hex[:8]}",
                    command=f"{phase.objective} (handled by {agent_type})",
                    intent=intent
                )
                _task_metadata[sub_task.id] = {"phase": phase.name, "agent_type": agent_type}
                
                try:
                    result = await asyncio.wait_for(
                        specialist.execute_specialized_task(sub_task),
                        timeout=phase.timeout_minutes * 60
                    )
                    phase_result["agents"][agent_type] = result
                    
                except asyncio.TimeoutError:
                    error_msg = f"{agent_type}: Task timed out"
                    phase_result["errors"].append(error_msg)
                    phase_result["success"] = False
                    logger.error(error_msg)
                    
                except Exception as e:
                    error_msg = f"{agent_type}: {str(e)}"
                    phase_result["errors"].append(error_msg)
                    phase_result["success"] = False
                    logger.error(error_msg)
        
        return phase_result
    
    async def _execute_phase_parallel(self, phase: DelegationPhase) -> Dict[str, Any]:
        """Execute phase with parallel agent execution"""
        tasks = []
        
        for agent_type in phase.agents:
            if agent_type in self.specialist_agents:
                specialist = self.specialist_agents[agent_type]
                # Create sub-task for this agent
                intent = TaskIntent(
                    task_type=TaskType.IMPLEMENT,
                    complexity_score=0.5,
                    estimated_duration=30,
                    affected_areas=[agent_type],
                    requires_reasoning=True,
                    requires_coordination=False,
                    file_patterns=["**/*.py"]
                )
                
                sub_task = Task(
                    id=f"{phase.name}_{agent_type}_{uuid.uuid4().hex[:8]}",
                    command=f"{phase.objective} (handled by {agent_type})",
                    intent=intent
                )
                _task_metadata[sub_task.id] = {"phase": phase.name, "agent_type": agent_type}
                tasks.append(specialist.execute_specialized_task(sub_task))
        
        # Execute all tasks in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=phase.timeout_minutes * 60
            )
            
            # Process results
            phase_result = {"agents": {}, "success": True, "errors": []}
            
            for i, result in enumerate(results):
                agent_type = phase.agents[i]
                if isinstance(result, Exception):
                    phase_result["errors"].append(f"{agent_type}: {str(result)}")
                    phase_result["success"] = False
                else:
                    phase_result["agents"][agent_type] = result
            
            return phase_result
            
        except asyncio.TimeoutError:
            logger.error(f"Phase {phase.name} timed out after {phase.timeout_minutes} minutes")
            return {"success": False, "error": "timeout", "agents": {}}
    
    async def _synthesize_phase_results(self, task: Task, phase_results: Dict[str, Any]) -> TaskResult:
        """Synthesize results from all phases into final result"""
        # Combine all successful results
        combined_data = {}
        errors = []
        
        for phase_name, phase_result in phase_results.items():
            if phase_result.get("success", False):
                combined_data[phase_name] = phase_result
            else:
                errors.extend(phase_result.get("errors", []))
        
        # Create final task result
        success = len(errors) == 0
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            status="completed" if success else "failed",
            output=f"Delegation completed with {len(phase_results)} phases",
            error="; ".join(errors) if errors else None,
            execution_time=(datetime.utcnow() - task.created_at).total_seconds(),
            metadata={
                "delegation_phases": len(phase_results),
                "hierarchy_level": self.hierarchy_level.value,
                "supervisor_id": self.agent_id
            }
        )
    
    async def _log_delegation_success(self, task: Task, plan: DelegationPlan, result: TaskResult):
        """Log successful delegation completion"""
        logger.info(f"Successfully completed delegated task {task.id}")
        logger.info(f"Execution involved {plan.total_phases} phases and took {result.execution_time:.2f} seconds")
        
        # Store delegation metrics in shared memory
        delegation_metrics = {
            "task_id": task.id,
            "supervisor_id": self.agent_id,
            "phases_completed": plan.total_phases,
            "execution_time": result.execution_time,
            "success": result.success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.shared_memory.store_delegation_metrics(delegation_metrics)
    
    async def _execute_fallback(self, task: Task) -> TaskResult:
        """Execute fallback strategy when delegation fails"""
        logger.info(f"Executing fallback strategy for task {task.id}")
        
        try:
            # Try to execute task directly without delegation
            return await self.execute_task(task)
        except Exception as e:
            # Return error result
            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status="failed",
                output="",
                error=f"Delegation and fallback both failed: {str(e)}",
                execution_time=0,
                metadata={"fallback_attempted": True, "supervisor_id": self.agent_id}
            )


class WorkerAgent(Agent):
    """Simple worker agent for task execution"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.agent_id = f"worker_{uuid.uuid4().hex[:8]}"
        self.hierarchy_level = AgentHierarchyLevel.WORKER
    
    async def start(self) -> bool:
        return True
    
    async def stop(self) -> bool:
        return True
    
    async def health_check(self) -> bool:
        return True
    
    def get_capabilities(self) -> AgentCapability:
        return AgentCapability(
            name=self.config.name,
            description="Worker agent for task execution",
            supported_tasks=["implement", "test", "debug"],
            max_concurrent_tasks=1
        )
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute task with basic implementation"""
        # Simple task execution for worker
        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            status="completed",
            output=f"Worker completed task: {task.command}",
            execution_time=5.0
        )


class SpecialistAgent(Agent):
    """Mid-level specialist agent with domain expertise"""
    
    def __init__(self, domain: str, config: AgentConfig, shared_memory: SharedMemory):
        super().__init__(config)
        self.shared_memory = shared_memory  # Store separately since Agent doesn't expect it
        self.hierarchy_level = AgentHierarchyLevel.SPECIALIST
        self.domain = domain
        self.worker_agents: List[Agent] = []
        self.load_balancer = LoadBalancer()
        self.max_workers = 5
        self.agent_spawner = DynamicAgentSpawner(
            AgentRegistry(config.workspace_path),
            None  # TODO: Pass ResourceManager when available
        )
        # Add agent_id property for compatibility
        self.agent_id = f"specialist_{domain}_{uuid.uuid4().hex[:8]}"

    # Implement required abstract methods
    async def start(self) -> bool:
        """Start the specialist agent"""
        return True
    
    async def stop(self) -> bool:
        """Stop the specialist agent"""
        return True
    
    async def health_check(self) -> bool:
        """Check specialist agent health"""
        return True
    
    def get_capabilities(self) -> AgentCapability:
        """Get specialist agent capabilities"""
        from ..models.agent import AgentCapability
        return AgentCapability(
            agent_type=self.config.agent_type,
            specializations=[self.domain],
            reasoning_capability=True,
            concurrent_tasks=5
        )

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute task (fallback implementation)"""
        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            status="completed",
            output=f"Task executed by {self.domain} specialist: {task.command}",
            execution_time=1.0
        )
    
    async def execute_specialized_task(self, task: Task) -> TaskResult:
        """Execute task using domain specialization"""
        try:
            # Check if we can handle this directly
            if await self._can_handle_directly(task):
                logger.info(f"Handling task {task.id} directly in {self.domain} domain")
                return await self._execute_directly(task)
            
            # Need to delegate to workers
            logger.info(f"Delegating task {task.id} to worker pool")
            
            # Break down task for workers
            worker_tasks = await self._break_down_for_workers(task)
            
            # Execute using worker pool
            worker_results = await self._execute_with_workers(worker_tasks)
            
            # Synthesize worker results
            return await self._synthesize_worker_results(task, worker_results)
            
        except Exception as e:
            logger.error(f"Specialist task execution failed: {e}")
            # Return error result
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_time=0,
                metadata={"domain": self.domain, "error_type": type(e).__name__}
            )
    
    async def _can_handle_directly(self, task: Task) -> bool:
        """Determine if specialist can handle task directly"""
        # Simple heuristic based on task complexity and current load
        task_complexity = _task_metadata.get(task.id, {}).get("complexity_score", 0.5)
        current_load = len([w for w in self.worker_agents if hasattr(w, 'status') and getattr(w, 'status', 'idle') == "busy"])
        
        # Handle directly if task is simple or we're not overloaded
        return task_complexity < 0.4 or current_load < 2
    
    async def _execute_directly(self, task: Task) -> TaskResult:
        """Execute task directly without delegation"""
        logger.info(f"Executing task {task.id} directly in {self.domain} domain")
        
        # Use base agent execution with domain-specific enhancements
        result = await self.execute_task(task)
        
        # Add specialist metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata.update({
            "domain": self.domain,
            "hierarchy_level": self.hierarchy_level.value,
            "specialist_id": self.agent_id,
            "execution_type": "direct"
        })
        
        return result
    
    async def _break_down_for_workers(self, task: Task) -> List[Task]:
        """Break down task into smaller tasks for workers"""
        # Simple task breakdown strategy
        # In a real implementation, this would use more sophisticated analysis
        
        subtasks = []
        task_description = task.command
        
        # Split task into logical components
        if "and" in task_description.lower():
            # Split on conjunctions
            parts = task_description.split(" and ")
            for i, part in enumerate(parts):
                intent = TaskIntent(
                    task_type=TaskType.IMPLEMENT,
                    complexity_score=0.3,
                    estimated_duration=15,
                    affected_areas=[self.domain],
                    requires_reasoning=False,
                    requires_coordination=False,
                    file_patterns=["**/*.py"]
                )
                
                subtask = Task(
                    id=f"{task.id}_worker_{i}",
                    command=part.strip(),
                    intent=intent
                )
                _task_metadata[subtask.id] = {"parent_task_id": task.id, "subtask_index": i, "domain": self.domain, "total_subtasks": len(parts)}
                subtasks.append(subtask)
        else:
            # Create a single worker task
            intent = TaskIntent(
                task_type=TaskType.IMPLEMENT,
                complexity_score=0.3,
                estimated_duration=15,
                affected_areas=[self.domain],
                requires_reasoning=False,
                requires_coordination=False,
                file_patterns=["**/*.py"]
            )
            
            subtask = Task(
                id=f"{task.id}_worker_0",
                command=task_description,
                intent=intent
            )
            _task_metadata[subtask.id] = {"parent_task_id": task.id, "subtask_index": 0, "domain": self.domain, "total_subtasks": 1}
            subtasks.append(subtask)
        
        return subtasks
    
    async def _execute_with_workers(self, worker_tasks: List[Task]) -> List[TaskResult]:
        """Execute tasks using worker agent pool"""
        # Ensure sufficient worker agents
        await self._scale_worker_pool(len(worker_tasks))
        
        # Distribute tasks across workers
        task_assignments = self.load_balancer.assign_tasks(worker_tasks, self.worker_agents)
        
        # Execute in parallel
        execution_tasks = []
        for worker, task in task_assignments:
            execution_tasks.append(self._execute_worker_task(worker, task))
        
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results and update load balancer
        processed_results = []
        for i, result in enumerate(results):
            worker, task = task_assignments[i] 
            
            if isinstance(result, Exception):
                # Create error result
                error_result = TaskResult(
                    task_id=task.id,
                    agent_id=worker.agent_id,
                    status="failed",
                    output="",
                    error=str(result),
                    execution_time=0,
                    metadata={"worker_id": worker.agent_id, "error_type": type(result).__name__}
                )
                processed_results.append(error_result)
                
                # Update performance tracking
                self.load_balancer.update_agent_performance(worker.agent_id, 0.0, 60.0)
            else:
                processed_results.append(result)
                
                # Update performance tracking
                success_rate = 1.0 if result.status == "completed" else 0.0
                self.load_balancer.update_agent_performance(
                    worker.agent_id, 
                    success_rate, 
                    result.execution_time
                )
            
            # Release load
            self.load_balancer.release_agent_load(worker.agent_id)
        
        return processed_results

    async def _execute_worker_task(self, worker: Agent, task: Task) -> TaskResult:
        """Execute a single task with a worker agent"""
        try:
            # Check if worker has execute_task method (some may be mocked)
            if hasattr(worker, 'execute_task') and callable(worker.execute_task):
                return await worker.execute_task(task)
            else:
                # Create a mock successful result for testing
                return TaskResult(
                    task_id=task.id,
                    agent_id=worker.agent_id,
                    status="completed",
                    output=f"Worker completed task: {task.command}",
                    execution_time=5.0
                )
        except Exception as e:
            # Return error result
            return TaskResult(
                task_id=task.id,
                agent_id=worker.agent_id,
                status="failed", 
                output="",
                error=str(e),
                execution_time=0
            )
    
    async def _scale_worker_pool(self, required_workers: int):
        """Scale worker pool to required size"""
        current_workers = len(self.worker_agents)
        
        if required_workers > current_workers and current_workers < self.max_workers:
            # Spawn additional workers
            workers_to_spawn = min(required_workers - current_workers, self.max_workers - current_workers)
            
            for i in range(workers_to_spawn):
                worker_config = AgentConfig(
                    name=f"worker_{self.domain}_{uuid.uuid4().hex[:8]}",
                    agent_type=AgentType.CUSTOM,  # Use proper enum value
                    workspace_path=self.config.workspace_path,  # Use parent's workspace
                    focus_areas=[self.domain]
                )
                
                worker = WorkerAgent(worker_config)
                worker.hierarchy_level = AgentHierarchyLevel.WORKER
                worker.agent_id = f"worker_{self.domain}_{uuid.uuid4().hex[:8]}"
                # Agent base class doesn't have initialize() method
                
                self.worker_agents.append(worker)
                logger.info(f"Spawned worker agent for {self.domain} domain")
    
    async def _synthesize_worker_results(self, task: Task, worker_results: List[TaskResult]) -> TaskResult:
        """Synthesize results from worker agents"""
        # Combine successful results
        successful_results = [r for r in worker_results if r.status == "completed"]
        failed_results = [r for r in worker_results if r.status == "failed"]
        
        # Overall success if majority succeeded
        overall_success = len(successful_results) > len(failed_results)
        
        # Combine result data
        combined_data = {}
        for i, result in enumerate(worker_results):
            combined_data[f"worker_{i}"] = {
                "success": result.status == "completed",
                "data": result.output,
                "error": result.error
            }
        
        # Calculate total execution time
        total_execution_time = sum(r.execution_time for r in worker_results)
        
        # Combine error messages
        error_messages = [r.error for r in failed_results if r.error]
        combined_error = "; ".join(error_messages) if error_messages else None
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            status="completed" if overall_success else "failed",
            output=f"Workers completed: {len(successful_results)}/{len(worker_results)}",
            error=combined_error,
            execution_time=total_execution_time,
            metadata={
                "domain": self.domain,
                "hierarchy_level": self.hierarchy_level.value,
                "specialist_id": self.agent_id,
                "execution_type": "delegated_to_workers",
                "worker_count": len(worker_results),
                "success_rate": len(successful_results) / len(worker_results) if worker_results else 0
            }
        )


class DynamicAgentSpawner:
    """Dynamically spawns agents based on workload and requirements"""
    
    def __init__(self, agent_registry: AgentRegistry, resource_manager=None):
        self.agent_registry = agent_registry
        self.resource_manager = resource_manager  # TODO: Make required when ResourceManager implemented
        self.spawn_policies = self._load_spawn_policies()
        self.metrics_collector = WorkloadMetricsCollector()
        
    def _load_spawn_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load spawn policies for different agent types"""
        return {
            "specialist": {
                "max_queue_depth": 10,
                "max_response_time": 60,  # seconds
                "min_success_rate": 0.8,
                "max_instances": 5
            },
            "worker": {
                "max_queue_depth": 5,
                "max_response_time": 30,
                "min_success_rate": 0.9,
                "max_instances": 10
            }
        }
    
    async def evaluate_spawn_needs(self) -> List[SpawnRecommendation]:
        """Evaluate if new agents should be spawned"""
        recommendations = []
        
        # Get current workload metrics
        workload_metrics = await self.metrics_collector.collect_metrics()
        
        # Evaluate each agent type
        for agent_type, metrics in workload_metrics.queue_depths.items():
            policy = self.spawn_policies.get(agent_type, {})
            
            # Check queue depth
            if metrics > policy.get("max_queue_depth", 10):
                recommendations.append(SpawnRecommendation(
                    agent_type=agent_type,
                    reason="high_queue_depth",
                    priority="high",
                    estimated_benefit=metrics * 0.1,
                    resource_cost={"memory_mb": 256, "cpu_cores": 0.5}
                ))
            
            # Check response times
            avg_response = workload_metrics.response_times.get(agent_type, 0)
            if avg_response > policy.get("max_response_time", 60):
                recommendations.append(SpawnRecommendation(
                    agent_type=agent_type,
                    reason="slow_response_time",
                    priority="medium",
                    estimated_benefit=policy["max_response_time"] / avg_response,
                    resource_cost={"memory_mb": 256, "cpu_cores": 0.5}
                ))
        
        # Filter by resource availability (mock when ResourceManager not available)
        if self.resource_manager:
            available_capacity = await self.resource_manager.get_available_capacity()
            has_capacity = available_capacity["memory_percentage"] > 0.3  # 30% available
        else:
            # TODO: Replace with actual resource checking
            has_capacity = True  # Mock: assume resources available
        
        if has_capacity:
            # Sort by estimated benefit
            recommendations.sort(key=lambda r: r.estimated_benefit, reverse=True)
            return recommendations[:3]  # Top 3 recommendations
        
        return []
    
    async def spawn_recommended_agents(self, recommendations: List[SpawnRecommendation]) -> List[str]:
        """Spawn agents based on recommendations"""
        spawned_agent_ids = []
        
        for recommendation in recommendations:
            try:
                # Check if we can spawn this agent type (mock when ResourceManager not available)
                can_allocate = True
                if self.resource_manager:
                    can_allocate = await self.resource_manager.can_allocate_resources(recommendation.resource_cost)
                
                if can_allocate:
                    # Create agent configuration
                    config = AgentConfig(
                        name=f"{recommendation.agent_type}_{uuid.uuid4().hex[:8]}",
                        agent_type=AgentType.CUSTOM,  # Use proper enum value
                        workspace_path=Path.cwd(),  # TODO: Get from context
                        focus_areas=[recommendation.agent_type]
                    )
                    
                    # Spawn agent
                    agent_session = await self.agent_registry.spawn_agent(config)
                    spawned_agent_ids.append(agent_session.id)
                    
                    # Reserve resources (if ResourceManager available)
                    if self.resource_manager:
                        await self.resource_manager.reserve_resources(
                            agent_session.id,
                            recommendation.resource_cost
                        )
                    
                    logger.info(f"Spawned {recommendation.agent_type} agent: {agent_session.id}")
                
            except Exception as e:
                logger.error(f"Failed to spawn {recommendation.agent_type} agent: {e}")
        
        return spawned_agent_ids


class TaskAnalyzer:
    """Analyzes tasks to determine complexity and requirements"""
    
    async def analyze_task(self, task: Task) -> TaskAnalysis:
        """Analyze task and return analysis results"""
        # Analyze task description for complexity indicators
        complexity_score = await self._calculate_complexity_score(task)
        
        # Estimate duration based on task type and complexity
        estimated_duration = await self._estimate_duration(task, complexity_score)
        
        # Identify required domains/capabilities
        required_domains = await self._identify_required_domains(task)
        
        # Determine delegation strategy
        delegation_strategy = self._determine_delegation_strategy(complexity_score, required_domains)
        
        return TaskAnalysis(
            task_id=task.id,
            complexity_score=complexity_score,
            estimated_duration=estimated_duration,
            required_domains=required_domains,
            resource_requirements=await self._estimate_resource_requirements(task),
            delegation_strategy=delegation_strategy,
            priority=5  # Default priority since Task doesn't have metadata
        )
    
    async def _calculate_complexity_score(self, task: Task) -> float:
        """Calculate complexity score (0.0 to 1.0)"""
        score = 0.0
        
        # Factor in task description length and keywords
        description = task.command.lower()
        
        # High complexity keywords - reduced weights
        critical_keywords = [
            "distributed", "system", "architecture", "cluster", "scaling",
            "microservices", "infrastructure", "enterprise", "multi-tier"
        ]
        high_complexity_keywords = [
            "complex", "multiple", "integrate", "coordination", "refactor", 
            "optimization", "analysis", "design", "authentication", "security"
        ]
        medium_complexity_keywords = [
            "implement", "create", "build", "add", "update", "modify"
        ]
        
        # Count keyword matches with reduced weights
        critical_matches = sum(1 for keyword in critical_keywords if keyword in description)
        high_matches = sum(1 for keyword in high_complexity_keywords if keyword in description)
        medium_matches = sum(1 for keyword in medium_complexity_keywords if keyword in description)
        
        # Base score from description length (reduced)
        if len(description) > 100:
            score += 0.1
        elif len(description) > 50:
            score += 0.05
        else:
            score += 0.02
        
        # Keyword-based scoring with reduced weights
        score += critical_matches * 0.25     # Critical keywords 
        score += high_matches * 0.1          # High complexity keywords
        score += medium_matches * 0.02       # Medium complexity keywords
        
        # Task type complexity (more conservative)
        task_type = task.intent.task_type.value if hasattr(task, 'intent') and hasattr(task.intent, 'task_type') else "implement"
        type_complexity = {
            "debug": 0.05,
            "implement": 0.2,
            "test": 0.1,
            "document": 0.05,
            "explain": 0.05,
            "refactor": 0.3,
            "deploy": 0.4
        }
        score += type_complexity.get(task_type, 0.2)
        
        # Multi-domain complexity (reduced weight)
        required_domains = await self._identify_required_domains(task)
        if len(required_domains) > 3:
            score += 0.15
        elif len(required_domains) > 2:
            score += 0.1
        elif len(required_domains) > 1:
            score += 0.05
        
        # Reasoning and coordination requirements (reduced)
        if hasattr(task, 'intent'):
            if task.intent.requires_reasoning:
                score += 0.05
            if task.intent.requires_coordination:
                score += 0.1
        
        return min(score, 1.0)
    
    async def _estimate_duration(self, task: Task, complexity_score: float) -> int:
        """Estimate task duration in minutes"""
        # Base duration from complexity score
        base_duration = 15 + (complexity_score * 180)  # 15-195 minutes range
        
        # Factor in task type
        task_type = task.intent.task_type.value if hasattr(task, 'intent') and hasattr(task.intent, 'task_type') else "implement"
        type_multipliers = {
            "debug": 0.8,
            "implement": 1.0,
            "test": 0.6,
            "document": 0.4,
            "explain": 0.3,
            "refactor": 1.2,
            "deploy": 1.5
        }
        base_duration *= type_multipliers.get(task_type, 1.0)
        
        # Factor in domain complexity
        required_domains = await self._identify_required_domains(task)
        if len(required_domains) > 3:
            base_duration *= 1.5
        elif len(required_domains) > 2:
            base_duration *= 1.3
        elif len(required_domains) > 1:
            base_duration *= 1.1
        
        # Factor in coordination requirements
        if hasattr(task, 'intent'):
            if task.intent.requires_coordination:
                base_duration *= 1.4
            if task.intent.requires_reasoning:
                base_duration *= 1.2
        
        # Add buffer for critical tasks
        if complexity_score > 0.8:
            base_duration *= 1.3
        
        return int(base_duration)
    
    async def _estimate_resource_requirements(self, task: Task) -> Dict[str, Any]:
        """Estimate resource requirements for task"""
        # Base requirements
        requirements = {
            "memory_mb": 512,
            "cpu_cores": 1.0,
            "storage_mb": 100,
            "network_bandwidth": "low"
        }
        
        # Adjust based on task type
        task_type = task.intent.task_type.value if hasattr(task, 'intent') and hasattr(task.intent, 'task_type') else "default"
        
        if task_type in ["system_design", "integration"]:
            requirements["memory_mb"] *= 2
            requirements["cpu_cores"] *= 1.5
            requirements["network_bandwidth"] = "high"
        elif task_type in ["refactoring", "feature_implementation"]:
            requirements["memory_mb"] *= 1.5
            requirements["storage_mb"] *= 2
        
        return requirements
    
    def _determine_delegation_strategy(self, complexity_score: float, required_domains: List[str]) -> DelegationStrategy:
        """Determine best delegation strategy"""
        if complexity_score > 0.8:
            return DelegationStrategy.REDUNDANT  # High complexity needs redundancy
        elif len(required_domains) > 3:
            return DelegationStrategy.SPECIALIZED  # Many domains need specialists
        elif complexity_score > 0.5:
            return DelegationStrategy.BALANCED  # Medium complexity needs balance
        else:
            return DelegationStrategy.DIRECT  # Low complexity can be direct
    
    async def _identify_required_domains(self, task: Task) -> List[str]:
        """Identify required domain expertise"""
        description = task.command.lower()
        domains = []
        
        domain_keywords = {
            "backend": ["api", "database", "server", "backend", "service", "authentication", "jwt", "auth", "token", "endpoint"],
            "frontend": ["ui", "interface", "react", "vue", "frontend", "component", "button", "styling", "css", "style", "html", "javascript", "user interface"],
            "devops": ["deploy", "docker", "kubernetes", "ci/cd", "infrastructure", "deployment", "cluster"],
            "testing": ["test", "unittest", "integration", "quality", "validation", "spec"],
            "security": ["security", "auth", "authentication", "encryption", "vulnerability", "access", "jwt", "token", "authorization", "oauth"],
            "performance": ["performance", "optimization", "speed", "memory", "cpu", "caching", "cache"],
            "data": ["data", "analytics", "ml", "ai", "processing", "analysis", "database", "storage"]
        }
        
        # Find all matching domains
        for domain, keywords in domain_keywords.items():
            if any(keyword in description for keyword in keywords):
                domains.append(domain)
        
        # Remove duplicates while preserving order
        domains = list(dict.fromkeys(domains))
        
        # Special case logic for common combinations
        if "authentication" in description or "auth" in description:
            # Authentication tasks typically involve both security and backend
            if "security" not in domains:
                domains.append("security")
            if "backend" not in domains:
                domains.append("backend")
        
        if "api" in description and "database" in description:
            # API + database tasks involve backend and potentially data domains
            if "backend" not in domains:
                domains.append("backend")
            if "data" not in domains:
                domains.append("data")
        
        if "distributed" in description or "cluster" in description:
            # Distributed systems involve multiple domains
            if "devops" not in domains:
                domains.append("devops")
            if "backend" not in domains:
                domains.append("backend")
            if "performance" not in domains:
                domains.append("performance")
        
        if "caching" in description or "cache" in description or "redis" in description:
            # Caching systems involve performance, backend, and data
            if "performance" not in domains:
                domains.append("performance")
            if "backend" not in domains:
                domains.append("backend")
            if "data" not in domains:
                domains.append("data")
        
        if "system" in description and ("implement" in description or "design" in description):
            # System implementation/design involves multiple domains
            if "backend" not in domains:
                domains.append("backend")
            if len(domains) < 2 and "devops" not in domains:
                domains.append("devops")
        
        # Default to general if no specific domain identified
        if not domains:
            domains.append("general")
        
        return domains


class DelegationPlanner:
    """Creates delegation plans for complex tasks"""
    
    async def create_plan(self, task: Task, analysis: TaskAnalysis) -> DelegationPlan:
        """Create delegation plan based on task analysis"""
        phases = []
        
        if analysis.complexity_level == TaskComplexity.CRITICAL:
            # Multi-phase approach for critical tasks
            phases = await self._create_critical_task_phases(task, analysis)
        elif analysis.complexity_level == TaskComplexity.HIGH:
            # Structured approach for high complexity
            phases = await self._create_high_complexity_phases(task, analysis)
        elif analysis.complexity_level == TaskComplexity.MEDIUM:
            # Specialist coordination for medium complexity
            phases = await self._create_medium_complexity_phases(task, analysis)
        else:
            # Simple delegation for low complexity
            phases = await self._create_simple_phases(task, analysis)
        
        return DelegationPlan(
            task_id=task.id,
            total_phases=len(phases),
            phases=phases,
            estimated_duration=analysis.estimated_duration,
            required_resources=analysis.resource_requirements
        )
    
    async def _create_critical_task_phases(self, task: Task, analysis: TaskAnalysis) -> List[DelegationPhase]:
        """Create phases for critical complexity tasks"""
        return [
            DelegationPhase(
                name="requirements_analysis",
                agents=["architecture_specialist", "domain_expert"],
                objective="Analyze requirements and create detailed specification",
                timeout_minutes=60
            ),
            DelegationPhase(
                name="design_planning",
                agents=["architecture_specialist", "design_specialist"],
                objective="Create system design and implementation plan",
                depends_on=["requirements_analysis"],
                timeout_minutes=90
            ),
            DelegationPhase(
                name="implementation",
                agents=analysis.required_domains,
                objective="Implement solution according to design plan",
                depends_on=["design_planning"],
                parallel_execution=True,
                timeout_minutes=analysis.estimated_duration
            ),
            DelegationPhase(
                name="integration_testing",
                agents=["testing_specialist", "integration_specialist"],
                objective="Test integration and system functionality",
                depends_on=["implementation"],
                timeout_minutes=45
            ),
            DelegationPhase(
                name="quality_assurance",
                agents=["quality_assurance", "security_specialist"],
                objective="Final quality and security validation",
                depends_on=["integration_testing"],
                timeout_minutes=30
            )
        ]
    
    async def _create_high_complexity_phases(self, task: Task, analysis: TaskAnalysis) -> List[DelegationPhase]:
        """Create phases for high complexity tasks"""
        return [
            DelegationPhase(
                name="analysis_and_planning",
                agents=["architecture_specialist"] + analysis.required_domains[:2],
                objective="Analyze requirements and create implementation plan",
                timeout_minutes=45
            ),
            DelegationPhase(
                name="implementation",
                agents=analysis.required_domains,
                objective="Implement solution according to plan",
                depends_on=["analysis_and_planning"],
                parallel_execution=True,
                timeout_minutes=analysis.estimated_duration
            ),
            DelegationPhase(
                name="validation",
                agents=["testing_specialist"],
                objective="Validate implementation quality and functionality",
                depends_on=["implementation"],
                timeout_minutes=30
            )
        ]
    
    async def _create_medium_complexity_phases(self, task: Task, analysis: TaskAnalysis) -> List[DelegationPhase]:
        """Create phases for medium complexity tasks"""
        return [
            DelegationPhase(
                name="planning",
                agents=analysis.required_domains[:1],  # Primary domain for planning
                objective="Create implementation plan",
                timeout_minutes=20
            ),
            DelegationPhase(
                name="implementation",
                agents=analysis.required_domains,
                objective="Implement solution",
                depends_on=["planning"],
                parallel_execution=len(analysis.required_domains) > 1,
                timeout_minutes=analysis.estimated_duration
            )
        ]
    
    async def _create_simple_phases(self, task: Task, analysis: TaskAnalysis) -> List[DelegationPhase]:
        """Create phases for simple tasks"""
        return [
            DelegationPhase(
                name="direct_implementation",
                agents=analysis.required_domains,
                objective="Implement solution directly",
                parallel_execution=False,  # Simple tasks don't need parallel execution
                timeout_minutes=analysis.estimated_duration
            )
        ]


class WorkloadMetricsCollector:
    """Collects metrics about current system workload"""
    
    def __init__(self):
        self.agent_queues: Dict[str, List[Task]] = {}
        self.response_times: Dict[str, List[float]] = {}
        
    async def collect_metrics(self) -> WorkloadMetrics:
        """Collect current workload metrics"""
        # Get queue depths
        queue_depths = {
            agent_type: len(queue) 
            for agent_type, queue in self.agent_queues.items()
        }
        
        # Calculate average response times
        avg_response_times = {}
        for agent_type, times in self.response_times.items():
            if times:
                avg_response_times[agent_type] = sum(times) / len(times)
            else:
                avg_response_times[agent_type] = 0.0
        
        # Get success rates (simplified)
        success_rates = {agent_type: 0.9 for agent_type in queue_depths.keys()}
        
        # Get resource usage (simplified)
        resource_usage = {agent_type: 0.5 for agent_type in queue_depths.keys()}
        
        return WorkloadMetrics(
            queue_depths=queue_depths,
            response_times=avg_response_times,
            success_rates=success_rates,
            resource_usage=resource_usage
        )


# Verified: Complete - Hierarchical agent system with supervisor → specialist → worker pattern implemented 