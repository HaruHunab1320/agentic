"""
Intelligent Multi-Agent Coordinator with Feedback Loops

This module implements progressive task execution with real-time agent feedback,
dynamic task generation, and dependency-aware scheduling.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from agentic.models.task import Task, TaskResult, TaskIntent, TaskType
from agentic.utils.logging import LoggerMixin


class DiscoveryType(str, Enum):
    """Types of discoveries agents can report"""
    API_READY = "api_ready"
    TEST_FAILING = "test_failing"
    TEST_PASSING = "test_passing"
    BUG_FOUND = "bug_found"
    IMPLEMENTATION_COMPLETE = "implementation_complete"
    DEPENDENCY_MISSING = "dependency_missing"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    CODE_SMELL = "code_smell"
    INTEGRATION_READY = "integration_ready"
    DOCUMENTATION_NEEDED = "documentation_needed"


class TaskPhase(str, Enum):
    """Execution phases for adaptive coordination"""
    EXPLORATION = "exploration"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    INTEGRATION = "integration"
    VALIDATION = "validation"
    REFINEMENT = "refinement"


class VerificationPhase(str, Enum):
    """Phases specific to verification loops"""
    PRE_VERIFICATION = "pre_verification"
    VERIFICATION = "verification"
    FIX_GENERATION = "fix_generation"
    FIX_EXECUTION = "fix_execution"
    POST_VERIFICATION = "post_verification"


@dataclass
class AgentDiscovery:
    """A discovery reported by an agent during execution"""
    agent_id: str
    discovery_type: DiscoveryType
    severity: str  # "info", "warning", "critical"
    context: Dict[str, Any]
    suggestions: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityMetrics:
    """Track quality metrics over time"""
    test_pass_rate: float = 0.0
    code_coverage: float = 0.0
    lint_score: float = 0.0
    build_success: bool = False
    system_health_score: float = 0.0
    total_failures: int = 0
    fixed_failures: int = 0
    verification_iterations: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score"""
        scores = []
        if self.test_pass_rate > 0:
            scores.append(self.test_pass_rate)
        if self.lint_score > 0:
            scores.append(self.lint_score)
        if self.build_success:
            scores.append(1.0)
        if self.system_health_score > 0:
            scores.append(self.system_health_score)
        
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class TaskDependency:
    """Dynamic task dependency"""
    task_id: str
    depends_on: Set[str]  # Task IDs this depends on
    produces: Set[str]    # What this task produces (e.g., "auth_api", "user_model")
    satisfied: bool = False


class DynamicDependencyGraph:
    """Manages task dependencies dynamically"""
    
    def __init__(self):
        self.dependencies: Dict[str, TaskDependency] = {}
        self.completed_products: Set[str] = set()  # Things that have been produced
        self.task_products: Dict[str, Set[str]] = defaultdict(set)  # task_id -> products
        
    def add_task(self, task: Task, depends_on: Set[str] = None, produces: Set[str] = None):
        """Add a task with its dependencies"""
        dep = TaskDependency(
            task_id=task.id,
            depends_on=depends_on or set(),
            produces=produces or set()
        )
        self.dependencies[task.id] = dep
        
        # Check if dependencies are already satisfied
        self._check_satisfaction(task.id)
    
    def mark_complete(self, task_id: str, products: Set[str] = None):
        """Mark a task as complete and update dependencies"""
        if task_id in self.dependencies:
            # Add products to completed set
            if products:
                self.completed_products.update(products)
                self.task_products[task_id] = products
            else:
                # Use declared products
                self.completed_products.update(self.dependencies[task_id].produces)
            
            # Check all tasks to see if new ones are ready
            for tid in self.dependencies:
                self._check_satisfaction(tid)
    
    def _check_satisfaction(self, task_id: str):
        """Check if a task's dependencies are satisfied"""
        dep = self.dependencies.get(task_id)
        if dep:
            # Check if all required products are available
            dep.satisfied = dep.depends_on.issubset(self.completed_products)
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute"""
        ready = []
        for task_id, dep in self.dependencies.items():
            if dep.satisfied and task_id not in self.task_products:
                # Task is ready and not yet completed
                ready.append(task_id)
        return ready
    
    def add_dependency(self, task_id: str, new_deps: Set[str]):
        """Add new dependencies to an existing task"""
        if task_id in self.dependencies:
            self.dependencies[task_id].depends_on.update(new_deps)
            self._check_satisfaction(task_id)


class VerificationLoopController(LoggerMixin):
    """Controls verification loops with progressive improvement"""
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        from .verification_coordinator import VerificationCoordinator
        self.verification_coordinator = VerificationCoordinator(workspace_path)
        self.max_verification_iterations = 5
        self.min_quality_threshold = 0.8
        self.quality_history: List[QualityMetrics] = []
        
    async def run_verification_loop(self, phase: TaskPhase) -> Tuple[bool, QualityMetrics, List[Task]]:
        """Run a verification loop and return success status, metrics, and fix tasks"""
        self.logger.info(f"Starting verification loop for phase: {phase}")
        
        # Run verification
        verification_result = await self.verification_coordinator.verify_system()
        
        # Calculate quality metrics
        metrics = self._calculate_metrics(verification_result)
        self.quality_history.append(metrics)
        
        # Generate fix tasks if needed
        fix_tasks = []
        if not verification_result.success:
            fix_tasks = await self.verification_coordinator.analyze_failures(verification_result)
            
        success = metrics.overall_quality >= self.min_quality_threshold
        
        return success, metrics, fix_tasks
    
    def _calculate_metrics(self, result) -> QualityMetrics:
        """Calculate quality metrics from verification result"""
        # Calculate test pass rate
        total_tests = sum(r.total_tests for r in result.test_results.values())
        passed_tests = sum(r.passed_tests for r in result.test_results.values())
        test_pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Calculate other metrics
        build_success = result.test_results.get('build', type('', (), {'passed': False})()).passed
        
        # System health score
        health_checks = len(result.system_health)
        healthy_systems = sum(1 for healthy in result.system_health.values() if healthy)
        system_health_score = healthy_systems / health_checks if health_checks > 0 else 0.0
        
        return QualityMetrics(
            test_pass_rate=test_pass_rate,
            build_success=build_success,
            system_health_score=system_health_score,
            total_failures=result.total_failures,
            verification_iterations=len(self.quality_history) + 1
        )
    
    def should_continue_fixing(self) -> bool:
        """Determine if we should continue fixing or stop"""
        if len(self.quality_history) >= self.max_verification_iterations:
            self.logger.warning("Max verification iterations reached")
            return False
        
        # Check if quality is improving
        if len(self.quality_history) >= 2:
            recent_quality = self.quality_history[-1].overall_quality
            previous_quality = self.quality_history[-2].overall_quality
            
            # Stop if quality is not improving
            if recent_quality <= previous_quality:
                self.logger.info("Quality not improving, stopping verification loop")
                return False
        
        return True


class FeedbackProcessor(LoggerMixin):
    """Processes agent feedback to generate new tasks"""
    
    def __init__(self):
        super().__init__()
        self.processors: Dict[DiscoveryType, Callable] = {
            DiscoveryType.API_READY: self._process_api_ready,
            DiscoveryType.TEST_FAILING: self._process_test_failing,
            DiscoveryType.BUG_FOUND: self._process_bug_found,
            DiscoveryType.IMPLEMENTATION_COMPLETE: self._process_implementation_complete,
            DiscoveryType.DEPENDENCY_MISSING: self._process_dependency_missing,
            DiscoveryType.SECURITY_ISSUE: self._process_security_issue,
            DiscoveryType.INTEGRATION_READY: self._process_integration_ready,
        }
    
    def _create_task(self, task_id: str, command: str, task_type: TaskType, 
                     complexity: float = 0.5, duration: int = 30,
                     metadata: Optional[Dict] = None) -> Task:
        """Helper to create a task with proper intent and metadata"""
        intent = TaskIntent(
            task_type=task_type,
            complexity_score=complexity,
            estimated_duration=duration,
            requires_coordination=complexity > 0.7
        )
        
        coordination_context = {}
        if metadata:
            coordination_context['metadata'] = metadata
            
        return Task(
            id=task_id,
            command=command,
            intent=intent,
            coordination_context=coordination_context
        )
    
    async def process(self, discovery: AgentDiscovery) -> List[Task]:
        """Process a discovery and generate follow-up tasks"""
        processor = self.processors.get(discovery.discovery_type)
        if processor:
            return await processor(discovery)
        else:
            self.logger.warning(f"No processor for discovery type: {discovery.discovery_type}")
            return []
    
    async def _process_api_ready(self, discovery: AgentDiscovery) -> List[Task]:
        """Generate tasks when API endpoints are ready"""
        tasks = []
        endpoints = discovery.context.get("endpoints", [])
        base_path = discovery.context.get("base_path", "/api")
        
        # Generate integration test task
        test_intent = TaskIntent(
            task_type=TaskType.TEST,
            complexity_score=0.6,
            estimated_duration=30,
            requires_coordination=False
        )
        test_task = Task(
            id=f"test_api_{uuid.uuid4().hex[:8]}",
            command=f"Create comprehensive integration tests for the following API endpoints: {endpoints}. "
                   f"Test authentication, validation, error cases, and success scenarios.",
            intent=test_intent,
            coordination_context={
                "metadata": {
                    "generated_from": discovery.agent_id,
                    "phase": TaskPhase.VALIDATION,
                    "depends_on": {"api_endpoints"},
                    "produces": {"api_tests"}
                }
            }
        )
        test_task.agent_type_hint = "aider_testing"
        tasks.append(test_task)
        
        # Generate frontend integration task if applicable
        if discovery.context.get("needs_frontend", True):
            frontend_intent = TaskIntent(
                task_type=TaskType.IMPLEMENT,
                complexity_score=0.7,
                estimated_duration=45,
                requires_coordination=True
            )
            frontend_task = Task(
                id=f"frontend_api_{uuid.uuid4().hex[:8]}",
                command=f"Integrate the frontend with the new API endpoints at {base_path}: {endpoints}. "
                       f"Create service functions, update API client, and handle responses.",
                intent=frontend_intent,
                coordination_context={
                    "metadata": {
                        "generated_from": discovery.agent_id,
                        "phase": TaskPhase.INTEGRATION,
                        "depends_on": {"api_endpoints", "frontend_base"},
                        "produces": {"frontend_api_integration"}
                    }
                }
            )
            frontend_task.agent_type_hint = "aider_frontend"
            tasks.append(frontend_task)
        
        # Generate API documentation task
        doc_task = self._create_task(
            task_id=f"doc_api_{uuid.uuid4().hex[:8]}",
            command=f"Document the API endpoints {endpoints}: request/response formats, "
                   f"authentication requirements, error codes, and example usage.",
            task_type=TaskType.DOCUMENT,
            complexity=0.4,
            duration=20,
            metadata={
                "generated_from": discovery.agent_id,
                "phase": TaskPhase.REFINEMENT,
                "depends_on": {"api_endpoints"},
                "produces": {"api_documentation"}
            }
        )
        doc_task.agent_type_hint = "claude_code"
        tasks.append(doc_task)
        
        return tasks
    
    async def _process_test_failing(self, discovery: AgentDiscovery) -> List[Task]:
        """Generate tasks to fix failing tests"""
        tasks = []
        test_name = discovery.context.get("test_name")
        error = discovery.context.get("error")
        
        # Generate debugging task
        debug_task = Task(
            id=f"debug_test_{uuid.uuid4().hex[:8]}",
            command=f"Debug and fix the failing test '{test_name}'. Error: {error}. "
                   f"Analyze the root cause and implement a fix.",
            metadata={
                "generated_from": discovery.agent_id,
                "phase": TaskPhase.REFINEMENT,
                "priority": "high",
                "depends_on": set(),
                "produces": {"test_fix"}
            }
        )
        debug_task.agent_type_hint = "aider_backend"  # Or appropriate type
        tasks.append(debug_task)
        
        return tasks
    
    async def _process_bug_found(self, discovery: AgentDiscovery) -> List[Task]:
        """Generate tasks to fix bugs"""
        tasks = []
        bug_description = discovery.context.get("description")
        severity = discovery.severity
        
        # High priority fix for critical bugs
        if severity == "critical":
            fix_task = Task(
                id=f"fix_bug_{uuid.uuid4().hex[:8]}",
                command=f"CRITICAL BUG: {bug_description}. Fix immediately. "
                       f"Affected files: {discovery.affected_files}",
                metadata={
                    "generated_from": discovery.agent_id,
                    "phase": TaskPhase.REFINEMENT,
                    "priority": "critical",
                    "depends_on": set(),
                    "produces": {"bug_fix"}
                }
            )
            fix_task.agent_type_hint = "aider_backend"
            tasks.append(fix_task)
            
            # Also create a test to prevent regression
            test_task = Task(
                id=f"test_bugfix_{uuid.uuid4().hex[:8]}",
                command=f"Create a test to ensure the bug '{bug_description}' doesn't regress.",
                metadata={
                    "generated_from": discovery.agent_id,
                    "phase": TaskPhase.VALIDATION,
                    "depends_on": {"bug_fix"},
                    "produces": {"regression_test"}
                }
            )
            test_task.agent_type_hint = "aider_testing"
            tasks.append(test_task)
        
        return tasks
    
    async def _process_implementation_complete(self, discovery: AgentDiscovery) -> List[Task]:
        """Generate tasks when implementation is complete"""
        tasks = []
        component = discovery.context.get("component")
        
        # Generate test task
        test_task = Task(
            id=f"test_{component}_{uuid.uuid4().hex[:8]}",
            command=f"Create comprehensive unit and integration tests for {component}.",
            metadata={
                "generated_from": discovery.agent_id,
                "phase": TaskPhase.VALIDATION,
                "depends_on": {f"{component}_implementation"},
                "produces": {f"{component}_tests"}
            }
        )
        test_task.agent_type_hint = "aider_testing"
        tasks.append(test_task)
        
        return tasks
    
    async def _process_dependency_missing(self, discovery: AgentDiscovery) -> List[Task]:
        """Generate tasks for missing dependencies"""
        tasks = []
        missing = discovery.context.get("dependency")
        
        # Create task to implement missing dependency
        dep_task = Task(
            id=f"impl_dep_{uuid.uuid4().hex[:8]}",
            command=f"Implement missing dependency: {missing}. "
                   f"This is blocking other work.",
            metadata={
                "generated_from": discovery.agent_id,
                "phase": TaskPhase.IMPLEMENTATION,
                "priority": "high",
                "depends_on": set(),
                "produces": {missing}
            }
        )
        dep_task.agent_type_hint = "aider_backend"
        tasks.append(dep_task)
        
        return tasks
    
    async def _process_security_issue(self, discovery: AgentDiscovery) -> List[Task]:
        """Generate tasks for security issues"""
        tasks = []
        issue = discovery.context.get("issue")
        
        # Critical security fix
        security_task = Task(
            id=f"sec_fix_{uuid.uuid4().hex[:8]}",
            command=f"SECURITY ISSUE: {issue}. Fix immediately following security best practices.",
            metadata={
                "generated_from": discovery.agent_id,
                "phase": TaskPhase.REFINEMENT,
                "priority": "critical",
                "depends_on": set(),
                "produces": {"security_fix"}
            }
        )
        security_task.agent_type_hint = "claude_code"  # Claude for security analysis
        tasks.append(security_task)
        
        return tasks
    
    async def _process_integration_ready(self, discovery: AgentDiscovery) -> List[Task]:
        """Generate tasks when components are ready for integration"""
        tasks = []
        components = discovery.context.get("components", [])
        
        # Create integration task
        integration_task = Task(
            id=f"integrate_{uuid.uuid4().hex[:8]}",
            command=f"Integrate the following components: {components}. "
                   f"Ensure proper communication and error handling.",
            metadata={
                "generated_from": discovery.agent_id,
                "phase": TaskPhase.INTEGRATION,
                "depends_on": set(components),
                "produces": {"integrated_system"}
            }
        )
        integration_task.agent_type_hint = "aider_backend"
        tasks.append(integration_task)
        
        # Create integration test
        test_task = Task(
            id=f"test_integration_{uuid.uuid4().hex[:8]}",
            command=f"Create end-to-end tests for integrated components: {components}.",
            metadata={
                "generated_from": discovery.agent_id,
                "phase": TaskPhase.VALIDATION,
                "depends_on": {"integrated_system"},
                "produces": {"integration_tests"}
            }
        )
        test_task.agent_type_hint = "aider_testing"
        tasks.append(test_task)
        
        return tasks


class EnhancedFeedbackProcessor(FeedbackProcessor):
    """Enhanced feedback processor with verification-specific handlers"""
    
    def __init__(self, verification_controller: VerificationLoopController):
        super().__init__()
        self.verification_controller = verification_controller
        
        # Add verification-specific processors
        self.processors[DiscoveryType.TEST_FAILING] = self._process_test_failing_enhanced
        self.processors[DiscoveryType.TEST_PASSING] = self._process_test_passing
        
    async def _process_test_failing_enhanced(self, discovery: AgentDiscovery) -> List[Task]:
        """Enhanced test failure processing with verification context"""
        tasks = await self._process_test_failing(discovery)
        
        # Add verification metadata to tasks
        for task in tasks:
            task.metadata = task.metadata or {}
            task.metadata["verification_iteration"] = len(self.verification_controller.quality_history)
            task.metadata["quality_score"] = self.verification_controller.quality_history[-1].overall_quality if self.verification_controller.quality_history else 0.0
            
        return tasks
    
    async def _process_test_passing(self, discovery: AgentDiscovery) -> List[Task]:
        """Process test passing discoveries"""
        # Log success but don't generate new tasks
        self.logger.info(f"Test passing reported by {discovery.agent_id}: {discovery.context}")
        return []


class ProgressiveTaskGenerator(LoggerMixin):
    """Generates tasks progressively based on system state"""
    
    def __init__(self):
        super().__init__()
        self.phase_templates = self._initialize_phase_templates()
    
    def _create_task(self, task_id: str, command: str, task_type: TaskType, 
                     complexity: float = 0.5, duration: int = 30,
                     metadata: Optional[Dict] = None) -> Task:
        """Helper to create a task with proper intent and metadata"""
        intent = TaskIntent(
            task_type=task_type,
            complexity_score=complexity,
            estimated_duration=duration,
            requires_coordination=complexity > 0.7
        )
        
        coordination_context = {}
        if metadata:
            coordination_context['metadata'] = metadata
            
        return Task(
            id=task_id,
            command=command,
            intent=intent,
            coordination_context=coordination_context
        )
    
    def _initialize_phase_templates(self) -> Dict[TaskPhase, List[Dict]]:
        """Initialize task templates for each phase"""
        return {
            TaskPhase.EXPLORATION: [
                {
                    "pattern": "analyze_codebase",
                    "command": "Analyze the existing codebase structure and identify integration points for {objective}",
                    "produces": {"codebase_analysis"}
                }
            ],
            TaskPhase.DESIGN: [
                {
                    "pattern": "design_architecture",
                    "command": "Design the architecture for {objective}. Create component diagrams and API contracts.",
                    "produces": {"architecture_design", "api_contracts"}
                }
            ],
            TaskPhase.IMPLEMENTATION: [
                {
                    "pattern": "implement_backend",
                    "command": "Implement backend components for {objective} based on the architecture design.",
                    "depends_on": {"architecture_design"},
                    "produces": {"backend_implementation"}
                },
                {
                    "pattern": "implement_frontend",
                    "command": "Implement frontend components for {objective} based on the API contracts.",
                    "depends_on": {"api_contracts"},
                    "produces": {"frontend_implementation"}
                }
            ],
            TaskPhase.VALIDATION: [
                {
                    "pattern": "create_tests",
                    "command": "Create comprehensive test suite for {component}.",
                    "depends_on": {"{component}_implementation"},
                    "produces": {"{component}_tests"}
                }
            ]
        }
    
    async def generate_initial_tasks(self, objective: str, phase: TaskPhase) -> List[Task]:
        """Generate initial tasks for a given phase"""
        tasks = []
        templates = self.phase_templates.get(phase, [])
        
        for template in templates:
            # Determine task type from pattern
            task_type = TaskType.IMPLEMENT
            if "test" in template["pattern"]:
                task_type = TaskType.TEST
            elif "design" in template["pattern"]:
                task_type = TaskType.REFACTOR
            elif "analyze" in template["pattern"]:
                task_type = TaskType.EXPLAIN
            
            task = self._create_task(
                task_id=f"{template['pattern']}_{uuid.uuid4().hex[:8]}",
                command=template["command"].format(objective=objective),
                task_type=task_type,
                metadata={
                    "phase": phase,
                    "pattern": template["pattern"],
                    "depends_on": template.get("depends_on", set()),
                    "produces": template.get("produces", set())
                }
            )
            
            # Set agent hint based on pattern
            if "backend" in template["pattern"]:
                task.agent_type_hint = "aider_backend"
            elif "frontend" in template["pattern"]:
                task.agent_type_hint = "aider_frontend"
            elif "test" in template["pattern"]:
                task.agent_type_hint = "aider_testing"
            else:
                task.agent_type_hint = "claude_code"
            
            tasks.append(task)
        
        return tasks


class IntelligentCoordinator(LoggerMixin):
    """Coordinates agents with intelligent feedback loops and progressive execution"""
    
    def __init__(self, agent_registry, shared_memory, workspace_path: Optional[Path] = None, enable_verification: bool = False):
        super().__init__()
        self.agent_registry = agent_registry
        self.shared_memory = shared_memory
        self.workspace_path = workspace_path
        self.enable_verification = enable_verification
        self.dependency_graph = DynamicDependencyGraph()
        
        # Initialize verification if enabled
        if enable_verification and workspace_path:
            self.verification_controller = VerificationLoopController(workspace_path)
            self.feedback_processor = EnhancedFeedbackProcessor(self.verification_controller)
            self.verification_phase = None
            self.fix_iteration = 0
            self.phase_verifications: Dict[TaskPhase, List[QualityMetrics]] = defaultdict(list)
        else:
            self.feedback_processor = FeedbackProcessor()
            self.verification_controller = None
            
        self.task_generator = ProgressiveTaskGenerator()
        self.active_tasks: Dict[str, Task] = {}
        self.discoveries: List[AgentDiscovery] = []
        self.current_phase = TaskPhase.EXPLORATION
        
    async def execute_with_intelligence(self, objective: str) -> Dict[str, Any]:
        """Execute an objective with intelligent coordination"""
        self.logger.info(f"Starting intelligent execution for: {objective}")
        
        # Initialize execution context
        start_time = datetime.utcnow()
        completed_tasks = []
        all_discoveries = []
        phase_history = [self.current_phase]
        
        # Generate initial exploration tasks
        initial_tasks = await self.task_generator.generate_initial_tasks(
            objective, 
            TaskPhase.EXPLORATION
        )
        
        # Add to dependency graph
        for task in initial_tasks:
            self._add_task_to_graph(task)
        
        # Main execution loop
        iteration = 0
        max_iterations = 50  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get tasks ready for execution
            ready_task_ids = self.dependency_graph.get_ready_tasks()
            if not ready_task_ids and not self.active_tasks:
                # Check if we should move to next phase
                if await self._should_advance_phase():
                    self.current_phase = self._get_next_phase()
                    phase_history.append(self.current_phase)
                    
                    # Generate tasks for new phase
                    new_tasks = await self.task_generator.generate_initial_tasks(
                        objective,
                        self.current_phase
                    )
                    
                    for task in new_tasks:
                        self._add_task_to_graph(task)
                    
                    continue
                else:
                    # No more tasks and shouldn't advance - we're done
                    break
            
            # Execute ready tasks
            if ready_task_ids:
                tasks_to_execute = [self.active_tasks[tid] for tid in ready_task_ids 
                                   if tid in self.active_tasks]
                
                # Execute tasks and collect feedback
                results = await self._execute_tasks_with_feedback(tasks_to_execute)
                
                # Process results
                for task_id, (result, discoveries) in results.items():
                    # Mark task complete
                    task = self.active_tasks.get(task_id)
                    if task:
                        products = set()
                        if task.coordination_context and 'metadata' in task.coordination_context:
                            products = task.coordination_context['metadata'].get("produces", set())
                        elif hasattr(task, 'metadata') and task.metadata:
                            products = task.metadata.get("produces", set())
                        self.dependency_graph.mark_complete(task_id, products)
                    
                    completed_tasks.append((task_id, result))
                    all_discoveries.extend(discoveries)
                    
                    # Process discoveries to generate new tasks
                    for discovery in discoveries:
                        new_tasks = await self.feedback_processor.process(discovery)
                        for new_task in new_tasks:
                            self._add_task_to_graph(new_task)
            
            # Brief pause to prevent CPU spinning
            await asyncio.sleep(0.1)
        
        # Generate execution summary
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "objective": objective,
            "duration_seconds": duration,
            "iterations": iteration,
            "tasks_completed": len(completed_tasks),
            "discoveries_made": len(all_discoveries),
            "phases_executed": phase_history,
            "final_phase": self.current_phase.value,
            "completed_tasks": completed_tasks,
            "discoveries": all_discoveries
        }
    
    def _add_task_to_graph(self, task: Task):
        """Add a task to the dependency graph and active tasks"""
        self.active_tasks[task.id] = task
        
        depends_on = set()
        produces = set()
        
        if task.coordination_context and 'metadata' in task.coordination_context:
            metadata = task.coordination_context['metadata']
            depends_on = metadata.get("depends_on", set())
            produces = metadata.get("produces", set())
        elif hasattr(task, 'metadata') and task.metadata:
            depends_on = task.metadata.get("depends_on", set())
            produces = task.metadata.get("produces", set())
        
        self.dependency_graph.add_task(task, depends_on, produces)
    
    async def _execute_tasks_with_feedback(self, tasks: List[Task]) -> Dict[str, tuple]:
        """Execute tasks and collect agent feedback"""
        from agentic.models.agent import AgentConfig, AgentType
        
        results = {}
        
        # Execute tasks using the agent registry
        for task in tasks:
            self.logger.info(f"Executing task {task.id}: {task.command[:50]}...")
            
            try:
                # Determine agent type from task hint
                agent_type = AgentType.CLAUDE_CODE  # Default
                if hasattr(task, 'agent_type_hint'):
                    if task.agent_type_hint == 'aider_backend':
                        agent_type = AgentType.AIDER_BACKEND
                    elif task.agent_type_hint == 'aider_frontend':
                        agent_type = AgentType.AIDER_FRONTEND
                    elif task.agent_type_hint == 'aider_testing':
                        agent_type = AgentType.AIDER_TESTING
                    elif task.agent_type_hint == 'claude_code':
                        agent_type = AgentType.CLAUDE_CODE
                
                # Create agent config
                config = AgentConfig(
                    agent_type=agent_type,
                    name=f"intelligent_{agent_type.value}_{task.id[:8]}",
                    workspace_path=self.agent_registry.workspace_path,
                    focus_areas=task.affected_areas if task.affected_areas else ["general"],
                    ai_model_config={"model": "sonnet" if agent_type == AgentType.CLAUDE_CODE else "gemini/gemini-2.5-pro-preview-06-05"}
                )
                
                # Get or spawn agent
                session = await self.agent_registry.get_or_spawn_agent(config)
                
                if session.status != "active":
                    # Failed to spawn agent
                    result = TaskResult(
                        task_id=task.id,
                        agent_id="none",
                        status="failed",
                        output="",
                        error=f"Failed to spawn agent: {session.error}"
                    )
                    results[task.id] = (result, [])
                    continue
                
                # Get agent instance
                agent_instance = self.agent_registry.get_agent_by_id(session.id)
                if not agent_instance:
                    result = TaskResult(
                        task_id=task.id,
                        agent_id=session.id,
                        status="failed",
                        output="",
                        error="Agent instance not found"
                    )
                    results[task.id] = (result, [])
                    continue
                
                # Set up discovery callback to collect discoveries
                collected_discoveries = []
                
                def discovery_callback(discovery):
                    # Convert to AgentDiscovery format
                    agent_discovery = AgentDiscovery(
                        agent_id=session.id,
                        discovery_type=self._map_from_agent_discovery(discovery.type),
                        severity=discovery.severity,
                        context=discovery.context,
                        suggestions=[discovery.suggested_action] if discovery.suggested_action else [],
                        affected_files=[discovery.file_path] if discovery.file_path else [],
                        timestamp=discovery.timestamp
                    )
                    collected_discoveries.append(agent_discovery)
                
                agent_instance.set_discovery_callback(discovery_callback)
                
                # Execute task
                result = await agent_instance.execute_task(task)
                
                # Collect results
                results[task.id] = (result, collected_discoveries)
                
            except Exception as e:
                self.logger.error(f"Task {task.id} execution failed: {e}")
                result = TaskResult(
                    task_id=task.id,
                    agent_id="error",
                    status="failed",
                    output="",
                    error=str(e)
                )
                results[task.id] = (result, [])
        
        return results
    
    async def _should_advance_phase(self) -> bool:
        """Determine if we should advance to the next phase"""
        # Simple heuristic - in reality would be more sophisticated
        completed_products = self.dependency_graph.completed_products
        
        if self.current_phase == TaskPhase.EXPLORATION:
            return "codebase_analysis" in completed_products
        elif self.current_phase == TaskPhase.DESIGN:
            return "architecture_design" in completed_products
        elif self.current_phase == TaskPhase.IMPLEMENTATION:
            return "backend_implementation" in completed_products
        
        return False
    
    def _get_next_phase(self) -> TaskPhase:
        """Get the next logical phase"""
        phase_order = [
            TaskPhase.EXPLORATION,
            TaskPhase.DESIGN,
            TaskPhase.IMPLEMENTATION,
            TaskPhase.INTEGRATION,
            TaskPhase.VALIDATION,
            TaskPhase.REFINEMENT
        ]
        
        try:
            current_idx = phase_order.index(self.current_phase)
            if current_idx < len(phase_order) - 1:
                return phase_order[current_idx + 1]
        except ValueError:
            pass
        
        return self.current_phase
    
    async def report_discovery(self, discovery: AgentDiscovery):
        """Public method for agents to report discoveries"""
        self.discoveries.append(discovery)
        
        # Generate follow-up tasks immediately
        new_tasks = await self.feedback_processor.process(discovery)
        for task in new_tasks:
            self._add_task_to_graph(task)
        
        self.logger.info(f"Processed discovery from {discovery.agent_id}: "
                        f"{discovery.discovery_type} -> {len(new_tasks)} new tasks")
    
    def _map_from_agent_discovery(self, agent_discovery_type) -> DiscoveryType:
        """Map from agent discovery type to intelligent coordinator discovery type"""
        # Import here to avoid circular imports
        from agentic.models.agent import DiscoveryType as AgentDiscoveryType
        
        mapping = {
            AgentDiscoveryType.API_READY: DiscoveryType.API_READY,
            AgentDiscoveryType.TEST_NEEDED: DiscoveryType.DOCUMENTATION_NEEDED,
            AgentDiscoveryType.BUG_FOUND: DiscoveryType.BUG_FOUND,
            AgentDiscoveryType.SECURITY_ISSUE: DiscoveryType.SECURITY_ISSUE,
            AgentDiscoveryType.PERFORMANCE_ISSUE: DiscoveryType.PERFORMANCE_ISSUE,
            AgentDiscoveryType.REFACTOR_OPPORTUNITY: DiscoveryType.CODE_SMELL,
            AgentDiscoveryType.DEPENDENCY_UPDATE: DiscoveryType.DEPENDENCY_MISSING,
            AgentDiscoveryType.DOCUMENTATION_NEEDED: DiscoveryType.DOCUMENTATION_NEEDED,
            AgentDiscoveryType.CONFIG_ISSUE: DiscoveryType.DEPENDENCY_MISSING,
            AgentDiscoveryType.INTEGRATION_POINT: DiscoveryType.INTEGRATION_READY,
        }
        return mapping.get(agent_discovery_type, DiscoveryType.CODE_SMELL)
    
    # Verification-specific methods (only used when verification is enabled)
    async def _should_verify(self) -> bool:
        """Determine if we should run verification"""
        if not self.enable_verification or not self.verification_controller:
            return False
            
        # Run verification at key phase transitions
        if self.current_phase in [TaskPhase.IMPLEMENTATION, TaskPhase.INTEGRATION, TaskPhase.VALIDATION]:
            # Check if enough tasks completed since last verification
            completed_since_last = len(self.dependency_graph.get_completed_tasks())
            return completed_since_last >= 3
            
        return False
    
    async def _run_verification_cycle(self) -> Tuple[bool, QualityMetrics, List[Task]]:
        """Run a verification cycle"""
        if not self.verification_controller:
            raise RuntimeError("Verification not enabled")
            
        return await self.verification_controller.run_verification_loop(self.current_phase)
    
    def _get_prioritized_ready_tasks(self) -> List[str]:
        """Get ready tasks prioritized by fix tasks first"""
        ready_tasks = self.dependency_graph.get_ready_tasks()
        
        if not self.enable_verification:
            return ready_tasks
            
        # Prioritize fix tasks
        fix_tasks = []
        normal_tasks = []
        
        for task_id in ready_tasks:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.metadata and task.metadata.get("is_fix_task"):
                    fix_tasks.append(task_id)
                else:
                    normal_tasks.append(task_id)
            else:
                normal_tasks.append(task_id)
        
        return fix_tasks + normal_tasks
    
    async def execute_with_verification(self, objective: str) -> Dict[str, Any]:
        """Execute with continuous verification and quality improvement"""
        if not self.enable_verification:
            # Fall back to normal execution
            return await self.execute_with_intelligence(objective)
            
        self.logger.info(f"Starting intelligent execution with verification for: {objective}")
        
        # Initialize execution context
        start_time = datetime.utcnow()
        completed_tasks = []
        all_discoveries = []
        phase_history = [self.current_phase]
        verification_results = []
        
        # Generate initial exploration tasks
        initial_tasks = await self.task_generator.generate_initial_tasks(
            objective, 
            TaskPhase.EXPLORATION
        )
        
        # Add to dependency graph
        for task in initial_tasks:
            self._add_task_to_graph(task)
        
        # Main execution loop with verification
        iteration = 0
        max_iterations = 50
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if we should run verification
            if await self._should_verify():
                success, metrics, fix_tasks = await self._run_verification_cycle()
                verification_results.append({
                    "iteration": iteration,
                    "phase": self.current_phase.value,
                    "metrics": metrics,
                    "success": success
                })
                
                # Add quality metrics to phase history
                self.phase_verifications[self.current_phase].append(metrics)
                
                if not success and fix_tasks:
                    # Add fix tasks with high priority
                    for fix_task in fix_tasks:
                        fix_task.metadata = fix_task.metadata or {}
                        fix_task.metadata["priority"] = "high"
                        fix_task.metadata["is_fix_task"] = True
                        fix_task.metadata["fix_iteration"] = self.fix_iteration
                        self._add_task_to_graph(fix_task)
                    
                    self.fix_iteration += 1
                    
                    # Check if we should continue fixing
                    if not self.verification_controller.should_continue_fixing():
                        self.logger.warning("Stopping fix iterations - max attempts or no improvement")
                        # Move to next phase anyway
                        if await self._should_advance_phase(force=True):
                            self.current_phase = self._get_next_phase()
                            phase_history.append(self.current_phase)
                            self.fix_iteration = 0
                else:
                    # Verification passed or no fixes needed
                    self.fix_iteration = 0
            
            # Get tasks ready for execution (prioritized if verification enabled)
            ready_task_ids = self._get_prioritized_ready_tasks()
            
            if not ready_task_ids and not self.active_tasks:
                # Check if we should move to next phase
                if await self._should_advance_phase():
                    self.current_phase = self._get_next_phase()
                    phase_history.append(self.current_phase)
                    
                    # Generate tasks for new phase
                    new_tasks = await self.task_generator.generate_initial_tasks(
                        objective,
                        self.current_phase
                    )
                    
                    for task in new_tasks:
                        self._add_task_to_graph(task)
                    
                    continue
                else:
                    # No more tasks - final verification if enabled
                    if self.current_phase == TaskPhase.REFINEMENT:
                        success, final_metrics, _ = await self._run_verification_cycle()
                        verification_results.append({
                            "iteration": iteration,
                            "phase": "final",
                            "metrics": final_metrics,
                            "success": success
                        })
                        self.logger.info(f"Final verification: {success}")
                    break
            
            # Execute ready tasks (same as normal execution)
            results = await self._execute_tasks(ready_task_ids)
            
            # Process results
            for task_id, (result, discoveries) in results.items():
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                    self.dependency_graph.mark_completed(task_id, result.success)
                    
                    completed_tasks.append((task_id, result))
                    all_discoveries.extend(discoveries)
                    
                    # Process discoveries to generate new tasks
                    for discovery in discoveries:
                        new_tasks = await self.feedback_processor.process(discovery)
                        for new_task in new_tasks:
                            self._add_task_to_graph(new_task)
            
            # Brief pause to prevent CPU spinning
            await asyncio.sleep(0.1)
        
        # Generate execution summary with verification info
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "objective": objective,
            "duration_seconds": duration,
            "iterations": iteration,
            "completed_tasks": len(completed_tasks),
            "total_discoveries": len(all_discoveries),
            "phase_progression": [p.value for p in phase_history],
            "final_phase": self.current_phase.value,
            "success": len(self.dependency_graph.failed_tasks) == 0,
            "verification_enabled": True,
            "verification_results": verification_results,
            "final_quality": verification_results[-1]["metrics"].overall_quality if verification_results else None
        }
    
    async def _should_advance_phase(self, force: bool = False) -> bool:
        """Enhanced phase advancement with force option"""
        if force:
            return True
        return await self._should_advance_phase()