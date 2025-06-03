# Phase 2: Multi-Agent Coordination (Weeks 3-4)

> **Implement multiple specialized Aider agents with intelligent command routing and Claude Code integration**

## 🎯 Objectives
- Implement multiple specialized Aider agents (frontend, backend, testing)
- Add intelligent command routing based on intent analysis
- Integrate Claude Code for complex reasoning tasks
- Create basic agent coordination mechanisms
- Establish inter-agent communication patterns

## 📦 Deliverables

### 2.1 Multi-Agent Architecture
**Goal**: Support multiple specialized agents working simultaneously

**Agent Types to Implement**:
- **Frontend Agent**: React/Vue components, styling, UI logic
- **Backend Agent**: APIs, databases, server logic  
- **Testing Agent**: Unit tests, integration tests, test utilities
- **Reasoning Agent**: Claude Code for debugging and analysis

**Technical Requirements**:
- [ ] Agent registry for managing multiple agent instances
- [ ] Agent lifecycle management (spawn, monitor, terminate)
- [ ] Agent specialization configuration
- [ ] Process isolation and resource management
- [ ] Agent health monitoring and recovery

**Agent Registry Implementation**:
```python
class AgentRegistry:
    """Manages multiple agent instances"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.active_sessions: Dict[str, AgentSession] = {}
    
    def register_agent_type(self, agent_type: str, agent_class: Type[Agent]):
        """Register a new agent type"""
        self.agent_types[agent_type] = agent_class
    
    async def spawn_agent(self, config: AgentConfig) -> AgentSession:
        """Spawn a new agent instance"""
        agent_class = self.agent_types[config.agent_type]
        agent = agent_class(config, self.shared_memory)
        
        session = AgentSession(
            agent_config=config,
            workspace=config.workspace_path,
            status="starting"
        )
        
        success = await agent.start()
        if success:
            session.status = "active"
            self.active_sessions[session.id] = session
            self.agents[session.id] = agent
        else:
            session.status = "failed"
        
        return session
    
    async def terminate_agent(self, session_id: str) -> bool:
        """Terminate an agent session"""
        if session_id in self.agents:
            agent = self.agents[session_id]
            await agent.stop()
            del self.agents[session_id]
            del self.active_sessions[session_id]
            return True
        return False
    
    def get_agents_by_capability(self, capability: str) -> List[AgentSession]:
        """Find agents with specific capability"""
        return [
            session for session in self.active_sessions.values()
            if capability in session.agent_config.focus_areas
        ]
```

### 2.2 Command Router
**Goal**: Intelligently route commands to appropriate agents based on analysis

**Routing Logic**:
- [ ] Intent classification (implement, debug, refactor, explain, test)
- [ ] File path analysis for agent selection
- [ ] Command complexity scoring
- [ ] Agent capability matching
- [ ] Load balancing for multiple capable agents

**Intent Classification**:
```python
class IntentClassifier:
    """Analyzes user commands to determine intent and routing"""
    
    def __init__(self):
        self.debug_keywords = ["debug", "fix", "error", "bug", "issue", "problem"]
        self.implement_keywords = ["add", "create", "build", "implement", "make"]
        self.refactor_keywords = ["refactor", "reorganize", "restructure", "cleanup"]
        self.explain_keywords = ["explain", "describe", "what", "how", "why"]
        self.test_keywords = ["test", "spec", "unit test", "integration test"]
    
    async def analyze_intent(self, command: str) -> TaskIntent:
        """Analyze command to determine intent"""
        command_lower = command.lower()
        
        # Determine task type
        task_type = TaskType.IMPLEMENT  # default
        if any(keyword in command_lower for keyword in self.debug_keywords):
            task_type = TaskType.DEBUG
        elif any(keyword in command_lower for keyword in self.explain_keywords):
            task_type = TaskType.EXPLAIN
        elif any(keyword in command_lower for keyword in self.test_keywords):
            task_type = TaskType.TEST
        elif any(keyword in command_lower for keyword in self.refactor_keywords):
            task_type = TaskType.REFACTOR
        
        # Analyze complexity
        complexity_score = self._calculate_complexity(command)
        
        # Determine affected areas
        affected_areas = self._identify_affected_areas(command)
        
        # Check if reasoning is required
        requires_reasoning = task_type in [TaskType.DEBUG, TaskType.EXPLAIN] or complexity_score > 0.7
        
        # Check if coordination is required
        requires_coordination = len(affected_areas) > 1 or any(
            keyword in command_lower for keyword in ["across", "throughout", "all", "entire"]
        )
        
        return TaskIntent(
            task_type=task_type,
            complexity_score=complexity_score,
            estimated_duration=self._estimate_duration(command, complexity_score),
            affected_areas=affected_areas,
            requires_reasoning=requires_reasoning,
            requires_coordination=requires_coordination
        )
    
    def _identify_affected_areas(self, command: str) -> List[str]:
        """Identify which areas of codebase are affected"""
        areas = []
        command_lower = command.lower()
        
        # Frontend indicators
        if any(keyword in command_lower for keyword in [
            "component", "ui", "frontend", "react", "vue", "css", "style", "page", "form"
        ]):
            areas.append("frontend")
        
        # Backend indicators  
        if any(keyword in command_lower for keyword in [
            "api", "backend", "server", "database", "endpoint", "service", "auth"
        ]):
            areas.append("backend")
        
        # Testing indicators
        if any(keyword in command_lower for keyword in [
            "test", "spec", "unit", "integration", "e2e"
        ]):
            areas.append("testing")
        
        return areas if areas else ["general"]
```

**Command Router Implementation**:
```python
class CommandRouter:
    """Routes commands to appropriate agents"""
    
    def __init__(self, agent_registry: AgentRegistry, project_structure: ProjectStructure):
        self.agent_registry = agent_registry
        self.project_structure = project_structure
        self.intent_classifier = IntentClassifier()
    
    async def route_command(self, command: str) -> ExecutionPlan:
        """Create execution plan for command"""
        intent = await self.intent_classifier.analyze_intent(command)
        
        # Select appropriate agents
        selected_agents = self._select_agents(intent)
        
        # Create tasks
        if intent.requires_coordination:
            tasks = await self._create_coordinated_tasks(command, intent, selected_agents)
        else:
            tasks = await self._create_simple_tasks(command, intent, selected_agents)
        
        # Plan execution order
        execution_order = self._plan_execution_order(tasks, intent)
        
        return ExecutionPlan(
            command=command,
            tasks=tasks,
            execution_order=execution_order,
            estimated_duration=sum(task.intent.estimated_duration for task in tasks)
        )
    
    def _select_agents(self, intent: TaskIntent) -> List[AgentSession]:
        """Select best agents for the task"""
        selected = []
        
        if intent.requires_reasoning:
            # Prefer Claude Code for reasoning tasks
            reasoning_agents = self.agent_registry.get_agents_by_capability("reasoning")
            if reasoning_agents:
                selected.append(reasoning_agents[0])
        
        # Select agents by affected areas
        for area in intent.affected_areas:
            area_agents = self.agent_registry.get_agents_by_capability(area)
            if area_agents:
                # Pick least busy agent
                selected.append(min(area_agents, key=lambda a: len(a.current_tasks)))
        
        return selected
```

### 2.3 Claude Code Integration
**Goal**: Integrate Claude Code for complex reasoning and debugging tasks

**Integration Features**:
- [ ] Claude Code process management
- [ ] Command routing for reasoning tasks  
- [ ] Output parsing and formatting
- [ ] Error handling and recovery
- [ ] Thinking mode utilization ("think", "ultrathink")

**Claude Code Agent**:
```python
class ClaudeCodeAgent(Agent):
    """Claude Code agent for reasoning and debugging"""
    
    def __init__(self, config: AgentConfig, shared_memory: SharedMemory):
        super().__init__(config, shared_memory)
        self.process = None
        self.session_file = None
    
    async def start(self) -> bool:
        """Start Claude Code session"""
        # Create session directory
        session_dir = self.config.workspace_path / ".claude-code"
        session_dir.mkdir(exist_ok=True)
        
        cmd = [
            "claude",
            "--headless",
            f"--workspace={self.config.workspace_path}"
        ]
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.config.workspace_path
        )
        
        return self.process is not None
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute reasoning task via Claude Code"""
        command = task.command
        
        # Add thinking directive for complex tasks
        if task.intent.complexity_score > 0.7:
            command = f"ultrathink {command}"
        elif task.intent.complexity_score > 0.5:
            command = f"think hard {command}"
        
        # Add project context
        command_with_context = self._add_project_context(command)
        
        # Send command
        self.process.stdin.write(f"{command_with_context}\n".encode())
        await self.process.stdin.drain()
        
        # Read response
        output = await self._read_claude_response()
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.session.id,
            status="completed",
            output=output,
            execution_time=time.time() - task.started_at.timestamp()
        )
    
    def _add_project_context(self, command: str) -> str:
        """Add relevant project context to command"""
        context_parts = []
        
        # Add project structure overview
        tech_stack = self.shared_memory.get_project_structure().tech_stack
        context_parts.append(f"Project uses: {', '.join(tech_stack.languages + tech_stack.frameworks)}")
        
        # Add recent changes context
        recent_changes = self.shared_memory.get_recent_changes(limit=5)
        if recent_changes:
            context_parts.append("Recent changes:")
            for change in recent_changes:
                context_parts.append(f"- {change['description']}")
        
        if context_parts:
            context = "\n".join(context_parts)
            return f"Context:\n{context}\n\nTask: {command}"
        
        return command
```

### 2.4 Agent Coordination
**Goal**: Coordinate multiple agents working on related tasks

**Coordination Features**:
- [ ] Shared workspace management
- [ ] Basic conflict detection (same file edits)
- [ ] Sequential vs parallel task execution
- [ ] Progress synchronization and reporting
- [ ] Rollback coordination on failures

**Coordination Engine**:
```python
class CoordinationEngine:
    """Coordinates multiple agents working together"""
    
    def __init__(self, agent_registry: AgentRegistry, shared_memory: SharedMemory):
        self.agent_registry = agent_registry
        self.shared_memory = shared_memory
        self.active_executions: Dict[str, ExecutionContext] = {}
    
    async def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute a multi-agent plan"""
        execution_id = str(uuid.uuid4())
        context = ExecutionContext(
            execution_id=execution_id,
            plan=plan,
            status="running",
            started_at=datetime.utcnow()
        )
        
        self.active_executions[execution_id] = context
        
        try:
            # Execute in planned order
            for parallel_group in plan.execution_order:
                await self._execute_parallel_group(parallel_group, context)
            
            context.status = "completed"
            
        except Exception as e:
            context.status = "failed"
            context.error = str(e)
            await self._rollback_execution(context)
        
        finally:
            del self.active_executions[execution_id]
        
        return ExecutionResult(
            execution_id=execution_id,
            status=context.status,
            completed_tasks=context.completed_tasks,
            failed_tasks=context.failed_tasks,
            total_duration=context.total_duration
        )
    
    async def _execute_parallel_group(self, task_ids: List[str], context: ExecutionContext):
        """Execute a group of tasks in parallel"""
        tasks = [context.plan.get_task(task_id) for task_id in task_ids]
        
        # Check for conflicts before starting
        conflicts = self._detect_conflicts(tasks)
        if conflicts:
            await self._resolve_conflicts(conflicts, tasks)
        
        # Execute tasks in parallel
        results = await asyncio.gather(
            *[self._execute_single_task(task, context) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                context.failed_tasks.append(task.id)
                raise result
            else:
                context.completed_tasks.append(task.id)
    
    def _detect_conflicts(self, tasks: List[Task]) -> List[ConflictDetection]:
        """Detect potential conflicts between tasks"""
        conflicts = []
        
        # Check for file conflicts
        file_map = {}
        for task in tasks:
            estimated_files = self._estimate_affected_files(task)
            for file_path in estimated_files:
                if file_path in file_map:
                    conflicts.append(ConflictDetection(
                        conflict_type="file_conflict",
                        affected_files=[file_path],
                        conflicting_agents=[file_map[file_path], task.assigned_agents[0]],
                        severity="medium",
                        auto_resolvable=False
                    ))
                else:
                    file_map[file_path] = task.assigned_agents[0]
        
        return conflicts
```

## 🛠 Technical Implementation

### Enhanced Agent Base Class
```python
class Agent(ABC):
    """Enhanced base agent with coordination support"""
    
    def __init__(self, config: AgentConfig, shared_memory: SharedMemory):
        self.config = config
        self.shared_memory = shared_memory
        self.session: Optional[AgentSession] = None
        self.current_tasks: List[Task] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the agent session"""
        pass
    
    @abstractmethod  
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a specific task"""
        pass
    
    async def send_message(self, recipient_id: str, message: Dict):
        """Send message to another agent"""
        self.shared_memory.send_inter_agent_message(
            sender_id=self.session.id,
            recipient_id=recipient_id,
            message=message
        )
    
    async def receive_messages(self) -> List[Dict]:
        """Receive messages from other agents"""
        return await self.shared_memory.get_messages_for_agent(self.session.id)
    
    async def notify_progress(self, task_id: str, progress: float, message: str):
        """Notify coordination engine of progress"""
        await self.shared_memory.update_task_progress(task_id, progress, message)
```

### Specialized Aider Agents
```python
class AiderFrontendAgent(AiderAgent):
    """Specialized Aider agent for frontend development"""
    
    def get_capabilities(self) -> AgentCapability:
        return AgentCapability(
            agent_type=AgentType.AIDER_FRONTEND,
            specializations=["react", "vue", "css", "javascript", "typescript"],
            supported_languages=["javascript", "typescript", "css", "html"],
            reasoning_capability=False
        )
    
    def _get_focus_files(self) -> List[Path]:
        """Get frontend-specific files"""
        patterns = [
            "src/components/**/*",
            "src/pages/**/*", 
            "src/styles/**/*",
            "public/**/*",
            "*.css",
            "*.scss",
            "*.jsx",
            "*.tsx"
        ]
        return self._glob_patterns(patterns)

class AiderBackendAgent(AiderAgent):
    """Specialized Aider agent for backend development"""
    
    def get_capabilities(self) -> AgentCapability:
        return AgentCapability(
            agent_type=AgentType.AIDER_BACKEND,
            specializations=["api", "database", "server", "auth"],
            supported_languages=["python", "javascript", "go", "rust", "java"],
            reasoning_capability=False
        )
    
    def _get_focus_files(self) -> List[Path]:
        """Get backend-specific files"""
        patterns = [
            "src/api/**/*",
            "src/models/**/*",
            "src/services/**/*", 
            "src/controllers/**/*",
            "*.py",
            "*.js",
            "*.go",
            "*.rs"
        ]
        return self._glob_patterns(patterns)
```

## 📊 Success Criteria

### Functional Requirements
- [ ] **Multiple Agents**: Successfully spawn and manage 3+ agents simultaneously
- [ ] **Command Routing**: 90% of commands routed to appropriate agents
- [ ] **Claude Code Integration**: Complex debugging tasks use Claude Code
- [ ] **Coordination**: Agents work together without major conflicts
- [ ] **Progress Tracking**: Real-time visibility into agent activities

### Performance Requirements
- [ ] **Agent Startup**: Agents start within 30 seconds
- [ ] **Command Processing**: Intent analysis completes in <5 seconds
- [ ] **Parallel Execution**: Multiple agents work simultaneously
- [ ] **Resource Usage**: Reasonable memory and CPU usage
- [ ] **Error Recovery**: Graceful handling of agent failures

### Quality Requirements
- [ ] **Test Coverage**: >85% coverage including integration tests
- [ ] **Error Handling**: Comprehensive error scenarios covered
- [ ] **Documentation**: All new APIs documented
- [ ] **Type Safety**: Full type hints and mypy compliance
- [ ] **Logging**: Detailed logging for debugging and monitoring

## 🧪 Test Cases

### Multi-Agent Coordination Tests
```python
@pytest.mark.asyncio
async def test_spawn_multiple_agents():
    """Test spawning multiple specialized agents"""
    registry = AgentRegistry()
    
    # Spawn frontend agent
    frontend_config = AgentConfig(
        agent_type=AgentType.AIDER_FRONTEND,
        name="frontend",
        workspace_path=Path("."),
        focus_areas=["components", "styling"]
    )
    frontend_session = await registry.spawn_agent(frontend_config)
    
    # Spawn backend agent
    backend_config = AgentConfig(
        agent_type=AgentType.AIDER_BACKEND,
        name="backend", 
        workspace_path=Path("."),
        focus_areas=["api", "database"]
    )
    backend_session = await registry.spawn_agent(backend_config)
    
    assert frontend_session.status == "active"
    assert backend_session.status == "active"
    assert len(registry.active_sessions) == 2

@pytest.mark.asyncio
async def test_command_routing():
    """Test intelligent command routing"""
    router = CommandRouter(mock_registry, mock_project_structure)
    
    # Test frontend routing
    plan = await router.route_command("fix the login form styling")
    assert any("frontend" in agent.focus_areas for agent in plan.selected_agents)
    
    # Test backend routing  
    plan = await router.route_command("add authentication to the API")
    assert any("backend" in agent.focus_areas for agent in plan.selected_agents)
    
    # Test coordination routing
    plan = await router.route_command("add user profiles across the entire app")
    assert len(plan.selected_agents) > 1

@pytest.mark.asyncio
async def test_claude_code_integration():
    """Test Claude Code for reasoning tasks"""
    claude_agent = ClaudeCodeAgent(claude_config, shared_memory)
    await claude_agent.start()
    
    debug_task = Task(
        command="debug the race condition in user authentication",
        intent=TaskIntent(
            task_type=TaskType.DEBUG,
            complexity_score=0.8,
            requires_reasoning=True
        )
    )
    
    result = await claude_agent.execute_task(debug_task)
    assert result.status == "completed"
    assert "race condition" in result.output.lower()
```

### Conflict Detection Tests
```python
def test_file_conflict_detection():
    """Test detection of file conflicts between agents"""
    coordination_engine = CoordinationEngine(registry, shared_memory)
    
    task1 = Task(
        command="modify login.js authentication logic",
        assigned_agents=["frontend-agent"]
    )
    task2 = Task(
        command="refactor login.js error handling", 
        assigned_agents=["backend-agent"]
    )
    
    conflicts = coordination_engine._detect_conflicts([task1, task2])
    
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "file_conflict"
    assert "login.js" in [str(f) for f in conflicts[0].affected_files]
```

## 🚀 Implementation Order

### Week 3: Multi-Agent Foundation
1. **Day 15-16**: Agent registry and lifecycle management
2. **Day 17-18**: Specialized Aider agents (frontend, backend, testing)
3. **Day 19**: Intent classification and command analysis
4. **Day 20-21**: Command router implementation

### Week 4: Advanced Coordination  
1. **Day 22-23**: Claude Code agent integration
2. **Day 24-25**: Coordination engine and conflict detection
3. **Day 26-27**: Inter-agent communication and progress tracking
4. **Day 28**: Integration testing and bug fixes

## 🎯 Phase 2 Demo Script

After completion, this workflow should work:

```bash
# Initialize with multiple agents
cd my-fullstack-app
agentic init
# Output: ✅ Configured frontend, backend, and testing agents

# Execute coordinated command  
agentic "add user profile feature with avatar upload"
# Output:
# 🤖 Analyzing command... requires frontend + backend coordination
# [Backend Agent]: Creating user profile API endpoints...
# [Frontend Agent]: Building profile UI components...
# [Testing Agent]: Writing profile feature tests...
# ✅ User profile feature implemented across stack

# Execute debugging command
agentic "debug the slow API response on user login"
# Output: 
# 🧠 Routing to reasoning agent for analysis...
# [Claude Code]: Analyzing authentication flow...
# [Claude Code]: Found N+1 query issue in user permissions...
# [Backend Agent]: Implementing fix with eager loading...
# ✅ Login performance improved by 80%

# Check agent status
agentic status
# Output:
# 🟢 Frontend Agent: Active (React specialist)
# 🟢 Backend Agent: Active (Node.js API specialist)  
# 🟢 Testing Agent: Active (Jest/Cypress specialist)
# 🧠 Reasoning Agent: Active (Claude Code)
```

## 🔍 Phase 2 Completion Checklist

**Before moving to Phase 3:**
- [ ] Multiple agents spawn and run simultaneously
- [ ] Commands route correctly based on intent analysis
- [ ] Claude Code handles reasoning tasks effectively
- [ ] Basic coordination prevents major conflicts
- [ ] Progress tracking shows real-time agent status
- [ ] Integration tests pass for multi-agent scenarios
- [ ] Performance acceptable with 3-4 active agents
- [ ] Error handling covers agent failures and recovery
- [ ] Documentation updated for new features

**Phase 2 establishes the foundation for true multi-agent AI development workflows.**