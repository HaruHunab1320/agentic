# Agentic Technical Architecture

> **Detailed technical specification for implementing the multi-agent AI development orchestrator**

This document provides the technical foundation for implementing Agentic, including system architecture, data models, APIs, and integration patterns.

---

## 🏗 System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Agentic CLI                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Command   │  │  Progress   │  │   Config    │        │
│  │   Parser    │  │  Display    │  │  Manager    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                Orchestrator Core                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Project   │  │   Command   │  │    Agent    │        │
│  │  Analyzer   │  │   Router    │  │  Registry   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Coordination│  │   Shared    │  │   Conflict  │        │
│  │   Engine    │  │   Memory    │  │  Resolver   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                   Agent Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Claude    │  │    Aider    │  │    Aider    │        │
│  │    Code     │  │   Backend   │  │  Frontend   │        │
│  │   Agent     │  │    Agent    │  │    Agent    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Aider    │  │   Custom    │  │   Plugin    │        │
│  │   Testing   │  │    Agent    │  │    Agent    │        │
│  │    Agent    │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                  Tool Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Aider    │  │   Claude    │  │     Git     │        │
│  │  Processes  │  │    Code     │  │  Operations │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    File     │  │   Process   │  │   External  │        │
│  │   System    │  │  Management │  │    APIs     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. CLI Interface
- **Command Parser**: Parse and validate user commands
- **Progress Display**: Real-time feedback using Rich library
- **Configuration Manager**: Handle user and project settings

#### 2. Orchestrator Core
- **Project Analyzer**: Understand codebase structure and dependencies
- **Command Router**: Intelligent routing to appropriate agents
- **Agent Registry**: Manage agent lifecycle and capabilities
- **Coordination Engine**: Synchronize agent activities
- **Shared Memory**: Cross-agent context and learning
- **Conflict Resolver**: Handle concurrent modifications

#### 3. Agent Layer
- **Base Agent**: Abstract interface for all agents
- **Claude Code Agent**: Deep reasoning and debugging specialist
- **Aider Agents**: Specialized for different domains (backend, frontend, testing)
- **Plugin Agents**: Community-contributed agents

#### 4. Tool Layer
- **Process Management**: Spawn and monitor external tools
- **File System Operations**: Safe file manipulation
- **Git Integration**: Version control operations
- **External APIs**: AI model APIs and other integrations

---

## 📊 Data Models

### Core Entities

```python
from enum import Enum
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path
import datetime

class TaskType(str, Enum):
    """Types of development tasks"""
    IMPLEMENT = "implement"
    DEBUG = "debug"
    REFACTOR = "refactor"
    EXPLAIN = "explain"
    TEST = "test"
    DOCUMENT = "document"

class AgentType(str, Enum):
    """Types of available agents"""
    CLAUDE_CODE = "claude_code"
    AIDER_BACKEND = "aider_backend"
    AIDER_FRONTEND = "aider_frontend"
    AIDER_TESTING = "aider_testing"
    AIDER_DEVOPS = "aider_devops"
    CUSTOM = "custom"

class TechStack(BaseModel):
    """Detected technology stack"""
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    testing_frameworks: List[str] = Field(default_factory=list)
    build_tools: List[str] = Field(default_factory=list)
    deployment_tools: List[str] = Field(default_factory=list)

class ProjectStructure(BaseModel):
    """Project structure analysis"""
    root_path: Path
    tech_stack: TechStack
    entry_points: List[Path] = Field(default_factory=list)
    config_files: List[Path] = Field(default_factory=list)
    source_directories: List[Path] = Field(default_factory=list)
    test_directories: List[Path] = Field(default_factory=list)
    documentation_files: List[Path] = Field(default_factory=list)
    dependency_files: List[Path] = Field(default_factory=list)

class DependencyGraph(BaseModel):
    """File and module dependencies"""
    nodes: Dict[str, Dict] = Field(default_factory=dict)
    edges: List[Dict[str, str]] = Field(default_factory=list)
    
    def get_dependents(self, file_path: str) -> List[str]:
        """Get files that depend on the given file"""
        return [edge["target"] for edge in self.edges if edge["source"] == file_path]
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """Get files that the given file depends on"""
        return [edge["source"] for edge in self.edges if edge["target"] == file_path]

class TaskIntent(BaseModel):
    """Analyzed intent of user command"""
    task_type: TaskType
    complexity_score: float = Field(ge=0.0, le=1.0)
    estimated_duration: int  # minutes
    affected_areas: List[str] = Field(default_factory=list)
    requires_reasoning: bool = False
    requires_coordination: bool = False

class AgentCapability(BaseModel):
    """Agent capabilities and specializations"""
    agent_type: AgentType
    specializations: List[str] = Field(default_factory=list)
    supported_languages: List[str] = Field(default_factory=list)
    max_context_tokens: int = 100000
    concurrent_tasks: int = 1
    reasoning_capability: bool = False

class AgentConfig(BaseModel):
    """Configuration for an agent instance"""
    agent_type: AgentType
    name: str
    workspace_path: Path
    focus_areas: List[str] = Field(default_factory=list)
    model_config: Dict = Field(default_factory=dict)
    tool_config: Dict = Field(default_factory=dict)
    max_tokens: int = 100000
    temperature: float = 0.1

class Task(BaseModel):
    """A task to be executed by agents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    command: str
    intent: TaskIntent
    assigned_agents: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)  # Task IDs
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

class AgentSession(BaseModel):
    """Active agent session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_config: AgentConfig
    process_id: Optional[int] = None
    status: str = "inactive"  # inactive, starting, active, busy, error, stopped
    current_task: Optional[str] = None  # Task ID
    workspace: Path
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    last_activity: Optional[datetime.datetime] = None
    
class ExecutionPlan(BaseModel):
    """Plan for executing a command across multiple agents"""
    command: str
    tasks: List[Task] = Field(default_factory=list)
    execution_order: List[List[str]] = Field(default_factory=list)  # Parallel groups
    estimated_duration: int = 0  # minutes
    risk_factors: List[str] = Field(default_factory=list)

class ConflictDetection(BaseModel):
    """Detected conflicts between agent operations"""
    conflict_type: str  # "file_conflict", "dependency_conflict", "logic_conflict"
    affected_files: List[Path] = Field(default_factory=list)
    conflicting_agents: List[str] = Field(default_factory=list)
    severity: str = "low"  # low, medium, high, critical
    resolution_strategy: Optional[str] = None
    auto_resolvable: bool = False

class SharedContext(BaseModel):
    """Shared context between agents"""
    project_context: Dict = Field(default_factory=dict)
    recent_changes: List[Dict] = Field(default_factory=list)
    learned_patterns: List[Dict] = Field(default_factory=list)
    active_tasks: Dict[str, Task] = Field(default_factory=dict)
    agent_states: Dict[str, Dict] = Field(default_factory=dict)
```

---

## 🔧 Core Interfaces

### Agent Interface

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

class Agent(ABC):
    """Base class for all agents"""
    
    def __init__(self, config: AgentConfig, shared_memory: 'SharedMemory'):
        self.config = config
        self.shared_memory = shared_memory
        self.session: Optional[AgentSession] = None
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the agent session"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the agent session"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a specific task"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if agent is healthy and responsive"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities"""
        pass
    
    async def stream_progress(self, task: Task) -> AsyncGenerator[str, None]:
        """Stream progress updates during task execution"""
        # Default implementation - agents can override
        yield f"Starting task: {task.command}"
        result = await self.execute_task(task)
        yield f"Completed task: {result.status}"

class TaskResult(BaseModel):
    """Result of task execution"""
    task_id: str
    agent_id: str
    status: str  # completed, failed, partial
    output: str = ""
    error: Optional[str] = None
    files_modified: List[Path] = Field(default_factory=list)
    execution_time: float = 0.0  # seconds
    tokens_used: int = 0
    cost: float = 0.0  # USD
```

### Tool Integration Interfaces

```python
class AiderAgent(Agent):
    """Aider-based agent implementation"""
    
    def __init__(self, config: AgentConfig, shared_memory: 'SharedMemory'):
        super().__init__(config, shared_memory)
        self.process: Optional[subprocess.Popen] = None
        self.focus_files: List[Path] = []
    
    async def start(self) -> bool:
        """Start Aider process with specific configuration"""
        cmd = [
            "aider",
            f"--model={self.config.model_config.get('model', 'claude-3-5-sonnet')}",
            f"--max-tokens={self.config.max_tokens}",
            "--no-auto-commits",  # We handle commits
        ]
        
        # Add focus files based on agent specialization
        self.focus_files = self._get_focus_files()
        cmd.extend([str(f) for f in self.focus_files])
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.config.workspace_path
        )
        
        return self.process is not None
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute task via Aider"""
        if not self.process:
            raise RuntimeError("Agent not started")
        
        # Send command to Aider
        command_with_context = self._add_context_to_command(task.command)
        self.process.stdin.write(f"{command_with_context}\n".encode())
        await self.process.stdin.drain()
        
        # Read response
        output = ""
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            output += line.decode()
            # Check for completion markers
            if "aider>" in line.decode():
                break
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.session.id,
            status="completed",
            output=output
        )
    
    def _get_focus_files(self) -> List[Path]:
        """Get files this agent should focus on"""
        focus_areas = self.config.focus_areas
        project_structure = self.shared_memory.get_project_structure()
        
        files = []
        for area in focus_areas:
            if area == "backend":
                files.extend(self._find_backend_files(project_structure))
            elif area == "frontend":
                files.extend(self._find_frontend_files(project_structure))
            elif area == "testing":
                files.extend(self._find_test_files(project_structure))
        
        return files
    
    def _add_context_to_command(self, command: str) -> str:
        """Add relevant context to command"""
        context = self.shared_memory.get_relevant_context(command)
        if context:
            return f"Context: {context}\n\nTask: {command}"
        return command

class ClaudeCodeAgent(Agent):
    """Claude Code agent implementation"""
    
    async def start(self) -> bool:
        """Start Claude Code session"""
        # Claude Code integration logic
        pass
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute task via Claude Code"""
        # Use Claude Code for reasoning-heavy tasks
        pass
```

---

## 🧠 Intelligence Components

### Project Analyzer

```python
class ProjectAnalyzer:
    """Analyzes project structure and creates intelligent routing decisions"""
    
    def __init__(self):
        self.tech_stack_detectors = {
            'javascript': JavaScriptDetector(),
            'python': PythonDetector(),
            'typescript': TypeScriptDetector(),
            'rust': RustDetector(),
            'go': GoDetector(),
        }
    
    async def analyze_project(self, project_path: Path) -> ProjectStructure:
        """Comprehensive project analysis"""
        tech_stack = await self._detect_tech_stack(project_path)
        structure = await self._analyze_structure(project_path, tech_stack)
        dependencies = await self._build_dependency_graph(project_path, tech_stack)
        
        return ProjectStructure(
            root_path=project_path,
            tech_stack=tech_stack,
            dependency_graph=dependencies,
            **structure
        )
    
    async def _detect_tech_stack(self, project_path: Path) -> TechStack:
        """Detect technologies used in project"""
        stack = TechStack()
        
        # Check for package files
        package_files = {
            'package.json': 'javascript',
            'requirements.txt': 'python',
            'Cargo.toml': 'rust',
            'go.mod': 'go',
            'composer.json': 'php',
        }
        
        for file_name, language in package_files.items():
            if (project_path / file_name).exists():
                stack.languages.append(language)
                await self._analyze_package_file(project_path / file_name, stack)
        
        # Analyze source files
        for detector_name, detector in self.tech_stack_detectors.items():
            if await detector.detect(project_path):
                await detector.analyze_stack(project_path, stack)
        
        return stack
    
    def recommend_agents(self, project_structure: ProjectStructure) -> List[AgentConfig]:
        """Recommend agent configuration based on project"""
        agents = []
        tech_stack = project_structure.tech_stack
        
        # Always recommend a reasoning agent
        agents.append(AgentConfig(
            agent_type=AgentType.CLAUDE_CODE,
            name="reasoning",
            workspace_path=project_structure.root_path,
            focus_areas=["debugging", "analysis", "explanation"]
        ))
        
        # Recommend based on tech stack
        if any(lang in tech_stack.languages for lang in ['javascript', 'typescript']):
            if 'react' in tech_stack.frameworks or 'vue' in tech_stack.frameworks:
                agents.append(AgentConfig(
                    agent_type=AgentType.AIDER_FRONTEND,
                    name="frontend",
                    workspace_path=project_structure.root_path,
                    focus_areas=["components", "styling", "state"]
                ))
        
        if any(lang in tech_stack.languages for lang in ['python', 'node', 'go', 'rust']):
            agents.append(AgentConfig(
                agent_type=AgentType.AIDER_BACKEND,
                name="backend",
                workspace_path=project_structure.root_path,
                focus_areas=["api", "database", "business-logic"]
            ))
        
        # Always recommend testing agent if tests exist
        if project_structure.test_directories:
            agents.append(AgentConfig(
                agent_type=AgentType.AIDER_TESTING,
                name="testing",
                workspace_path=project_structure.root_path,
                focus_areas=["unit-tests", "integration-tests", "test-utilities"]
            ))
        
        return agents

class CommandRouter:
    """Routes commands to appropriate agents based on intent analysis"""
    
    def __init__(self, project_structure: ProjectStructure):
        self.project_structure = project_structure
        self.intent_classifier = IntentClassifier()
    
    async def route_command(self, command: str, available_agents: List[AgentSession]) -> ExecutionPlan:
        """Create execution plan for command"""
        intent = await self.intent_classifier.analyze_intent(command)
        
        # Select appropriate agents
        selected_agents = self._select_agents(intent, available_agents)
        
        # Create tasks
        tasks = await self._create_tasks(command, intent, selected_agents)
        
        # Determine execution order
        execution_order = self._plan_execution_order(tasks)
        
        return ExecutionPlan(
            command=command,
            tasks=tasks,
            execution_order=execution_order,
            estimated_duration=sum(task.intent.estimated_duration for task in tasks)
        )
    
    def _select_agents(self, intent: TaskIntent, available_agents: List[AgentSession]) -> List[AgentSession]:
        """Select best agents for the task"""
        selected = []
        
        if intent.requires_reasoning:
            # Prefer Claude Code for reasoning tasks
            reasoning_agents = [a for a in available_agents if a.agent_config.agent_type == AgentType.CLAUDE_CODE]
            if reasoning_agents:
                selected.append(reasoning_agents[0])
        
        if intent.requires_coordination:
            # Select multiple specialized agents
            for area in intent.affected_areas:
                area_agents = [a for a in available_agents if area in a.agent_config.focus_areas]
                if area_agents:
                    selected.append(area_agents[0])
        
        # Fallback: select based on task type
        if not selected:
            if intent.task_type == TaskType.DEBUG:
                claude_agents = [a for a in available_agents if a.agent_config.agent_type == AgentType.CLAUDE_CODE]
                if claude_agents:
                    selected.append(claude_agents[0])
            else:
                # Default to first available Aider agent
                aider_agents = [a for a in available_agents if "aider" in a.agent_config.agent_type.value]
                if aider_agents:
                    selected.append(aider_agents[0])
        
        return selected

class SharedMemory:
    """Manages shared context and learning across agents"""
    
    def __init__(self):
        self.context: SharedContext = SharedContext()
        self.session_storage: Dict = {}
    
    def update_project_context(self, key: str, value: Any):
        """Update project-wide context"""
        self.context.project_context[key] = value
    
    def record_change(self, agent_id: str, change: Dict):
        """Record a change made by an agent"""
        change_record = {
            'timestamp': datetime.datetime.utcnow(),
            'agent_id': agent_id,
            **change
        }
        self.context.recent_changes.append(change_record)
        
        # Keep only recent changes (last 100)
        if len(self.context.recent_changes) > 100:
            self.context.recent_changes = self.context.recent_changes[-100:]
    
    def get_relevant_context(self, command: str) -> Dict:
        """Get context relevant to a command"""
        # Simple keyword matching for now
        # TODO: Use embeddings for semantic similarity
        relevant = {}
        
        for change in self.context.recent_changes[-10:]:  # Last 10 changes
            if any(keyword in change.get('description', '') for keyword in command.split()):
                relevant[f"recent_change_{change['timestamp']}"] = change
        
        return relevant
    
    def learn_pattern(self, pattern: Dict):
        """Learn from successful patterns"""
        self.context.learned_patterns.append({
            'timestamp': datetime.datetime.utcnow(),
            **pattern
        })
```

---

## 🔌 Integration Patterns

### CLI Integration

```python
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

@click.group()
@click.pass_context
def cli(ctx):
    """Agentic - Multi-agent AI development orchestrator"""
    ctx.ensure_object(dict)
    ctx.obj['console'] = Console()

@cli.command()
@click.option('--agents', '-a', multiple=True, help='Specific agents to spawn')
@click.pass_context
def init(ctx, agents):
    """Initialize agentic in current project"""
    console = ctx.obj['console']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing project...", total=None)
        
        # Project analysis
        analyzer = ProjectAnalyzer()
        structure = asyncio.run(analyzer.analyze_project(Path.cwd()))
        
        progress.update(task, description="Recommending agents...")
        recommended_agents = analyzer.recommend_agents(structure)
        
        progress.update(task, description="Creating configuration...")
        config = AgenticConfig(
            project=structure,
            agents=recommended_agents
        )
        
        config.save(Path.cwd() / '.agentic' / 'config.yml')
        
    console.print("✅ Agentic initialized successfully!")
    console.print(f"📊 Detected: {', '.join(structure.tech_stack.languages)}")
    console.print(f"🤖 Recommended agents: {', '.join(a.name for a in recommended_agents)}")

@cli.command()
@click.argument('command')
@click.option('--agent', '-a', help='Specific agent to use')
@click.option('--dry-run', is_flag=True, help='Show execution plan without running')
@click.pass_context
def run(ctx, command, agent, dry_run):
    """Execute command via AI agents"""
    console = ctx.obj['console']
    
    # Load configuration
    config_path = Path.cwd() / '.agentic' / 'config.yml'
    if not config_path.exists():
        console.print("❌ Not initialized. Run 'agentic init' first.")
        return
    
    # Start orchestrator
    orchestrator = Orchestrator(config_path)
    
    if dry_run:
        plan = asyncio.run(orchestrator.create_execution_plan(command))
        console.print(f"📋 Execution Plan for: {command}")
        console.print(f"⏱️  Estimated duration: {plan.estimated_duration} minutes")
        console.print(f"🤖 Agents: {', '.join(plan.assigned_agents)}")
        return
    
    # Execute command
    asyncio.run(orchestrator.execute_command(command))
```

### Configuration Management

```yaml
# .agentic/config.yml
project:
  name: "my-awesome-app"
  root_path: "/path/to/project"
  tech_stack:
    languages: ["typescript", "python"]
    frameworks: ["react", "fastapi"]
    testing_frameworks: ["jest", "pytest"]

agents:
  reasoning:
    type: "claude_code"
    focus_areas: ["debugging", "analysis"]
    model:
      name: "claude-4"
      max_tokens: 200000
      temperature: 0.1
  
  frontend:
    type: "aider_frontend"
    focus_areas: ["components", "styling", "state"]
    model:
      name: "claude-4"
      max_tokens: 100000
    focus_files:
      - "src/components/**/*.tsx"
      - "src/pages/**/*.tsx"
      - "src/styles/**/*.css"
  
  backend:
    type: "aider_backend"  
    focus_areas: ["api", "database", "business-logic"]
    model:
      name: "claude-4"
      max_tokens: 100000
    focus_files:
      - "src/api/**/*.py"
      - "src/models/**/*.py"
      - "src/services/**/*.py"

routing:
  complexity_threshold: 0.7
  parallel_execution: true
  conflict_detection: true

integrations:
  git:
    auto_commit: true
    commit_message_template: "feat: {summary}"
  
  notifications:
    slack_webhook: "https://hooks.slack.com/..."
```

This technical architecture provides a solid foundation for implementing Agentic. The modular design allows for incremental development while maintaining flexibility for future enhancements.

Key architectural decisions:
- **Event-driven coordination** for agent communication
- **Plugin-based extensibility** for community contributions  
- **Configuration-driven behavior** for customization
- **Async/await patterns** for performance
- **Type safety** with Pydantic models
- **Rich CLI experience** with progress indicators

The architecture balances simplicity for initial implementation with extensibility for advanced features.