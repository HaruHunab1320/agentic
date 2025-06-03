# Phase 1: Foundation & MVP (Weeks 1-2)

> **Establish core project structure and implement basic CLI with Aider integration**

## 🎯 Objectives
- Establish core project structure with proper Python packaging
- Implement basic CLI interface for user interaction
- Create project analysis capabilities to understand codebases
- Demonstrate proof-of-concept with single Aider agent integration
- Set up development infrastructure (testing, CI/CD, documentation)

## 📦 Deliverables

### 1.1 Project Setup
**Goal**: Create professional Python project structure with development tooling

**Tasks**:
- [ ] Initialize repository with proper Python package structure
- [ ] Set up CI/CD pipeline using GitHub Actions
- [ ] Configure development environment with pre-commit hooks
- [ ] Set up testing framework with pytest
- [ ] Create documentation structure with Sphinx or MkDocs
- [ ] Configure linting and formatting (black, flake8, mypy)

**Files to Create**:
```
agentic/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .github/
│   └── workflows/
│       ├── test.yml
│       └── release.yml
├── .pre-commit-config.yaml
├── src/agentic/
│   ├── __init__.py
│   ├── cli.py
│   └── py.typed
├── tests/
│   ├── __init__.py
│   └── conftest.py
├── docs/
│   ├── index.md
│   └── api.md
└── examples/
```

**Expected Output**: Fully configured Python project ready for development

### 1.2 Core CLI Interface
**Goal**: Create intuitive command-line interface for user interaction

**Commands to Implement**:
```bash
agentic init          # Initialize project configuration
agentic analyze       # Analyze current codebase
agentic spawn         # Manually spawn agents
agentic "command"     # Execute command via agents
agentic status        # Show agent status
agentic stop          # Stop all agents
```

**Technical Requirements**:
- Use Click framework for CLI parsing
- Use Rich library for beautiful terminal output
- Implement proper error handling and help messages
- Support configuration via files and environment variables
- Include progress indicators for long-running operations

**Example Usage**:
```bash
$ agentic init
🔍 Analyzing project structure...
📊 Detected: React frontend, Node.js backend
🤖 Recommended agents: frontend, backend, testing
✅ Configuration saved to .agentic/config.yml

$ agentic "fix the authentication bug in login.js"
🤖 Spawning frontend agent...
[Frontend Agent]: Analyzing login.js...
[Frontend Agent]: Found authentication issue on line 45...
[Frontend Agent]: Implementing fix...
✅ Authentication bug fixed
```

### 1.3 Project Analyzer
**Goal**: Intelligently analyze codebases to understand structure and technology stack

**Analysis Capabilities**:
- [ ] Technology stack detection (React, Node.js, Python, etc.)
- [ ] File structure mapping (source, tests, configs, docs)
- [ ] Dependency analysis from package files
- [ ] Test framework identification
- [ ] Git repository integration and status
- [ ] Code complexity and size estimation

**Technology Detectors**:
```python
class TechStackDetector:
    def detect_javascript(self, project_path: Path) -> bool:
        """Detect JavaScript/Node.js projects"""
        return (project_path / "package.json").exists()
    
    def detect_python(self, project_path: Path) -> bool:
        """Detect Python projects"""
        return any([
            (project_path / "requirements.txt").exists(),
            (project_path / "pyproject.toml").exists(),
            (project_path / "setup.py").exists()
        ])
    
    def detect_react(self, project_path: Path) -> bool:
        """Detect React projects"""
        package_json = project_path / "package.json"
        if package_json.exists():
            with open(package_json) as f:
                content = json.load(f)
                deps = {**content.get("dependencies", {}), **content.get("devDependencies", {})}
                return "react" in deps
        return False
```

**Output Format**:
```python
ProjectStructure(
    root_path=Path("/path/to/project"),
    tech_stack=TechStack(
        languages=["javascript", "typescript"],
        frameworks=["react", "express"],
        testing_frameworks=["jest", "cypress"],
        build_tools=["webpack", "vite"]
    ),
    source_directories=[Path("src"), Path("components")],
    test_directories=[Path("tests"), Path("__tests__")],
    entry_points=[Path("src/index.js"), Path("src/App.js")]
)
```

### 1.4 Basic Aider Integration
**Goal**: Integrate with Aider to execute commands through AI agent

**Integration Features**:
- [ ] Spawn Aider process with project-specific configuration
- [ ] Pass user commands to Aider session
- [ ] Capture and format Aider output for display
- [ ] Handle Aider process lifecycle (start, monitor, stop)
- [ ] Manage Aider workspace and file context
- [ ] Error handling for Aider failures

**Aider Agent Implementation**:
```python
class AiderAgent:
    def __init__(self, workspace: Path, model: str = "claude-4"):
        self.workspace = workspace
        self.model = model
        self.process = None
    
    async def start(self) -> bool:
        """Start Aider process"""
        cmd = [
            "aider",
            f"--model={self.model}",
            "--no-auto-commits",
            "--no-pretty"
        ]
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.workspace
        )
        return self.process is not None
    
    async def execute_command(self, command: str) -> str:
        """Send command to Aider and return response"""
        if not self.process:
            raise RuntimeError("Agent not started")
        
        self.process.stdin.write(f"{command}\n".encode())
        await self.process.stdin.drain()
        
        # Read response until prompt returns
        output = ""
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            decoded_line = line.decode()
            output += decoded_line
            if "aider>" in decoded_line:
                break
        
        return output
```

## 🛠 Technical Requirements

### Core Dependencies
```toml
# pyproject.toml
[project]
name = "agentic"
version = "0.1.0"
description = "Multi-agent AI development orchestrator"
dependencies = [
    "click>=8.0.0",
    "rich>=13.0.0", 
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "gitpython>=3.1.0",
    "psutil>=5.9.0",
    "aiofiles>=23.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0"
]

[project.scripts]
agentic = "agentic.cli:main"
```

### Project Structure
```
src/agentic/
├── __init__.py              # Package initialization
├── cli.py                   # Main CLI entry point  
├── core/
│   ├── __init__.py
│   ├── analyzer.py          # Project analysis logic
│   ├── config.py            # Configuration management
│   └── exceptions.py        # Custom exceptions
├── agents/
│   ├── __init__.py
│   ├── base.py             # Base agent interface
│   └── aider.py            # Aider agent implementation
└── utils/
    ├── __init__.py
    ├── git.py              # Git utilities
    ├── process.py          # Process management
    └── logging.py          # Logging configuration
```

### Configuration Format
```yaml
# .agentic/config.yml
project:
  name: "my-project"
  root_path: "/path/to/project"
  tech_stack:
    languages: ["javascript", "typescript"]
    frameworks: ["react"]
    testing_frameworks: ["jest"]

agents:
  default:
    type: "aider"
    model: "claude-4"
    workspace: "."
    auto_commit: false

cli:
  output_format: "rich"  # rich, plain, json
  log_level: "info"
  progress_indicators: true
```

## 📊 Success Criteria

### Functional Requirements
- [ ] **Project Analysis**: Correctly identify tech stack for 5+ different project types
- [ ] **CLI Interface**: All commands work with proper help and error messages
- [ ] **Aider Integration**: Successfully spawn Aider and execute basic commands
- [ ] **Configuration**: Save and load project configuration properly
- [ ] **Error Handling**: Graceful failures with helpful error messages

### Technical Requirements  
- [ ] **Code Quality**: 100% type hints, passes mypy strict mode
- [ ] **Test Coverage**: >80% unit test coverage
- [ ] **Performance**: Project analysis completes in <10 seconds for typical projects
- [ ] **Reliability**: No crashes during normal operation
- [ ] **Documentation**: All public APIs documented with examples

### User Experience
- [ ] **Installation**: `pip install agentic` works smoothly
- [ ] **First Use**: `agentic init` creates working configuration in <30 seconds
- [ ] **Command Execution**: Basic commands complete in reasonable time
- [ ] **Output**: Clear, formatted output using Rich library
- [ ] **Help**: Comprehensive help available for all commands

## 🧪 Test Cases

### 1. Project Analysis Tests
```python
def test_detect_react_project():
    """Test React project detection"""
    # Create temp directory with package.json containing React
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        package_json = {
            "dependencies": {"react": "^18.0.0"}
        }
        (project_path / "package.json").write_text(json.dumps(package_json))
        
        analyzer = ProjectAnalyzer()
        structure = analyzer.analyze_project(project_path)
        
        assert "javascript" in structure.tech_stack.languages
        assert "react" in structure.tech_stack.frameworks

def test_detect_python_project():
    """Test Python project detection"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "requirements.txt").write_text("fastapi==0.68.0\n")
        
        analyzer = ProjectAnalyzer()
        structure = analyzer.analyze_project(project_path)
        
        assert "python" in structure.tech_stack.languages
```

### 2. CLI Interface Tests
```python
def test_init_command(runner):
    """Test agentic init command"""
    with runner.isolated_filesystem():
        # Create a simple project
        Path("package.json").write_text('{"dependencies": {"react": "^18.0.0"}}')
        
        result = runner.invoke(cli, ['init'])
        
        assert result.exit_code == 0
        assert "Analyzing project" in result.output
        assert Path(".agentic/config.yml").exists()

def test_analyze_command(runner):
    """Test agentic analyze command"""
    with runner.isolated_filesystem():
        # Setup project
        Path("package.json").write_text('{"dependencies": {"react": "^18.0.0"}}')
        Path("src").mkdir()
        
        result = runner.invoke(cli, ['analyze'])
        
        assert result.exit_code == 0
        assert "React" in result.output
```

### 3. Aider Integration Tests
```python
@pytest.mark.asyncio
async def test_aider_agent_start():
    """Test starting Aider agent"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        agent = AiderAgent(workspace)
        
        # Mock Aider process for testing
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_subprocess.return_value = mock_process
            
            result = await agent.start()
            
            assert result is True
            assert agent.process is not None

@pytest.mark.asyncio 
async def test_aider_command_execution():
    """Test executing command through Aider"""
    agent = AiderAgent(Path("."))
    agent.process = AsyncMock()
    agent.process.stdin.write = AsyncMock()
    agent.process.stdin.drain = AsyncMock()
    agent.process.stdout.readline = AsyncMock(
        side_effect=[b"Analyzing code...\n", b"aider> "]
    )
    
    result = await agent.execute_command("fix the bug")
    
    assert "Analyzing code" in result
    agent.process.stdin.write.assert_called_once()
```

### 4. Error Handling Tests
```python
def test_init_without_project(runner):
    """Test init in empty directory"""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init'])
        
        # Should still work but with minimal detection
        assert result.exit_code == 0
        assert "No specific tech stack detected" in result.output

def test_command_without_init(runner):
    """Test running command without initialization"""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code != 0
        assert "not initialized" in result.output.lower()
```

## 🚀 Implementation Order

### Week 1: Foundation
1. **Day 1-2**: Project setup, repository structure, CI/CD
2. **Day 3-4**: Basic CLI framework with Click and Rich
3. **Day 5**: Project analyzer core logic
4. **Day 6-7**: Technology detection and file analysis

### Week 2: Integration
1. **Day 8-9**: Aider agent implementation and process management  
2. **Day 10-11**: Command routing and execution flow
3. **Day 12-13**: Configuration system and error handling
4. **Day 14**: Testing, documentation, and polish

## 🎯 MVP Demo Script

After Phase 1 completion, this should work:

```bash
# Initialize in a React project
cd my-react-app
agentic init
# Output: ✅ Detected React + TypeScript project, configured frontend agent

# Analyze the project  
agentic analyze
# Output: 📊 React app with 45 components, Jest tests, TypeScript

# Execute a simple command
agentic "add a loading spinner to the login form"
# Output: 🤖 Frontend agent working... ✅ Loading spinner added to LoginForm.tsx

# Check status
agentic status  
# Output: 🟢 Frontend agent active, last command: 30s ago
```

## 🔍 Phase 1 Completion Checklist

**Before moving to Phase 2, ensure:**
- [ ] All CLI commands work without errors
- [ ] Project analysis correctly identifies major tech stacks  
- [ ] Single Aider agent can execute basic commands
- [ ] Configuration saves/loads properly
- [ ] Unit tests pass with >80% coverage
- [ ] Documentation is complete and accurate
- [ ] CI/CD pipeline is green
- [ ] No critical bugs or performance issues

**Phase 1 deliverables must be working before Phase 2 begins.**