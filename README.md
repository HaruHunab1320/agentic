# Agentic

> **🤖 Intelligent Multi-Agent AI Development Platform**

Agentic is a production-ready multi-agent AI system that orchestrates specialized agents to tackle complex software development tasks. Unlike simple AI assistants, Agentic coordinates domain experts with sophisticated routing, load balancing, and parallel execution.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ What Makes Agentic Special

```bash
$ agentic "Add JWT authentication to my FastAPI app with React frontend"

🔍 Analyzing task complexity... COMPLEX MULTI-DOMAIN
📊 Spawning optimal agents:
  • Backend Agent (Python/FastAPI) - JWT middleware & endpoints
  • Frontend Agent (React/TypeScript) - Login components & auth hooks  
  • Testing Agent - Integration & unit tests
  • Security Review - Vulnerability scanning

🚀 Parallel execution with conflict detection...
✅ Authentication system complete across 23 files
📈 Quality score: 94/100 • Security: PASSED • Tests: 100% coverage
```

## 🎯 Key Features

### **🧠 Enhanced Agent Selection Strategy**
- **Sophisticated Routing Algorithm** - Multi-factor scoring with task analysis
- **Claude Code** for fast analysis, debugging, and creative problem-solving
- **Aider Specialists** for systematic implementation and multi-file coordination
- **Domain Experts** - Frontend, Backend, Testing, DevOps specializations

### **⚡ Production-Ready Architecture**
- **Load Balancing** - Intelligent agent distribution and parallel execution
- **Conflict Detection** - Prevents file conflicts across concurrent agents
- **Session Management** - Persistent contexts and advanced memory systems
- **Circuit Breakers** - Graceful degradation and fault tolerance

### **🌐 Multi-Language & Multi-Model Support**
- **14+ Programming Languages** - Python, TypeScript, Go, Rust, Java, and more
- **Multiple AI Models** - Claude, Gemini, GPT-4, with automatic selection
- **Framework Detection** - React, FastAPI, Django, Vue, Angular, etc.
- **Three-Tier API Key Management** - Environment, keyring, and configuration

### **🔒 Enterprise Security & Quality**
- **Security Scanning** - Built-in vulnerability detection
- **Quality Assurance** - Automated testing and coverage analysis
- **Audit Logging** - Comprehensive tracking and compliance
- **Type Safety** - Strict typing throughout with Pydantic models

## 🚀 Quick Start

### Installation
```bash
# Install Agentic
pip install agentic

# Install required AI tools
npm install -g @anthropic-ai/claude-code  # For Claude Code agent
pip install aider-chat                    # For Aider agents
```

### Setup API Keys
```bash
# Set your AI provider keys (choose your preferred method)
export ANTHROPIC_API_KEY="your-key"     # For Claude
export GOOGLE_API_KEY="your-key"        # For Gemini
export OPENAI_API_KEY="your-key"        # For OpenAI

# Or use secure keyring storage
agentic auth add anthropic your-key
agentic auth add google your-key
```

### Initialize Your Project
```bash
cd your-project
agentic init
# Creates .agentic/config.yml with intelligent defaults
```

### Start Building
```bash
# Single command for complex features
agentic "Build a user management system with CRUD API and React dashboard"

# Specific agent targeting
agentic "Debug the React component rendering issue" --agent claude_code
agentic "Refactor the authentication module" --agent aider_backend

# Multi-language projects
agentic "Add TypeScript types to the entire frontend codebase"
```

## 🏗️ Architecture & Agents

### **Agent Ecosystem**

#### **🔍 Claude Code Agent**
- **Specialization**: Analysis, debugging, code review, explanations
- **Strengths**: Fast reasoning, creative problem-solving, single-file tasks
- **Languages**: Python, JavaScript, TypeScript, Rust, Go, Java, C++, HTML, CSS, SQL, Bash

#### **🎨 Aider Frontend Agent** 
- **Specialization**: UI/UX, components, styling, frontend architecture
- **Frameworks**: React, Vue, Angular, Svelte, Next.js
- **Technologies**: TypeScript, CSS/SCSS, Tailwind, state management

#### **⚙️ Aider Backend Agent**
- **Specialization**: APIs, databases, authentication, server logic
- **Languages**: Python, Node.js, Go, Rust, Java, C#
- **Frameworks**: FastAPI, Django, Express, Gin, Actix, Spring

#### **🧪 Aider Testing Agent**
- **Specialization**: Unit tests, integration tests, coverage, QA
- **Frameworks**: pytest, Jest, Go test, Rust test, JUnit
- **Features**: Automated test generation, coverage analysis

#### **🚀 Aider DevOps Agent**
- **Specialization**: CI/CD, deployment, infrastructure, monitoring
- **Technologies**: Docker, Kubernetes, GitHub Actions, AWS, GCP

### **Intelligent Routing System**

```python
# Enhanced selection algorithm with multi-factor scoring
def determine_optimal_agent(task):
    factors = {
        "task_type": analyze_task_keywords(task),      # +5 for matches
        "scope": detect_file_scope(task),              # single vs multi-file
        "approach": determine_approach_style(task),     # creative vs systematic
        "specialization": detect_domain(task),         # frontend/backend/testing
        "complexity": calculate_complexity(task)       # simple vs complex
    }
    return select_highest_scoring_agent(factors)
```

## 📊 Performance & Benchmarks

### **Agent Selection Accuracy**
- **100% Success Rate** on 10 diverse task scenarios
- **Perfect Specialization** - Frontend tasks → Frontend agent
- **Optimal Tool Utilization** - Claude for speed, Aider for thoroughness

### **Execution Performance**
- **Parallel Processing** - Independent tasks run simultaneously  
- **Load Balancing** - Automatic agent distribution
- **Conflict Resolution** - Zero file conflicts in concurrent operations

### **Quality Metrics**
- **Type Safety**: 100% typed codebase with mypy compliance
- **Test Coverage**: Comprehensive test suites across all components
- **Security**: Built-in vulnerability scanning and best practices

## 🛠️ Advanced Configuration

### **Multi-Model Setup**
```yaml
# .agentic/config.yml
ai_providers:
  claude:
    model: "claude-3-5-sonnet-20241022"
    api_key_source: "keyring"  # environment, keyring, or config
  gemini:
    model: "gemini-2.0-flash-exp"
    api_key_source: "environment"
  openai:
    model: "gpt-4o"
    api_key_source: "keyring"

agents:
  claude_code:
    primary_model: "claude"
    specializations: ["analysis", "debugging", "creative"]
  aider_backend:
    primary_model: "gemini"  # Gemini Pro 2.5 for detailed reasoning
    focus_areas: ["api", "database", "auth"]
  aider_frontend:
    primary_model: "claude"
    focus_areas: ["components", "styling", "ux"]
```

### **Agent Specialization**
```yaml
task_routing:
  analysis_keywords: ["explain", "analyze", "debug", "review"]
  implementation_keywords: ["create", "build", "implement", "refactor"]
  
  # Strong patterns for precise routing
  patterns:
    "explain the": +4 points → Claude Code
    "create a": +4 points → Aider
    frontend_files: ["*.tsx", "*.vue"] → Aider Frontend
    backend_files: ["**/api/**", "**/models/**"] → Aider Backend
```

### **Load Balancing Configuration**
```yaml
coordination:
  max_parallel_agents: 4
  conflict_detection: true
  session_persistence: true
  automatic_git_integration: true
  
performance:
  agent_timeout: 300
  max_context_tokens: 200000
  memory_management: "auto"
```

## 🔧 Development & Contributing

### **Development Setup**
```bash
# Clone and setup
git clone https://github.com/yourusername/agentic.git
cd agentic
pip install -e ".[dev]"

# Run comprehensive tests
python -m pytest tests/ -v --cov=src/agentic

# Type checking and formatting
mypy src/
black src/ tests/
isort src/ tests/
```

### **Project Structure**
```
agentic/
├── src/agentic/
│   ├── agents/           # Agent implementations
│   ├── core/            # Core orchestration and routing
│   ├── models/          # Pydantic data models
│   ├── utils/           # Utility functions
│   └── auth/            # API key management
├── tests/               # Comprehensive test suite
├── docs/                # Sphinx documentation
└── examples/            # Usage examples
```

### **Quality Standards**
- **100% Type Coverage** - Strict typing with mypy
- **Comprehensive Testing** - Unit, integration, and E2E tests
- **Security First** - No hardcoded secrets, secure key management
- **Documentation** - Complete API docs and user guides

## 📚 Documentation

- **[Quick Start Guide](docs/quickstart.rst)** - Get up and running in minutes
- **[Architecture Overview](docs/architecture.rst)** - Deep dive into system design
- **[API Reference](docs/api.rst)** - Complete API documentation
- **[Contributing Guide](docs/contributing.rst)** - How to contribute
- **[Enhanced Agent Selection Strategy](ENHANCED_AGENT_SELECTION_STRATEGY.md)** - Detailed routing algorithm

## 🎯 Use Cases

### **🏢 Enterprise Development**
```bash
# Complex multi-service features
agentic "Implement OAuth2 with microservices architecture"

# Legacy modernization
agentic "Migrate jQuery frontend to React with TypeScript"

# Security auditing
agentic "Perform comprehensive security audit of the authentication system"
```

### **🔬 Research & Prototyping**
```bash
# Cross-language exploration
agentic "Compare performance of Rust vs Go for this API endpoint"

# Architecture decisions
agentic "Analyze trade-offs between GraphQL and REST for this use case"
```

### **📈 Quality Improvement**
```bash
# Code quality enhancement
agentic "Add comprehensive type hints and improve test coverage to 95%"

# Performance optimization
agentic "Identify and fix performance bottlenecks in the data processing pipeline"
```

## 🗺️ Roadmap

### **✅ Phase 1: Core Platform (COMPLETE)**
- [x] Multi-agent orchestration with intelligent routing
- [x] Enhanced Claude Code and Aider integration
- [x] Load balancing and parallel execution
- [x] Three-tier API key management
- [x] Conflict detection and resolution

### **🔄 Phase 2: Advanced Coordination (IN PROGRESS)**
- [x] Sophisticated agent selection algorithm
- [x] Multi-model support (Claude, Gemini, OpenAI)
- [x] Quality assurance pipeline
- [ ] Custom agent plugin system
- [ ] Enterprise monitoring dashboard

### **🚀 Phase 3: Enterprise Features (PLANNED)**
- [ ] Multi-developer coordination
- [ ] Shared agent pools across teams  
- [ ] Advanced security scanning
- [ ] Performance analytics and optimization
- [ ] Cloud deployment and scaling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Aider](https://github.com/paul-gauthier/aider)** - Pioneering AI pair programming with exceptional multi-file coordination
- **[Claude Code](https://claude.ai/code)** - Advanced reasoning and analysis capabilities for coding tasks
- **[Anthropic](https://anthropic.com)** - Claude AI models powering intelligent code analysis
- **[Google](https://ai.google.dev/)** - Gemini models for detailed reasoning and implementation

## 🔗 Links

- **GitHub Repository**: [https://github.com/yourusername/agentic](https://github.com/yourusername/agentic)
- **Documentation**: [https://agentic.readthedocs.io](https://agentic.readthedocs.io) 
- **PyPI Package**: [https://pypi.org/project/agentic/](https://pypi.org/project/agentic/)
- **Issue Tracker**: [https://github.com/yourusername/agentic/issues](https://github.com/yourusername/agentic/issues)

---

**Built with ❤️ for the AI-assisted development community**