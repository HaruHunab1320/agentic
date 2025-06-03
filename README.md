# Agentic

> **Multi-agent AI development workflows from a single CLI**

Agentic orchestrates specialized AI agents to work together on your codebase. Instead of juggling multiple tools and contexts, command a coordinated team of AI specialists from one interface.

## ✨ The Vision

```bash
$ agentic .
🔍 Analyzing codebase...
📊 Detected: React frontend, Node.js backend, Jest tests
🤖 Spawning specialized agents...

You: "Add user authentication across the stack"
[Backend Agent]: Creating auth middleware and JWT handling...
[Frontend Agent]: Building login/signup components...
[Testing Agent]: Writing auth integration tests...
[Coordinator]: Ensuring consistency across implementations...

✅ Authentication system implemented across 12 files
🔄 All tests passing, ready for review
```

## 🚀 Why Agentic?

**Single Interface, Multiple Specialists**
- Command a team of AI agents from one CLI
- Each agent specializes in different aspects (backend, frontend, testing, etc.)
- Intelligent routing sends tasks to the right specialist

**Best-in-Class AI Models**
- Uses Claude 4 (industry-leading 72.5% SWE-bench score) via both Aider and Claude Code
- Leverages each tool's strengths: Aider for coordination, Claude Code for deep reasoning
- Automatic model selection based on task complexity

**Intelligent Coordination**
- Analyzes your project structure and dependencies
- Prevents conflicts between agents working on related code
- Maintains consistency across multi-file changes
- Built-in rollback and conflict resolution

## 🛠 How It Works

### Project Analysis
Agentic maps your entire codebase to understand:
- Technology stack and frameworks
- File dependencies and relationships  
- Testing strategies and patterns
- Code style and conventions

### Smart Routing
Commands are intelligently routed based on:
- **Complex debugging** → Claude Code (superior reasoning)
- **Multi-file refactoring** → Aider (coordinated changes)
- **Legacy code analysis** → Claude Code (better explanations)
- **Cross-system features** → Multiple Aider agents (coordination)

### Agent Coordination
- Shared workspace management
- Real-time conflict detection
- Progress synchronization
- Automatic git integration

## 🎯 Use Cases

**Large Feature Development**
```bash
agentic "Build a real-time chat system with React frontend and WebSocket backend"
```

**Legacy Code Modernization** 
```bash
agentic "Migrate this jQuery codebase to modern React with TypeScript"
```

**Bug Investigation**
```bash
agentic "Debug the race condition in user session management"
```

**Cross-Stack Refactoring**
```bash
agentic "Rename all user-related entities to customer throughout the codebase"
```

## 🚀 Quick Start

### Installation
```bash
pip install agentic
```

### Prerequisites
- [Aider](https://github.com/paul-gauthier/aider) installed
- [Claude Code](https://claude.ai/code) installed  
- Anthropic API key

### Initialize Your Project
```bash
cd your-project
agentic init
```

### Start Commanding Your AI Team
```bash
agentic "Add comprehensive error handling to the API layer"
```

## 🏗 Architecture

```
┌─────────────────┐
│   Single CLI    │
│   Interface     │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │ Agentic   │
    │Orchestrator│
    └─────┬─────┘
          │
    ┌─────▼─────────────────────────┐
    │     Intelligent Router       │
    │  (Analyzes task complexity)   │
    └┬────────────────────────────┬┘
     │                            │
┌────▼────┐                 ┌─────▼─────┐
│ Claude  │                 │   Aider   │
│  Code   │                 │ Sessions  │
│ Agent   │                 │           │
├─────────┤                 ├───────────┤
│• Deep   │                 │• Backend  │
│  Reason │                 │• Frontend │
│• Debug  │                 │• Testing  │
│• Explain│                 │• DevOps   │
└─────────┘                 └───────────┘
```

## 🎛 Configuration

### Agent Specialization
Configure agents for your tech stack:

```yaml
# .agentic/config.yml
agents:
  backend:
    focus: ["api", "database", "auth"]
    tools: ["aider"]
  frontend:
    focus: ["components", "state", "styling"] 
    tools: ["aider"]
  testing:
    focus: ["unit", "integration", "e2e"]
    tools: ["aider"]
  reasoning:
    focus: ["debugging", "analysis", "explanation"]
    tools: ["claude-code"]
```

### Project Detection
Automatic detection supports:
- **Frontend**: React, Vue, Angular, Svelte
- **Backend**: Node.js, Python, Go, Rust, Java
- **Databases**: PostgreSQL, MongoDB, Redis
- **Testing**: Jest, Pytest, Go test, Rust test

## 📊 Benchmarks

Agentic combines the strengths of leading AI coding tools:

| Tool | SWE-bench Score | Specialization |
|------|-----------------|----------------|
| Claude 4 via Aider | 72.5% | Multi-file coordination |
| Claude Code | 72.5% | Deep reasoning & debugging |
| **Agentic (Combined)** | **Best of both** | **Full-stack orchestration** |

## 🗺 Roadmap

### Phase 1: Core Orchestration ✅ (Current)
- [x] Project analysis and agent spawning
- [x] Basic command routing
- [x] Aider integration
- [ ] Claude Code integration
- [ ] Conflict detection

### Phase 2: Advanced Coordination
- [ ] Cross-agent memory sharing
- [ ] Advanced dependency analysis  
- [ ] Custom agent definitions
- [ ] Plugin system

### Phase 3: Team Features
- [ ] Multi-developer coordination
- [ ] Shared agent pools
- [ ] Enterprise integrations
- [ ] Performance analytics

## 🤝 Contributing

We believe the future of software development involves AI agents as team members. Help us build that future:

1. **🐛 Report Issues**: Found a bug or have a feature request?
2. **💡 Suggest Agents**: What specialist agents would help your workflow?
3. **🔧 Contribute Code**: See [CONTRIBUTING.md](CONTRIBUTING.md)
4. **📖 Improve Docs**: Help others understand agentic development

### Development Setup
```bash
git clone https://github.com/yourusername/agentic
cd agentic
pip install -e ".[dev]"
pytest
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[Aider](https://github.com/paul-gauthier/aider)** - Pioneering AI pair programming
- **[Claude Code](https://claude.ai/code)** - Advanced reasoning for coding
- **[Anthropic](https://anthropic.com)
