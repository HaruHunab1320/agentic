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

## Features

- **Multi-Agent Architecture**: Specialized agents for different development tasks
- **Enhanced Claude Code Integration**: Full-featured Claude Code CLI integration with memory, sessions, and extended thinking
- **Task Planning & Execution**: Intelligent task breakdown and parallel execution
- **Git Integration**: Automated version control workflows
- **Project Memory**: Persistent memory for project-specific conventions and preferences

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/agentic.git
cd agentic

# Install dependencies
pip install -r requirements.txt

# Install Claude Code CLI (required for enhanced features)
npm install -g @anthropic-ai/claude-code
```

## Quick Start

```bash
# Start an interactive session
python -m agentic

# Execute a specific task
python -m agentic exec "analyze the architecture of this project"

# Use enhanced Claude Code features
python -m agentic claude "refactor the authentication module"
```

## Claude Code Integration

Agentic provides enhanced Claude Code integration with advanced features:

### Memory Management

When you first use Claude Code with Agentic, it automatically creates a `CLAUDE.md` file with sensible defaults. You can customize this file for your project:

```bash
# Copy the comprehensive template
cp examples/claude_memory_template.md CLAUDE.md

# Add to .gitignore to keep preferences local
echo "CLAUDE.md" >> .gitignore
```

### Best Practices

1. **Keep Memory Local**: Add `CLAUDE.md` to your `.gitignore` - it should contain personal/team preferences, not be committed to tool repos
2. **Customize for Your Project**: Use the template in `examples/claude_memory_template.md` as a starting point
3. **Session Persistence**: Complex tasks automatically use session IDs for multi-turn conversations
4. **Extended Thinking**: Architecture and refactoring tasks automatically trigger deeper analysis

### Example Memory Setup

```markdown
# My Project - Claude Code Memory

## Project Overview
This is a React/Node.js e-commerce application.

## Coding Standards
- Use TypeScript for all new code
- Follow ESLint configuration
- Prefer functional components with hooks

## Team Conventions
- Use conventional commits
- Squash merge pull requests
- Always update tests with changes
```

## Architecture

### Agents

- **ClaudeCodeAgent**: Enhanced Claude Code CLI integration with memory and sessions
- **GitAgent**: Git operations and workflow automation
- **TaskPlannerAgent**: Intelligent task breakdown and coordination

### Core Components

- **Task Management**: Structured task representation with intent analysis
- **Agent Coordination**: Multi-agent workflow orchestration
- **Session Management**: Persistent sessions for complex multi-turn tasks
- **Memory System**: Project-specific memory for conventions and preferences

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤖 Model Configuration

Agentic supports multiple AI models and makes it easy to configure them for different agents and tasks.

### **Supported Models**

#### **🌟 Gemini (Google)**
- `gemini/gemini-1.5-pro-latest` - Latest Pro model (recommended)
- `gemini/gemini-1.5-flash-latest` - Fast Flash model
- `gemini-1.5-pro` - Alias for Pro model
- `gemini` - Default Gemini alias

#### **🧠 Claude (Anthropic)**  
- `claude-3-5-sonnet` - Latest Sonnet model (default)
- `claude-3-haiku` - Fast Haiku model
- `claude-3-opus` - Most capable Opus model
- `claude` - Default Claude alias

#### **🤖 OpenAI**
- `gpt-4o` - Latest GPT-4 Omni
- `gpt-4-0125-preview` - GPT-4 Turbo
- `gpt-3.5-turbo` - GPT-3.5 Turbo

### **Quick Setup for Gemini**

1. **Set your API key:**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
# Or for Aider specifically:
export GEMINI_API_KEY="your-gemini-api-key"
```

2. **Set Gemini as your default model:**
```bash
agentic model set gemini
```

3. **Test the configuration:**
```bash
agentic model test gemini
```

### **Model Configuration Commands**

#### **List Available Models**
```bash
# See all supported models and current configuration
agentic model list
```

#### **Set Model for All Agents**
```bash
# Set default model for all agents
agentic model set gemini-1.5-pro
agentic model set claude-3-5-sonnet
agentic model set gpt-4o
```

#### **Set Model for Specific Agent**
```bash
# Set model for just the backend agent
agentic model set gemini --agent backend

# Set model for frontend agent
agentic model set claude-3-5-sonnet --agent frontend
```

#### **Test Model Configuration**
```bash
# Test if a model is working correctly
agentic model test gemini
agentic model test claude-3-5-sonnet
```

### **Per-Project Configuration**

Models can be configured per project via the `.agentic/config.yml` file:

```yaml
models:
  primary_model: "gemini/gemini-1.5-pro-latest"
  fallback_model: "claude-3-haiku"
  temperature: 0.1
  max_tokens: 100000

agents:
  backend:
    ai_model_config:
      model: "gemini-1.5-pro"
  frontend:
    ai_model_config:
      model: "claude-3-5-sonnet"
  testing:
    ai_model_config:
      model: "gpt-4o"
```

### **Environment Variables**

You can also configure models via environment variables:

```bash
# Global model configuration
export AGENTIC_PRIMARY_MODEL="gemini"
export AGENTIC_TEMPERATURE="0.1"

# Model-specific API keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export GEMINI_API_KEY="your-gemini-key"  # Alternative for Gemini
```

### **Model Selection Best Practices**

- **🚀 Gemini Flash**: Best for fast iterations and simple coding tasks
- **🧠 Gemini Pro**: Excellent for complex reasoning and architecture decisions
- **✨ Claude Sonnet**: Great balance of capability and speed for general coding
- **🎯 Claude Haiku**: Fast and efficient for simple refactoring tasks
- **💪 Claude Opus**: Most capable for complex system design
- **🔥 GPT-4o**: Strong alternative with good coding capabilities

## 🔐 API Key Setup

Agentic supports **three secure methods** for managing API keys. Choose the one that works best for you:

### **🏆 Method 1: Secure Storage (Recommended)**

Uses your operating system's secure credential storage (Keychain on macOS, Credential Manager on Windows, Secret Service on Linux).

```bash
# Set API keys securely
agentic keys set gemini          # For Gemini models
agentic keys set anthropic       # For Claude models  
agentic keys set openai          # For GPT models

# Set globally (all projects) vs project-specific
agentic keys set gemini --global        # Available everywhere
agentic keys set gemini                 # Only this project

# View configured keys (masked for security)
agentic keys list

# Remove keys
agentic keys remove gemini
```

### **📄 Method 2: .env File**

For developers who prefer environment files:

```bash
# Create template
agentic keys env-template

# Copy and edit
cp .env.example .env
# Edit .env with your API keys
```

Example `.env` file:
```bash
# Google/Gemini API Key
GOOGLE_API_KEY=your_actual_api_key_here

# Anthropic API Key  
ANTHROPIC_API_KEY=your_actual_api_key_here

# OpenAI API Key
OPENAI_API_KEY=your_actual_api_key_here
```

### **🌍 Method 3: Environment Variables**

Set environment variables directly:

```bash
# For current session
export GOOGLE_API_KEY="your_key_here"

# Permanently (add to your shell profile)
echo 'export GOOGLE_API_KEY="your_key_here"' >> ~/.bashrc
```

### **🔄 Fallback Priority**

Agentic automatically finds your API keys in this order:
1. **Keyring** (project-specific)
2. **Keyring** (global)  
3. **.env file** (project directory)
4. **Environment variables**

### **🧪 Test Your Setup**

```bash
# Test if your Gemini setup works
agentic model test gemini

# Check which keys are configured
agentic keys list
```

## 🤖 Model Configuration
