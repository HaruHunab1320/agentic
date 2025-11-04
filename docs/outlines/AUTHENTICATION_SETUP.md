# 🔐 Agentic Authentication & Configuration Guide

## Authentication Options Overview

Agentic supports **multiple AI integrations** with different authentication methods:

### 🤖 **Option 1: Anthropic API (Recommended)**
- **What**: Direct API access to Claude models via Aider
- **Authentication**: API key (`ANTHROPIC_API_KEY`)
- **Usage**: Powers Aider agents for specialized development tasks
- **Cost**: Pay-per-token usage
- **Setup**: Get API key from [console.anthropic.com](https://console.anthropic.com/)

### 🖥️ **Option 2: Claude Desktop App + Claude Code CLI (Primary)** 
- **What**: Claude Code CLI tool with desktop app authentication
- **Authentication**: Claude Pro/Team subscription + CLI login
- **Usage**: Advanced coding, refactoring, and project analysis
- **Cost**: Monthly subscription (Pro: $20/month, Team: $30/month)
- **Setup**: Install Claude Desktop + Claude Code CLI

### 🔧 **Option 3: Both (Recommended for Teams)**
- **Best of both worlds**: Specialized Aider agents + powerful Claude Code sessions
- **Use Aider for**: Backend development, testing, focused tasks
- **Use Claude Code for**: Code review, refactoring, complex analysis

---

## 🚀 Quick Setup Guide

### **Step 1: Install Claude Code CLI**

Claude Code is the **primary** coding agent in Agentic. Install it first:

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --help

# Login with your Claude account
claude
# Follow the login prompts
```

### **Step 2: Set Up API Keys (Optional but Recommended)**

For enhanced capabilities with Aider agents:

```bash
# Add to your ~/.zshrc or ~/.bashrc
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
export OPENAI_API_KEY="your-openai-api-key-here"  # Optional

# Reload your shell
source ~/.zshrc
```

### **Step 3: Get Your Accounts**

**Claude Desktop + Pro Account (Primary):**
1. Sign up for [Claude Pro/Team](https://claude.ai/upgrade) ($20-30/month)
2. Download [Claude Desktop](https://claude.ai/download)
3. Install and sign in with your Claude account
4. Claude Code CLI will use your desktop authentication

**Anthropic API (Optional Enhancement):**
1. Visit [console.anthropic.com](https://console.anthropic.com/)
2. Create account and verify email
3. Navigate to **API Keys** section
4. Click **Create Key** (pay-per-token)

---

## 🏗️ Architecture & Authentication Flow

### **How Claude Code Authentication Works:**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Claude Pro    │    │   Claude         │    │   Claude Code   │
│   Account       │───▶│   Desktop App    │───▶│   CLI Sessions  │
│   Login         │    │   Authentication │    │   (Agentic)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Anthropic     │    │   Environment    │───▶│   Aider         │
│   API Key       │───▶│   Variables      │    │   Subprocess    │
│   (Optional)    │    │   .agentic/      │    │   (Specialized) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Agent Types & Authentication:**

| Agent Type | Authentication | Purpose |
|------------|----------------|---------|
| **Claude Code Agent** | Claude Desktop Login | Primary coding, refactoring, analysis |
| **Claude Reasoning Agent** | Claude Desktop Login | Internal analysis, debugging, explanations |
| **Aider Backend Agent** | Anthropic API Key | Backend development via Aider |
| **Aider Frontend Agent** | Anthropic API Key | Frontend development via Aider |
| **Aider Testing Agent** | Anthropic API Key | Testing and QA via Aider |

---

## 🧪 Testing Authentication

### **1. Test Claude Code CLI:**
```bash
# Test basic functionality
claude --help

# Check authentication status
claude
> /status

# Test with a simple query
claude -p "explain what this project does"

# Test in your project directory
cd /path/to/your/project
claude "analyze this codebase structure"
```

### **2. Test Aider with API Key (Optional):**
```bash
# Create a test file
echo "def hello(): pass" > test.py

# Test with Anthropic API
aider --anthropic-api-key $ANTHROPIC_API_KEY \
      --model claude-3-5-sonnet \
      --no-git \
      --yes \
      test.py \
      --message "Add a docstring to this function"
```

### **3. Test Agentic CLI:**
```bash
# Basic commands
agentic --help
agentic config show

# Initialize workspace
agentic init

# Test Claude Code integration
agentic spawn claude_code

# Test command execution
agentic exec "refactor the authentication system"
```

---

## 🛠️ Current Implementation Status

### **✅ Fully Implemented:**
- **Claude Code CLI integration** - Primary coding agent with subprocess management
- **Anthropic API integration** via Aider - Specialized development tasks
- **Configuration management** (YAML + environment variables)
- **Agent specializations** (claude_code, frontend, backend, testing, reasoning)
- **CLI command routing** and orchestration
- **Project analysis** and intelligent agent selection

### **🎯 Authentication Priority:**
1. **Claude Code CLI** - Core functionality (install and login required)
2. **Anthropic API** - Enhanced capabilities (optional but recommended)
3. **OpenAI API** - Alternative models (completely optional)

---

## 🔧 Detailed Implementation

### **Claude Code Integration (Primary):**
```python
# From src/agentic/agents/claude_code_agent.py
class ClaudeCodeAgent(Agent):
    async def _build_claude_command(self, task: Task) -> List[str]:
        cmd = ["claude"]
        
        # Use print mode for automation
        if task.metadata.get("mode") != "interactive":
            cmd.extend(["-p"])
        
        # Add allowed tools for coding tasks
        coding_tools = [
            "Edit",  # File editing
            "Bash(git *)",  # Git operations
            "Write"  # File writing
        ]
        
        for tool in coding_tools:
            cmd.extend(["--allowedTools", tool])
        
        # Set model and prompt
        cmd.extend(["--model", "sonnet"])
        cmd.append(self._build_task_prompt(task))
        
        return cmd
```

### **Aider Integration (Enhancement):**
```python
# From src/agentic/agents/aider_agents.py
class BaseAiderAgent(Agent):
    def _setup_aider_args(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.aider_args = [
            "aider",
            "--yes",
            "--no-git",
            f"--anthropic-api-key={api_key}",
            f"--model={self.config.ai_model_config.get('model', 'claude-3-5-sonnet')}",
        ]
```

---

## 📋 Setup Checklist

### **Required for Basic Functionality:**
- [ ] **Claude Pro/Team subscription** ($20-30/month)
- [ ] **Claude Desktop app** installed and authenticated
- [ ] **Claude Code CLI** installed (`npm install -g @anthropic-ai/claude-code`)
- [ ] **Claude Code authenticated** (run `claude` and login)
- [ ] **Python 3.10+** with Agentic installed

### **Optional for Enhanced Features:**
- [ ] **Anthropic API Key** for specialized Aider agents
- [ ] **OpenAI API Key** for GPT model support
- [ ] **Git repository** for change tracking
- [ ] **Node.js** for Claude Code CLI

### **Verification Commands:**
```bash
# Check Claude Code
claude --help
claude -p "test authentication"

# Check API keys (optional)
echo "Anthropic: ${ANTHROPIC_API_KEY:0:10}..."
echo "OpenAI: ${OPENAI_API_KEY:0:10}..."

# Check Agentic installation
agentic --version

# Test end-to-end
python test_agentic_simple.py
```

---

## 🎯 Recommendations

### **For Immediate Use:**
1. **Start with Claude Code CLI** - This is the primary agent and requires subscription
2. **Test with simple coding tasks** to verify authentication
3. **Add API keys later** for enhanced Aider agent capabilities

### **For Advanced Users:**
1. **Use both authentication methods** for maximum flexibility
2. **Claude Code for**: Complex refactoring, code review, analysis
3. **Aider agents for**: Specialized backend/frontend/testing tasks

### **Example Usage:**
```bash
# Primary Claude Code usage
agentic spawn claude_code
agentic exec "analyze and refactor the user authentication system"

# Enhanced Aider usage (with API key)
agentic spawn backend
agentic exec "implement a new REST API endpoint for user profiles"

# Combined workflow
agentic exec "claude_code should review the backend changes, then aider_testing should create comprehensive tests"
```

---

## 🔮 Future Roadmap

### **Phase 1: Enhanced Claude Code Integration**
- Advanced session management and persistence
- Custom CLAUDE.md memory integration
- Multi-session coordination

### **Phase 2: Desktop App Features**
- Real-time code suggestions
- Intelligent refactoring assistance
- Advanced context sharing

### **Phase 3: Team Features**
- Shared Claude Desktop sessions
- Team-wide agent coordination
- Advanced project templates

**Bottom Line**: 
1. **Primary**: Get Claude Pro + Claude Code CLI for core functionality
2. **Enhancement**: Add Anthropic API key for specialized Aider agents  
3. **Optional**: OpenAI API for additional model choices

**The Claude Code CLI is now the primary coding agent in Agentic** - it provides the most sophisticated coding capabilities and integrates directly with your Claude Pro subscription. 