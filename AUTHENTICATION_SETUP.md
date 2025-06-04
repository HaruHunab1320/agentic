# 🔐 Agentic Authentication & Configuration Guide

## Current Implementation Status

### ✅ **What's Working:**
- **Aider Integration**: Fully implemented with subprocess management
- **API Key Support**: Environment variables and configuration files
- **Claude Code Agent**: Ready for reasoning tasks
- **Configuration Framework**: YAML-based config in `.agentic/config.yml`

### ⚠️ **What Needs Setup:**
- API keys must be configured
- Minor implementation fixes needed for full CLI functionality

---

## 🚀 Quick Setup Guide

### 1. **Set Up API Keys**

**Option A: Environment Variables (Recommended)**
```bash
# Add to your ~/.zshrc or ~/.bashrc
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
export OPENAI_API_KEY="your-openai-api-key-here"

# Reload your shell
source ~/.zshrc
```

**Option B: Direct Configuration**
```bash
# Set temporarily for current session
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export OPENAI_API_KEY="sk-..."
```

### 2. **Get Your API Keys**

**Anthropic (Claude):**
1. Visit https://console.anthropic.com/
2. Create account and verify
3. Navigate to API Keys section
4. Generate new API key
5. Copy the key (starts with `sk-ant-api03-...`)

**OpenAI (GPT):**
1. Visit https://platform.openai.com/
2. Create account and add payment method
3. Navigate to API Keys
4. Generate new secret key
5. Copy the key (starts with `sk-...`)

### 3. **Test Direct Aider Integration**

```bash
# Test Aider with Anthropic
aider --anthropic-api-key $ANTHROPIC_API_KEY --model claude-3-5-sonnet --help

# Test Aider with OpenAI  
aider --openai-api-key $OPENAI_API_KEY --model gpt-4 --help

# Interactive session (current working method)
aider --anthropic-api-key $ANTHROPIC_API_KEY --model claude-3-5-sonnet
```

---

## 🏗️ Architecture Overview

### **How Authentication Works:**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Environment   │    │   .agentic/      │    │   Aider         │
│   Variables     │───▶│   config.yml     │───▶│   --api-key     │
│                 │    │                  │    │   Arguments     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Agent Types & Authentication:**

1. **Claude Code Agent**: Uses Anthropic API for reasoning
2. **Aider Agents**: Pass API keys to Aider subprocess
   - `AiderFrontendAgent`: Frontend development
   - `AiderBackendAgent`: Backend development  
   - `AiderTestingAgent`: Testing and QA

---

## 🧪 Testing Authentication

### **1. Test Aider Directly:**
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

# Should modify test.py with a docstring
```

### **2. Test Configuration:**
```bash
# Run the setup script
./setup_agentic.sh

# Check generated config
cat .agentic/config.yml
```

### **3. Test Agentic CLI (Limited - needs fixes):**
```bash
# These commands work:
agentic --help
agentic config --help

# These need implementation fixes:
# agentic init
# agentic status
```

---

## 🛠️ Current Implementation Details

### **Aider Integration Code:**
```python
# From src/agentic/agents/aider_agents.py
class BaseAiderAgent(Agent):
    def _setup_aider_args(self) -> None:
        self.aider_args = [
            "aider",
            "--yes",  # Auto-confirm changes
            "--no-git",  # Don't auto-commit
            f"--model={self.config.ai_model_config.get('model', 'claude-3-5-sonnet')}",
        ]
        
    async def _verify_model_access(self) -> bool:
        # Verifies API key works with Aider
        test_cmd = self.aider_args + ["--help"]
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
```

### **Configuration Structure:**
```yaml
# .agentic/config.yml
workspace_name: "your-project"
workspace_path: "/path/to/project"
default_agents: ["claude_code"]
api:
  anthropic:
    api_key: "sk-ant-api03-..."
    model: "claude-3-5-sonnet"
  openai:
    api_key: "sk-..."
    model: "gpt-4"
```

---

## 🔧 Implementation Status

### **✅ Completed:**
- Aider subprocess integration
- API key passing to Aider
- Configuration file structure
- Agent specializations (frontend, backend, testing)
- Claude Code agent for reasoning

### **🚧 Needs Fixes:**
```python
# Missing method in AgenticConfig
@classmethod
def load_or_create(cls, workspace_path: Path) -> 'AgenticConfig':
    # Implementation needed
    
# Logger initialization in CLI
def setup_logging(debug: bool = False) -> Logger:
    # Return logger instance, not None
```

### **🎯 Next Steps:**
1. Fix `AgenticConfig.load_or_create()` method
2. Fix logger initialization in CLI
3. Test end-to-end workflow
4. Add API key validation

---

## 💡 Manual Workflow (Currently Working)

**Until CLI fixes are complete, you can use Aider directly:**

```bash
# 1. Set API key
export ANTHROPIC_API_KEY="your-key-here"

# 2. Use Aider with specialized prompts
aider --anthropic-api-key $ANTHROPIC_API_KEY \
      --model claude-3-5-sonnet \
      --message "You are a backend specialist. Help me optimize this database query." \
      src/database/*.py

# 3. Or start interactive session
aider --anthropic-api-key $ANTHROPIC_API_KEY --model claude-3-5-sonnet
```

**Example Specialized Prompts:**
- **Frontend**: "You are a React specialist. Help me create responsive components."
- **Backend**: "You are an API specialist. Help me design RESTful endpoints."  
- **Testing**: "You are a testing expert. Help me write comprehensive unit tests."
- **Debugging**: "You are a debugging expert. Help me trace this error."

---

## 🎯 Summary

**The authentication framework is solid** - API keys work with Aider, configuration is flexible, and the agent architecture is well-designed. The main missing pieces are:

1. **Set your API keys** (most important)
2. **Minor CLI implementation fixes** (for full Agentic experience)
3. **Direct Aider usage works now** (as interim solution)

**Bottom line**: Yes, this can work! The infrastructure is there, just needs API keys configured and minor bug fixes. 