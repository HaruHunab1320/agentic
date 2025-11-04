# Gemini Agent Integration

This document describes the Google Gemini CLI integration with the Agentic framework.

## Overview

The Gemini agent brings powerful multimodal capabilities to Agentic, including:

- **1M token context window** - 5x larger than Claude Code
- **Multimodal processing** - Analyze images, PDFs, sketches, diagrams
- **Google Search grounding** - Access real-time information
- **Session management** - Save and resume conversations
- **Memory system** - Similar to CLAUDE.md for persistent context

## Installation

### Prerequisites

1. Node.js 18 or higher
2. Gemini API key or Google account for authentication

### Setup

```bash
# Install Gemini CLI globally
npm install -g @google/gemini-cli

# Set API key (optional - can authenticate interactively)
export GEMINI_API_KEY="your-api-key"
```

## Usage

### Spawning a Gemini Agent

```python
from agentic.models.agent import AgentConfig, AgentType

config = AgentConfig(
    agent_type=AgentType.GEMINI,
    name="multimodal-analyst",
    workspace_path=Path("/path/to/project"),
    focus_areas=["multimodal", "research", "documentation"],
    ai_model_config={"model": "gemini-2.5-pro"},
    api_key="your-api-key"  # Optional
)
```

### Task Examples

#### Multimodal Analysis
```python
task = Task(
    type=TaskType.ANALYSIS,
    description="Analyze architecture diagram",
    prompt="Analyze the system architecture from this diagram and suggest improvements",
    context={
        "media": ["architecture.png"],
        "files": ["src/system/config.py"]
    }
)
```

#### Research with Web Search
```python
task = Task(
    type=TaskType.RESEARCH,
    description="Research latest framework updates",
    prompt="Research the latest updates to React 19 and summarize breaking changes",
    context={"use_search": True}
)
```

#### PDF Processing
```python
task = Task(
    type=TaskType.FEATURE,
    description="Generate code from specification",
    prompt="Generate the API endpoints based on this specification document",
    context={"media": ["api_spec.pdf"]}
)
```

## Capabilities Comparison

| Feature | Claude Code | Aider | Gemini |
|---------|------------|-------|---------|
| Context Window | 200k | 8k-32k | 1M |
| Multimodal | ❌ | ❌ | ✅ |
| Web Search | ❌ | ❌ | ✅ |
| Session Memory | ✅ | ❌ | ✅ |
| File Editing | ✅ | ✅ | ✅ |
| Git Integration | ✅ | ✅ | ✅ |

## When to Use Gemini

The task analyzer automatically selects Gemini for:

- **Multimodal tasks** - Processing images, PDFs, diagrams
- **Research tasks** - Requiring current information
- **Large context** - Working with massive codebases
- **Documentation** - Analyzing or generating from visual specs
- **Architecture** - Understanding system designs from diagrams

## Integration with Other Agents

Gemini works seamlessly with other agents in multi-agent workflows:

```python
# Example: Architecture analysis + implementation
# 1. Gemini analyzes architecture diagram
# 2. Claude Code designs the implementation approach  
# 3. Aider agents implement the code
```

## Advanced Features

### Session Management
```python
# Save current session
await gemini_agent.save_session("feature-x-analysis")

# Resume later
await gemini_agent.resume_session("feature-x-analysis")
```

### Memory Management
```python
# Add to agent's memory
await gemini_agent.add_memory("Always use TypeScript strict mode in this project")
```

## Limitations

- Requires Node.js runtime
- API rate limits: 60 requests/minute, 1000/day
- No direct code execution (uses shell commands)

## Troubleshooting

### Authentication Issues
```bash
# Use interactive auth if API key not set
npx @google/gemini-cli

# Or set API key
export GEMINI_API_KEY="your-key"
```

### Installation Problems
```bash
# Clear npm cache
npm cache clean --force

# Reinstall
npm install -g @google/gemini-cli
```

## Future Enhancements

- [ ] Streaming response support
- [ ] Better multimodal result parsing
- [ ] Integration with MCP servers
- [ ] Custom tool connections