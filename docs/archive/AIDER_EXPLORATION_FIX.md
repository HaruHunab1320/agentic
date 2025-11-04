# Aider Agent Exploration Capabilities Fix

## Problem
When users ask Aider agents to "examine the current state of the code base" or similar exploration tasks, the agents couldn't see any files because:
1. No specific files were mentioned in the command
2. Aider runs with `--exit` flag (non-interactive mode)
3. No repository context was provided

## Solution Implemented

### 1. Exploration Detection
Added detection for exploration/analysis keywords:
```python
exploration_keywords = ['examine', 'analyze', 'explore', 'review', 'understand', 
                       'familiarize', 'inspect', 'current state', 'codebase',
                       'architecture', 'list files', 'what files', 'show me']
```

### 2. Read-Only File Access
When exploration is detected, the system now:
- Uses Aider's `--read` flag to provide read-only access to files
- Automatically includes:
  - README files (README.md, README.rst, etc.)
  - Configuration files (package.json, tsconfig.json, pyproject.toml, etc.)
  - Sample source files from common directories (src/, lib/, app/, packages/)

### 3. Enhanced Instructions
For exploration tasks, the agent receives special instructions:
```
You have been given read-only access to key project files. 
Please examine the codebase structure, understand the architecture, and provide a comprehensive analysis.
Focus on identifying the project's current state, technologies used, and any remaining work needed.
```

## Benefits
- Aider can now examine codebases without needing specific file names
- Maintains non-interactive `--exit` mode for predictable behavior
- Provides appropriate context for different types of queries
- Separates exploration (read-only) from editing tasks

## Example Usage
When you ask:
> "examine the current state of the code base, familiarize yourself with the architecture"

Aider will now:
1. Detect this as an exploration task
2. Get read-only access to project files
3. Provide a comprehensive analysis of the codebase
4. List remaining tasks based on what it finds

## Technical Details
The fix modifies `_build_enhanced_aider_command()` in `aider_agents.py` to:
1. Check for exploration keywords
2. Add `--read` flags for relevant files when no specific files are targeted
3. Provide appropriate context in the message

This allows Aider to use its full analytical capabilities while maintaining our automated, non-interactive execution model.