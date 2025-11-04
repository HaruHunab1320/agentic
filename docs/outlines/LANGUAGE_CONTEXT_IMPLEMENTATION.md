# Language Context Awareness Implementation

## Overview

The Agentic system now has comprehensive language and framework awareness, allowing it to:
1. **Detect** the language/framework of existing projects
2. **Infer** the appropriate language for new code generation
3. **Ask** for clarification when ambiguous
4. **Pass** language context to all agents

This solves the fundamental issue where the system (built in Python) would default to Python for all code generation tasks.

## Key Components

### 1. Project Indexer (`src/agentic/core/project_indexer.py`)

The `ProjectIndexer` class provides deep project understanding:

- **File Indexing**: Scans all project files, respecting .gitignore
- **Language Detection**: Identifies languages by file extensions
- **Framework Detection**: Detects frameworks through config files and code patterns
- **Dependency Mapping**: Builds a graph of file dependencies
- **Caching**: Maintains a cached index for performance

Key features:
```python
# Index the project
indexer = ProjectIndexer(workspace_path)
index = await indexer.index_project()

# Get project context
context = indexer.get_project_context()
# Returns: primary_language, frameworks, project_type, etc.
```

### 2. Language Selector (`src/agentic/core/language_selector.py`)

The `LanguageSelector` handles intelligent language/framework selection:

- **Command Analysis**: Detects language hints in user commands
- **Context Inference**: Uses project context when command is ambiguous
- **Clarification Prompts**: Generates user-friendly language selection prompts
- **Response Parsing**: Interprets user language preferences

Example:
```python
selector = LanguageSelector()

# Detect from command
lang = selector.detect_language_from_command("create a react component")
# Returns: "javascript"

# Infer with context  
lang, fw = selector.infer_language_from_context(command, project_context)

# Get clarification if needed
prompt = selector.get_clarification_prompt(command, project_context)
```

### 3. Enhanced Task Model (`src/agentic/models/task.py`)

Tasks now carry language context:

```python
class Task(BaseModel):
    # ... existing fields ...
    target_language: Optional[str] = Field(default=None)
    target_framework: Optional[str] = Field(default=None)
    project_context: Optional[ProjectContext] = Field(default=None)
```

### 4. Coordination Engine Updates

The coordination engine now:
- Indexes the project on first command
- Adds project context to all tasks
- Passes language preferences through the execution pipeline

### 5. Agent Prompt Updates

Both Aider and Claude Code agents now include language context in their prompts:

**Aider agents**: 
```python
# In _build_specialized_message
if task.target_language:
    base_message += f"Target Language: {task.target_language}\n"
```

**Claude Code agents**:
```python
# In _build_natural_prompt
if task.target_language:
    return f"[Target language: {task.target_language}] {task.command}"
```

## How It Works

1. **User enters command**: "create a todo app"

2. **Project indexing**: System scans workspace to understand current project
   - Detects React + TypeScript project
   - Identifies npm as package manager

3. **Language detection**: Checks if command specifies language
   - No explicit language found

4. **Context inference**: Uses project context
   - Primary language: TypeScript
   - Framework: React

5. **Clarification** (if needed): Asks user for preference
   - Shows relevant options based on project type

6. **Task creation**: Creates task with language context
   - `task.target_language = "typescript"`
   - `task.target_framework = "react"`

7. **Agent execution**: Agents receive language context
   - Generate TypeScript React components, not Python

## Benefits

1. **Language Agnostic**: Works with any programming language
2. **Context Aware**: Understands your project's technology stack
3. **Smart Defaults**: Uses project language when not specified
4. **User Control**: Can override with explicit language mentions
5. **Framework Support**: Handles framework-specific code generation

## Usage Examples

### Automatic Detection
```bash
# In a React project
agentic chat
> create a new component

# Generates React component in project's language (JS/TS)
```

### Explicit Language
```bash
> create a python flask API endpoint

# Overrides project default, generates Python code
```

### Clarification Flow
```bash
> create a web application

# System asks:
# I need to know what language/framework you'd like to use
# Suggestions:
#   - React (TypeScript)
#   - Vue.js
#   - Angular
# Your choice: react typescript
```

## Configuration

The system stores preferences for the session:
- `default_language`: Remembered after first selection
- `default_framework`: Remembered after first selection

Project context is cached in `.agentic/project_index.json` for performance.

## Future Enhancements

1. **Multi-language projects**: Better support for polyglot repositories
2. **Style detection**: Detect coding style preferences
3. **Template system**: Language-specific templates
4. **Migration support**: Help migrate between languages/frameworks