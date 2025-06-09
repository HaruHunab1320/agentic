# Claude Code CLI Flags Documentation

This document details how Agentic leverages Claude Code's CLI flags for optimal multi-agent execution.

## Flags We Use

### Core Execution Flags

#### `--print, -p`
- **Purpose**: Run Claude Code in non-interactive mode
- **Why we use it**: Essential for autonomous agent execution - prevents Claude from waiting for user input
- **Implementation**: Always used in our `execute_task` method

#### `--output-format json`
- **Purpose**: Get structured JSON output instead of plain text
- **Why we use it**: Enables reliable parsing of Claude's responses
- **Implementation**: Added to all print mode executions for better output handling
- **JSON Structure**: We parse `messages[].content` from the response

#### `--verbose`
- **Purpose**: Show full turn-by-turn output
- **Why we use it**: Provides real-time activity updates for our swarm monitor
- **Implementation**: Enables activity streaming to show what Claude is actually doing

#### `--max-turns 10`
- **Purpose**: Limit the number of autonomous turns
- **Why we use it**: Prevents runaway execution and infinite loops
- **Implementation**: Set to 10 turns as a reasonable limit for most tasks

### Model Selection

#### `--model`
- **Purpose**: Select the AI model (sonnet, opus, or full model name)
- **Usage**: We map our config models to Claude's format:
  - `claude-sonnet` → `sonnet`
  - `opus` → `opus`
- **Default**: Uses Claude's default if not specified

## Flags We Don't Use (But Could)

### Tool Permission Flags

#### `--allowedTools` / `--disallowedTools`
- **Purpose**: Pre-authorize specific tools without prompting
- **Potential use**: Could speed up execution by pre-approving common tools
- **Example**: `--allowedTools "Write" "Edit" "Bash(git *)"` 

#### `--dangerously-skip-permissions`
- **Purpose**: Skip all permission prompts
- **Why we avoid it**: Security risk - better to use specific allowed tools

### Session Management

#### `--resume` / `--continue`
- **Purpose**: Resume previous sessions
- **Why we don't use**: Each agent task starts fresh to avoid context pollution
- **Potential use**: Could enable long-running agent sessions

### Advanced Output

#### `--output-format stream-json`
- **Purpose**: Stream JSON events in real-time
- **Potential use**: Could provide even better activity monitoring
- **Trade-off**: More complex parsing but real-time updates

## Implementation Details

### Current Command Structure
```python
cmd = [
    "claude",
    "-p",                          # Print mode
    "--output-format", "json",     # JSON output
    "--verbose",                   # Activity details
    "--max-turns", "10",          # Turn limit
    "--model", model_name,        # Model selection
    prompt                        # The actual task
]
```

### JSON Output Parsing
```python
# We parse the JSON response to extract:
output_data = json.loads(stdout)
messages = output_data.get('messages', [])
assistant_messages = [m for m in messages if m['role'] == 'assistant']
content = assistant_messages[-1]['content']
```

### Activity Monitoring
The `--verbose` flag provides turn-by-turn updates that we parse for the swarm monitor:
- Tool usage events
- File operations
- Test execution status
- Error messages

## Best Practices

1. **Always use `--print` mode** for autonomous execution
2. **Prefer `--output-format json`** for reliable parsing
3. **Include `--verbose`** when activity monitoring is needed
4. **Set `--max-turns`** to prevent infinite loops
5. **Avoid `--dangerously-skip-permissions`** for security

## Future Enhancements

1. **Stream JSON**: Could switch to `--output-format stream-json` for real-time updates
2. **Tool Whitelisting**: Use `--allowedTools` to pre-approve common operations
3. **Session Persistence**: Explore `--resume` for long-running agent tasks
4. **Custom Permission Handler**: Use `--permission-prompt-tool` with an MCP tool

## Security Considerations

- We never use `--dangerously-skip-permissions` to maintain security
- Tool permissions are handled through Claude's normal flow
- Each agent runs in its own process with limited permissions
- File operations are restricted to the workspace directory