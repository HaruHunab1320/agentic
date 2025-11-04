# Claude Code Real-Time Monitoring Enhancement

## Overview
Enhanced the Claude Code agent to parse Claude's JSON output in real-time and display detailed activities in the swarm monitor. This provides users with moment-by-moment visibility into what Claude is doing.

## Key Changes

### 1. Enhanced JSON Stream Parsing
- Added JSON buffer accumulation in `read_stream()` method
- Detects JSON output by looking for `{"messages":` pattern
- Accumulates multi-line JSON data until complete
- Parses JSON and extracts tool uses and text responses

### 2. New Activity Parsing Methods

#### `_parse_claude_json_stream(json_data)`
- Parses Claude's JSON output structure
- Extracts tool use blocks and text blocks
- Sends formatted activities to swarm monitor

#### `_format_tool_activity(tool_name, tool_input)`
- Maps Claude Code tool names to human-readable activities
- Supports all Claude Code tools:
  - Read, Write, Edit, MultiEdit
  - Bash (with command descriptions)
  - Grep, Glob, LS
  - NotebookRead, NotebookEdit
  - WebFetch, WebSearch
  - TodoRead, TodoWrite
- Uses emojis for visual clarity:
  - 📖 Reading files
  - ✍️ Writing files
  - ✏️ Editing files
  - 🧪 Running tests
  - 🔍 Searching
  - ⚡ Executing commands

#### `_extract_activity_from_text(text)`
- Extracts meaningful activities from Claude's text responses
- Identifies planning statements ("I'll...", "Let me...")
- Filters out long explanations
- Focuses on action-oriented phrases

### 3. Streaming for All Print Mode Tasks
- Extended real-time streaming to all tasks using print mode
- Not just test tasks anymore
- Provides consistent monitoring experience

### 4. Enhanced Swarm Monitor
- Added activity truncation (80 chars max)
- Prevents duplicate consecutive activities
- Maintains clean activity history

## Benefits

1. **Real-time Visibility**: Users see exactly what Claude is doing as it happens
2. **Better Debugging**: Easy to see where Claude might be stuck or taking time
3. **Progress Tracking**: Clear indication of task progress through activities
4. **Tool Usage Insights**: See which tools Claude uses and how

## Example Activities Shown

```
📖 Reading src/agentic/core/swarm_monitor.py
🔍 Searching for 'monitor' in *.py files
✏️ Editing config.py - replacing 'old_value...'
🧪 Running tests: npm test
⚡ Lists files in current directory
Planning to analyze the test framework...
Starting to search for skipped tests...
```

## Testing

Use the provided test script:
```bash
python test_claude_monitoring.py
```

This will demonstrate the real-time monitoring capabilities with a sample task.