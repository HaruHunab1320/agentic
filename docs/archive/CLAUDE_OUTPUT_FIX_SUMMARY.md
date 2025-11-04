# Claude Code Output Streaming Fix Summary

## Problem
When running Claude Code in query mode (e.g., "can you tell me what we would have to build..."), the output was stalling and not showing meaningful real-time updates. The CLI would show initialization logs but then appear to hang while Claude was processing.

## Root Causes
1. **Verbose flag causing issues**: The `--verbose` flag was causing Claude Code to output too much information that was interfering with JSON parsing
2. **JSON output format for queries**: Using `--output-format json` for analysis queries prevented natural streaming of Claude's thought process
3. **Lack of periodic updates**: No feedback was shown while Claude was thinking/processing

## Fixes Applied

### 1. Removed verbose flag (line 747)
```python
# Remove verbose flag for query mode - it can cause stalling
# cmd.extend(["--verbose"])
```

### 2. Conditional JSON format (lines 745-748)
```python
# For analysis tasks, don't use JSON format to get better streaming
if not any(word in task.command.lower() for word in ['tell me', 'what', 'explain', 'analyze', 'find']):
    # Use JSON output format for better parsing of action tasks
    cmd.extend(["--output-format", "json"])
```

### 3. Added periodic thinking updates (lines 1232-1264)
When Claude is processing without streaming output, show periodic updates:
```python
async def provide_periodic_updates():
    update_messages = [
        "Claude is thinking...",
        "Analyzing the codebase...",
        "Formulating response...",
        "Processing information...",
        "Gathering insights..."
    ]
```

### 4. Improved activity detection (lines 1185-1191)
Better detection of Claude's thinking patterns in streaming output:
```python
if any(pattern in line_text.lower() for pattern in [
    'thinking', 'analyzing', 'reading', 'looking', 'checking',
    'searching', 'found', 'let me', "i'll", 'processing'
]):
    self._monitor.update_agent_activity(self._monitor_agent_id, f"Claude: {line_text[:100]}...")
```

### 5. Enhanced task-specific feedback (lines 669-678)
More specific initial activity messages based on query type:
```python
elif 'explain' in command_lower or 'analyze' in command_lower or 'tell me' in command_lower:
    self._monitor.update_agent_activity(self._monitor_agent_id, "Reading and understanding codebase...")
else:
    self._monitor.update_agent_activity(self._monitor_agent_id, "Processing your query...")
```

## Expected Behavior After Fix

1. **Immediate feedback**: Users will see "Processing your query..." or task-specific messages immediately
2. **Periodic updates**: While Claude thinks, users see rotating status messages every 3 seconds
3. **Better streaming**: For analysis queries, output streams more naturally without JSON formatting
4. **Activity detection**: When Claude's output is available, meaningful activities are shown
5. **No more stalling**: Removed verbose flag prevents output parsing issues

## Testing

To test the improvements:
```bash
$ agentic
[agentic] > can you tell me what we would have to build to get the 1 skipped test to be ready to test?
```

You should now see:
- Initial "Processing your query..." message
- Periodic updates while Claude thinks
- Real-time activity updates as Claude works
- Final formatted response