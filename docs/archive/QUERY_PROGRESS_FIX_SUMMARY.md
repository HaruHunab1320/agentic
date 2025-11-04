# Query Progress Display Fix Summary

## Problem
When running single-agent queries (questions/analysis), users saw no progress indicators while Claude was processing. The output would appear to hang after showing "Using analysis timeout of 120s for analysis task" until the final result appeared.

## Root Cause
The swarm monitor was disabled for single-agent question tasks (`enable_monitoring: False`), which meant none of the activity updates from the Claude Code agent were being displayed.

## Fixes Applied

### 1. Chat Interface - Added Progress Spinner (chat_interface.py)
```python
# Show processing indicator for single-agent queries with Live display
if is_question_task:
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    
    # Create a live display with spinner for single-agent queries
    self._query_live = Live(
        Spinner("dots", text=Text("Processing query...", style="cyan")),
        refresh_per_second=4
    )
    self._query_live.start()
```

### 2. Enabled Monitoring for All Tasks (chat_interface.py)
Changed from disabling monitor for questions to always enabling it:
```python
context = {
    'preferred_agent': query_analysis.suggested_agent,
    'enable_monitoring': True,  # Enable monitoring to show progress
    'lightweight_monitor': is_question_task
}
```

### 3. Claude Code Agent Improvements (claude_code_agent.py)
- Removed `--verbose` flag that was causing stalling
- Conditional JSON format (skip for analysis queries)
- Added periodic "thinking" updates
- Better activity detection from output streams

### 4. Proper Cleanup
Added cleanup in finally block to stop the spinner:
```python
finally:
    # Stop the query live display if it exists
    if hasattr(self, '_query_live') and self._query_live:
        self._query_live.stop()
        self._query_live = None
```

## Expected Behavior
Users will now see:
1. Initial "🤔 Processing query..." message
2. Animated spinner while Claude thinks
3. Live activity updates if monitoring is working
4. Clean transition to the final result

## Note
Since agentic is installed in dev mode, these changes should take effect immediately. The progress spinner provides visual feedback even if the detailed Claude activity monitoring isn't fully working.