# Chat Interface Loop Fix

## Problem
After task completion, the chat interface was entering an infinite loop where:
- Leftover display lines from the swarm monitor were being read as user input
- Each line triggered a new agent execution
- Lines like "│ 📊 Execution Summary" were treated as commands

## Root Cause
The swarm monitor display wasn't being properly cleared when stopping, leaving display content in the terminal buffer that the chat interface would then read as input.

## Solution Implemented

### 1. Track Display Line Count
Added `self._last_line_count` to remember how many lines the display used.

### 2. Enhanced Stop Monitoring
In `stop_monitoring()`, added proper cleanup for non-alternate screen mode:
```python
# Clear the display area completely
if self._last_line_count > 0:
    # Move up to the start of the display
    sys.stderr.write(f"\033[{self._last_line_count}A")
    # Clear each line
    for _ in range(self._last_line_count):
        sys.stderr.write("\033[2K")  # Clear entire line
        sys.stderr.write("\n")
    # Move back up to the start
    sys.stderr.write(f"\033[{self._last_line_count}A")
```

### 3. Proper Stream Separation
- Monitor display uses stderr (not read by chat interface)
- Final summary uses stdout (visible to user)
- Clear separation prevents display content from being interpreted as input

## Results
- No more infinite loops after task completion
- Clean terminal state after monitor stops
- Display content won't be interpreted as user commands
- Chat interface remains responsive without spurious inputs

## Testing
1. Run a command that uses swarm monitor
2. Wait for completion
3. Verify the chat prompt returns cleanly
4. No more automatic queries from display lines

The fix ensures that the swarm monitor display is completely cleared from the terminal when it stops, preventing any leftover content from being read as input by the chat interface.