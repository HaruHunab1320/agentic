# Swarm Monitor Carriage Return Fix

## Problem
The swarm monitor was spamming the terminal with new lines instead of updating the display in place. This created a scrolling effect that made it difficult to track agent progress.

## Solution Implemented

### 1. Enhanced Update Loop (`_update_loop` method)
- Track the number of lines in the previous display
- Use ANSI escape sequences to move cursor up and overwrite previous content
- Clear lines properly to avoid artifacts

### 2. Key Changes:
```python
# Move cursor up to start of previous display
if last_line_count > 0:
    sys.stderr.write(f"\033[{last_line_count}A")  # Move up N lines
    sys.stderr.write("\033[0G")  # Move to start of line

# Clear each line and write new content
for i, line in enumerate(lines):
    sys.stderr.write("\033[2K")  # Clear entire line
    sys.stderr.write(line)
    if i < len(lines) - 1:  # Don't add newline after last line
        sys.stderr.write("\n")
```

### 3. Start Monitoring Adjustment
- When not using alternate screen, only add a single newline instead of clearing screen
- This prevents jarring screen clears in normal terminal mode

## ANSI Escape Sequences Used
- `\033[nA` - Move cursor up n lines
- `\033[0G` - Move cursor to beginning of line
- `\033[2K` - Clear entire line
- `\033[?1049h/l` - Enter/exit alternate screen buffer
- `\033[2J` - Clear entire screen
- `\033[H` - Move cursor to home position

## Benefits
1. **Clean Updates**: Display updates in place without scrolling
2. **Preserved Context**: User can see command output above the monitor
3. **Terminal Friendly**: Works in both alternate screen and normal modes
4. **No Spam**: No more newline flooding in the terminal

## Testing
The fix was tested with:
- Multiple agents running concurrently
- Activity updates streaming in real-time
- Terminal resizing during execution
- Both with and without alternate screen mode

## Usage
The fix is automatic. To disable alternate screen mode (useful in CI/testing):
```bash
export AGENTIC_NO_ALTERNATE_SCREEN=1
```

This ensures the swarm monitor provides a clean, professional display that updates smoothly without cluttering the terminal.