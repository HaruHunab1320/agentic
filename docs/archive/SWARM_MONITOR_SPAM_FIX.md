# Swarm Monitor Display Spam Fix

## Problem
The swarm monitor was showing multiple displays that weren't updating in place:
- Multiple "Agent Swarm Execution" headers appearing
- Display showing before any agents were active
- Carriage returns not working properly

## Root Causes
1. Monitor starting before task analysis was complete
2. Display showing empty state with 0 agents
3. Line counting issues in carriage return implementation

## Fixes Applied

### 1. Fixed Display Order in coordination_engine.py
```python
# BEFORE: Monitor started before task analysis
await self.swarm_monitor.start_monitoring()
# ... then update_task_analysis()

# AFTER: Task analysis first, then start monitor
self.swarm_monitor.update_task_analysis(...)
await self.swarm_monitor.start_monitoring()
```

### 2. Enhanced Update Loop in swarm_monitor_enhanced.py
- Added delay before first display to avoid empty state
- Skip display if no agents registered on first display
- Remove trailing empty lines from output
- Proper line counting with carriage returns

### 3. Removed Initial Newline
- Removed the newline when starting monitoring in non-alternate screen mode
- First display now adds its own newline to separate from command

## Results
- Monitor only shows when agents are active
- Display updates in place without spamming
- Clean separation between command and monitor display
- No more duplicate headers

## Testing
To verify the fix:
1. Run a command: `agentic "build a typescript package"`
2. Monitor should appear only after agents start
3. Display should update in place
4. No duplicate "Agent Swarm Execution" headers

The swarm monitor now provides a clean, professional display that updates smoothly without cluttering the terminal.