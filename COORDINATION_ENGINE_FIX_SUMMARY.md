# Coordination Engine "Unknown Error" Fix Summary

## Problem
When running tests, the coordination engine would fail with "Unknown error" after successfully registering an agent.

## Root Causes

### 1. Agent Spawn Failures Not Handled Properly
- When `_find_best_agent_for_task` failed to spawn an agent, it returned `None`
- Tasks with no assigned agent were marked as failed but had no proper error result
- The swarm monitor would show "Unknown error" because no TaskResult was created

### 2. Non-existent Error Field
- The code tried to access `session.error` but AgentSession class had no error field
- This would cause additional errors when trying to log spawn failures

### 3. Claude Code Authentication Issues
- Claude Code requires authentication before use
- In automated mode, it would hang waiting for browser authentication
- This caused agent spawn to fail silently

## Fixes Applied

### 1. Proper Error Handling for Failed Agent Assignment
```python
# coordination_engine.py - Create proper error result when no agent available
if agent is None:
    return TaskResult(
        task_id=task.id,
        agent_id="none",
        status="failed",
        output="",
        error="No agent available to execute this task. Agent spawn failed."
    )
```

### 2. Added Error Field to AgentSession
```python
# models/agent.py - Added error field to track spawn failures
error: Optional[str] = Field(default=None, description="Error message if session failed")
```

### 3. Fixed Session Error Access
```python
# coordination_engine.py - Removed access to non-existent session.error
self.logger.error(f"Agent spawn failed for {agent_type.value} - status: {session.status}")
```

### 4. Better Error Propagation
- Tasks without agents now get proper TaskResult with error message
- Failed agent spawns are logged with meaningful errors
- The swarm monitor displays actual error instead of "Unknown error"

## Diagnosing Claude Code Issues

Run the diagnostic script to check Claude Code setup:

```bash
python diagnose_claude_auth.py
```

This will check:
1. Claude CLI installation
2. Claude version
3. Authentication status
4. Agent spawn capability

## Common Issues and Solutions

### Claude Not Authenticated
```bash
# Run Claude manually to authenticate
claude

# Complete the browser authentication flow
```

### Claude Not Installed
```bash
npm install -g @anthropic-ai/claude-code
```

### Running in CI/Automated Environment
```bash
# Set automated mode to fail fast instead of hanging
export AGENTIC_AUTOMATED_MODE=true
```

## Testing the Fix

1. Ensure Claude is authenticated:
   ```bash
   claude --version  # Should complete without prompts
   ```

2. Run a test command:
   ```bash
   agentic
   > run the tests
   ```

3. The swarm monitor should now show:
   - Proper agent initialization
   - Meaningful error messages if spawn fails
   - No more "Unknown error" messages

## What Changed

1. **Better Error Messages**: Instead of "Unknown error", you'll see specific reasons like:
   - "No agent available to execute this task. Agent spawn failed."
   - "Claude Code not authenticated. Run 'claude' manually to authenticate."
   - "Claude Code not set up. Run 'claude' manually to complete setup."

2. **Proper Task Results**: Failed tasks now have proper TaskResult objects with error details

3. **No More Crashes**: Fixed AttributeError from accessing non-existent session.error field

4. **Clearer Diagnostics**: Enhanced logging shows exactly why agent spawn failed