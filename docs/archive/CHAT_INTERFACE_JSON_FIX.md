# Chat Interface JSON Output Fix

## Problem
Claude's JSON array output was being printed raw after the parsed content was already displayed, resulting in duplicate output. Additionally, analysis tasks were limited to only 3 turns and 30 seconds timeout, which wasn't enough for thorough code exploration.

## Solution

### 1. Fixed Duplicate JSON Output (chat_interface.py)
- Modified the output handling logic to prevent Claude's JSON output from being processed twice
- Added a check to skip the generic JSON pretty-printing when Claude's output has already been parsed by the enhanced parser
- The fix ensures that after the enhanced parser extracts and displays Claude's thinking process, actions, and summary, the raw JSON is not printed again

### 2. Increased Analysis Task Limits (claude_code_agent.py)
- Increased `--max-turns` from 3 to 10 for analysis tasks (commands containing 'tell me', 'what', 'explain', 'analyze', 'find')
- Increased timeout from 30 seconds to 120 seconds (2 minutes) for analysis tasks
- This allows Claude Code to perform more thorough exploration when analyzing code

## Files Modified
1. `/Users/jakobgrant/Workspaces/agentic/src/agentic/core/chat_interface.py`
   - Lines 468-491: Added conditional logic to skip JSON printing for already-parsed Claude output

2. `/Users/jakobgrant/Workspaces/agentic/src/agentic/agents/claude_code_agent.py`
   - Line 752: Changed max_turns from "3" to "10" for analysis tasks
   - Line 1087: Changed timeout from 30 to 120 seconds for analysis tasks

## Testing
To test the fix, run an analysis command in the Agentic chat interface:
```bash
agentic chat
# Then type: analyze the orchestrator.py file and tell me about its main functions
```

The output should now show:
1. "✅ Task completed successfully!"
2. Claude's Process (thinking steps)
3. Actions taken
4. The formatted answer
5. Statistics (turns, tokens, cache hit)

Without the raw JSON array being printed afterwards.

## Benefits
- Cleaner, more readable output in the chat interface
- Claude can now perform deeper analysis with up to 10 turns
- More time allowed for complex code exploration tasks
- Better user experience with properly formatted responses