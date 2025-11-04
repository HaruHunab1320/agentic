# Single Agent Query Experience Fix

## Issues Addressed

1. **Swarm Monitor Running for Simple Questions**
   - The swarm monitor was showing updates every 2 seconds for single-agent question tasks
   - This created a cluttered experience for simple queries that don't need monitoring

2. **Raw JSON Output Not Being Parsed**
   - When Claude Code hit max turns limit, it returned raw JSON array format
   - This wasn't being properly parsed and displayed to the user

## Solutions Implemented

### 1. Disabled Swarm Monitor for Question Tasks

**File**: `src/agentic/core/chat_interface.py`

Added logic to detect question-type queries and disable the swarm monitor:

```python
# For single-agent question tasks, disable the swarm monitor
is_question_task = (query_analysis.query_type in ["question", "explanation", "analysis"] and 
                   not is_multi_agent)

# Pass enable_monitoring flag to orchestrator
context = {
    'preferred_agent': query_analysis.suggested_agent,
    'enable_monitoring': not is_question_task  # Disable monitor for simple questions
}
```

### 2. Enhanced JSON Array Format Support

**File**: `src/agentic/agents/output_parser.py`

Added support for Claude Code's JSON array format (returned when hitting max turns):

```python
# Handle JSON array format (new Claude Code format when hitting max turns)
if isinstance(data, list):
    # Extract assistant messages and usage info from the array
    assistant_messages = []
    usage_info = None
    
    for msg in data:
        if isinstance(msg, dict):
            if msg.get('type') == 'assistant':
                content = msg.get('content', '')
                if content:
                    assistant_messages.append(content)
            elif msg.get('type') == 'system' and msg.get('subtype') == 'usage':
                usage_info = msg.get('usage', {})
```

**File**: `src/agentic/core/chat_interface.py`

Updated JSON detection to handle both objects and arrays:

```python
# Check if output is JSON (Claude Code format) 
output_stripped = result.output.strip()
if (output_stripped.startswith('{') or output_stripped.startswith('[')) and hasattr(result, 'agent_id') and 'claude' in str(result.agent_id).lower():
```

### 3. Improved Query Type Detection

**File**: `src/agentic/core/query_analyzer.py`

Enhanced question detection to properly classify "what does X do?" type queries:

```python
# Common question patterns that should remain questions
if any(phrase in query for phrase in [
    "tell me what", "can you tell me", "what does", "what is", 
    "how does", "why does", "explain", "describe"
]):
    return "question"
```

## Benefits

1. **Cleaner Experience**: Single-agent questions no longer show the distracting swarm monitor
2. **Proper Output Parsing**: Claude's responses are properly extracted even when hitting turn limits
3. **Better Query Classification**: Questions about functionality are correctly identified as questions

## Testing

The fixes were tested with various query types:
- ✅ Simple questions ("what does X do?") - No monitor, clean output
- ✅ Analysis queries ("find all skipped tests") - No monitor, parsed output
- ✅ Implementation tasks ("create tests") - Monitor shown as expected
- ✅ JSON array output parsing - Correctly extracts assistant messages

## User Experience Improvements

Before:
- Swarm monitor updating every 2 seconds for simple questions
- Raw JSON array shown when Claude hits turn limit
- "What does X do?" classified as implementation task

After:
- Clean, monitor-free experience for questions
- Properly formatted responses from Claude
- Correct query type detection for better routing