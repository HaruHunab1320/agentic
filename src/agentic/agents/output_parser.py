"""
Agent Output Parser

Parses output from different agent types to extract key information
like files created, actions taken, and summaries.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from agentic.utils.logging import LoggerMixin


class AgentOutputParser(LoggerMixin):
    """Base class for parsing agent outputs"""
    
    def parse_output(self, output: str, agent_type: str) -> Dict[str, Any]:
        """Parse agent output based on agent type"""
        if not output:
            return self._empty_result()
        
        # Route to appropriate parser
        if 'claude' in agent_type.lower():
            return self._parse_claude_output(output)
        elif 'aider' in agent_type.lower():
            return self._parse_aider_output(output)
        else:
            return self._parse_generic_output(output)
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty parsing result"""
        return {
            'files_created': [],
            'files_modified': [],
            'actions': [],
            'summary': '',
            'errors': [],
            'metadata': {}
        }
    
    def _parse_claude_output(self, output: str) -> Dict[str, Any]:
        """Parse Claude Code agent output with enhanced extraction"""
        result = self._empty_result()
        
        # Check if it's JSON output (object or array)
        output_stripped = output.strip()
        if output_stripped.startswith('{') or output_stripped.startswith('['):
            try:
                data = json.loads(output_stripped)
                
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
                    
                    # If we found assistant messages, create a summary
                    if assistant_messages:
                        # Combine all assistant messages
                        combined_text = '\n\n'.join(assistant_messages)
                        result['summary'] = combined_text[:500]
                        
                        # Extract any tool uses or actions mentioned in the text
                        for msg in assistant_messages:
                            # Look for common action patterns
                            if 'skipped test' in msg.lower() or 'found test' in msg.lower():
                                result['actions'].append('Found skipped test')
                            if 'need to build' in msg.lower() or 'needs to be built' in msg.lower():
                                result['actions'].append('Identified missing implementation')
                    
                    # Add usage metadata if available
                    if usage_info:
                        result['metadata']['turns'] = usage_info.get('claude_api_calls', 0)
                        result['metadata']['total_tokens'] = usage_info.get('total_tokens', 0)
                        result['metadata']['cache_read_tokens'] = usage_info.get('cache_read_tokens', 0)
                    
                    return result
                
                # Handle regular JSON object format
                if not isinstance(data, dict):
                    # Fallback for unexpected formats
                    result['summary'] = str(data)[:500]
                    return result
                
                # Extract from messages array (new format)
                if 'messages' in data and isinstance(data['messages'], list):
                    thinking_lines = []
                    summary_lines = []
                    
                    for msg in data['messages']:
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', [])
                            
                            # Handle content as list or string
                            if isinstance(content, str):
                                content = [{'type': 'text', 'text': content}]
                            elif not isinstance(content, list):
                                continue
                            
                            for block in content:
                                if isinstance(block, dict):
                                    if block.get('type') == 'tool_use':
                                        self._process_claude_tool_use(block, result)
                                        # Add descriptive action for the tool use
                                        action_desc = self._describe_tool_use(block)
                                        if action_desc:
                                            thinking_lines.append(action_desc)
                                    elif block.get('type') == 'text':
                                        text = block.get('text', '').strip()
                                        if text:
                                            # Capture Claude's thinking process
                                            if any(indicator in text.lower() for indicator in 
                                                   ['i\'ll', 'i will', 'let me', 'looking at', 'analyzing', 
                                                    'checking', 'searching', 'found', 'noticed']):
                                                thinking_lines.append(text[:200])
                                            # Capture final answers/conclusions
                                            else:
                                                summary_lines.append(text)
                                elif isinstance(block, str) and block.strip():
                                    summary_lines.append(block.strip())
                    
                    # Build comprehensive summary
                    if thinking_lines:
                        result['metadata']['thinking_process'] = thinking_lines
                    
                    if summary_lines:
                        # Use the last substantial response as summary
                        for line in reversed(summary_lines):
                            if len(line) > 20:  # Skip very short lines
                                result['summary'] = line[:500]  # Allow longer summaries
                                break
                    
                    # If no good summary found, combine all text
                    if not result['summary'] and (thinking_lines or summary_lines):
                        all_text = thinking_lines + summary_lines
                        result['summary'] = ' '.join(all_text)[:500]
                
                # Legacy format support
                elif 'conversation' in data:
                    for turn in data['conversation']:
                        if turn.get('role') == 'assistant' and 'content' in turn:
                            for content in turn['content']:
                                if content.get('type') == 'tool_use':
                                    self._process_claude_tool_use(content, result)
                                elif content.get('type') == 'text':
                                    text = content.get('text', '')
                                    if text and not result['summary']:
                                        result['summary'] = text.split('\n')[0][:200]
                
                # Extract usage metadata
                if 'usage' in data:
                    result['metadata']['turns'] = data['usage'].get('claude_api_calls', 0)
                    result['metadata']['total_tokens'] = data['usage'].get('total_tokens', 0)
                    result['metadata']['cache_creation_tokens'] = data['usage'].get('cache_creation_tokens', 0)
                    result['metadata']['cache_read_tokens'] = data['usage'].get('cache_read_tokens', 0)
                
                return result
                
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse Claude JSON output")
        
        # Fallback to text parsing
        return self._parse_generic_output(output)
    
    def _process_claude_tool_use(self, content: Dict[str, Any], result: Dict[str, Any]):
        """Process a Claude tool use entry"""
        tool_name = content.get('name', '')
        tool_input = content.get('input', {})
        
        if tool_name == 'Write':
            file_path = tool_input.get('file_path')
            if file_path:
                result['files_created'].append(file_path)
                result['actions'].append(f"Created {Path(file_path).name}")
                
        elif tool_name in ['Edit', 'MultiEdit']:
            file_path = tool_input.get('file_path')
            if file_path:
                result['files_modified'].append(file_path)
                if tool_name == 'MultiEdit':
                    edits = tool_input.get('edits', [])
                    result['actions'].append(f"Modified {Path(file_path).name} ({len(edits)} changes)")
                else:
                    result['actions'].append(f"Modified {Path(file_path).name}")
                
        elif tool_name == 'Bash':
            command = tool_input.get('command', '')
            description = tool_input.get('description', '')
            if command and not command.startswith('cd'):
                # Use description if available, otherwise shorten command
                if description:
                    result['actions'].append(f"Executed: {description}")
                else:
                    # Shorten long commands
                    if len(command) > 50:
                        command = command[:47] + '...'
                    result['actions'].append(f"Executed: {command}")
                
        elif tool_name == 'Read':
            file_path = tool_input.get('file_path')
            if file_path:
                result['actions'].append(f"Analyzed {Path(file_path).name}")
                
        elif tool_name == 'Grep':
            pattern = tool_input.get('pattern', '')
            path = tool_input.get('path', '.')
            result['actions'].append(f"Searched for '{pattern}' in {Path(path).name}")
            
        elif tool_name == 'Glob':
            pattern = tool_input.get('pattern', '')
            result['actions'].append(f"Found files matching '{pattern}'")
            
        elif tool_name == 'LS':
            path = tool_input.get('path', '.')
            result['actions'].append(f"Listed contents of {Path(path).name}")
            
        elif tool_name == 'WebSearch':
            query = tool_input.get('query', '')
            result['actions'].append(f"Searched web for: {query[:50]}..." if len(query) > 50 else f"Searched web for: {query}")
            
        elif tool_name == 'WebFetch':
            url = tool_input.get('url', '')
            result['actions'].append(f"Fetched content from {url}")
            
        elif tool_name == 'TodoRead':
            result['actions'].append("Checked todo list")
            
        elif tool_name == 'TodoWrite':
            todos = tool_input.get('todos', [])
            result['actions'].append(f"Updated todo list ({len(todos)} items)")
    
    def _parse_aider_output(self, output: str) -> Dict[str, Any]:
        """Parse Aider agent output"""
        result = self._empty_result()
        
        lines = output.split('\n')
        
        for line in lines:
            # Look for file operations
            if 'Added' in line and ' to the chat' in line:
                # Extract filename from "Added path/to/file.py to the chat"
                match = re.search(r'Added (.+?) to the chat', line)
                if match:
                    file_path = match.group(1)
                    result['actions'].append(f"Added {Path(file_path).name} to context")
            
            elif 'Edited' in line or 'Updated' in line:
                # Look for edited files
                parts = line.split()
                for part in parts:
                    if '.' in part and '/' in part:  # Likely a file path
                        result['files_modified'].append(part)
                        result['actions'].append(f"Modified {Path(part).name}")
                        break
            
            elif 'Created' in line or 'Wrote' in line:
                # Look for created files
                parts = line.split()
                for part in parts:
                    if '.' in part and '/' in part:  # Likely a file path
                        result['files_created'].append(part)
                        result['actions'].append(f"Created {Path(part).name}")
                        break
            
            # Extract summary from completion message
            elif 'successfully' in line.lower() or 'completed' in line.lower():
                if not result['summary']:
                    result['summary'] = line.strip()[:200]
            
            # Check for errors
            elif 'error' in line.lower() or 'failed' in line.lower():
                result['errors'].append(line.strip())
        
        # Extract commit info if present
        commit_match = re.search(r'Commit ([a-f0-9]+) ', output)
        if commit_match:
            result['metadata']['commit'] = commit_match.group(1)
            result['actions'].append(f"Committed changes: {commit_match.group(1)[:7]}")
        
        return result
    
    def _parse_generic_output(self, output: str) -> Dict[str, Any]:
        """Generic output parser for unknown agent types"""
        result = self._empty_result()
        
        lines = output.split('\n')
        
        # Look for common patterns
        for line in lines:
            line_lower = line.lower()
            
            # File operations
            if any(word in line_lower for word in ['created', 'wrote', 'generated']):
                # Try to extract file path
                match = re.search(r'["\']?([/\w\-_.]+\.\w+)["\']?', line)
                if match:
                    file_path = match.group(1)
                    result['files_created'].append(file_path)
                    result['actions'].append(f"Created {Path(file_path).name}")
            
            elif any(word in line_lower for word in ['modified', 'updated', 'changed']):
                # Try to extract file path
                match = re.search(r'["\']?([/\w\-_.]+\.\w+)["\']?', line)
                if match:
                    file_path = match.group(1)
                    result['files_modified'].append(file_path)
                    result['actions'].append(f"Modified {Path(file_path).name}")
            
            # Commands
            elif 'executed' in line_lower or 'ran' in line_lower:
                result['actions'].append(line.strip()[:80])
            
            # Errors
            elif 'error' in line_lower or 'failed' in line_lower:
                result['errors'].append(line.strip())
            
            # Summary candidates
            elif any(word in line_lower for word in ['completed', 'finished', 'done']):
                if not result['summary']:
                    result['summary'] = line.strip()[:200]
        
        # If no summary found, use first non-empty line
        if not result['summary']:
            for line in lines:
                if line.strip():
                    result['summary'] = line.strip()[:200]
                    break
        
        return result
    
    def _describe_tool_use(self, tool_use: Dict[str, Any]) -> Optional[str]:
        """Generate a natural language description of what Claude is doing"""
        tool_name = tool_use.get('name', '')
        tool_input = tool_use.get('input', {})
        
        descriptions = {
            'Read': lambda: f"Reading {Path(tool_input.get('file_path', '')).name} to understand the code",
            'Write': lambda: f"Creating {Path(tool_input.get('file_path', '')).name}",
            'Edit': lambda: f"Updating {Path(tool_input.get('file_path', '')).name}",
            'MultiEdit': lambda: f"Making {len(tool_input.get('edits', []))} changes to {Path(tool_input.get('file_path', '')).name}",
            'Bash': lambda: tool_input.get('description') or f"Running: {tool_input.get('command', '')[:50]}",
            'Grep': lambda: f"Searching for '{tool_input.get('pattern', '')}' in the codebase",
            'Glob': lambda: f"Looking for files matching '{tool_input.get('pattern', '')}'",
            'LS': lambda: f"Exploring directory structure",
            'WebSearch': lambda: f"Searching the web for: {tool_input.get('query', '')[:50]}",
            'WebFetch': lambda: f"Fetching information from {tool_input.get('url', '')}",
            'TodoRead': lambda: "Checking the task list",
            'TodoWrite': lambda: f"Updating the task list with {len(tool_input.get('todos', []))} items"
        }
        
        if tool_name in descriptions:
            try:
                return descriptions[tool_name]()
            except:
                return None
        return None
    
    def extract_key_actions(self, output: str, agent_type: str, max_actions: int = 5) -> List[str]:
        """Extract only the key actions from agent output"""
        parsed = self.parse_output(output, agent_type)
        return parsed['actions'][:max_actions]
    
    def extract_file_changes(self, output: str, agent_type: str) -> Tuple[List[str], List[str]]:
        """Extract files created and modified"""
        parsed = self.parse_output(output, agent_type)
        return parsed['files_created'], parsed['files_modified']