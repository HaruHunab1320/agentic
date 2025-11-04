"""
Execution Summary Formatter for Multi-Agent Results

Provides clean, unified summaries of multi-agent task execution results
instead of raw JSON dumps.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic.models.task import TaskResult
from agentic.core.coordination_engine import ExecutionResult
from agentic.utils.logging import LoggerMixin


@dataclass
class AgentSummary:
    """Summary of a single agent's execution"""
    agent_id: str
    agent_name: str
    agent_type: str
    role: str
    tasks_completed: int
    files_created: List[str]
    files_modified: List[str]
    key_actions: List[str]
    errors: List[str]
    duration: float


class ExecutionSummaryFormatter(LoggerMixin):
    """Formats execution results into clean, readable summaries"""
    
    def __init__(self):
        super().__init__()
        self.console = Console()
    
    def format_execution_summary(self, result: ExecutionResult, agent_sessions: Dict[str, Any]) -> None:
        """Format and display a comprehensive execution summary"""
        
        # Build agent summaries
        agent_summaries = self._build_agent_summaries(result, agent_sessions)
        
        # Display overall status
        self._display_overall_status(result)
        
        # Display per-agent summaries
        self._display_agent_summaries(agent_summaries)
        
        # Display file changes summary
        self._display_file_changes(agent_summaries)
        
        # Display agent responses if no files were created/modified
        # This ensures we see the actual answer to questions
        total_files = sum(len(s.files_created) + len(s.files_modified) for s in agent_summaries)
        if total_files == 0 and result.task_results:
            self._display_agent_responses(result)
        
        # Display any errors
        if result.failed_tasks:
            self._display_errors(result)
        
        # Display verification status if available
        if hasattr(result, 'verification_status') and result.verification_status:
            self._display_verification_status(result.verification_status)
    
    def _build_agent_summaries(self, result: ExecutionResult, agent_sessions: Dict[str, Any]) -> List[AgentSummary]:
        """Build summaries for each agent that participated"""
        agent_summaries = {}
        
        for task_id, task_result in result.task_results.items():
            agent_id = task_result.agent_id
            
            if agent_id not in agent_summaries:
                # Get agent info from sessions
                agent_info = agent_sessions.get(agent_id, {})
                agent_config = agent_info.get('agent_config', {})
                
                agent_summaries[agent_id] = AgentSummary(
                    agent_id=agent_id,
                    agent_name=agent_config.get('name', 'Unknown'),
                    agent_type=agent_config.get('agent_type', {}).get('value', 'unknown'),
                    role=self._extract_role_from_task(task_result),
                    tasks_completed=0,
                    files_created=[],
                    files_modified=[],
                    key_actions=[],
                    errors=[],
                    duration=0.0
                )
            
            summary = agent_summaries[agent_id]
            
            # Update task count
            if task_result.status == "completed":
                summary.tasks_completed += 1
            
            # Extract actions and files from output
            actions, files = self._parse_agent_output(task_result)
            summary.key_actions.extend(actions)
            summary.files_created.extend(files.get('created', []))
            summary.files_modified.extend(files.get('modified', []))
            
            # Collect errors
            if task_result.error:
                summary.errors.append(task_result.error)
        
        return list(agent_summaries.values())
    
    def _parse_agent_output(self, task_result: TaskResult) -> tuple[List[str], Dict[str, List[str]]]:
        """Parse agent output to extract key actions and file changes"""
        # Use the dedicated output parser
        from agentic.agents.output_parser import AgentOutputParser
        
        parser = AgentOutputParser()
        
        # Determine agent type from agent_id
        agent_type = 'unknown'
        if 'claude' in task_result.agent_id.lower():
            agent_type = 'claude_code'
        elif 'aider' in task_result.agent_id.lower():
            agent_type = 'aider'
        
        # Parse the output
        parsed = parser.parse_output(task_result.output or '', agent_type)
        
        # Return actions and files
        files = {
            'created': parsed['files_created'],
            'modified': parsed['files_modified']
        }
        
        actions = parsed['actions']
        
        # If no actions found, at least note the task
        if not actions:
            actions.append("Completed task successfully")
        
        return actions[:5], files  # Limit to 5 key actions
    
    def _extract_role_from_task(self, task_result: TaskResult) -> str:
        """Extract the role from task result"""
        agent_id_lower = task_result.agent_id.lower()
        
        # Check for specific role indicators in agent ID
        if 'frontend' in agent_id_lower:
            return "Frontend Developer"
        elif 'backend' in agent_id_lower:
            return "Backend Developer"
        elif 'test' in agent_id_lower or 'qa' in agent_id_lower:
            return "QA Engineer"
        elif 'claude' in agent_id_lower:
            return "System Architect"
        elif 'architect' in agent_id_lower:
            return "System Architect"
        elif 'analyst' in agent_id_lower:
            return "Code Analyst"
        elif 'security' in agent_id_lower:
            return "Security Analyst"
        elif 'performance' in agent_id_lower:
            return "Performance Analyst"
        elif 'devops' in agent_id_lower:
            return "DevOps Engineer"
        else:
            return "Developer"
    
    def _display_overall_status(self, result: ExecutionResult):
        """Display overall execution status"""
        status_color = "green" if result.status == "completed" else "red"
        status_icon = "✅" if result.status == "completed" else "❌"
        
        total_tasks = len(result.completed_tasks) + len(result.failed_tasks)
        
        summary_text = f"""
[bold {status_color}]{status_icon} Execution {result.status.title()}[/bold {status_color}]

Tasks: {len(result.completed_tasks)}/{total_tasks} completed
Duration: {result.total_duration:.1f}s
        """
        
        self.console.print(Panel(summary_text.strip(), title="Execution Summary", border_style=status_color))
    
    def _display_agent_summaries(self, summaries: List[AgentSummary]):
        """Display per-agent execution summaries"""
        if not summaries:
            return
        
        table = Table(title="Agent Activity Summary", show_header=True, header_style="bold magenta")
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Role", style="green")
        table.add_column("Tasks", justify="center")
        table.add_column("Key Actions", style="yellow")
        table.add_column("Files", style="blue")
        
        for summary in summaries:
            # Format key actions
            actions_text = "\n".join(f"• {action}" for action in summary.key_actions[:3])
            if len(summary.key_actions) > 3:
                actions_text += f"\n• ... and {len(summary.key_actions) - 3} more"
            
            # Format file counts
            file_count = len(summary.files_created) + len(summary.files_modified)
            files_text = f"{file_count} files"
            if summary.files_created:
                files_text += f"\n({len(summary.files_created)} new)"
            
            # Add row
            table.add_row(
                summary.agent_name,
                summary.role,
                str(summary.tasks_completed),
                actions_text or "No actions recorded",
                files_text
            )
        
        self.console.print(table)
    
    def _display_file_changes(self, summaries: List[AgentSummary]):
        """Display summary of all file changes"""
        all_created = []
        all_modified = []
        
        for summary in summaries:
            all_created.extend(summary.files_created)
            all_modified.extend(summary.files_modified)
        
        # Remove duplicates
        all_created = list(set(all_created))
        all_modified = list(set(all_modified))
        
        if not all_created and not all_modified:
            return
        
        file_summary = ""
        
        if all_created:
            file_summary += f"[bold green]Files Created ({len(all_created)}):[/bold green]\n"
            for file_path in sorted(all_created)[:10]:
                file_summary += f"  + {Path(file_path).name}\n"
            if len(all_created) > 10:
                file_summary += f"  ... and {len(all_created) - 10} more\n"
        
        if all_modified:
            if file_summary:
                file_summary += "\n"
            file_summary += f"[bold yellow]Files Modified ({len(all_modified)}):[/bold yellow]\n"
            for file_path in sorted(all_modified)[:10]:
                file_summary += f"  ~ {Path(file_path).name}\n"
            if len(all_modified) > 10:
                file_summary += f"  ... and {len(all_modified) - 10} more\n"
        
        self.console.print(Panel(file_summary.strip(), title="File Changes", border_style="blue"))
    
    def _display_errors(self, result: ExecutionResult):
        """Display any errors that occurred"""
        error_text = "[bold red]Errors Encountered:[/bold red]\n\n"
        
        for task_id in result.failed_tasks:
            task_result = result.task_results.get(task_id)
            if task_result and task_result.error:
                error_text += f"• Task {task_id[:8]}: {task_result.error}\n"
        
        self.console.print(Panel(error_text.strip(), title="Errors", border_style="red"))
    
    def _display_verification_status(self, status: str):
        """Display verification status if available"""
        if status == "passed":
            self.console.print("\n[bold green]✅ Verification: All tests passed[/bold green]")
        elif status == "failed_after_retries":
            self.console.print("\n[bold yellow]⚠️ Verification: Some issues remain after fixes[/bold yellow]")
    
    def _display_claude_response(self, task_result: TaskResult):
        """Display Claude Code response with detailed activity log"""
        output = task_result.output.strip()
        
        # Try to parse as JSON
        if output.startswith('{'):
            try:
                import json
                data = json.loads(output)
                
                # Use the output parser to extract information
                from agentic.agents.output_parser import AgentOutputParser
                parser = AgentOutputParser()
                parsed = parser.parse_output(output, 'claude_code')
                
                # Display Claude's thinking process if available
                thinking_process = parsed.get('metadata', {}).get('thinking_process', [])
                if thinking_process:
                    self.console.print("\n[bold cyan]Claude's Process:[/bold cyan]")
                    for thought in thinking_process[:5]:  # Show first 5 thoughts
                        self.console.print(f"  💭 {thought}")
                    if len(thinking_process) > 5:
                        self.console.print(f"  ... and {len(thinking_process) - 5} more steps")
                
                # Display actions taken
                if parsed['actions']:
                    self.console.print("\n[bold yellow]Actions Taken:[/bold yellow]")
                    for action in parsed['actions'][:10]:  # Show first 10 actions
                        self.console.print(f"  ✓ {action}")
                    if len(parsed['actions']) > 10:
                        self.console.print(f"  ... and {len(parsed['actions']) - 10} more actions")
                
                # Display the final answer/summary
                if parsed['summary']:
                    self.console.print("\n[bold green]Answer:[/bold green]")
                    # Format the summary nicely
                    summary_lines = parsed['summary'].split('\n')
                    for line in summary_lines:
                        if line.strip():
                            self.console.print(f"  {line.strip()}")
                
                # Display any errors
                if parsed['errors']:
                    self.console.print("\n[bold red]Errors:[/bold red]")
                    for error in parsed['errors']:
                        self.console.print(f"  ❌ {error}")
                
                # Display usage statistics
                if 'usage' in data:
                    usage = data['usage']
                    self.console.print("\n[dim]Statistics:[/dim]")
                    if 'claude_api_calls' in usage:
                        self.console.print(f"  [dim]API Calls: {usage['claude_api_calls']}[/dim]")
                    if 'total_tokens' in usage:
                        self.console.print(f"  [dim]Total Tokens: {usage['total_tokens']:,}[/dim]")
                    if 'cache_read_tokens' in usage and usage['cache_read_tokens'] > 0:
                        self.console.print(f"  [dim]Cache Hits: {usage['cache_read_tokens']:,} tokens[/dim]")
                
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to text display
                self.console.print(f"\n[cyan]Response from Claude:[/cyan]")
                self.console.print(output)
        else:
            # Not JSON, just display as text
            self.console.print(f"\n[cyan]Response from Claude:[/cyan]")
            self.console.print(output)
    
    def _display_agent_responses(self, result: ExecutionResult):
        """Display the actual responses from agents when they're answering questions"""
        self.console.print(Panel("[bold]Agent Responses[/bold]", border_style="blue"))
        
        for task_id, task_result in result.task_results.items():
            if task_result.status == "completed" and task_result.output:
                # Try to extract meaningful content from the output
                output = task_result.output.strip()
                
                # For Claude agents, parse JSON and show detailed activity
                if "claude" in task_result.agent_id.lower():
                    self._display_claude_response(task_result)
                # For Aider agents, look for the actual response
                elif "aider" in task_result.agent_id.lower():
                    # Look for lines that aren't just status messages
                    lines = output.split('\n')
                    meaningful_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Skip common Aider status messages
                        if any(skip in line.lower() for skip in [
                            'added', 'to the chat', 'tokens:', 'cost:', 
                            'git working', 'cur working', 'repo root'
                        ]):
                            continue
                        # Keep lines that look like actual content
                        if line and not line.startswith('>'):
                            meaningful_lines.append(line)
                    
                    if meaningful_lines:
                        response = '\n'.join(meaningful_lines)
                        self.console.print(f"\n[cyan]Response:[/cyan]")
                        self.console.print(response)
                    else:
                        # If no meaningful lines found, show the whole output
                        self.console.print(f"\n[cyan]Full Output:[/cyan]")
                        self.console.print(output)
                
                # For other agents, just show the output
                else:
                    self.console.print(f"\n[cyan]Response from {task_result.agent_id}:[/cyan]")
                    self.console.print(output)