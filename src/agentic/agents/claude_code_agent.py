"""
Claude Code Agent - Enhanced CLI Integration

Integrates with the Claude Code CLI tool for sophisticated coding tasks.
Leverages Claude Code's full feature set including memory, sessions, and extended thinking.
"""
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from agentic.models.agent import Agent, AgentConfig, AgentCapability, AgentType, AgentSession
from agentic.models.task import Task, TaskResult


class ClaudeCodeAgent(Agent):
    """Enhanced agent that leverages Claude Code CLI's full feature set."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.process: Optional[subprocess.Popen] = None
        self.temp_files: List[Path] = []
        
        # Enhanced Claude Code settings
        self.claude_model = config.ai_model_config.get('model', 'sonnet')
        self.session_id: Optional[str] = None
        self.memory_initialized = False
        
        # Enhanced tool permissions based on task types
        self.tool_presets = {
            "coding": ["Edit", "Write", "Bash(git *)", "Bash(find *)", "Bash(grep *)"],
            "git": ["Bash(git *)", "Edit", "Write"],
            "analysis": ["Bash(find *)", "Bash(grep *)", "Bash(cat *)", "Bash(ls *)"],
            "testing": ["Bash(pytest *)", "Bash(npm test)", "Bash(cargo test)", "Edit", "Write"],
            "refactoring": ["Edit", "Write", "Bash(git *)", "Bash(grep *)", "Bash(find *)"],
            "debugging": ["Edit", "Bash(gdb *)", "Bash(lldb *)", "Bash(cargo check)", "Bash(npm run *)"],
            "documentation": ["Edit", "Write", "Bash(find *)", "Bash(grep *)"]
        }
        
        self.project_root = config.workspace_path
    
    def get_capabilities(self) -> AgentCapability:
        """Return enhanced agent capabilities."""
        return AgentCapability(
            agent_type=AgentType.CLAUDE_CODE,
            specializations=[
                "coding", "refactoring", "analysis", "debugging", 
                "code_review", "architecture", "documentation", "git_operations",
                "testing", "project_management", "extended_thinking"
            ],
            supported_languages=[
                "python", "javascript", "typescript", "rust", "go",
                "java", "cpp", "c", "html", "css", "sql", "bash", "yaml", "json"
            ],
            max_context_tokens=200000,  # Claude has large context
            concurrent_tasks=1,  # Claude Code works best with sequential tasks
            reasoning_capability=True,
            file_editing_capability=True,
            code_execution_capability=True,
            memory_capability=True,  # NEW: Memory support
            session_persistence=True,  # NEW: Session persistence
            git_integration=True  # NEW: Git integration
        )
    
    async def health_check(self) -> bool:
        """Check if Claude Code CLI is healthy and responsive."""
        try:
            # Test basic claude command
            process = await asyncio.create_subprocess_exec(
                "claude", "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.debug("Claude Code CLI health check passed")
                return True
            else:
                self.logger.error(f"Claude Code CLI health check failed: {stderr.decode()}")
                return False
                
        except FileNotFoundError:
            self.logger.error("Claude Code CLI not found - install with: npm install -g @anthropic-ai/claude-code")
            return False
        except Exception as e:
            self.logger.error(f"Claude Code CLI health check error: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the enhanced Claude Code agent session."""
        try:
            # Check health first
            if not await self.health_check():
                return False
            
            # Initialize project memory if needed
            await self._ensure_memory_setup()
            
            # Create session
            self.session = AgentSession(
                agent_config=self.config,
                workspace=self.project_root,
                status="starting"
            )
            
            # Change to project directory
            os.chdir(self.project_root)
            
            # Generate session ID for persistence
            self.session_id = f"agentic_{uuid.uuid4().hex[:8]}"
            
            self.session.mark_active()
            self.logger.info(f"Enhanced Claude Code agent {self.config.name} started with session {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Claude Code agent: {e}")
            if self.session:
                self.session.mark_error(str(e))
            return False

    async def _ensure_memory_setup(self) -> None:
        """Ensure CLAUDE.md memory files are properly set up."""
        try:
            project_memory = self.project_root / "CLAUDE.md"
            
            if not project_memory.exists():
                self.logger.info("Creating CLAUDE.md with default template")
                
                # Create sensible default that works for any project
                memory_content = f"""# {self.project_root.name} - Claude Code Memory

> **Note**: This file was auto-generated by Agentic's Claude Code agent.
> You can customize it for your project. Add it to .gitignore to keep preferences local.
> See examples/claude_memory_template.md for a comprehensive template.

## Project Overview
This project uses Agentic's Claude Code integration for AI-powered development tasks.

## Basic Guidelines
- **Code Quality**: Write clean, maintainable code
- **Type Safety**: Use type hints where applicable
- **Documentation**: Document complex logic and public APIs
- **Testing**: Include tests for new functionality
- **Consistency**: Follow existing patterns in the codebase

## AI Assistant Instructions
- Always consider the full project context before making changes
- Follow the existing code style and conventions
- Prefer safe, incremental improvements
- Explain your reasoning for significant changes

## Personal Notes
<!-- Add your personal preferences and project-specific instructions here -->
"""
                
                project_memory.write_text(memory_content)
                self.logger.info(f"Created default memory at {project_memory}")
                self.logger.info("💡 Tip: Customize CLAUDE.md for your project and add it to .gitignore")
                self.logger.info("💡 See examples/claude_memory_template.md for a comprehensive template")
            else:
                self.logger.debug(f"Memory file already exists at {project_memory}")
            
            self.memory_initialized = True
            
        except Exception as e:
            self.logger.warning(f"Failed to setup memory: {e}")
    
    async def stop(self) -> bool:
        """Stop the Claude Code agent session."""
        try:
            # Cleanup any running processes
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            
            # Cleanup temporary files
            for temp_file in self.temp_files:
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            
            self.temp_files.clear()
            
            if self.session:
                self.session.status = "stopped"
            
            self.logger.info(f"Claude Code agent {self.config.name} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping Claude Code agent: {e}")
            return False
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task using Claude Code with enhanced CLI features"""
        try:
            # Determine optimal execution mode based on task characteristics
            execution_mode = self._determine_execution_mode(task)
            
            # Build advanced Claude Code command
            cmd = self._build_enhanced_claude_command(task, execution_mode)
            
            self.logger.info(f"Executing {execution_mode} task: {task.command[:50]}...")
            self.logger.debug(f"Running command: {' '.join(cmd[:5])}...")  # Log first 5 args for security
            
            # Set up environment with API keys
            env = os.environ.copy()
            self._set_environment_variables(env)
            
            # Execute Claude Code with appropriate handling
            result = await self._execute_claude_with_advanced_features(cmd, env, task, execution_mode)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                status="failed",
                error=str(e),
                agent_id=self.session.id if self.session else "unknown"
            )
    
    def _determine_execution_mode(self, task: Task) -> str:
        """Determine optimal execution mode based on task characteristics"""
        command_lower = task.command.lower()
        
        # Quick analysis tasks → print mode for fast results
        quick_tasks = [
            'explain', 'analyze', 'what does', 'how does', 'describe',
            'summarize', 'review', 'debug', 'find', 'check'
        ]
        
        # Interactive tasks → interactive mode for iterative work
        interactive_tasks = [
            'help me', 'work with me', 'iterative', 'step by step',
            'refactor', 'improve', 'optimize'
        ]
        
        # Determine mode
        if any(task in command_lower for task in quick_tasks):
            return "print"  # One-shot mode for quick results
        elif any(task in command_lower for task in interactive_tasks):
            return "interactive"  # Interactive mode for complex work
        else:
            # Default to print mode for most automated tasks
            return "print"
    
    def _build_enhanced_claude_command(self, task: Task, execution_mode: str) -> List[str]:
        """Build enhanced Claude Code command with advanced CLI features"""
        cmd = ["claude"]
        
        # Set model if specified in agent config
        if hasattr(self, 'ai_model_config') and self.ai_model_config.get('model'):
            model = self.ai_model_config['model']
            # Map our model names to Claude-compatible names
            if model in ['claude', 'claude-sonnet', 'sonnet']:
                cmd.extend(["--model", "sonnet"])
            elif model == 'opus':
                cmd.extend(["--model", "opus"])
            # For other models, let Claude use its default
        
        if execution_mode == "print":
            # Print mode for quick, automated tasks
            cmd.extend(["-p"])
            
            # Use JSON output format for better parsing in automation
            cmd.extend(["--output-format", "json"])
            
            # Limit turns for automated execution
            cmd.extend(["--max-turns", "3"])
            
            # Enable verbose logging for debugging
            cmd.extend(["--verbose"])
            
        elif execution_mode == "interactive":
            # Interactive mode - let Claude handle session management
            # Add continue flag if this is part of an ongoing session
            if hasattr(self, 'session_id') and self.session_id:
                cmd.extend(["--continue"])
        
        # Add the task as the query
        enhanced_prompt = self._build_enhanced_prompt(task)
        cmd.append(enhanced_prompt)
        
        return cmd
    
    def _build_enhanced_prompt(self, task: Task) -> str:
        """Build enhanced prompt with agent context and specialized instructions"""
        prompt_parts = []
        
        # Agent identity and specialization
        prompt_parts.append(f"You are a Claude Code agent specialized in: {', '.join(self.focus_areas)}.")
        
        # Add context about the current workspace
        if hasattr(self, 'workspace_path'):
            prompt_parts.append(f"Working in project directory: {self.workspace_path}")
        
        # Add agent-specific context based on focus areas
        context_additions = []
        if "analysis" in self.focus_areas:
            context_additions.append("Provide detailed analysis with specific examples and code references.")
        if "debugging" in self.focus_areas:
            context_additions.append("Focus on identifying root causes and providing actionable solutions.")
        if "optimization" in self.focus_areas:
            context_additions.append("Suggest performance improvements and best practices.")
        if "code_review" in self.focus_areas:
            context_additions.append("Evaluate code quality, security, and maintainability.")
            
        if context_additions:
            prompt_parts.append("Additional context: " + " ".join(context_additions))
        
        # Add the main task
        prompt_parts.append(f"Task: {task.command}")
        
        # For analysis tasks, add specific instructions
        if any(keyword in task.command.lower() for keyword in ['analyze', 'explain', 'review']):
            prompt_parts.append("""
Please provide:
1. Clear explanation of what you found
2. Specific code examples with line references when applicable  
3. Actionable recommendations
4. Any potential issues or improvements identified""")
        
        return "\n\n".join(prompt_parts)
    
    async def _execute_claude_with_advanced_features(self, cmd: List[str], env: dict, task: Task, execution_mode: str) -> TaskResult:
        """Execute Claude Code with advanced features and proper output handling"""
        try:
            # Execute the command with appropriate timeout
            timeout = 300 if execution_mode == "interactive" else 120  # 5 min for interactive, 2 min for print
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.workspace_path)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return TaskResult(
                    task_id=task.id,
                    status="failed",
                    error=f"Task timed out after {timeout} seconds",
                    agent_id=self.session.id if self.session else "unknown"  
                )
            
            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')
            
            # Parse output based on execution mode
            success, parsed_output = self._parse_claude_output(
                output, error_output, process.returncode, execution_mode
            )
            
            if success:
                self.logger.info("Task completed successfully")
                return TaskResult(
                    task_id=task.id,
                    status="completed",
                    output=parsed_output,
                    agent_id=self.session.id if self.session else "unknown"
                )
            else:
                self.logger.error(f"Task failed: {parsed_output}")
                return TaskResult(
                    task_id=task.id,
                    status="failed",
                    error=parsed_output,
                    agent_id=self.session.id if self.session else "unknown"
                )
                
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return TaskResult(
                task_id=task.id,
                status="failed",
                error=str(e),
                agent_id=self.session.id if self.session else "unknown"
            )
    
    def _parse_claude_output(self, stdout: str, stderr: str, return_code: int, execution_mode: str) -> tuple[bool, str]:
        """Parse Claude Code output with mode-specific handling"""
        
        # Check for obvious errors first
        if return_code != 0:
            if stderr:
                return False, stderr
            else:
                return False, "Command failed with non-zero exit code"
        
        # Handle JSON output format (used in print mode)
        if execution_mode == "print" and stdout.strip().startswith('{'):
            try:
                import json
                output_data = json.loads(stdout)
                
                # Extract the actual response content
                if isinstance(output_data, dict):
                    if 'content' in output_data:
                        return True, output_data['content']
                    elif 'response' in output_data:
                        return True, output_data['response']
                    elif 'message' in output_data:
                        return True, output_data['message']
                    else:
                        # Return formatted JSON as fallback
                        return True, json.dumps(output_data, indent=2)
                else:
                    return True, str(output_data)
                    
            except json.JSONDecodeError:
                # Fall through to text parsing
                pass
        
        # Handle text output (interactive mode or fallback)
        if stdout.strip():
            # Clean up the output by removing excessive whitespace and formatting
            lines = stdout.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Skip empty lines and CLI formatting
                if line.strip() and not line.startswith('Aider ') and not line.startswith('───'):
                    cleaned_lines.append(line.strip())
            
            if cleaned_lines:
                return True, '\n'.join(cleaned_lines)
            else:
                return True, stdout.strip()
        
        # If we get here, there's no meaningful output
        if stderr:
            return False, stderr
        else:
            return False, "No output received from Claude Code"
    
    def get_supported_tasks(self) -> List[str]:
        """Get enhanced list of supported task types."""
        return [
            "analysis", "refactoring", "debugging", "code_review",
            "documentation", "testing", "optimization", "architecture",
            "git_operations", "project_setup", "extended_thinking",
            "memory_management", "session_planning"
        ]
    
    async def add_memory(self, memory_content: str, memory_type: str = "project") -> bool:
        """Add content to Claude Code memory."""
        try:
            if memory_type == "project":
                memory_file = self.project_root / "CLAUDE.md"
            else:
                memory_file = Path.home() / ".claude" / "CLAUDE.md"
            
            if not memory_file.parent.exists():
                memory_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to existing memory
            current_content = memory_file.read_text() if memory_file.exists() else ""
            updated_content = f"{current_content}\n\n## Added Memory\n{memory_content}"
            
            memory_file.write_text(updated_content)
            self.logger.info(f"Added memory to {memory_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add memory: {e}")
            return False

    async def continue_session(self, new_task: Task) -> TaskResult:
        """Continue the previous session with a new task."""
        if not self.session_id:
            return await self.execute_task(new_task)
        
        # Build command with session continuation
        cmd = ["claude", "--continue", "--print", "--output-format", "json"]
        
        if self.claude_model:
            cmd.extend(["--model", self.claude_model])
        
        prompt = f"Continuing session {self.session_id}: {new_task.command}"
        cmd.append(prompt)
        
        # Execute with session continuation
        return await self._execute_with_command(cmd, new_task)

    async def _execute_with_command(self, cmd: List[str], task: Task) -> TaskResult:
        """Execute a specific command and return TaskResult."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                result = await self._parse_claude_output(stdout.decode('utf-8').strip(), task)
                return TaskResult(
                    task_id=task.id,
                    agent_id=self.config.name,
                    status="completed",
                    output=result["output"],
                    metadata=result
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    agent_id=self.config.name,
                    status="failed",
                    output=stderr.decode('utf-8').strip(),
                    error=stderr.decode('utf-8').strip()
                )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                agent_id=self.config.name,
                status="failed",
                output=str(e),
                error=str(e)
            )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get enhanced agent status."""
        if not self.session:
            return {"status": "not_started"}
        
        return {
            "status": self.session.status,
            "current_task": self.session.current_task,
            "workspace": str(self.session.workspace),
            "session_id": self.session_id,
            "memory_initialized": self.memory_initialized,
            "created_at": self.session.created_at.isoformat(),
            "last_activity": self.session.last_activity.isoformat() if self.session.last_activity else None,
            "capabilities": self.get_capabilities().model_dump(),
            "health": await self.health_check(),
            "enhanced_features": {
                "memory_support": True,
                "session_persistence": True,
                "extended_thinking": True,
                "json_output": True,
                "git_integration": True
            }
        }