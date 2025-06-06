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
            
            # Handle Claude Code authentication/trust if needed
            if not await self._ensure_claude_authentication():
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
            
            # Generate session ID for persistence (Claude Code expects full UUID format)
            self.session_id = str(uuid.uuid4())
            
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
    
    async def _ensure_claude_authentication(self) -> bool:
        """Ensure Claude Code is authenticated and trusted for this workspace"""
        try:
            # Always handle potential first-time setup by running a quick test
            # This will either complete immediately or trigger the setup wizard
            await self._handle_claude_first_time_setup()
            
            # Test if Claude Code can access the workspace without prompting
            test_cmd = [
                "claude", 
                "--version"  # Simple command that shouldn't trigger trust dialog
            ]
            
            process = await asyncio.create_subprocess_exec(
                *test_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            
            if process.returncode == 0:
                self.logger.debug("Claude Code authentication check passed")
                return True
            
            # If version check fails, try to handle trust dialog automatically
            self.logger.info("Claude Code may need authentication - attempting automatic trust setup")
            
            # Try running a simple Claude command that might trigger the trust dialog
            auth_cmd = [
                "claude",
                "--help"
            ]
            
            # For automation, we need to provide input to automatically trust
            process = await asyncio.create_subprocess_exec(
                *auth_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            # Send "1" and Enter to select "Yes, proceed" if prompted
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=b"1\n"), 
                    timeout=30
                )
                
                if process.returncode == 0:
                    self.logger.info("Claude Code authentication successful")
                    return True
                else:
                    self.logger.warning(f"Claude Code authentication may have failed: {stderr.decode()}")
                    # Don't fail completely - try to continue
                    return True
                    
            except asyncio.TimeoutError:
                process.kill()
                self.logger.warning("Claude Code authentication timed out - continuing anyway")
                return True
                
        except Exception as e:
            self.logger.warning(f"Claude Code authentication check failed: {e}")
            # Don't fail completely - the user can manually authenticate if needed
            return True
    
    async def _handle_claude_first_time_setup(self) -> None:
        """Handle Claude Code first-time setup (theme selection)"""
        try:
            # Check if setup might be needed by looking for the Welcome message
            test_cmd = ["claude", "--version"]
            
            process = await asyncio.create_subprocess_exec(
                *test_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            # If version works quickly, setup is already done
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=2)
                if process.returncode == 0:
                    self.logger.debug("Claude Code already set up")
                    return
            except asyncio.TimeoutError:
                # Kill the version check
                process.kill()
                await process.wait()
            
            # Now run actual setup - the version check hanging means setup is needed
            self.logger.info("Claude Code needs setup - selecting dark mode automatically")
            setup_cmd = ["claude", "-p", "echo setup"]
            
            process = await asyncio.create_subprocess_exec(
                *setup_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            # Wait a bit for the prompt to appear, then send "1" for dark mode
            await asyncio.sleep(0.5)
            process.stdin.write(b"1\n")
            await process.stdin.drain()
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
                self.logger.info("Claude Code first-time setup completed")
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                self.logger.warning("Claude Code setup timed out")
                
        except Exception as e:
            self.logger.warning(f"Failed to handle Claude Code setup: {e}")
    
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
            self.logger.debug(f"Full command: {' '.join(cmd)}")  # Log full command for debugging
            self.logger.debug(f"Working directory: {self.workspace_path}")
            
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
        
        # File creation tasks → print mode with explicit file creation prompts
        file_creation_tasks = [
            'create', 'write', 'build', 'implement', 'generate'
        ]
        
        # Interactive tasks → interactive mode for iterative work only
        interactive_tasks = [
            'help me', 'work with me', 'iterative', 'step by step',
            'refactor', 'improve', 'optimize'
        ]
        
        # Determine mode
        if any(task in command_lower for task in quick_tasks):
            return "print"  # One-shot mode for quick results
        elif any(task in command_lower for task in file_creation_tasks):
            return "print"  # Use print mode for automated file creation
        elif any(task in command_lower for task in interactive_tasks):
            return "interactive"  # Interactive mode for complex work
        else:
            # Default to print mode for automated tasks
            return "print"
    
    def _build_enhanced_claude_command(self, task: Task, execution_mode: str) -> List[str]:
        """Build enhanced Claude Code command with native session persistence"""
        cmd = ["claude"]
        
        # LEVERAGE NATIVE SESSION PERSISTENCE
        # For multi-agent coordination, start fresh to avoid conversation state issues
        # The continue flag can cause problems if there are dangling tool_use blocks
        pass  # Start with fresh session
        
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
            
            # Explicitly allow Write tool for file creation tasks
            if any(keyword in task.command.lower() for keyword in ['create', 'write', 'build', 'implement']):
                cmd.extend(["--allowedTools", "Write,Edit,MultiEdit,NotebookEdit"])
                # Add max turns to allow multiple file operations
                cmd.extend(["--max-turns", "20"])
            
            # ENHANCED: Use memory features for context
            enhanced_prompt = self._build_enhanced_prompt_with_memory(task)
            cmd.append(enhanced_prompt)
            
        elif execution_mode == "interactive":
            # Interactive mode - leverage full Claude Code features
            # Add the task as input to interactive session
            enhanced_prompt = self._build_enhanced_prompt_with_memory(task)
            cmd.append(enhanced_prompt)
        
        return cmd
    
    def _build_enhanced_prompt_with_memory(self, task: Task) -> str:
        """Build prompt that leverages Claude Code's memory features"""
        prompt_parts = []
        
        # LEVERAGE MEMORY FEATURE: Use # prefix for quick memory
        memory_context = f"# Agentic Agent Context: {', '.join(self.focus_areas)} specialist"
        prompt_parts.append(memory_context)
        
        # Add inter-agent communication context if available
        if hasattr(self, '_shared_context'):
            shared_memory = f"# Shared Context from other agents: {self._shared_context}"
            prompt_parts.append(shared_memory)
        
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
        
        # Add the main task with explicit file creation instruction
        prompt_parts.append(f"Task: {task.command}")
        
        # CRITICAL: Explicit file creation instruction for multi-agent coordination
        if any(keyword in task.command.lower() for keyword in ['create', 'write', 'build', 'implement']):
            prompt_parts.append("IMPORTANT: You MUST use the Write tool to create actual files. Do not just show code examples. Use Write to save each file to the filesystem. Create all necessary files including package.json, main.py, components, etc.")
        
        # IMPORTANT: Add memory update instruction
        prompt_parts.append("# Please update memory with any important findings or decisions using the /memory command")
        
        return "\n\n".join(prompt_parts)
    
    async def set_shared_context(self, context: str) -> None:
        """Set shared context from other agents for inter-agent communication"""
        self._shared_context = context
        self.logger.info(f"Updated shared context for {self.config.name}: {context[:50]}...")
    
    async def get_memory_content(self) -> Optional[str]:
        """Get current memory content for inter-agent sharing"""
        try:
            # Use Claude Code to extract memory
            cmd = ["claude", "-p", "/memory show"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode('utf-8')
            else:
                self.logger.warning(f"Failed to get memory content: {stderr.decode()}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting memory content: {e}")
            return None
    
    async def handle_interactive_input(self, task: Task) -> TaskResult:
        """Handle interactive scenarios where agent needs user input"""
        try:
            self.logger.info(f"Starting interactive session for task: {task.command[:50]}...")
            
            # Build command for interactive mode
            cmd = self._build_enhanced_claude_command(task, "interactive")
            
            # Check if we're in a context where we can handle interactivity
            if hasattr(self, '_input_handler') and self._input_handler:
                # Use custom input handler (e.g., from main CLI)
                return await self._execute_with_input_handler(cmd, task)
            else:
                # Fall back to automated mode with clear indication
                self.logger.warning("Interactive input needed but no handler available - using automated mode")
                return await self._execute_automated_with_fallback(cmd, task)
                
        except Exception as e:
            self.logger.error(f"Interactive input handling failed: {e}")
            return TaskResult(
                task_id=task.id,
                status="failed",
                error=f"Interactive input handling failed: {e}",
                agent_id=self.session.id if self.session else "unknown"
            )
    
    async def _execute_with_input_handler(self, cmd: List[str], task: Task) -> TaskResult:
        """Execute with custom input handler for interactive scenarios"""
        try:
            # Start Claude Code process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            # Stream output and handle input requests
            output_lines = []
            while process.returncode is None:
                try:
                    # Read output with timeout
                    line = await asyncio.wait_for(
                        process.stdout.readline(), 
                        timeout=1.0
                    )
                    
                    if line:
                        line_str = line.decode('utf-8').strip()
                        output_lines.append(line_str)
                        
                        # Check if Claude is asking for input
                        if self._is_input_request(line_str):
                            user_input = await self._input_handler(line_str)
                            if user_input:
                                process.stdin.write(f"{user_input}\n".encode())
                                await process.stdin.drain()
                    
                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break
                    continue
            
            # Get any remaining output
            stdout, stderr = await process.communicate()
            if stdout:
                output_lines.extend(stdout.decode('utf-8').split('\n'))
            
            success = process.returncode == 0
            output = '\n'.join(output_lines)
            
            return TaskResult(
                task_id=task.id,
                status="completed" if success else "failed",
                output=output,
                error=stderr.decode('utf-8') if stderr and not success else None,
                agent_id=self.session.id if self.session else "unknown"
            )
            
        except Exception as e:
            self.logger.error(f"Execution with input handler failed: {e}")
            return TaskResult(
                task_id=task.id,
                status="failed",
                error=str(e),
                agent_id=self.session.id if self.session else "unknown"
            )
    
    async def _execute_automated_with_fallback(self, cmd: List[str], task: Task) -> TaskResult:
        """Execute with automated fallback when interactive input isn't available"""
        try:
            # Modify command for automated execution
            automated_cmd = ["claude", "-p"]
            
            # Add session persistence
            if hasattr(self, 'session_id') and self.session_id:
                automated_cmd.extend(["-r", self.session_id])
            else:
                automated_cmd.extend(["-c"])
            
            # Add automated instruction
            automated_prompt = f"""
{self._build_enhanced_prompt_with_memory(task)}

IMPORTANT: This is an automated execution. Please:
1. Make reasonable assumptions if you need input
2. Explain your assumptions clearly
3. Provide the best solution you can without user interaction
4. If you absolutely need user input, clearly state what's needed and continue with a reasonable default
"""
            
            automated_cmd.append(automated_prompt)
            
            # Execute automated command
            process = await asyncio.create_subprocess_exec(
                *automated_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            output = stdout.decode('utf-8') if stdout else ""
            error = stderr.decode('utf-8') if stderr else None
            
            if not success and error:
                self.logger.warning(f"Automated fallback had issues: {error}")
            
            return TaskResult(
                task_id=task.id,
                status="completed" if success else "failed",
                output=output,
                error=error,
                agent_id=self.session.id if self.session else "unknown"
            )
            
        except Exception as e:
            self.logger.error(f"Automated fallback execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                status="failed",
                error=str(e),
                agent_id=self.session.id if self.session else "unknown"
            )
    
    def _is_input_request(self, line: str) -> bool:
        """Check if a line indicates Claude is requesting user input"""
        input_indicators = [
            "?",  # Questions
            "please confirm",
            "would you like",
            "do you want",
            "should i",
            "press enter",
            "y/n",
            "(y/n)",
            "continue?",
            "proceed?",
        ]
        
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in input_indicators)
    
    def set_input_handler(self, handler):
        """Set custom input handler for interactive scenarios"""
        self._input_handler = handler
        self.logger.info("Interactive input handler configured")
    
    def _set_environment_variables(self, env: dict) -> None:
        """Set up environment variables for Claude Code execution"""
        # Set agent-specific environment variables
        env.setdefault('AGENTIC_AGENT_TYPE', self.config.agent_type.value)
        env.setdefault('AGENTIC_AGENT_NAME', self.config.name)
        
        # Set workspace path
        env.setdefault('AGENTIC_WORKSPACE', str(self.config.workspace_path))
        
        # Set terminal environment for better Claude Code experience
        env.setdefault('TERM', 'xterm-256color')
        env.setdefault('PYTHONUNBUFFERED', '1')
        
        # Claude Code specific environment setup
        if hasattr(self, 'session_id') and self.session_id:
            env.setdefault('AGENTIC_SESSION_ID', self.session_id)
        
        self.logger.debug("Environment variables configured for Claude Code execution")
    
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
            # For multi-agent coordination, allow much longer timeouts
            timeout = 1800 if execution_mode == "interactive" else 600  # 30 min for interactive, 10 min for print
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.workspace_path)
            )
            
            try:
                # Log the process ID for debugging
                self.logger.debug(f"Claude Code process started with PID: {process.pid}")
                
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
            error_msg = f"Command failed with exit code {return_code}"
            if stderr:
                error_msg += f"\nStderr: {stderr}"
            if stdout:
                error_msg += f"\nStdout: {stdout}"
            return False, error_msg
        
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