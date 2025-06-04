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
    
    def __init__(self, name: str, config: Dict[str, Any]):
        # Convert dict config to AgentConfig
        agent_config = AgentConfig(
            agent_type=AgentType.CLAUDE_CODE,
            name=name,
            workspace_path=Path(config.get("project_root", ".")),
            focus_areas=config.get("focus_areas", ["coding", "analysis", "refactoring"]),
            ai_model_config={"model": config.get("claude_model", "sonnet")},
            tool_config=config.get("tool_config", {}),
            max_tokens=config.get("max_tokens", 100000),
            temperature=config.get("temperature", 0.1)
        )
        
        super().__init__(agent_config)
        self.logger = logging.getLogger(__name__)
        self.process: Optional[subprocess.Popen] = None
        self.temp_files: List[Path] = []
        
        # Enhanced Claude Code settings
        self.claude_model = config.get("claude_model", "sonnet")
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
        
        self.project_root = Path(config.get("project_root", "."))
    
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
        """Execute a task using enhanced Claude Code CLI features."""
        if not self.session or not self.session.is_available:
            return TaskResult(
                task_id=task.id,
                agent_id=self.config.name,
                status="failed",
                output="Agent not available",
                error="Agent session not available",
                metadata={"error": "Agent session not available"}
            )
        
        self.session.mark_busy(task.id)
        
        try:
            # Build enhanced claude command
            cmd = await self._build_enhanced_command(task)
            
            self.logger.info(f"Executing enhanced Claude Code command for task {task.id}")
            self.logger.debug(f"Command: {' '.join(cmd)}")
            
            # Execute the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            stdout, stderr = await process.communicate()
            
            # Process results with enhanced parsing
            if process.returncode == 0:
                result = await self._parse_claude_output(stdout.decode('utf-8').strip(), task)
                
                self.session.mark_idle()
                return TaskResult(
                    task_id=task.id,
                    agent_id=self.config.name,
                    status="completed",
                    output=result["output"],
                    metadata={
                        "claude_model": self.claude_model,
                        "session_id": self.session_id,
                        "tools_used": result.get("tools_used", []),
                        "thinking_time": result.get("thinking_time"),
                        "files_modified": result.get("files_modified", []),
                        "output_length": len(result["output"])
                    }
                )
            else:
                error_msg = stderr.decode('utf-8').strip()
                self.logger.error(f"Claude Code execution failed: {error_msg}")
                
                self.session.mark_idle()
                return TaskResult(
                    task_id=task.id,
                    agent_id=self.config.name,
                    status="failed",
                    output=f"Claude Code execution failed: {error_msg}",
                    error=error_msg,
                    metadata={"error": error_msg, "return_code": process.returncode, "session_id": self.session_id}
                )
                
        except Exception as e:
            self.logger.error(f"Error executing task {task.id}: {e}")
            self.session.mark_idle()
            return TaskResult(
                task_id=task.id,
                agent_id=self.config.name,
                status="failed",
                output=f"Execution error: {str(e)}",
                error=str(e),
                metadata={"error": str(e), "session_id": self.session_id}
            )

    async def _build_enhanced_command(self, task: Task) -> List[str]:
        """Build enhanced Claude Code command with full feature utilization."""
        cmd = ["claude"]
        
        # Add enhanced options first
        # Use print mode for automation with JSON output for better parsing  
        cmd.extend(["--print", "--output-format", "json"])
        
        # Add model
        if self.claude_model:
            cmd.extend(["--model", self.claude_model])
        
        # TODO: Fix allowedTools syntax - currently causing argument parsing issues
        # Add task-specific tool permissions as comma-separated list
        # allowed_tools = self._get_task_tools(task)
        # if allowed_tools:
        #     tools_str = ",".join(allowed_tools)
        #     cmd.extend(["--allowedTools", tools_str])
        
        # Add enhanced prompt with thinking triggers as the final argument
        prompt = await self._build_enhanced_prompt(task)
        cmd.append(prompt)
        
        return cmd

    def _get_task_tools(self, task: Task) -> List[str]:
        """Get appropriate tools based on task type and content."""
        task_type = task.intent.task_type.value if task.intent else "general"
        command_lower = task.command.lower()
        
        # Start with base tools for the task type
        tools = self.tool_presets.get(task_type, self.tool_presets["coding"]).copy()
        
        # Add specific tools based on command content
        if "git" in command_lower or "commit" in command_lower or "branch" in command_lower:
            tools.extend(self.tool_presets["git"])
        
        if "test" in command_lower or "pytest" in command_lower:
            tools.extend(self.tool_presets["testing"])
        
        if "document" in command_lower or "readme" in command_lower:
            tools.extend(self.tool_presets["documentation"])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(tools))

    async def _build_enhanced_prompt(self, task: Task) -> str:
        """Build enhanced prompt with thinking triggers and context."""
        task_type = task.intent.task_type.value if task.intent else "general"
        command = task.command
        
        # Determine if this needs extended thinking
        needs_thinking = any(keyword in command.lower() for keyword in [
            "architecture", "design", "refactor", "complex", "plan", "strategy",
            "analyze", "review", "optimize", "debug"
        ])
        
        prompt_parts = []
        
        # Add thinking trigger for complex tasks
        if needs_thinking:
            if task_type in ["implement", "refactor"]:
                prompt_parts.append("Think deeply about this task before proceeding.")
            else:
                prompt_parts.append("Think about the best approach for this task.")
        
        # Add main task
        prompt_parts.append(f"Task: {command}")
        
        # Add context and task type
        if hasattr(task, 'context') and task.context:
            prompt_parts.append(f"Context: {task.context}")
        
        prompt_parts.append(f"Task type: {task_type}")
        
        # Add task-specific instructions
        if task_type == "explain":
            prompt_parts.append("Analyze the codebase thoroughly and provide detailed insights.")
        elif task_type == "refactor":
            prompt_parts.append("Refactor the code while maintaining functionality. Consider architecture improvements.")
        elif task_type == "debug":
            prompt_parts.append("Identify and fix bugs systematically. Explain your debugging process.")
        elif task_type == "document":
            prompt_parts.append("Create comprehensive documentation that helps other developers.")
        elif task_type == "implement":
            prompt_parts.append("Think about system design, scalability, and maintainability.")
        
        # Add memory and project context
        prompt_parts.append(f"Project directory: {self.project_root}")
        if self.memory_initialized:
            prompt_parts.append("Use the CLAUDE.md memory file for project-specific guidelines.")
        
        prompt_parts.append("Work within this project context and maintain consistency with existing patterns.")
        
        # Add session continuity for complex tasks
        if self.session_id and task_type in ["refactor", "implement", "debug"]:
            prompt_parts.append(f"Session ID: {self.session_id} (for potential follow-up tasks)")
        
        return "\n\n".join(prompt_parts)

    async def _parse_claude_output(self, output: str, task: Task) -> Dict[str, Any]:
        """Parse Claude Code's JSON output for enhanced metadata."""
        try:
            # Try to parse as JSON first
            if output.startswith('{') and output.endswith('}'):
                data = json.loads(output)
                return {
                    "output": data.get("content", output),
                    "tools_used": data.get("tools_used", []),
                    "files_modified": data.get("files_modified", []),
                    "thinking_time": data.get("thinking_time"),
                    "session_info": data.get("session_info")
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback to text parsing
        return {
            "output": output,
            "tools_used": [],
            "files_modified": [],
            "thinking_time": None
        }
    
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