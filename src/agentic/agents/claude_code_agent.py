"""
Claude Code Agent - CLI Integration

Integrates with the Claude Code CLI tool for sophisticated coding tasks.
Spawns 'claude' subprocess sessions with proper authentication and project context.
"""
import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

from agentic.models.agent import Agent, AgentConfig, AgentCapability, AgentType, AgentSession
from agentic.models.task import Task, TaskResult


class ClaudeCodeAgent(Agent):
    """Agent that integrates with Claude Code CLI for sophisticated coding tasks."""
    
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
        
        # Claude Code specific settings
        self.claude_model = config.get("claude_model", "sonnet")
        self.allowed_tools = config.get("allowed_tools", ["Edit", "Bash(git *)", "Write"])
        self.project_root = Path(config.get("project_root", "."))
        
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities."""
        return AgentCapability(
            agent_type=AgentType.CLAUDE_CODE,
            specializations=[
                "coding", "refactoring", "analysis", "debugging", 
                "code_review", "architecture", "documentation"
            ],
            supported_languages=[
                "python", "javascript", "typescript", "rust", "go",
                "java", "cpp", "c", "html", "css", "sql", "bash"
            ],
            max_context_tokens=200000,  # Claude has large context
            concurrent_tasks=1,  # Claude Code works best with sequential tasks
            reasoning_capability=True,
            file_editing_capability=True,
            code_execution_capability=True  # Claude Code can execute bash commands
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
        """Start the Claude Code agent session."""
        try:
            # Check health first
            if not await self.health_check():
                return False
            
            # Create session
            self.session = AgentSession(
                agent_config=self.config,
                workspace=self.project_root,
                status="starting"
            )
            
            # Change to project directory
            os.chdir(self.project_root)
            
            self.session.mark_active()
            self.logger.info(f"Claude Code agent {self.config.name} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Claude Code agent: {e}")
            if self.session:
                self.session.mark_error(str(e))
            return False
    
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
        """Execute a task using Claude Code CLI."""
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
            # Prepare the prompt
            prompt = self._prepare_prompt(task)
            
            # Build claude command
            cmd = ["claude", "--print", "--output-format", "text"]
            
            # Add model if specified
            if self.claude_model:
                cmd.extend(["--model", self.claude_model])
            
            # Add allowed tools
            if self.allowed_tools:
                cmd.extend(["--allowedTools", *self.allowed_tools])
            
            # Add the prompt
            cmd.append(prompt)
            
            self.logger.info(f"Executing Claude Code command for task {task.id}")
            
            # Execute the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            stdout, stderr = await process.communicate()
            
            # Process results
            if process.returncode == 0:
                result = stdout.decode('utf-8').strip()
                
                self.session.mark_idle()
                return TaskResult(
                    task_id=task.id,
                    agent_id=self.config.name,
                    status="completed",
                    output=result,
                    metadata={
                        "claude_model": self.claude_model,
                        "allowed_tools": self.allowed_tools,
                        "output_length": len(result)
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
                    metadata={"error": error_msg, "return_code": process.returncode}
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
                metadata={"error": str(e)}
            )
    
    def _prepare_prompt(self, task: Task) -> str:
        """Prepare a prompt for Claude Code based on the task."""
        # Base prompt with task description
        prompt_parts = [
            f"Task: {task.command}",
            f"Command: {task.command}",
        ]
        
        # Add context if available
        if hasattr(task, 'context') and task.context:
            prompt_parts.append(f"Context: {task.context}")
        
        # Add metadata information from the intent
        task_type = task.intent.task_type.value if task.intent else "general"
        prompt_parts.append(f"Task type: {task_type}")
        
        # Add specific instructions based on task type
        if task_type == "explain":
            prompt_parts.append("Please analyze the codebase and provide detailed insights.")
        elif task_type == "refactor":
            prompt_parts.append("Please refactor the code while maintaining functionality.")
        elif task_type == "debug":
            prompt_parts.append("Please identify and fix any bugs or issues.")
        elif task_type == "document":
            prompt_parts.append("Please improve or create documentation.")
        
        # Add project context
        prompt_parts.extend([
            f"Project directory: {self.project_root}",
            "Please work within this project context and maintain consistency with existing code patterns."
        ])
        
        return "\n\n".join(prompt_parts)
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task types."""
        return [
            "analysis", "refactoring", "debugging", "code_review",
            "documentation", "testing", "optimization", "architecture"
        ]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        if not self.session:
            return {"status": "not_started"}
        
        return {
            "status": self.session.status,
            "current_task": self.session.current_task,
            "workspace": str(self.session.workspace),
            "created_at": self.session.created_at.isoformat(),
            "last_activity": self.session.last_activity.isoformat() if self.session.last_activity else None,
            "capabilities": self.get_capabilities().model_dump(),
            "health": await self.health_check()
        }