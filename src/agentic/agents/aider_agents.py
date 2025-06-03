"""
Specialized Aider Agents for different development areas
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from agentic.models.agent import Agent, AgentConfig, AgentType
from agentic.models.task import Task, TaskResult
from agentic.utils.logging import LoggerMixin


class BaseAiderAgent(Agent):
    """Base class for Aider-based agents"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.aider_process: Optional[subprocess.Popen] = None
        self.aider_args: List[str] = []
        self._setup_aider_args()
    
    def _setup_aider_args(self) -> None:
        """Setup common Aider arguments"""
        self.aider_args = [
            "aider",
            "--yes",  # Auto-confirm changes
            "--no-git",  # Don't auto-commit
            f"--model={self.config.ai_model_config.get('model', 'claude-3-5-sonnet')}",
        ]
        
        # Add focus area specific arguments
        if self.config.focus_areas:
            # For now, we'll use focus areas as hints in prompts rather than CLI args
            pass
    
    async def start(self) -> bool:
        """Start the Aider agent"""
        try:
            self.logger.info(f"Starting Aider agent: {self.config.name}")
            
            # Verify aider is available
            result = subprocess.run(["which", "aider"], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("Aider not found in PATH. Please install aider.")
                return False
            
            # Verify model access
            if not await self._verify_model_access():
                self.logger.warning("Could not verify model access, but continuing...")
            
            self.logger.info(f"Aider agent {self.config.name} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Aider agent {self.config.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the Aider agent"""
        try:
            if self.aider_process and self.aider_process.poll() is None:
                self.aider_process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process(self.aider_process)), 
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    self.aider_process.kill()
                    await asyncio.create_task(self._wait_for_process(self.aider_process))
            
            self.logger.info(f"Aider agent {self.config.name} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping Aider agent {self.config.name}: {e}")
            return False
    
    async def _wait_for_process(self, process: subprocess.Popen) -> None:
        """Wait for process to terminate"""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def _verify_model_access(self) -> bool:
        """Verify that the AI model is accessible"""
        try:
            # Test with a simple aider command
            test_cmd = self.aider_args + ["--help"]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            self.logger.warning(f"Model access verification failed: {e}")
            return False
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task using Aider"""
        self.logger.info(f"Executing task: {task.command[:100]}...")
        
        try:
            # Prepare Aider command with task-specific context
            aider_command = await self._prepare_aider_command(task)
            
            # Execute the command
            result = await self._run_aider_command(aider_command, task)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                agent_id=self.session.id if self.session else "unknown",
                success=False,
                output="",
                error=str(e)
            )
    
    async def _prepare_aider_command(self, task: Task) -> List[str]:
        """Prepare Aider command for the specific task"""
        command = self.aider_args.copy()
        
        # Add working directory
        command.extend(["--cwd", str(self.config.workspace_path)])
        
        # Add the actual command/prompt
        prompt = await self._build_agent_prompt(task)
        command.extend(["--message", prompt])
        
        return command
    
    async def _build_agent_prompt(self, task: Task) -> str:
        """Build agent-specific prompt for the task"""
        # Base prompt with agent context
        prompt_parts = [
            f"You are a {self.config.name} agent specializing in: {', '.join(self.config.focus_areas)}.",
            f"Task: {task.command}",
        ]
        
        # Add agent-specific context
        agent_context = await self._get_agent_context(task)
        if agent_context:
            prompt_parts.append(f"Context: {agent_context}")
        
        return "\n\n".join(prompt_parts)
    
    async def _get_agent_context(self, task: Task) -> str:
        """Get agent-specific context for the task"""
        # Override in specialized agents
        return ""
    
    async def _run_aider_command(self, command: List[str], task: Task) -> TaskResult:
        """Run the Aider command and capture results"""
        try:
            self.logger.debug(f"Running command: {' '.join(command)}")
            
            # Run Aider command
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.workspace_path
            )
            
            stdout, stderr = await process.communicate()
            
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            success = process.returncode == 0
            
            # Log the output
            if success:
                self.logger.info(f"Task completed successfully")
                if stdout_text:
                    self.logger.debug(f"Output: {stdout_text[:500]}...")
            else:
                self.logger.error(f"Task failed with return code: {process.returncode}")
                if stderr_text:
                    self.logger.error(f"Error: {stderr_text[:500]}...")
            
            return TaskResult(
                task_id=task.id,
                agent_id=self.session.id if self.session else "unknown",
                success=success,
                output=stdout_text,
                error=stderr_text if not success else ""
            )
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                agent_id=self.session.id if self.session else "unknown",
                success=False,
                output="",
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy"""
        try:
            # Simple check - verify aider is still accessible
            result = subprocess.run(["aider", "--help"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False


class AiderFrontendAgent(BaseAiderAgent):
    """Specialized Aider agent for frontend development"""
    
    def __init__(self, config: AgentConfig):
        # Ensure config is set for frontend
        if config.agent_type != AgentType.AIDER_FRONTEND:
            config.agent_type = AgentType.AIDER_FRONTEND
        if not config.focus_areas:
            config.focus_areas = ["frontend", "components", "ui", "styling"]
        
        super().__init__(config)
    
    async def _get_agent_context(self, task: Task) -> str:
        """Get frontend-specific context"""
        context_parts = []
        
        # Focus on frontend files and technologies
        context_parts.append("Focus on frontend development including:")
        context_parts.append("- React/Vue/Angular components")
        context_parts.append("- CSS/styling and responsive design")
        context_parts.append("- User interface and user experience")
        context_parts.append("- Frontend build tools and configuration")
        context_parts.append("- Browser compatibility and performance")
        
        # Add file pattern hints
        context_parts.append("\nPrioritize working with these file patterns:")
        context_parts.append("- *.tsx, *.jsx, *.vue, *.svelte")
        context_parts.append("- *.css, *.scss, *.less, *.styled.ts")
        context_parts.append("- package.json, webpack.config.*, vite.config.*")
        
        return "\n".join(context_parts)


class AiderBackendAgent(BaseAiderAgent):
    """Specialized Aider agent for backend development"""
    
    def __init__(self, config: AgentConfig):
        # Ensure config is set for backend
        if config.agent_type != AgentType.AIDER_BACKEND:
            config.agent_type = AgentType.AIDER_BACKEND
        if not config.focus_areas:
            config.focus_areas = ["backend", "api", "database", "server"]
        
        super().__init__(config)
    
    async def _get_agent_context(self, task: Task) -> str:
        """Get backend-specific context"""
        context_parts = []
        
        # Focus on backend technologies
        context_parts.append("Focus on backend development including:")
        context_parts.append("- API design and implementation (REST, GraphQL)")
        context_parts.append("- Database design, queries, and migrations")
        context_parts.append("- Authentication and authorization")
        context_parts.append("- Server configuration and middleware")
        context_parts.append("- Performance optimization and caching")
        context_parts.append("- Error handling and logging")
        
        # Add file pattern hints
        context_parts.append("\nPrioritize working with these file patterns:")
        context_parts.append("- *.py, *.js, *.ts, *.go, *.rs, *.java")
        context_parts.append("- **/models/**, **/api/**, **/services/**")
        context_parts.append("- requirements.txt, go.mod, Cargo.toml, pom.xml")
        context_parts.append("- Database migration files and SQL scripts")
        
        return "\n".join(context_parts)


class AiderTestingAgent(BaseAiderAgent):
    """Specialized Aider agent for testing and quality assurance"""
    
    def __init__(self, config: AgentConfig):
        # Ensure config is set for testing
        if config.agent_type != AgentType.AIDER_TESTING:
            config.agent_type = AgentType.AIDER_TESTING
        if not config.focus_areas:
            config.focus_areas = ["testing", "qa", "quality", "coverage"]
        
        super().__init__(config)
    
    async def _get_agent_context(self, task: Task) -> str:
        """Get testing-specific context"""
        context_parts = []
        
        # Focus on testing and quality
        context_parts.append("Focus on testing and quality assurance including:")
        context_parts.append("- Unit tests with high coverage")
        context_parts.append("- Integration and end-to-end tests")
        context_parts.append("- Test-driven development (TDD)")
        context_parts.append("- Mocking and stubbing external dependencies")
        context_parts.append("- Performance and load testing")
        context_parts.append("- Code quality and static analysis")
        
        # Add file pattern hints
        context_parts.append("\nPrioritize working with these file patterns:")
        context_parts.append("- **/*test*.py, **/*spec*.js, **/*.test.ts")
        context_parts.append("- **/tests/**, **/spec/**, **/__tests__/**")
        context_parts.append("- pytest.ini, jest.config.*, cypress.json")
        context_parts.append("- .coverage, coverage.xml, test-results/")
        
        return "\n".join(context_parts) 