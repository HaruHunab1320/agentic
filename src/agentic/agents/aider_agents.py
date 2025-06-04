"""
Specialized Aider Agents for different development areas
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import os
import re

from agentic.models.agent import Agent, AgentConfig, AgentType, AgentCapability
from agentic.models.task import Task, TaskResult
from agentic.utils.logging import LoggerMixin


class BaseAiderAgent(Agent, LoggerMixin):
    """Base class for Aider-based agents"""
    
    def __init__(self, config: AgentConfig):
        Agent.__init__(self, config)
        LoggerMixin.__init__(self)
        self.aider_process: Optional[subprocess.Popen] = None
        self.aider_args: List[str] = []
        self._setup_aider_args()
    
    def _setup_aider_args(self) -> None:
        """Setup common Aider arguments"""
        # Get model from config with intelligent defaults
        model = self._get_model_for_aider()
        
        self.aider_args = [
            "aider",
            "--yes-always",  # Auto-confirm changes
            "--no-git",  # Don't auto-commit
            f"--model={model}",
        ]
        
        # Add additional model-specific configurations
        self._add_model_specific_args()
        
        # Set up API key environment variables
        self._setup_api_keys()
        
        # Add focus area specific arguments
        if self.config.focus_areas:
            # For now, we'll use focus areas as hints in prompts rather than CLI args
            pass
    
    def _setup_api_keys(self) -> None:
        """Setup API keys from credential store"""
        try:
            from agentic.utils.credentials import get_api_key
            
            # Determine which provider we need based on model
            model = self._get_model_for_aider()
            
            if "gemini" in model.lower() or "google" in model.lower():
                provider = "gemini"
                env_var = "GEMINI_API_KEY"
            elif "claude" in model.lower() or "anthropic" in model.lower():
                provider = "anthropic"
                env_var = "ANTHROPIC_API_KEY"
            elif "gpt" in model.lower() or "openai" in model.lower():
                provider = "openai"
                env_var = "OPENAI_API_KEY"
            else:
                # Default to trying all providers
                return
            
            # Get API key from credential store
            api_key = get_api_key(provider, self.config.workspace_path)
            
            if api_key:
                # Set environment variable for this process
                os.environ[env_var] = api_key
            else:
                # Warn user about missing API key
                print(f"⚠️ Warning: No {provider.upper()} API key found.")
                print(f"Set one with: agentic keys set {provider}")
                
        except ImportError:
            # Credentials module not available, continue without setup
            pass
        except Exception as e:
            # Log error but continue
            pass
    
    def _get_model_for_aider(self) -> str:
        """Get the appropriate model for Aider with intelligent fallbacks"""
        model_config = self.config.ai_model_config
        
        # Primary model from config
        primary_model = model_config.get('model', 'claude-3-5-sonnet')
        
        # Handle Gemini models - map to Aider's expected format
        gemini_model_map = {
            'gemini-1.5-pro': 'gemini/gemini-1.5-pro-latest',
            'gemini-1.5-flash': 'gemini/gemini-1.5-flash-latest',
            'gemini-pro': 'gemini/gemini-1.5-pro-latest',
            'gemini-flash': 'gemini/gemini-1.5-flash-latest',
            'gemini': 'gemini/gemini-1.5-pro-latest',  # Default Gemini
        }
        
        # Handle Claude models - ensure correct format
        claude_model_map = {
            'claude': 'claude-3-5-sonnet',
            'sonnet': 'claude-3-5-sonnet',
            'haiku': 'claude-3-haiku',
            'opus': 'claude-3-opus',
        }
        
        # Handle OpenAI models
        openai_model_map = {
            'gpt-4': 'gpt-4-0125-preview',
            'gpt-4o': 'gpt-4o',
            'gpt-3.5': 'gpt-3.5-turbo',
        }
        
        # Check if it's a mapped model
        if primary_model in gemini_model_map:
            return gemini_model_map[primary_model]
        elif primary_model in claude_model_map:
            return claude_model_map[primary_model]
        elif primary_model in openai_model_map:
            return openai_model_map[primary_model]
        else:
            # Return as-is, might be a fully qualified model name
            return primary_model
    
    def _add_model_specific_args(self) -> None:
        """Add model-specific arguments to Aider command"""
        model = self.config.ai_model_config.get('model', 'claude-3-5-sonnet')
        
        # Gemini-specific configurations
        if 'gemini' in model.lower():
            # Gemini models work well with these settings
            if self.config.ai_model_config.get('temperature'):
                # Note: Not all Aider integrations support temperature via CLI
                # But we can store it for future use
                pass
        
        # Claude-specific configurations  
        elif 'claude' in model.lower():
            # Claude models can handle larger contexts
            pass
        
        # OpenAI-specific configurations
        elif 'gpt' in model.lower():
            # OpenAI models might need specific timeout settings
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
                status="failed",
                output="",
                error=str(e)
            )
    
    async def _prepare_aider_command(self, task: Task) -> List[str]:
        """Prepare Aider command for the specific task"""
        command = self.aider_args.copy()
        
        # Add target files if mentioned in the task
        target_files = self._extract_target_files(task)
        if target_files:
            command.extend(target_files)
        
        # Add the actual command/prompt
        prompt = await self._build_agent_prompt(task)
        command.extend(["--message", prompt])
        
        return command
    
    def _extract_target_files(self, task: Task) -> List[str]:
        """Extract target files mentioned in the task command"""
        # Look for file patterns in the command
        file_patterns = [
            r'(\w+\.py)',  # Python files
            r'(\w+\.js)',  # JavaScript files
            r'(\w+\.ts)',  # TypeScript files
            r'(\w+\.jsx)', # JSX files
            r'(\w+\.tsx)', # TSX files
            r'(\w+\.go)',  # Go files
            r'(\w+\.rs)',  # Rust files
            r'(\w+\.java)', # Java files
            r'(\w+\.cpp)', # C++ files
            r'(\w+\.c)',   # C files
            r'(\w+\.h)',   # Header files
        ]
        
        files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, task.command, re.IGNORECASE)
            files.extend(matches)
        
        # Check if files exist in workspace
        existing_files = []
        for file in files:
            file_path = self.config.workspace_path / file
            if file_path.exists():
                existing_files.append(str(file))
        
        return existing_files
    
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
            
            # Prepare environment with current environment plus any API keys we've set
            env = os.environ.copy()
            
            # Run Aider command
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.workspace_path,
                env=env  # Pass the environment with API keys
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
                status="completed" if success else "failed",
                output=stdout_text,
                error=stderr_text if not success else ""
            )
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                agent_id=self.session.id if self.session else "unknown",
                status="failed",
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
    
    def get_capabilities(self) -> AgentCapability:
        """Get capabilities of this Aider agent"""
        return AgentCapability(
            agent_type=self.config.agent_type,
            specializations=self.config.focus_areas,
            supported_languages=["python", "javascript", "typescript", "go", "rust", "java"],
            max_context_tokens=self.config.max_tokens,
            concurrent_tasks=1,
            reasoning_capability=True,
            file_editing_capability=True,
            code_execution_capability=False  # Aider doesn't execute code directly
        )


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