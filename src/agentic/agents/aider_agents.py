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
            "--no-fancy-input",  # Disable fancy input for automation
            "--no-show-model-warnings",  # Suppress model warnings for automation
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
            
            # Load .env file to ensure environment variables are available
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            
            # Determine which provider we need based on model
            model = self._get_model_for_aider()
            
            # For Aider agents, we only use Gemini models
            if "gemini" in model.lower() or "google" in model.lower():
                provider = "gemini"
                env_var = "GEMINI_API_KEY"
            else:
                # All Aider agents default to Gemini, so always try gemini
                provider = "gemini"
                env_var = "GEMINI_API_KEY"
            
            # Get API key from credential store
            api_key = get_api_key(provider, self.config.workspace_path)
            
            if api_key:
                # Set environment variable for this process
                os.environ[env_var] = api_key
                print(f"✅ {provider.upper()} API key loaded successfully")
                print(f"✅ Set {env_var} = {api_key[:10]}...")
            else:
                # Warn user about missing API key
                print(f"⚠️ Warning: No {provider.upper()} API key found.")
                print(f"Set one with: agentic keys set {provider}")
                print(f"Or add {env_var} to your .env file")
                
        except ImportError:
            # Credentials module not available, continue without setup
            pass
        except Exception as e:
            # Log error but continue
            pass
    
    def _set_environment_variables(self, env: dict) -> None:
        """Set up environment variables for agent execution"""
        # This method provides additional environment setup
        # The main API key setup is already handled in _setup_api_keys()
        
        # Set agent-specific environment variables if needed
        env.setdefault('AGENTIC_AGENT_TYPE', self.config.agent_type.value)
        env.setdefault('AGENTIC_AGENT_NAME', self.config.name)
        
        # Set workspace path
        env.setdefault('AGENTIC_WORKSPACE', str(self.config.workspace_path))
        
        # Set terminal environment to help with Aider
        env.setdefault('TERM', 'xterm-256color')
        env.setdefault('PYTHONUNBUFFERED', '1')
        
        # Ensure API keys are available in subprocess environment
        # Re-run API key setup to make sure they're in the environment
        try:
            from agentic.utils.credentials import get_api_key
            from dotenv import load_dotenv
            load_dotenv()  # Load .env again
            
            # Get the model and provider
            model = self._get_model_for_aider()
            if "gemini" in model.lower() or "google" in model.lower():
                provider = "gemini"
                env_var = "GEMINI_API_KEY"
                
                api_key = get_api_key(provider, self.config.workspace_path)
                if api_key:
                    env[env_var] = api_key
                    print(f"🔑 Set {env_var} for subprocess")
                    
        except Exception as e:
            print(f"⚠️ Error setting API key in subprocess: {e}")
    
    def _get_model_for_aider(self) -> str:
        """Get the appropriate model for Aider with intelligent fallbacks"""
        model_config = self.config.ai_model_config
        
        # Primary model from config
        primary_model = model_config.get('model', 'claude-3-5-sonnet')
        
        # Handle Gemini models - map to Aider's expected format
        gemini_model_map = {
            'gemini-2.5-pro': 'gemini-2.5-pro',
            'gemini-1.5-pro': 'gemini/gemini-1.5-pro-latest',
            'gemini-1.5-flash': 'gemini/gemini-1.5-flash-latest',
            'gemini-pro': 'gemini-2.5-pro',  # Use stable 2.5 pro
            'gemini-flash': 'gemini/gemini-1.5-flash-latest',
            'gemini': 'gemini-2.5-pro',  # Default to stable 2.5 Pro
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
            
            # Handle Aider first-time setup if needed
            if not await self._ensure_aider_setup():
                self.logger.warning("Aider setup may have issues, but continuing...")
            
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
    
    async def _ensure_aider_setup(self) -> bool:
        """Ensure Aider is properly set up with API keys and model configuration"""
        try:
            # Test if Aider can run without prompting for setup
            test_cmd = ["aider", "--help"]
            
            process = await asyncio.create_subprocess_exec(
                *test_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_path)
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            
            if process.returncode == 0:
                self.logger.debug("Aider basic command works")
                
                # Test with our model to see if it prompts for setup
                model = self._get_model_for_aider()
                model_test_cmd = ["aider", "--model", model, "--help"]
                
                process = await asyncio.create_subprocess_exec(
                    *model_test_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, 
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.workspace_path)
                )
                
                # Send 'N' to decline OpenRouter setup and documentation if prompted
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input=b"N\nN\n"), 
                        timeout=15
                    )
                    
                    combined_output = stdout.decode() + stderr.decode()
                    
                    # Check if there were API key issues
                    if "No LLM model was specified" in combined_output or "Login to OpenRouter" in combined_output:
                        self.logger.info("Aider needs model/API key setup - our setup should handle this")
                        return True  # We'll handle this with our own API key setup
                    
                    return True
                    
                except asyncio.TimeoutError:
                    process.kill()
                    self.logger.warning("Aider model test timed out")
                    return True
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Aider setup check failed: {e}")
            return True  # Don't fail completely
    
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
        """Execute a task using Aider with enhanced capabilities"""
        try:
            # Extract target files from the command for better Aider targeting
            target_files = self._extract_target_files(task.command)
            
            # Build enhanced Aider command with better file targeting
            cmd = await self._build_enhanced_aider_command(task, target_files)
            
            self.logger.info(f"Executing task: {task.command[:50]}...")
            self.logger.debug(f"Running command: {' '.join(cmd[:5])}...")  # Log first 5 args only
            
            # Set up environment with API keys
            env = os.environ.copy()
            self._set_environment_variables(env)
            
            # Execute Aider with better error handling and output parsing
            result = await self._execute_aider_with_session_management(cmd, env, task)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                status="failed",
                error=str(e),
                agent_id=self.session.id if self.session else "unknown"
            )
    
    def _extract_target_files(self, command: str) -> List[str]:
        """Extract specific file targets from command for better Aider targeting"""
        import re
        
        target_files = []
        
        # Look for explicit file patterns
        file_patterns = [
            r'(\w+\.py)',           # Python files
            r'(\w+\.js)',           # JavaScript files  
            r'(\w+\.ts)',           # TypeScript files
            r'(\w+\.html)',         # HTML files
            r'(\w+\.css)',          # CSS files
            r'(\w+\.json)',         # JSON files
            r'(src/[\w/]+\.py)',    # Source path files
            r'(tests?/[\w/]+\.py)', # Test files
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, command, re.IGNORECASE)
            target_files.extend(matches)
        
        # Look for directory patterns that suggest file creation
        if 'create' in command.lower() or 'add' in command.lower():
            # Extract directory hints for new files
            dir_patterns = [
                r'src/([\w/]+)/',
                r'tests?/([\w/]+)/',
                r'(auth|api|models|services)/'
            ]
            
            for pattern in dir_patterns:
                matches = re.findall(pattern, command, re.IGNORECASE)
                if matches:
                    # Don't add directories directly, let Aider handle file creation
                    pass
        
        return list(set(target_files))  # Remove duplicates
    
    async def _build_enhanced_aider_command(self, task: Task, target_files: List[str]) -> List[str]:
        """Build enhanced Aider command with advanced features"""
        cmd = ["aider"]
        
        # Core Aider flags for automation
        cmd.extend(["--yes-always", "--no-git"])
        
        # Set model based on agent configuration
        if hasattr(self, 'ai_model_config') and self.ai_model_config.get('model'):
            model = self.ai_model_config['model']
            # Map our model names to Aider-compatible names
            if model == 'gemini':
                cmd.extend(["--model", "gemini-2.5-pro"])
            elif model in ['claude', 'claude-sonnet']:
                cmd.extend(["--model", "sonnet"])
            else:
                cmd.extend(["--model", model])
        else:
            # Default to Gemini 2.5 Pro for Aider agents
            cmd.extend(["--model", "gemini-2.5-pro"])
        
        # Add target files if we found any (following Aider best practices)
        if target_files:
            # Only add files that likely need editing, not extra context files
            relevant_files = [f for f in target_files if self._is_likely_edit_target(f, task.command)]
            if relevant_files:
                cmd.extend(relevant_files)
        
        # Build specialized message based on agent type and task
        message = self._build_specialized_message(task)
        cmd.extend(["--message", message])
        
        # Add exit flag to prevent hanging
        cmd.extend(["--exit"])
        
        return cmd
    
    def _is_likely_edit_target(self, filename: str, command: str) -> bool:
        """Determine if a file is likely to be edited based on command context"""
        command_lower = command.lower()
        
        # Files explicitly mentioned for creation/modification
        if any(action in command_lower for action in ['create', 'add', 'modify', 'update', 'edit']):
            if filename.lower() in command_lower:
                return True
        
        # Configuration and test files are usually edit targets
        if any(pattern in filename.lower() for pattern in ['config', 'test', 'spec']):
            return True
            
        # Source files in src/ are usually edit targets
        if filename.startswith('src/'):
            return True
            
        return False
    
    def _build_specialized_message(self, task: Task) -> str:
        """Build specialized message based on agent type and expertise"""
        base_message = f"You are a {self.agent_type.value} agent specializing in: {', '.join(self.focus_areas)}.\n\n"
        
        # Add task
        base_message += f"Task: {task.command}\n\n"
        
        # Add specialized context based on agent type
        if self.agent_type.value == "aider_frontend":
            base_message += """Context: Focus on frontend development including:
- UI/UX design and implementation
- Component architecture and reusability  
- Modern frontend frameworks (React, Vue, Angular)
- CSS/styling and responsive design
- Frontend build tools and optimization
- Accessibility and performance

Prioritize working with these file patterns:
- *.js, *.jsx, *.ts, *.tsx, *.vue, *.svelte
- *.css, *.scss, *.less, *.styled.*
- **/components/**, **/pages/**, **/views/**
- package.json, webpack.config.js, vite.config.*"""

        elif self.agent_type.value == "aider_backend":
            base_message += """Context: Focus on backend development including:
- API design and implementation (REST, GraphQL)
- Database design, queries, and migrations
- Authentication and authorization
- Server configuration and middleware
- Performance optimization and caching
- Error handling and logging

Prioritize working with these file patterns:
- *.py, *.js, *.ts, *.go, *.rs, *.java
- **/models/**, **/api/**, **/services/**
- requirements.txt, go.mod, Cargo.toml, pom.xml
- Database migration files and SQL scripts"""

        elif self.agent_type.value == "aider_testing":
            base_message += """Context: Focus on testing and quality assurance including:
- Unit testing and test-driven development
- Integration and end-to-end testing
- Test automation and CI/CD
- Code coverage and quality metrics
- Performance and load testing
- Bug reproduction and regression testing

Prioritize working with these file patterns:
- test_*.py, *_test.py, *.test.js, *.spec.*
- **/tests/**, **/test/**, **/__tests__/**
- pytest.ini, jest.config.*, karma.conf.*
- Coverage configuration and test utilities"""
        
        return base_message
    
    async def _execute_aider_with_session_management(self, cmd: List[str], env: dict, task: Task) -> TaskResult:
        """Execute Aider with better session management and output parsing"""
        try:
            # Execute the command with proper stdin handling
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.workspace_path)
            )
            
            stdout, stderr = await process.communicate(input=b'')
            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')
            
            # Parse Aider output for better result reporting
            success, parsed_output = self._parse_aider_output(output, error_output, process.returncode)
            
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
    
    def _parse_aider_output(self, stdout: str, stderr: str, return_code: int) -> tuple[bool, str]:
        """Parse Aider output to extract meaningful results"""
        
        # Check for success indicators
        success_indicators = [
            "Applied edit to",
            "Created file",
            "Modified file", 
            "Changes applied successfully"
        ]
        
        # Check for error indicators  
        error_indicators = [
            "Error:",
            "Failed to",
            "Could not",
            "Exception:",
            "Permission denied"
        ]
        
        # Harmless warnings that should be ignored
        harmless_warnings = [
            "Warning: Input is not a terminal",
            "Terminal warning"
        ]
        
        combined_output = stdout + stderr
        
        # Check if failure is due to harmless warnings only
        is_harmless_warning = any(warning in stderr for warning in harmless_warnings)
        has_real_errors = any(error in combined_output for error in error_indicators)
        
        # Determine success based on return code and output content
        if return_code == 0 or (is_harmless_warning and not has_real_errors):
            # Look for actual changes or meaningful output
            if any(indicator in combined_output for indicator in success_indicators):
                success = True
            elif "Tokens:" in combined_output:  # Aider ran but maybe no changes needed
                success = True
            elif stdout.strip():  # Has some output
                success = True
            else:
                success = False
        else:
            success = False
            
        # Extract the most relevant output
        if not success and stderr:
            parsed_output = stderr
        else:
            # Extract meaningful parts of stdout
            lines = stdout.split('\n')
            relevant_lines = []
            
            for line in lines:
                # Skip empty lines and progress indicators
                if line.strip() and not line.startswith('⠋') and not line.startswith('⠴'):
                    relevant_lines.append(line.strip())
            
            parsed_output = '\n'.join(relevant_lines) if relevant_lines else stdout
            
        return success, parsed_output
    
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