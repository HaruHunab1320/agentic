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
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

# fcntl is Unix-only, so we'll use a file-based locking mechanism
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from rich.console import Console

from agentic.models.agent import Agent, AgentConfig, AgentCapability, AgentType, AgentSession
from agentic.models.task import Task, TaskResult

console = Console()

# Global lock for Claude Code initialization to prevent concurrent config access
_claude_init_lock = asyncio.Lock()

# Semaphore for Claude Code processes - removed artificial limit to enable true swarm execution
# The coordination engine will handle resource management and conflict detection
# This allows unlimited agents to work in parallel as originally designed
_claude_process_semaphore = None  # No limit - let the swarm scale!

# File lock for .claude.json access
_claude_config_file_lock = asyncio.Lock()


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
        self._monitor = None
        self._monitor_agent_id = None
        self._activity_monitor = None
    
    def set_monitor(self, monitor, agent_id: str):
        """Set the swarm monitor for status updates"""
        self._monitor = monitor
        self._monitor_agent_id = agent_id
    
    def set_activity_monitor(self, activity_monitor):
        """Set the activity monitor for real-time updates"""
        self._activity_monitor = activity_monitor
    
    def set_progress_monitor(self, progress_monitor):
        """Set the simple progress monitor for lightweight updates"""
        self._progress_monitor = progress_monitor
    
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
            concurrent_tasks=100,  # Unlimited parallelism for swarm execution
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
            # Use global lock only to prevent concurrent setup wizards
            async with _claude_init_lock:
                # Just check authentication status - let Claude manage its own config
                auth_check_result = await self._check_authentication_status()
            
            if auth_check_result == "authenticated":
                self.logger.debug("Claude Code is already authenticated")
                return True
            elif auth_check_result == "needs_browser_auth":
                # Check if we're in automated/swarm execution mode
                is_automated = os.environ.get('AGENTIC_AUTOMATED_MODE', 'false').lower() == 'true'
                if is_automated:
                    self.logger.error("Claude Code requires browser authentication but running in automated mode")
                    self.logger.error("Please run 'claude' manually to complete authentication before using Agentic")
                    raise RuntimeError("Claude Code not authenticated. Run 'claude' manually to authenticate.")
                
                # Notify user about browser authentication
                self.logger.warning("⚠️ Claude Code requires browser authentication")
                console.print("\n[bold yellow]🔐 Claude Code Authentication Required[/bold yellow]")
                console.print("A browser window should open for authentication.")
                console.print("Please complete the authentication flow in your browser.")
                console.print("[dim]If the browser doesn't open, run 'claude' manually to authenticate.[/dim]\n")
                
                # Try to trigger the auth flow with stdin support
                auth_process = await asyncio.create_subprocess_exec(
                    "claude", "--version",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.project_root)
                )
                
                # Monitor the output for authentication completion
                authenticated = False
                start_time = asyncio.get_event_loop().time()
                timeout = 120  # 2 minutes
                
                while not authenticated and (asyncio.get_event_loop().time() - start_time) < timeout:
                    try:
                        # Read available output
                        line = await asyncio.wait_for(auth_process.stdout.readline(), timeout=1.0)
                        if line:
                            output_line = line.decode().strip()
                            self.logger.debug(f"Auth output: {output_line}")
                            
                            # Check for successful login indicators
                            if "logged in as" in output_line.lower():
                                console.print(f"[green]✅ {output_line}[/green]")
                                
                            # Check for "Press Enter to continue" prompt
                            if "press enter to continue" in output_line.lower():
                                self.logger.info("Authentication successful, sending Enter")
                                auth_process.stdin.write(b"\n")
                                await auth_process.stdin.drain()
                                authenticated = True
                                
                    except asyncio.TimeoutError:
                        # Check stderr for any messages
                        try:
                            stderr_line = await asyncio.wait_for(auth_process.stderr.readline(), timeout=0.1)
                            if stderr_line:
                                err_output = stderr_line.decode().strip()
                                if "browser didn't open" in err_output.lower():
                                    # Show the auth URL to user
                                    console.print(f"[yellow]{err_output}[/yellow]")
                        except asyncio.TimeoutError:
                            pass
                        continue
                
                if authenticated:
                    # Wait for process to complete after sending Enter
                    try:
                        await asyncio.wait_for(auth_process.wait(), timeout=5)
                        self.logger.info("✅ Claude Code authentication completed successfully")
                        console.print("[bold green]✅ Authentication successful![/bold green]")
                        return True
                    except asyncio.TimeoutError:
                        # Process might have completed successfully
                        auth_process.kill()
                        return True
                else:
                    auth_process.kill()
                    await auth_process.wait()
                    self.logger.error("Authentication timed out after 2 minutes")
                    console.print("[bold red]❌ Authentication timed out. Please run 'claude' manually to complete setup.[/bold red]")
                    raise RuntimeError("Claude Code authentication timed out. Please run 'claude' manually.")
            
            elif auth_check_result == "needs_setup":
                # Check if we're in automated/swarm execution mode
                is_automated = os.environ.get('AGENTIC_AUTOMATED_MODE', 'false').lower() == 'true'
                if is_automated:
                    self.logger.error("Claude Code requires initial setup but running in automated mode")
                    self.logger.error("Please run 'claude' manually to complete setup before using Agentic")
                    raise RuntimeError("Claude Code not set up. Run 'claude' manually to complete setup.")
                
                # Handle first-time setup
                self.logger.info("Claude Code needs first-time setup")
                # Show this message even if INFO is suppressed
                from rich.console import Console
                Console().print("[yellow]Setting up Claude Code for first-time use...[/yellow]")
                await self._handle_claude_first_time_setup()
                return True
            
            return True
                
        except Exception as e:
            self.logger.error(f"Claude Code authentication check failed: {e}")
            raise  # Re-raise to prevent agent from starting without auth
    
    async def _check_authentication_status(self) -> str:
        """Check if Claude Code is authenticated
        Returns: 'authenticated', 'needs_browser_auth', or 'needs_setup'
        """
        try:
            # Try a simple command that would fail if not authenticated
            test_process = await asyncio.create_subprocess_exec(
                "claude", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(test_process.communicate(), timeout=5)
                
                # Check output for authentication indicators
                stdout_text = stdout.decode()
                stderr_text = stderr.decode()
                combined_output = (stdout_text + stderr_text).lower()
                
                if test_process.returncode == 0 and ("claude" in stdout_text or "version" in stdout_text):
                    # Version command succeeded cleanly - we're authenticated
                    return "authenticated"
                elif "browser" in combined_output or "login" in combined_output or "authorize" in combined_output or "press enter" in combined_output:
                    # Needs browser authentication
                    return "needs_browser_auth"
                else:
                    # Needs setup wizard
                    return "needs_setup"
                    
            except asyncio.TimeoutError:
                # If it times out, likely waiting for input (setup wizard)
                test_process.kill()
                await test_process.wait()
                return "needs_setup"
                
        except FileNotFoundError:
            self.logger.error("Claude Code CLI not found")
            raise RuntimeError("Claude Code CLI not installed. Install with: npm install -g @anthropic-ai/claude-code")
        except Exception as e:
            self.logger.warning(f"Error checking auth status: {e}")
            return "needs_setup"
    
    
    async def _handle_claude_first_time_setup(self) -> None:
        """Handle Claude Code first-time setup (theme selection and additional options)"""
        try:
            # First ensure config is healthy
            await self._ensure_claude_config_health()
            
            # Get configuration for Claude Code setup
            from agentic.models.config import AgenticConfig
            try:
                config = AgenticConfig.load_or_create(self.project_root)
                claude_config = config.claude_code_config
                
                # Check if auto-setup is disabled
                if not claude_config.auto_setup:
                    self.logger.info("Claude Code auto-setup is disabled in configuration")
                    return
            except:
                # If config loading fails, use defaults
                claude_config = None
            
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
            self.logger.info("Claude Code needs setup - automating setup wizard")
            
            # Ensure we have a clean config before setup
            await self._ensure_claude_config_health()
            
            setup_cmd = ["claude", "-p", "echo setup"]
            
            process = await asyncio.create_subprocess_exec(
                *setup_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            # Step 1: Wait for theme selection prompt
            await asyncio.sleep(0.5)
            if claude_config and claude_config.theme == "light":
                process.stdin.write(b"2\n")  # Select light mode
                self.logger.debug("Selected light mode theme")
            else:
                process.stdin.write(b"1\n")  # Select dark mode (default)
                self.logger.debug("Selected dark mode theme")
            await process.stdin.drain()
            
            # Step 2: Wait for second prompt (likely model or feature selection)
            await asyncio.sleep(0.5)
            setup_option = str(claude_config.setup_option if claude_config else 1)
            process.stdin.write(f"{setup_option}\n".encode())
            await process.stdin.drain()
            self.logger.debug(f"Selected option {setup_option} for second setup step")
            
            # Step 3: Handle browser authentication if needed
            # This is handled in _ensure_claude_authentication
            
            # Step 4: Security notes - just press Enter
            await asyncio.sleep(0.8)
            self.logger.debug("Acknowledging security notes")
            process.stdin.write(b"\n")
            await process.stdin.drain()
            
            # Step 5: Terminal setup - select option 1 (recommended settings)
            await asyncio.sleep(0.5)
            self.logger.debug("Selecting recommended terminal settings")
            process.stdin.write(b"1\n")
            await process.stdin.drain()
            
            # Step 6: Trust folder - select option 1 (Yes, proceed)
            await asyncio.sleep(0.5)
            self.logger.debug("Trusting workspace folder")
            process.stdin.write(b"1\n")
            await process.stdin.drain()
            
            # Allow some time for any additional prompts
            for _ in range(2):  # Handle any remaining prompts
                await asyncio.sleep(0.3)
                process.stdin.write(b"1\n")  # Default to option 1
                await process.stdin.drain()
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
                self.logger.info("Claude Code first-time setup completed successfully")
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                self.logger.warning("Claude Code setup timed out - may need manual configuration")
            finally:
                # Always remove setup lock
                try:
                    setup_lock_path.unlink()
                except:
                    pass
                
        except Exception as e:
            self.logger.warning(f"Failed to handle Claude Code setup: {e}")
            # Remove setup lock on error
            try:
                setup_lock_path = Path.home() / ".claude_setup.lock"
                setup_lock_path.unlink()
            except:
                pass
    
    @contextmanager
    def _file_lock(self, file_path: Path, timeout: float = 10.0):
        """Context manager for file locking with timeout"""
        lock_path = file_path.with_suffix('.lock')
        start_time = time.time()
        
        while True:
            try:
                # Try to create lock file atomically
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                try:
                    yield
                finally:
                    os.close(fd)
                    try:
                        lock_path.unlink()
                    except:
                        pass
                break
            except FileExistsError:
                # Lock file exists, check if it's stale or wait
                if time.time() - start_time > timeout:
                    # Force remove stale lock
                    try:
                        lock_path.unlink()
                        self.logger.warning(f"Removed stale lock file: {lock_path}")
                    except:
                        pass
                    raise TimeoutError(f"Could not acquire lock for {file_path} after {timeout} seconds")
                time.sleep(0.1)
    
    async def _read_claude_config_safe(self, claude_config_path: Path) -> Optional[dict]:
        """Safely read Claude config with retries and locking"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with _claude_config_file_lock:
                    with self._file_lock(claude_config_path, timeout=5.0):
                        if not claude_config_path.exists():
                            return None
                        
                        with open(claude_config_path, 'r') as f:
                            content = f.read()
                            if not content.strip():
                                return None
                            return json.loads(content)
            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.debug(f"JSON decode error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(0.2 * (attempt + 1))
            except (IOError, OSError) as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.debug(f"IO error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(0.2 * (attempt + 1))
        return None
    
    async def _write_claude_config_safe(self, claude_config_path: Path, config_data: dict) -> bool:
        """Safely write Claude config with atomic write and locking"""
        try:
            async with _claude_config_file_lock:
                with self._file_lock(claude_config_path, timeout=5.0):
                    # Write to temp file first
                    temp_path = claude_config_path.with_suffix('.tmp')
                    with open(temp_path, 'w') as f:
                        json.dump(config_data, f, indent=2)
                    
                    # Atomic rename
                    temp_path.replace(claude_config_path)
                    return True
        except Exception as e:
            self.logger.error(f"Failed to write Claude config: {e}")
            return False
    
    async def _fix_corrupted_config(self) -> None:
        """Fix corrupted Claude configuration file if detected"""
        claude_config_path = Path.home() / ".claude.json"
        
        if claude_config_path.exists():
            try:
                # Try to read the config safely
                config_data = await self._read_claude_config_safe(claude_config_path)
                if config_data is not None:
                    self.logger.debug("Claude config is valid")
                    return
            except json.JSONDecodeError as e:
                self.logger.warning(f"Corrupted Claude config detected: {e}")
                
                # Backup the corrupted file with timestamp
                timestamp = int(time.time())
                backup_path = claude_config_path.with_suffix(f'.json.backup.{timestamp}')
                
                try:
                    async with _claude_config_file_lock:
                        with self._file_lock(claude_config_path):
                            if claude_config_path.exists():
                                claude_config_path.rename(backup_path)
                                self.logger.info(f"Backed up corrupted config to {backup_path}")
                except Exception as backup_error:
                    self.logger.error(f"Failed to backup corrupted config: {backup_error}")
                    # Try to just delete it
                    try:
                        async with _claude_config_file_lock:
                            with self._file_lock(claude_config_path):
                                if claude_config_path.exists():
                                    claude_config_path.unlink()
                                    self.logger.info("Deleted corrupted Claude config - will be recreated")
                    except Exception as delete_error:
                        self.logger.error(f"Failed to delete corrupted config: {delete_error}")
    
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
        # Ensure config health before execution
        if not await self._ensure_claude_config_health():
            self.logger.error("Failed to ensure Claude config health before task execution")
            return TaskResult(
                task_id=task.id,
                status="failed",
                error="Claude configuration is corrupted. Please run 'claude --version' manually to reinitialize.",
                agent_id=self.session.id if self.session else "unknown"
            )
        
        # Use semaphore if configured, otherwise allow unlimited concurrency
        if _claude_process_semaphore:
            async with _claude_process_semaphore:
                return await self._execute_task_impl(task)
        else:
            # No limit - swarm execution mode
            return await self._execute_task_impl(task)
    
    async def _execute_task_impl(self, task: Task) -> TaskResult:
        """Implementation of task execution"""
        try:
            command_lower = task.command.lower()
            
            # Update monitor status if available
            if self._monitor and self._monitor_agent_id:
                from agentic.core.swarm_monitor import AgentStatus
                self._monitor.update_agent_status(self._monitor_agent_id, AgentStatus.ANALYZING, "Analyzing task...")
                
                # Add specific activity based on task type
                if hasattr(self._monitor, 'update_agent_activity'):
                    if 'test' in command_lower:
                        self._monitor.update_agent_activity(self._monitor_agent_id, "Checking test framework and project structure...")
                    elif 'create' in command_lower or 'build' in command_lower:
                        self._monitor.update_agent_activity(self._monitor_agent_id, "Planning file structure and components...")
                    elif 'fix' in command_lower or 'debug' in command_lower:
                        self._monitor.update_agent_activity(self._monitor_agent_id, "Analyzing code for issues...")
                    elif 'explain' in command_lower or 'analyze' in command_lower or 'tell me' in command_lower:
                        self._monitor.update_agent_activity(self._monitor_agent_id, "Reading and understanding codebase...")
                    else:
                        self._monitor.update_agent_activity(self._monitor_agent_id, "Processing your query...")
            
            # Update simple progress monitor if available
            elif hasattr(self, '_progress_monitor') and self._progress_monitor:
                if 'test' in command_lower:
                    self._progress_monitor.update_status("Analyzing", "Checking test framework...")
                elif 'explain' in command_lower or 'analyze' in command_lower or 'tell me' in command_lower:
                    self._progress_monitor.update_status("Analyzing", "Reading project structure...")
                else:
                    self._progress_monitor.update_status("Processing", "Understanding your request...")
            
            # Determine optimal execution mode based on task characteristics
            execution_mode = self._determine_execution_mode(task)
            
            # Build advanced Claude Code command
            cmd = self._build_enhanced_claude_command(task, execution_mode)
            
            self.logger.info(f"Executing {execution_mode} task: {task.command[:50]}...")
            self.logger.debug(f"Full command: {' '.join(cmd)}")  # Log full command for debugging
            self.logger.debug(f"Working directory: {self.workspace_path}")
            
            # Update monitor to executing
            if self._monitor and self._monitor_agent_id:
                self._monitor.update_agent_status(self._monitor_agent_id, AgentStatus.EXECUTING, task.command[:50] + "...")
                
                # Update activity based on execution mode
                if hasattr(self._monitor, 'update_agent_activity'):
                    self._monitor.update_agent_activity(self._monitor_agent_id, "Executing task in print mode...")
            
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
        """Determine execution mode - always use print for automated execution"""
        # For multi-agent autonomous execution, always use print mode
        # This ensures predictable, non-interactive behavior
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
            # Print mode for automated tasks
            cmd.extend(["-p"])
            
            # For analysis tasks, don't use JSON format to get better streaming
            if not any(word in task.command.lower() for word in ['tell me', 'what', 'explain', 'analyze', 'find']):
                # Use JSON output format for better parsing of action tasks
                cmd.extend(["--output-format", "json"])
            
            # Remove verbose flag for query mode - it can cause stalling
            # cmd.extend(["--verbose"])
            
            # Don't limit turns for queries - let Claude use as many tools as needed
            # Only limit turns for implementation tasks to prevent runaway execution
            if not any(word in task.command.lower() for word in ['tell me', 'what', 'explain', 'analyze', 'find', 'why', 'how', 'when', 'where']):
                cmd.extend(["--max-turns", "10"])  # Limit only for implementation tasks
            
            # Use natural prompt without over-constraining
            prompt = self._build_natural_prompt(task)
            cmd.append(prompt)
            
        
        return cmd
    
    def _build_natural_prompt(self, task: Task) -> str:
        """Build a natural prompt that trusts Claude's capabilities"""
        # Just pass the command naturally - Claude understands context
        return task.command
    
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
            # Special handling for architect tasks which create many files
            command_lower = task.command.lower()
            is_architect_task = any(keyword in command_lower for keyword in ['architect', 'design the complete', 'technical specification', 'api specification'])
            
            # Set timeout based on task type
            if is_architect_task:
                timeout = 1800  # 30 minutes for complex architect tasks
                self.logger.info(f"Using extended timeout of {timeout}s for architect task")
            elif any(word in command_lower for word in ['tell me', 'what', 'explain', 'analyze', 'find', 'search']):
                timeout = 120  # 2 minutes for analysis tasks to allow thorough exploration
                self.logger.info(f"Using analysis timeout of {timeout}s for analysis task")
            else:
                timeout = 300  # 5 minutes for regular tasks
            
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
                
                # For architect tasks, check for completion periodically
                if is_architect_task:
                    self.logger.info("Monitoring architect task execution...")
                    stdout_data = []
                    stderr_data = []
                    
                    start_time = asyncio.get_event_loop().time()
                    while True:
                        try:
                            # Check if process has completed
                            if process.returncode is not None:
                                # Process completed, get any remaining output
                                remaining_stdout, remaining_stderr = await process.communicate()
                                if remaining_stdout:
                                    stdout_data.append(remaining_stdout)
                                if remaining_stderr:
                                    stderr_data.append(remaining_stderr)
                                break
                            
                            # Check timeout
                            elapsed = asyncio.get_event_loop().time() - start_time
                            if elapsed > timeout:
                                raise asyncio.TimeoutError()
                            
                            # Wait a bit before checking again
                            await asyncio.sleep(1.0)
                            
                            # Update progress periodically
                            if self._monitor and self._monitor_agent_id and int(elapsed) % 10 == 0:
                                # Note: Enhanced monitor calculates progress automatically
                                # Just update activity to show we're still working
                                if hasattr(self._monitor, 'update_agent_activity'):
                                    self._monitor.update_agent_activity(self._monitor_agent_id, f"Processing architect task... ({int(elapsed)}s)")
                                
                        except asyncio.TimeoutError:
                            raise
                    
                    stdout = b''.join(stdout_data)
                    stderr = b''.join(stderr_data)
                else:
                    # Normal execution for non-architect tasks
                    # Monitor output in real-time for all tasks when using JSON output
                    use_streaming = execution_mode == "print" or any(word in command_lower for word in ['test', 'tests', 'passing'])
                    
                    if use_streaming:
                        stdout_data = []
                        stderr_data = []
                        
                        # Create async tasks to read stdout and stderr
                        async def read_stream(stream, data_list, stream_name):
                            json_buffer = ""  # Buffer for accumulating JSON data
                            in_json_mode = False
                            nonlocal execution_mode  # Access execution_mode from outer scope
                            
                            while True:
                                line = await stream.readline()
                                if not line:
                                    break
                                data_list.append(line)
                                
                                # Parse and report activities
                                line_text = line.decode('utf-8', errors='replace').strip()
                                
                                # Detect JSON output start or Claude's thinking
                                if line_text.startswith('{') and '"messages"' in line_text:
                                    in_json_mode = True
                                    json_buffer = line_text
                                elif in_json_mode:
                                    json_buffer += line_text
                                    # Try to parse accumulated JSON
                                    if line_text.endswith('}'):
                                        try:
                                            await self._parse_claude_json_stream(json_buffer)
                                            json_buffer = ""
                                            in_json_mode = False
                                        except:
                                            # Continue accumulating
                                            pass
                                # Detect Claude thinking/working messages
                                elif self._monitor and self._monitor_agent_id and hasattr(self._monitor, 'update_agent_activity'):
                                    # Check for Claude's common patterns that indicate work is happening
                                    if any(pattern in line_text.lower() for pattern in [
                                        'thinking', 'analyzing', 'reading', 'looking', 'checking',
                                        'searching', 'found', 'let me', "i'll", 'processing'
                                    ]):
                                        self._monitor.update_agent_activity(self._monitor_agent_id, f"Claude: {line_text[:100]}...")
                                
                                # Regular activity parsing for non-JSON output
                                elif line_text and self._monitor and self._monitor_agent_id and hasattr(self._monitor, 'update_agent_activity'):
                                    # Detect specific activities
                                    # Skip Claude's interactive prompts
                                    if '?' in line_text and 'shortcuts' in line_text:
                                        continue  # Skip interactive prompt
                                    elif 'Working in' in line_text:
                                        self._monitor.update_agent_activity(self._monitor_agent_id, f"Working in: {line_text}")
                                    elif 'Model:' in line_text:
                                        self._monitor.update_agent_activity(self._monitor_agent_id, f"Using model: {line_text}")
                                    elif 'npm test' in line_text or 'jest' in line_text or 'running tests' in line_text.lower():
                                        self._monitor.update_agent_activity(self._monitor_agent_id, "Running test suite...")
                                    elif 'PASS' in line_text or '✓' in line_text:
                                        self._monitor.update_agent_activity(self._monitor_agent_id, f"Tests passing: {line_text[:50]}...")
                                    elif 'FAIL' in line_text or '✗' in line_text or '✖' in line_text:
                                        self._monitor.update_agent_activity(self._monitor_agent_id, f"Test failure: {line_text[:50]}...")
                                    elif 'passed' in line_text and 'failed' in line_text:
                                        self._monitor.update_agent_activity(self._monitor_agent_id, line_text[:60])
                                    elif 'Error:' in line_text or 'error:' in line_text:
                                        self._monitor.update_agent_activity(self._monitor_agent_id, f"Error: {line_text[:50]}...")
                                    elif 'Looking at' in line_text or 'Checking' in line_text:
                                        self._monitor.update_agent_activity(self._monitor_agent_id, line_text[:60])
                                    elif 'Running:' in line_text:
                                        self._monitor.update_agent_activity(self._monitor_agent_id, line_text[:60])
                                    elif 'Turn' in line_text and 'of' in line_text:
                                        # Verbose mode shows turns
                                        self._monitor.update_agent_activity(self._monitor_agent_id, f"Processing: {line_text}")
                                    elif 'Tool' in line_text and 'use:' in line_text:
                                        # Tool usage from verbose output
                                        self._monitor.update_agent_activity(self._monitor_agent_id, line_text[:60])
                                    else:
                                        # For query mode, show any other activity
                                        if execution_mode == "print" and line_text and len(line_text) > 10:
                                            self._monitor.update_agent_activity(self._monitor_agent_id, f"Claude: {line_text[:80]}...")
                        
                        # Read both streams concurrently
                        await asyncio.gather(
                            read_stream(process.stdout, stdout_data, "stdout"),
                            read_stream(process.stderr, stderr_data, "stderr"),
                            process.wait()
                        )
                        
                        stdout = b''.join(stdout_data)
                        stderr = b''.join(stderr_data)
                    else:
                        # Normal execution for non-test tasks with periodic updates
                        # Start a background task to provide periodic updates
                        async def provide_periodic_updates():
                            update_messages = [
                                "Claude is thinking...",
                                "Analyzing the codebase...",
                                "Formulating response...",
                                "Processing information...",
                                "Gathering insights..."
                            ]
                            message_index = 0
                            while process.returncode is None:
                                await asyncio.sleep(3)  # Update every 3 seconds
                                if self._monitor and self._monitor_agent_id and hasattr(self._monitor, 'update_agent_activity'):
                                    self._monitor.update_agent_activity(
                                        self._monitor_agent_id, 
                                        update_messages[message_index % len(update_messages)]
                                    )
                                    message_index += 1
                                elif hasattr(self, '_progress_monitor') and self._progress_monitor:
                                    self._progress_monitor.update_status(
                                        "Processing",
                                        update_messages[message_index % len(update_messages)]
                                    )
                                    message_index += 1
                        
                        # Start periodic updates
                        update_task = asyncio.create_task(provide_periodic_updates())
                        
                        try:
                            stdout, stderr = await asyncio.wait_for(
                                process.communicate(), 
                                timeout=timeout
                            )
                        finally:
                            # Cancel the update task
                            update_task.cancel()
                            try:
                                await update_task
                            except asyncio.CancelledError:
                                pass
            except asyncio.TimeoutError:
                self.logger.error(f"Task timed out after {timeout} seconds")
                process.kill()
                await process.wait()
                
                # For architect tasks, check if files were created despite timeout
                if is_architect_task:
                    # Check for architecture files
                    arch_files = ['README.md', 'ARCHITECTURE.md', 'API_SPEC.md', 'DATA_MODEL.md']
                    created_files = []
                    for filename in arch_files:
                        file_path = self.workspace_path / filename
                        if file_path.exists():
                            created_files.append(filename)
                    
                    if created_files:
                        self.logger.warning(f"Architect task timed out but created files: {created_files}")
                        # Consider it successful if core files were created
                        return TaskResult(
                            task_id=task.id,
                            status="completed",
                            output=f"Created architecture documentation: {', '.join(created_files)}",
                            agent_id=self.session.id if self.session else "unknown",
                            files_modified=[self.workspace_path / f for f in created_files]
                        )
                
                return TaskResult(
                    task_id=task.id,
                    status="failed",
                    error=f"Task timed out after {timeout} seconds",
                    agent_id=self.session.id if self.session else "unknown"  
                )
            
            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')
            
            # Check for config errors in stderr first
            if "claude.json" in error_output and ("invalid JSON" in error_output or "Unexpected token" in error_output or "SyntaxError" in error_output):
                self.logger.warning("Claude.json corruption detected in output")
                # Try to fix and suggest retry
                async with _claude_init_lock:
                    await self._fix_corrupted_config()
                return TaskResult(
                    task_id=task.id,
                    status="failed",
                    error="Claude configuration was corrupted and has been reset. Please retry the task.",
                    agent_id=self.session.id if self.session else "unknown"
                )
            
            # Parse output based on execution mode
            success, parsed_output = self._parse_claude_output(
                output, error_output, process.returncode, execution_mode
            )
            
            if success:
                self.logger.info("Task completed successfully")
                
                # Update monitor status to completed if available
                if self._monitor and self._monitor_agent_id:
                    from agentic.core.swarm_monitor import AgentStatus
                    self._monitor.update_agent_status(self._monitor_agent_id, AgentStatus.FINALIZING)
                    # Give a moment for finalization
                    await asyncio.sleep(0.5)
                    self._monitor.update_agent_status(self._monitor_agent_id, AgentStatus.COMPLETED)
                    # Note: Enhanced monitor calculates progress automatically from task completion
                    self.logger.info(f"Updated monitor status to COMPLETED for agent {self._monitor_agent_id}")
                
                # Extract files created from output
                files_created = self._extract_files_from_output(parsed_output)
                
                result = TaskResult(
                    task_id=task.id,
                    status="completed",
                    output=parsed_output,
                    agent_id=self.session.id if self.session else "unknown"
                )
                
                # Add files_modified attribute if files were created
                if files_created:
                    result.files_modified = files_created
                
                return result
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
            
            # Check if it's a JSON config error
            error_str = str(e)
            if "claude.json" in error_str and ("JSON" in error_str or "Unexpected token" in error_str or "SyntaxError" in error_str):
                # Try to fix the config and retry once
                self.logger.warning("Detected claude.json error, attempting to fix...")
                async with _claude_init_lock:
                    await self._fix_corrupted_config()
                
                # Return error suggesting retry
                return TaskResult(
                    task_id=task.id,
                    status="failed",
                    error="Claude configuration was corrupted and has been reset. Please retry the task.",
                    agent_id=self.session.id if self.session else "unknown"
                )
            return TaskResult(
                task_id=task.id,
                status="failed",
                error=str(e),
                agent_id=self.session.id if self.session else "unknown"
            )
    
    async def _parse_claude_json_stream(self, json_data: str):
        """Parse Claude's JSON output and extract tool uses and activities"""
        try:
            import json
            data = json.loads(json_data)
            
            # Handle the messages array if present
            if isinstance(data, dict) and 'messages' in data:
                for message in data.get('messages', []):
                    if message.get('role') == 'assistant':
                        content = message.get('content', [])
                        
                        # Process each content block
                        for block in content if isinstance(content, list) else [content]:
                            if isinstance(block, dict):
                                # Tool use block
                                if block.get('type') == 'tool_use':
                                    tool_name = block.get('name', 'Unknown tool')
                                    tool_input = block.get('input', {})
                                    
                                    # Create descriptive activity messages based on tool
                                    activity = self._format_tool_activity(tool_name, tool_input)
                                    if activity:
                                        if self._monitor and self._monitor_agent_id:
                                            self._monitor.update_agent_activity(self._monitor_agent_id, activity)
                                        elif hasattr(self, '_progress_monitor') and self._progress_monitor:
                                            self._progress_monitor.update_status("Working", activity)
                                
                                # Text block (Claude's thoughts)
                                elif block.get('type') == 'text':
                                    text = block.get('text', '').strip()
                                    if text and self._monitor and self._monitor_agent_id:
                                        # Extract key activities from Claude's text
                                        activity = self._extract_activity_from_text(text)
                                        if activity:
                                            self._monitor.update_agent_activity(self._monitor_agent_id, activity)
                            
                            # Handle string content (older format)
                            elif isinstance(block, str) and block.strip():
                                activity = self._extract_activity_from_text(block.strip())
                                if activity and self._monitor and self._monitor_agent_id:
                                    self._monitor.update_agent_activity(self._monitor_agent_id, activity)
        
        except Exception as e:
            self.logger.debug(f"Failed to parse Claude JSON stream: {e}")
    
    def _format_tool_activity(self, tool_name: str, tool_input: dict) -> Optional[str]:
        """Format tool usage into human-readable activity"""
        # Map Claude Code tool names to descriptive activities
        if tool_name in ["Read", "read_file"]:
            file_path = tool_input.get('file_path', tool_input.get('path', 'unknown file'))
            return f"📖 Reading {file_path}"
        
        elif tool_name in ["Write", "write_file"]:
            file_path = tool_input.get('file_path', tool_input.get('path', 'unknown file'))
            return f"✍️ Writing to {file_path}"
        
        elif tool_name in ["Edit", "edit_file"]:
            file_path = tool_input.get('file_path', tool_input.get('path', 'unknown file'))
            old_str = tool_input.get('old_string', '')[:30]
            return f"✏️ Editing {file_path}" + (f" - replacing '{old_str}...'" if old_str else "")
        
        elif tool_name == "MultiEdit":
            file_path = tool_input.get('file_path', 'unknown file')
            num_edits = len(tool_input.get('edits', []))
            return f"🔧 Making {num_edits} edits to {file_path}"
        
        elif tool_name in ["Bash", "run_command"]:
            command = tool_input.get('command', 'unknown command')
            description = tool_input.get('description', '')
            
            # Use description if available
            if description:
                return f"⚡ {description}"
            # Otherwise parse command
            elif 'test' in command.lower() or 'jest' in command.lower():
                return f"🧪 Running tests: {command[:60]}..."
            elif 'npm' in command or 'yarn' in command:
                return f"📦 Package command: {command[:60]}..."
            elif 'git' in command:
                return f"🔀 Git operation: {command[:60]}..."
            else:
                return f"⚡ Executing: {command[:60]}..."
        
        elif tool_name in ["Grep", "search_files"]:
            pattern = tool_input.get('pattern', '')
            path = tool_input.get('path', '.')
            include = tool_input.get('include', '')
            if include:
                return f"🔍 Searching for '{pattern}' in {include} files"
            else:
                return f"🔍 Searching for '{pattern}' in {path}"
        
        elif tool_name in ["Glob", "find_files"]:
            pattern = tool_input.get('pattern', '')
            path = tool_input.get('path', '.')
            return f"🔎 Finding files matching '{pattern}' in {path}"
        
        elif tool_name in ["LS", "list_files"]:
            path = tool_input.get('path', '.')
            return f"📁 Listing files in {path}"
        
        elif tool_name == "NotebookRead":
            notebook_path = tool_input.get('notebook_path', 'notebook')
            return f"📓 Reading Jupyter notebook {notebook_path}"
        
        elif tool_name == "NotebookEdit":
            notebook_path = tool_input.get('notebook_path', 'notebook')
            cell_number = tool_input.get('cell_number', '?')
            return f"📝 Editing cell {cell_number} in {notebook_path}"
        
        elif tool_name == "WebFetch":
            url = tool_input.get('url', '')
            return f"🌐 Fetching content from {url[:50]}..."
        
        elif tool_name == "WebSearch":
            query = tool_input.get('query', '')
            return f"🔍 Searching web for: {query}"
        
        elif tool_name == "TodoRead":
            return "📋 Checking todo list"
        
        elif tool_name == "TodoWrite":
            todos = tool_input.get('todos', [])
            return f"✅ Updating todo list ({len(todos)} items)"
        
        elif tool_name == "str_replace_based_edit_tool":
            file_path = tool_input.get('command', '').split()[-1] if tool_input.get('command') else 'file'
            return f"🔧 Modifying {file_path}"
        
        else:
            # Generic tool activity
            return f"🔧 Using {tool_name}"
    
    def _extract_activity_from_text(self, text: str) -> Optional[str]:
        """Extract meaningful activity from Claude's text output"""
        # Only extract first line or key phrases
        lines = text.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Skip very long lines (likely explanations)
        if len(first_line) > 100:
            # Look for key action phrases
            if "I'll" in first_line or "Let me" in first_line:
                # Extract the action part
                if "I'll" in first_line:
                    action_part = first_line.split("I'll", 1)[1].split('.')[0].strip()
                    return f"Planning to {action_part[:80]}..."
                elif "Let me" in first_line:
                    action_part = first_line.split("Let me", 1)[1].split('.')[0].strip()
                    return f"Starting to {action_part[:80]}..."
            return None
        
        # Return meaningful short activities
        if any(keyword in first_line.lower() for keyword in ['analyzing', 'searching', 'reading', 'writing', 'creating', 'found', 'detected']):
            return first_line[:80] + ("..." if len(first_line) > 80 else "")
        
        return None

    def _extract_files_from_output(self, output: str) -> List[Path]:
        """Extract file paths mentioned in Claude's output"""
        files = []
        
        # Look for common patterns in Claude's output
        import re
        
        # Pattern 1: "Created file: path/to/file"
        created_pattern = r"(?:Created|Wrote|Writing|Created file|File created)(?:\s+file)?:\s*([^\s]+)"
        # Pattern 2: "path/to/file created"
        file_created_pattern = r"([^\s]+\.(?:md|ts|js|json|py|yml|yaml))\s+created"
        # Pattern 3: Markdown/documentation files
        doc_pattern = r"([A-Z_]+\.md)"
        
        for pattern in [created_pattern, file_created_pattern, doc_pattern]:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                if match and not match.startswith('http'):
                    file_path = Path(match)
                    if not file_path.is_absolute():
                        file_path = self.workspace_path / file_path
                    files.append(file_path)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        
        return unique_files
    
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
        
        # Handle JSON output format (used in print mode with --output-format json)
        if execution_mode == "print":
            try:
                import json
                # Try to parse as JSON first
                output_data = json.loads(stdout.strip())
                
                # Handle JSON array format (new Claude Code format)
                if isinstance(output_data, list) and output_data:
                    # Extract assistant messages from the array
                    assistant_messages = []
                    usage_info = None
                    
                    for msg in output_data:
                        if isinstance(msg, dict):
                            if msg.get('type') == 'assistant':
                                content = msg.get('content', '')
                                if content:
                                    assistant_messages.append(content)
                            elif msg.get('type') == 'system' and msg.get('subtype') == 'usage':
                                usage_info = msg.get('usage', {})
                    
                    # Return concatenated assistant messages
                    if assistant_messages:
                        return True, '\n\n'.join(assistant_messages)
                    else:
                        # No assistant messages found, return the raw JSON
                        return True, stdout.strip()
                
                # For JSON output, we want to preserve the full structure
                # so our enhanced parser can extract all the details
                if isinstance(output_data, dict) and ('messages' in output_data or 'usage' in output_data):
                    # Return the full JSON for our enhanced parser to process
                    return True, stdout.strip()
                
                # Legacy format handling for other JSON structures
                if isinstance(output_data, dict):
                    if 'content' in output_data:
                        return True, output_data['content']
                    elif 'response' in output_data:
                        return True, output_data['response']
                    elif 'message' in output_data:
                        return True, output_data['message']
                    elif 'result' in output_data and 'content' in output_data['result']:
                        return True, output_data['result']['content']
                    else:
                        # Return formatted JSON as fallback
                        return True, json.dumps(output_data, indent=2)
                else:
                    return True, str(output_data)
                    
            except json.JSONDecodeError:
                # Fall through to text parsing if not valid JSON
                pass
        
        # Handle text output (fallback or non-JSON mode)
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
        # Special case for architect tasks - check if files were created
        if "architect" in execution_mode or "design" in str(stdout).lower():
            # Check for architecture files in the workspace
            arch_files = ['README.md', 'ARCHITECTURE.md', 'API_SPEC.md', 'DATA_MODEL.md']
            found_files = []
            workspace = getattr(self, 'workspace_path', Path.cwd())
            
            for filename in arch_files:
                if (workspace / filename).exists():
                    found_files.append(filename)
            
            if found_files:
                return True, f"Architecture documentation created: {', '.join(found_files)}"
        
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
    
    @contextmanager
    def _file_lock(self, file_path: Path, timeout: float = 10.0):
        """Context manager for file locking with timeout"""
        lock_path = file_path.with_suffix('.lock')
        start_time = time.time()
        
        # Try to create lock file exclusively
        while True:
            try:
                # Try to create the lock file exclusively
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                # Lock exists, check if it's stale
                if time.time() - start_time > timeout:
                    # Remove stale lock
                    try:
                        lock_path.unlink()
                        self.logger.warning(f"Removed stale lock file: {lock_path}")
                    except:
                        pass
                    raise TimeoutError(f"Failed to acquire lock on {file_path} after {timeout}s")
                # Brief sleep before retry
                time.sleep(0.1)
        
        try:
            yield
        finally:
            # Release lock
            try:
                lock_path.unlink()
            except:
                pass
    
    async def _read_claude_config_safe(self, claude_config_path: Path) -> Optional[dict]:
        """Safely read Claude config with retries and locking"""
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                with self._file_lock(claude_config_path):
                    if claude_config_path.exists():
                        with open(claude_config_path, 'r') as f:
                            return json.load(f)
                    return None
            except json.JSONDecodeError as e:
                self.logger.warning(f"Corrupted Claude config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Last attempt failed, config is corrupted
                    return None
            except Exception as e:
                self.logger.error(f"Error reading Claude config: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    return None
        
        return None
    
    async def _write_claude_config_safe(self, claude_config_path: Path, config_data: dict) -> bool:
        """Safely write Claude config with atomic write and locking"""
        try:
            # Write to temp file first
            temp_path = claude_config_path.with_suffix('.tmp')
            
            with self._file_lock(claude_config_path):
                # Write to temp file
                with open(temp_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # Atomic rename
                temp_path.replace(claude_config_path)
                
            return True
        except Exception as e:
            self.logger.error(f"Error writing Claude config: {e}")
            # Clean up temp file if it exists
            try:
                temp_path.unlink()
            except:
                pass
            return False
    
    async def _ensure_claude_config_health(self) -> bool:
        """Ensure Claude config is healthy, create default if needed"""
        claude_config_path = Path.home() / ".claude.json"
        
        # Try to read existing config
        config = await self._read_claude_config_safe(claude_config_path)
        
        if config is None:
            # Config doesn't exist or is corrupted, create default
            self.logger.info("Creating default Claude config")
            default_config = {
                "version": "1.0",
                "theme": "dark",
                "analytics": False,
                "created_at": datetime.now().isoformat()
            }
            
            # Back up corrupted config if it exists
            if claude_config_path.exists():
                backup_path = claude_config_path.with_suffix(f'.backup.{int(time.time())}')
                try:
                    claude_config_path.rename(backup_path)
                    self.logger.info(f"Backed up corrupted config to {backup_path}")
                except:
                    pass
            
            # Write default config
            return await self._write_claude_config_safe(claude_config_path, default_config)
        
        # Config exists and is valid
        return True