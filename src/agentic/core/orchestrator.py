"""
Main Orchestrator for coordinating agent activities
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic.core.agent_registry import AgentRegistry
from agentic.core.command_router import CommandRouter
from agentic.core.coordination_engine import CoordinationEngine, ExecutionResult
from agentic.core.intent_classifier import IntentClassifier
from agentic.core.project_analyzer import ProjectAnalyzer
from agentic.core.shared_memory import SharedMemory
from agentic.core.inter_agent_communication import InterAgentCommunicationHub
from agentic.core.autonomous_execution import ExecutionMode
from agentic.core.hierarchical_agents import SupervisorAgent, ResourceManager
from agentic.models.agent import AgentSession, AgentType
from agentic.models.config import AgenticConfig
from agentic.models.project import ProjectStructure
from agentic.models.task import ExecutionPlan, Task, TaskResult
from agentic.utils.logging import LoggerMixin


class Orchestrator(LoggerMixin):
    """
    Main orchestrator that coordinates all agent activities
    
    This class serves as the central coordination point for the Agentic system,
    managing agents, routing commands, and executing tasks.
    """
    
    def __init__(self, config: AgenticConfig):
        super().__init__()
        self.config = config
        
        # Initialize core components
        self.project_analyzer = None  # Will be initialized when needed with project_path
        self.agent_registry = AgentRegistry(workspace_path=config.workspace_path)
        self.shared_memory = SharedMemory()
        self.intent_classifier = IntentClassifier()
        self.command_router = CommandRouter(self.agent_registry)
        # Initialize coordination engine with safety features disabled by default
        # TODO: Enable safety features once database issues are resolved
        enable_safety = config.enable_safety if hasattr(config, 'enable_safety') else False
        self.coordination_engine = CoordinationEngine(
            self.agent_registry, 
            self.shared_memory, 
            config.workspace_path,
            enable_safety=enable_safety
        )
        self.logger.info(f"Initialized CoordinationEngine with safety features {'enabled' if enable_safety else 'disabled'}")
        
        # NEW: Inter-agent communication system
        self.communication_hub = InterAgentCommunicationHub(workspace_path=config.workspace_path)
        
        # State
        self._project_structure: Optional[ProjectStructure] = None
        self._is_initialized = False
        self._background_task: Optional[asyncio.Task] = None
        self._current_mode: ExecutionMode = ExecutionMode.AUTONOMOUS
        self._resource_manager: Optional[ResourceManager] = None
        self._supervisor_agent: Optional[SupervisorAgent] = None
    
    async def initialize(self, project_path: Optional[Path] = None) -> bool:
        """Initialize the orchestrator with project analysis"""
        try:
            self.logger.info("Initializing Agentic orchestrator...")
            
            # Determine project path
            if project_path is None:
                project_path = self.config.workspace_path
            
            # Initialize project analyzer with project path
            self.project_analyzer = ProjectAnalyzer(project_path)
            
            # Analyze project structure
            self._project_structure = await self.project_analyzer.analyze()
            self.shared_memory.set_project_structure(self._project_structure)
            self.logger.info(f"Project analyzed: {self._project_structure.root_path.name}")
            
            # Initialize agent registry
            await self.agent_registry.initialize()
            
            # Auto-spawn agents if enabled
            if self.config.auto_spawn_agents and self._project_structure:
                self.logger.info("Auto-spawning agents based on project structure...")
                spawned_agents = await self.agent_registry.auto_spawn_agents(self._project_structure)
                self.logger.info(f"Auto-spawned {len(spawned_agents)} agents")
                for agent in spawned_agents:
                    self.logger.info(f"  - {agent.agent_config.name} ({agent.agent_config.agent_type})")
            
            # Start background tasks
            self._background_task = asyncio.create_task(self._background_maintenance())
            
            self._is_initialized = True
            self.logger.info("Orchestrator initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    @property
    def current_mode(self) -> ExecutionMode:
        """Get the current execution mode"""
        return self._current_mode
    
    def enable_safety_features(self, enabled: bool = True) -> None:
        """Enable or disable safety features in the coordination engine"""
        self.coordination_engine.enable_safety = enabled
        self.logger.info(f"Safety features {'enabled' if enabled else 'disabled'}")
    
    def set_mode(self, mode: ExecutionMode) -> None:
        """Set the execution mode"""
        self.logger.info(f"Setting execution mode to: {mode.value}")
        self._current_mode = mode
        
        # Initialize resources for hierarchical mode if needed
        if mode == ExecutionMode.HIERARCHICAL and not self._resource_manager:
            self._resource_manager = ResourceManager()
            self.logger.info("Initialized ResourceManager for hierarchical mode")
    
    async def execute_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """Execute a single command"""
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        
        # Check if this is a mode switch command
        if command.strip().lower().startswith("/mode"):
            return await self._handle_mode_command(command)
        
        # Check if we're in hierarchical mode
        if self._current_mode == ExecutionMode.HIERARCHICAL:
            return await self.execute_hierarchical_command(command, context)
        
        self.logger.info(f"Executing command in {self._current_mode.value} mode: {command[:50]}...")
        
        # Check if this is a follow-up command that needs context
        if context and context.get('session_manager'):
            enriched_command = await self._enrich_command_with_context(command, context)
        else:
            enriched_command = command
        
        # Determine if we should show monitoring
        should_monitor = context and context.get('enable_monitoring', True)
        status_updater = context.get('status_updater') if context else None
        
        try:
            # Start monitoring if enabled
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                await self.coordination_engine.swarm_monitor.start_monitoring()
            # Classify intent - use enriched command
            intent = await self.intent_classifier.analyze_intent(enriched_command)
            self.logger.debug(f"Classified intent: {intent.task_type}, requires_coordination: {intent.requires_coordination}")
            
            # Check if this requires multi-agent coordination
            if intent.requires_coordination:
                self.logger.info("Intent requires multi-agent coordination, delegating to execute_multi_agent_command")
                # Stop monitoring as multi-agent command will start its own
                if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                    await self.coordination_engine.swarm_monitor.stop_monitoring()
                # Execute as multi-agent command
                return await self.execute_multi_agent_command(enriched_command, context)
            
            # Create task from intent with enriched command
            task = Task.from_intent(intent, enriched_command)
            
            # Check if we have any agents, if not spawn one
            available_agents = self.agent_registry.get_available_agents()
            if not available_agents:
                self.logger.info("No agents available, spawning a backend agent for the task...")
                try:
                    await self._auto_spawn_agent(intent)
                except Exception as e:
                    self.logger.error(f"Failed to auto-spawn agent: {e}")
                    return TaskResult(
                        task_id=task.id,
                        agent_id="none",
                        status="failed",
                        output="",
                        error=f"No agents available and failed to spawn: {e}"
                    )
            
            # Route to appropriate agent
            routing_decision = await self.command_router.route_command(command, context or {})
            
            if not routing_decision.primary_agent:
                return TaskResult(
                    task_id=task.id,
                    agent_id="none",
                    status="failed",
                    output="",
                    error="No suitable agent found for command"
                )
            
            # Execute with best agent
            best_agent = routing_decision.primary_agent
            agent_instance = self.agent_registry.get_agent_by_id(best_agent.id)
            
            if not agent_instance:
                return TaskResult(
                    task_id=task.id,
                    agent_id=best_agent.id,
                    status="failed",
                    output="",
                    error="Agent instance not found"
                )
            
            # Register agent with monitor if monitoring is enabled
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                monitor = self.coordination_engine.swarm_monitor
                from agentic.core.swarm_monitor_unified import AgentStatus
                
                # Determine a meaningful role based on agent type and task
                if "test" in command.lower():
                    role = "test_runner"
                elif "claude" in best_agent.agent_config.agent_type.value.lower():
                    role = "analyzer"
                elif "frontend" in best_agent.agent_config.agent_type.value.lower():
                    role = "frontend_dev"
                elif "backend" in best_agent.agent_config.agent_type.value.lower():
                    role = "backend_dev"
                else:
                    role = best_agent.agent_config.agent_type.value.replace("_", " ").title()
                
                # Register agent with monitor
                monitor.register_agent(
                    agent_id=best_agent.id,
                    agent_name=best_agent.agent_config.name,
                    agent_type=best_agent.agent_config.agent_type.value,
                    role=role
                )
                monitor.update_agent_status(best_agent.id, AgentStatus.INITIALIZING)
                
                # Set monitor reference in agent for status updates
                if hasattr(agent_instance, 'set_monitor'):
                    agent_instance.set_monitor(monitor, best_agent.id)
            
            # Set up simple progress monitor if we have a status updater but no full monitoring
            simple_monitor = None
            if not should_monitor and status_updater:
                from agentic.core.simple_progress import SimpleProgressMonitor
                simple_monitor = SimpleProgressMonitor(status_updater)
                simple_monitor.start_monitoring(best_agent.id, best_agent.agent_config.name)
                
                # Give agent a reference to update progress
                if hasattr(agent_instance, 'set_progress_monitor'):
                    agent_instance.set_progress_monitor(simple_monitor)
            
            # Register task and execute
            await self.shared_memory.register_task(task)
            
            # Update monitor - starting task
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                monitor = self.coordination_engine.swarm_monitor
                monitor.update_agent_status(best_agent.id, AgentStatus.SETTING_UP)
                
                # Use appropriate method based on monitor type
                if hasattr(monitor, 'start_agent_task'):
                    monitor.start_agent_task(best_agent.id, task.id, task.command[:50] + "...")
                else:
                    monitor.start_task(best_agent.id, task.id, task.command[:50] + "...")
            
            result = await agent_instance.execute_task(task)
            
            # Update monitor - task completed
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                monitor = self.coordination_engine.swarm_monitor
                is_success = result.status == "completed"
                
                # Use appropriate method based on monitor type
                if hasattr(monitor, 'complete_agent_task'):
                    monitor.complete_agent_task(best_agent.id, task.id, is_success)
                else:
                    monitor.complete_task(best_agent.id, task.id, is_success)
                
                if is_success:
                    monitor.update_agent_status(best_agent.id, AgentStatus.COMPLETED)
                else:
                    monitor.update_agent_status(best_agent.id, AgentStatus.FAILED)
            
            await self.shared_memory.complete_task(task.id, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return TaskResult(
                task_id=task.id if 'task' in locals() else "unknown",
                agent_id="orchestrator",
                status="failed",
                output="",
                error=str(e)
            )
        finally:
            # Stop monitoring if it was started
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                await self.coordination_engine.swarm_monitor.stop_monitoring()
            
            # Stop simple monitor if it was started
            if 'simple_monitor' in locals() and simple_monitor:
                await simple_monitor.stop_monitoring()
    
    async def execute_execution_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute a multi-task execution plan with coordination"""
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        
        self.logger.info(f"Executing plan with {len(plan.tasks)} tasks")
        
        try:
            # Execute using coordination engine
            result = await self.coordination_engine.execute_coordinated_tasks(
                plan.tasks,
                plan.parallel_groups
            )
            
            self.logger.info(f"Plan execution completed: {result.status}")
            return result
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            raise
    
    async def create_execution_plan(self, commands: List[str], 
                                  context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create an execution plan from multiple commands"""
        self.logger.info(f"Creating execution plan for {len(commands)} commands")
        
        tasks = []
        context = context or {}
        
        # Convert commands to tasks
        for command in commands:
            intent = await self.intent_classifier.analyze_intent(command)
            task = Task.from_intent(intent, command)
            tasks.append(task)
        
        # Create basic execution plan
        # The coordination engine will handle conflict detection and resolution
        plan = ExecutionPlan(
            id=f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            tasks=tasks,
            parallel_groups=None,  # Let coordination engine determine this
            estimated_duration=sum(task.estimated_duration or 60 for task in tasks)
        )
        
        return plan
    
    @property
    def agent_count(self) -> int:
        """Get the number of active agents"""
        return len(self.agent_registry.get_all_agents()) if self.agent_registry else 0
    
    @property
    def is_ready(self) -> bool:
        """Check if orchestrator is ready for operations"""
        return self._is_initialized
    
    async def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all agents"""
        if not self.agent_registry:
            return {}
        
        agent_status = {}
        all_agents = self.agent_registry.get_all_agents()
        
        for session in all_agents:
            agent_status[session.id] = {
                'name': session.agent_config.name,
                'type': session.agent_config.agent_type.value if hasattr(session.agent_config.agent_type, 'value') else str(session.agent_config.agent_type),
                'status': session.status,
                'focus_areas': session.agent_config.focus_areas,
                'last_activity': session.last_activity.isoformat() if session.last_activity else None
            }
        
        return agent_status
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all agents"""
        if not self.agent_registry:
            return {}
        
        health_status = {}
        all_agents = self.agent_registry.get_all_agents()
        
        for session in all_agents:
            # Basic health check - could be enhanced with actual health status
            health_status[session.id] = session.status == "active"
        
        return health_status
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "initialized": self._is_initialized,
            "project_analyzed": self._project_structure is not None,
            "project_name": self._project_structure.root_path.name if self._project_structure else None,
            "agents": {
                "total_agents": self.agent_count,
                "available_agents": len(self.agent_registry.get_available_agents()) if self.agent_registry else 0
            },
            "active_executions": len(self.coordination_engine.get_active_executions()) if hasattr(self.coordination_engine, 'get_active_executions') else 0,
            "recent_changes": len(self.shared_memory.get_recent_changes()) if hasattr(self.shared_memory, 'get_recent_changes') else 0,
            "task_progress": len(self.shared_memory.get_all_task_progress()) if hasattr(self.shared_memory, 'get_all_task_progress') else 0
        }
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent system activity"""
        changes = self.shared_memory.get_recent_changes(limit)
        task_progress = self.shared_memory.get_all_task_progress()
        
        # Combine and sort by timestamp
        activity = []
        
        # Add recent changes
        for change in changes:
            activity.append({
                "type": "change",
                "timestamp": change["timestamp"],
                "description": change["description"],
                "agent_id": change["agent_id"],
                "files": change["files"]
            })
        
        # Add recent task completions
        for task_id, progress in task_progress.items():
            if progress.get("status") == "completed" and "completed_at" in progress:
                activity.append({
                    "type": "task_completion",
                    "timestamp": progress["completed_at"],
                    "description": f"Task {task_id} completed",
                    "task_id": task_id,
                    "progress": progress
                })
        
        # Sort by timestamp (newest first) and limit
        activity.sort(key=lambda x: x["timestamp"], reverse=True)
        return activity[:limit]
    
    async def spawn_agent_session(self, agent_type: str, config: Optional[Dict[str, Any]] = None) -> AgentSession:
        """Spawn a new agent session or reuse an existing one"""
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        
        try:
            from agentic.models.agent import AgentConfig, AgentType
            
            # Convert string to AgentType if needed
            if isinstance(agent_type, str):
                agent_type = AgentType(agent_type)
            
            # Create agent config
            agent_config = AgentConfig(
                agent_type=agent_type,
                name=f"{agent_type.value.lower()}_session",
                workspace_path=self.config.workspace_path,
                focus_areas=config.get('focus_areas', []) if config else [],
                ai_model_config=config.get('ai_model_config', {"model": self.config.primary_model}) if config else {"model": self.config.primary_model},
                max_tokens=config.get('max_tokens', self.config.max_tokens) if config else self.config.max_tokens,
                temperature=config.get('temperature', self.config.temperature) if config else self.config.temperature
            )
            
            session = await self.agent_registry.get_or_spawn_agent(agent_config)
            self.logger.info(f"Spawned/reused {agent_type.value} agent: {session.id}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to spawn {agent_type} agent: {e}")
            raise
    
    async def stop_agent_session(self, session_id: str) -> bool:
        """Stop an agent session"""
        try:
            success = await self.agent_registry.stop_agent(session_id)
            if success:
                self.logger.info(f"Stopped agent session: {session_id}")
            else:
                self.logger.warning(f"Failed to stop agent session: {session_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"Error stopping agent session {session_id}: {e}")
            return False
    
    async def get_project_analysis(self) -> Optional[ProjectStructure]:
        """Get the current project analysis"""
        return self._project_structure
    
    async def refresh_project_analysis(self, project_path: Path) -> ProjectStructure:
        """Refresh the project analysis"""
        self.logger.info("Refreshing project analysis...")
        
        try:
            # Re-initialize project analyzer with new path
            self.project_analyzer = ProjectAnalyzer(project_path)
            self._project_structure = await self.project_analyzer.analyze()
            self.shared_memory.set_project_structure(self._project_structure)
            
            self.logger.info(f"Project analysis refreshed: {self._project_structure.root_path.name}")
            return self._project_structure
            
        except Exception as e:
            self.logger.error(f"Failed to refresh project analysis: {e}")
            raise
    
    async def _background_maintenance(self) -> None:
        """Background maintenance tasks"""
        while True:
            try:
                # Clean up old tasks and messages
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup tasks
                cleaned_tasks = self.shared_memory.cleanup_completed_tasks(older_than_hours=24)
                cleaned_messages = self.shared_memory.cleanup_old_messages(older_than_hours=1)
                cleaned_locks = await self.shared_memory.cleanup_stale_file_locks(threshold_minutes=30)
                
                if cleaned_tasks or cleaned_messages or cleaned_locks:
                    self.logger.debug(f"Maintenance: cleaned {cleaned_tasks} tasks, "
                                    f"{cleaned_messages} messages, {cleaned_locks} locks")
                
                # Check for stale agents
                stale_agents = self.shared_memory.get_stale_agents(threshold_minutes=10)
                for agent_id in stale_agents:
                    self.logger.warning(f"Agent {agent_id} appears to be stale")
                    # Could implement auto-restart logic here
                
            except Exception as e:
                self.logger.error(f"Background maintenance error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def execute_background_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute a command in the background without blocking"""
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        
        self.logger.info(f"Starting background execution: {command[:50]}...")
        
        # Use coordination engine for background execution
        task_id = await self.coordination_engine.execute_background_task(command, context)
        
        return task_id
    
    def get_background_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a background task"""
        return self.coordination_engine.get_background_task_status(task_id)
    
    def get_all_background_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all background tasks"""
        return self.coordination_engine.get_all_background_tasks()
    
    async def cancel_background_task(self, task_id: str) -> bool:
        """Cancel a running background task"""
        return await self.coordination_engine.cancel_background_task(task_id)
    
    async def execute_multi_agent_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute a command using multiple coordinated agents"""
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        
        self.logger.info(f"Executing multi-agent command: {command[:50]}...")
        
        try:
            return await self.coordination_engine.execute_multi_agent_command(command, context)
            
        except Exception as e:
            self.logger.error(f"Multi-agent command execution failed: {e}")
            raise
    
    # NEW: Inter-agent communication methods
    async def send_inter_agent_message(
        self,
        from_agent_id: str,
        message_type: str,
        content: str,
        to_agent_id: Optional[str] = None
    ) -> str:
        """Send a message between agents"""
        return await self.communication_hub.send_message(
            from_agent_id=from_agent_id,
            message_type=message_type,
            content=content,
            to_agent_id=to_agent_id
        )
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get inter-agent communication statistics"""
        return self.communication_hub.get_communication_stats()
    
    async def get_shared_context(self) -> str:
        """Get current shared context for agents"""
        return await self.communication_hub.get_shared_context()
    
    async def execute_interactive_command(self, command: str, input_handler=None) -> TaskResult:
        """Execute a command with interactive capabilities"""
        if not self._is_initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        
        self.logger.info(f"Executing interactive command: {command[:50]}...")
        
        try:
            # Classify intent
            intent = await self.intent_classifier.analyze_intent(command)
            self.logger.debug(f"Classified intent: {intent.task_type}")
            
            # Create task from intent
            task = Task.from_intent(intent, command)
            
            # Check if we have agents, if not spawn one
            available_agents = self.agent_registry.get_available_agents()
            if not available_agents:
                self.logger.info("No agents available, spawning agent for interactive task...")
                try:
                    await self._auto_spawn_agent(intent)
                except Exception as e:
                    self.logger.error(f"Failed to auto-spawn agent: {e}")
                    return TaskResult(
                        task_id=task.id,
                        agent_id="none",
                        status="failed",
                        output="",
                        error=f"No agents available and failed to spawn: {e}"
                    )
            
            # Route to appropriate agent
            routing_decision = await self.command_router.route_command(command, {})
            
            if not routing_decision.primary_agent:
                return TaskResult(
                    task_id=task.id,
                    agent_id="none",
                    status="failed",
                    output="",
                    error="No suitable agent found for command"
                )
            
            # Get agent instance
            best_agent = routing_decision.primary_agent
            agent_instance = self.agent_registry.get_agent_by_id(best_agent.id)
            
            if not agent_instance:
                return TaskResult(
                    task_id=task.id,
                    agent_id=best_agent.id,
                    status="failed",
                    output="",
                    error="Agent instance not found"
                )
            
            # Set up input handler for interactive execution
            if hasattr(agent_instance, 'set_input_handler') and input_handler:
                agent_instance.set_input_handler(input_handler)
            
            # Register task and execute
            await self.shared_memory.register_task(task)
            
            # Use interactive execution if agent supports it
            if hasattr(agent_instance, 'handle_interactive_input'):
                result = await agent_instance.handle_interactive_input(task)
            else:
                result = await agent_instance.execute_task(task)
            
            await self.shared_memory.complete_task(task.id, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Interactive command execution failed: {e}")
            return TaskResult(
                task_id=task.id if 'task' in locals() else "unknown",
                agent_id="orchestrator",
                status="failed",
                output="",
                error=str(e)
            )

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator"""
        self.logger.info("Shutting down orchestrator...")
        
        try:
            # Cancel the background maintenance task first
            if self._background_task and not self._background_task.done():
                self._background_task.cancel()
                try:
                    await self._background_task
                except asyncio.CancelledError:
                    pass
                self.logger.debug("Cancelled background maintenance task")
            
            # Cancel all background tasks
            background_tasks = self.get_all_background_tasks()
            for task_id in background_tasks:
                if not background_tasks[task_id].get("done", True):
                    await self.cancel_background_task(task_id)
                    self.logger.info(f"Cancelled background task {task_id}")
            
            # Stop all agents
            active_sessions = self.agent_registry.get_all_agents()
            for session in active_sessions:
                await self.agent_registry.stop_agent(session.id)
            
            # Cancel any active executions
            active_executions = self.coordination_engine.get_active_executions()
            if active_executions:
                self.logger.warning(f"Cancelling {len(active_executions)} active executions")
                # In a real implementation, we'd gracefully cancel these
            
            self.logger.info("Orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
    
    async def _auto_spawn_agent(self, intent) -> None:
        """Automatically spawn an appropriate agent based on task intent and characteristics"""
        from agentic.models.agent import AgentConfig
        
        # Analyze task characteristics
        command_lower = intent.command.lower() if hasattr(intent, 'command') else ""
        
        # Determine agent type based on multiple factors
        agent_type, focus_areas = self._determine_optimal_agent(intent, command_lower)
        
        # Create agent config using unified model configuration
        config = AgentConfig(
            agent_type=agent_type,
            name=f"auto_{agent_type.value.lower()}",
            workspace_path=self.config.workspace_path,
            focus_areas=focus_areas,
            ai_model_config={"model": self.config.primary_model},
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        # Get existing or spawn new agent (enables session persistence)
        session = await self.agent_registry.get_or_spawn_agent(config)
        
        # Register agent with communication hub
        self.communication_hub.register_agent(session)
        
        self.logger.info(f"Auto-spawned/reused {agent_type.value} agent: {session.id}")

    def _determine_optimal_agent(self, intent, command_lower: str) -> tuple:
        """Determine the optimal agent type based on task characteristics"""
        
        # Quick analysis/explanation tasks → Claude Code
        quick_analysis_keywords = [
            'explain', 'analyze', 'review', 'what does', 'how does', 'why does',
            'understand', 'summarize', 'describe', 'debug', 'find bug', 'troubleshoot',
            'examine', 'inspect', 'check', 'evaluate', 'assess', 'investigate'
        ]
        
        # Multi-file/complex implementation → Aider  
        complex_implementation_keywords = [
            'create system', 'build', 'implement feature', 'refactor', 'migrate',
            'add tests', 'authentication system', 'api endpoints', 'database',
            'develop', 'implement', 'generate', 'construct', 'establish'
        ]
        
        # File scope indicators
        single_file_indicators = [
            'file', 'function', 'class', 'method', 'specific', 'single',
            'one file', 'this file', 'in models.py', 'in config.py'
        ]
        
        multi_file_indicators = [
            'system', 'module', 'package', 'application', 'project', 'architecture',
            'tests', 'endpoints', 'models and', 'service and', 'complete',
            'multiple files', 'across files', 'entire project'
        ]
        
        # Creative vs systematic indicators
        creative_keywords = [
            'creative', 'innovative', 'alternative', 'better way', 'optimize',
            'improve', 'enhance', 'suggestion', 'ideas', 'ways to'
        ]
        
        systematic_keywords = [
            'step by step', 'thorough', 'comprehensive', 'detailed', 'complete',
            'systematic', 'methodical', 'following best practices', 'structured'
        ]
        
        # Initialize scoring
        claude_score = 0
        aider_score = 0
        
        # Score based on task type - increased weight for analysis tasks
        if any(keyword in command_lower for keyword in quick_analysis_keywords):
            claude_score += 5  # Increased from 3
        
        if any(keyword in command_lower for keyword in complex_implementation_keywords):
            aider_score += 3
            
        # Score based on file scope
        if any(indicator in command_lower for indicator in single_file_indicators):
            claude_score += 2
            
        if any(indicator in command_lower for indicator in multi_file_indicators):
            aider_score += 2
            
        # Score based on approach needed
        if any(keyword in command_lower for keyword in creative_keywords):
            claude_score += 2  # Increased from 1
            
        if any(keyword in command_lower for keyword in systematic_keywords):
            aider_score += 1
            
        # Strong indicators for explanation/analysis tasks
        explanation_patterns = [
            'explain the', 'what is', 'how does', 'why does', 'analyze the',
            'review the', 'understand the', 'describe the'
        ]
        
        if any(pattern in command_lower for pattern in explanation_patterns):
            claude_score += 4  # Strong preference for Claude on explanation tasks
            
        # Strong indicators for creation/implementation tasks
        creation_patterns = [
            'create a', 'build a', 'implement a', 'develop a', 'generate a',
            'add new', 'set up', 'establish'
        ]
        
        if any(pattern in command_lower for pattern in creation_patterns):
            aider_score += 4  # Strong preference for Aider on creation tasks
            
        # Consider intent properties if available
        if hasattr(intent, 'requires_reasoning') and intent.requires_reasoning:
            claude_score += 2
            
        if hasattr(intent, 'complexity_score') and intent.complexity_score > 0.5:
            aider_score += 2
            
        if hasattr(intent, 'estimated_duration') and intent.estimated_duration > 15:
            aider_score += 1  # Longer tasks better for Aider sessions
            
        # Check for multi-file patterns
        file_count_indicators = command_lower.count(' and ') + command_lower.count(', ')
        if file_count_indicators >= 2:
            aider_score += 2
            
        # Default focus areas and agent selection
        if claude_score > aider_score:
            agent_type = AgentType.CLAUDE_CODE
            focus_areas = ["analysis", "debugging", "optimization", "code_review"]
        else:
            # Determine Aider specialization
            if hasattr(intent, 'affected_areas') and intent.affected_areas:
                if "frontend" in intent.affected_areas:
                    agent_type = AgentType.AIDER_FRONTEND
                    focus_areas = ["frontend", "ui", "components", "styling"]
                elif "testing" in intent.affected_areas:
                    agent_type = AgentType.AIDER_TESTING  
                    focus_areas = ["testing", "qa", "test_automation"]
                else:
                    agent_type = AgentType.AIDER_BACKEND
                    focus_areas = ["backend", "api", "database", "architecture"]
            else:
                # Check command for specialization hints
                if any(word in command_lower for word in ['react', 'frontend', 'ui', 'component', 'css', 'html']):
                    agent_type = AgentType.AIDER_FRONTEND
                    focus_areas = ["frontend", "ui", "components", "styling"]
                elif any(word in command_lower for word in ['test', 'testing', 'unittest', 'pytest', 'spec']):
                    agent_type = AgentType.AIDER_TESTING
                    focus_areas = ["testing", "qa", "test_automation"]
                else:
                    agent_type = AgentType.AIDER_BACKEND  # Default
                    focus_areas = ["backend", "api", "database", "architecture"]
        
        return agent_type, focus_areas
    
    async def _enrich_command_with_context(self, command: str, context: Dict[str, Any]) -> str:
        """Enrich a command with previous context if it's a follow-up"""
        # Check if this is a follow-up command
        follow_up_indicators = [
            'now implement', 'great!', 'perfect!', 'thanks!', 'can you now',
            'please implement', 'go ahead', 'proceed with', 'let\'s implement',
            'based on that', 'using this analysis', 'with this information',
            'the missing logic', 'complete the'
        ]
        
        command_lower = command.lower()
        is_follow_up = any(indicator in command_lower for indicator in follow_up_indicators)
        
        if not is_follow_up:
            return command
        
        self.logger.info("Detected follow-up command, enriching with previous context")
        
        # Try to get previous analysis from session manager
        session_manager = context.get('session_manager')
        if session_manager:
            # Get recent analysis from session
            recent_analysis = session_manager.get_recent_analysis()
            if recent_analysis:
                # Prepend the analysis to the command
                enriched = f"Based on the following analysis:\n\n{recent_analysis}\n\nNow {command}"
                self.logger.info("Enriched command with previous analysis from session")
                return enriched
        
        # Try shared memory for recent task results
        recent_tasks = self.shared_memory.get_all_task_progress()
        for task_id, progress in recent_tasks.items():
            if progress.get('status') == 'completed' and progress.get('result'):
                result = progress['result']
                # Check if this was an analysis task
                if hasattr(result, 'output') and result.output:
                    output_lower = result.output.lower()
                    if any(word in output_lower for word in ['analysis', 'implementation', 'missing', 'need to']):
                        # Use this analysis as context
                        enriched = f"Based on the following analysis:\n\n{result.output[:2000]}\n\nNow {command}"
                        self.logger.info("Enriched command with previous analysis from shared memory")
                        return enriched
        
        return command
    
    async def _handle_mode_command(self, command: str) -> TaskResult:
        """Handle /mode command to switch execution modes"""
        parts = command.strip().lower().split()
        
        if len(parts) < 2:
            # Show current mode
            return TaskResult(
                task_id="mode_info",
                agent_id="orchestrator",
                status="completed",
                output=f"Current execution mode: {self._current_mode.value}\n"
                       f"Available modes: autonomous, interactive, supervised, hierarchical",
                execution_time=0
            )
        
        mode_name = parts[1]
        try:
            new_mode = ExecutionMode(mode_name)
            self.set_mode(new_mode)
            
            mode_descriptions = {
                ExecutionMode.AUTONOMOUS: "Agents work independently without intervention",
                ExecutionMode.INTERACTIVE: "Agents can ask questions and interact with you",
                ExecutionMode.SUPERVISED: "You monitor agent execution but don't interfere",
                ExecutionMode.HIERARCHICAL: "Tasks are delegated through supervisor→specialist→worker hierarchy"
            }
            
            return TaskResult(
                task_id="mode_change",
                agent_id="orchestrator",
                status="completed",
                output=f"✅ Execution mode changed to: {new_mode.value}\n"
                       f"Description: {mode_descriptions.get(new_mode, 'Unknown mode')}",
                execution_time=0
            )
        except ValueError:
            return TaskResult(
                task_id="mode_error",
                agent_id="orchestrator",
                status="failed",
                output="",
                error=f"Invalid mode: {mode_name}. Valid modes are: autonomous, interactive, supervised, hierarchical",
                execution_time=0
            )
    
    async def execute_hierarchical_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """Execute a command using hierarchical agent structure"""
        self.logger.info(f"Executing command in hierarchical mode: {command[:50]}...")
        
        try:
            # Initialize supervisor if not already created
            if not self._supervisor_agent:
                await self._initialize_supervisor_agent()
            
            # Check if this is a simple command that can be handled without full hierarchy
            if await self._can_handle_without_hierarchy(command):
                self.logger.info("Command is simple enough for direct execution, falling back to regular mode")
                # Temporarily switch to autonomous mode for this command
                original_mode = self._current_mode
                self._current_mode = ExecutionMode.AUTONOMOUS
                try:
                    # Use regular command execution
                    return await self._execute_regular_command(command, context)
                finally:
                    self._current_mode = original_mode
            
            # Create task from command
            intent = await self.intent_classifier.analyze_intent(command)
            task = Task.from_intent(intent, command)
            
            # Execute through supervisor
            self.logger.info("Delegating task to supervisor agent for hierarchical execution")
            result = await self._supervisor_agent.execute_complex_task(task)
            
            # Add mode information to result
            if not result.metadata:
                result.metadata = {}
            result.metadata["execution_mode"] = "hierarchical"
            result.metadata["supervisor_id"] = self._supervisor_agent.agent_id
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hierarchical execution failed: {e}")
            return TaskResult(
                task_id="hierarchical_error",
                agent_id="orchestrator",
                status="failed",
                output="",
                error=f"Hierarchical execution failed: {str(e)}",
                execution_time=0
            )
    
    async def _initialize_supervisor_agent(self) -> None:
        """Initialize the supervisor agent for hierarchical execution"""
        self.logger.info("Initializing supervisor agent for hierarchical execution")
        
        # Create supervisor config
        from agentic.models.agent import AgentConfig
        supervisor_config = AgentConfig(
            name="supervisor_main",
            agent_type=AgentType.CLAUDE_CODE,  # Supervisor uses Claude for high-level reasoning
            workspace_path=self.config.workspace_path,
            focus_areas=["coordination", "task_delegation", "architecture"],
            ai_model_config={"model": self.config.primary_model},
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        # Create supervisor agent
        self._supervisor_agent = SupervisorAgent(
            config=supervisor_config,
            shared_memory=self.shared_memory,
            resource_manager=self._resource_manager
        )
        
        self.logger.info(f"Supervisor agent initialized: {self._supervisor_agent.agent_id}")
    
    async def _can_handle_without_hierarchy(self, command: str) -> bool:
        """Determine if a command can be handled without full hierarchy"""
        command_lower = command.lower()
        
        # Simple commands that don't need hierarchy
        simple_patterns = [
            "explain", "what is", "show", "list", "describe",
            "help", "status", "version", "debug"
        ]
        
        # Check if it's a simple query
        if any(pattern in command_lower for pattern in simple_patterns):
            # Additional check - if it's asking about a complex system, still use hierarchy
            complex_indicators = ["system", "architecture", "distributed", "microservice"]
            if not any(indicator in command_lower for indicator in complex_indicators):
                return True
        
        # Short commands are usually simple
        if len(command.split()) < 5:
            return True
        
        return False
    
    async def _execute_regular_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """Execute command using regular (non-hierarchical) flow"""
        # This is the original execute_command logic without the mode check
        
        # Check if this is a follow-up command that needs context
        if context and context.get('session_manager'):
            enriched_command = await self._enrich_command_with_context(command, context)
        else:
            enriched_command = command
        
        # Determine if we should show monitoring
        should_monitor = context and context.get('enable_monitoring', True)
        status_updater = context.get('status_updater') if context else None
        
        try:
            # Start monitoring if enabled
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                await self.coordination_engine.swarm_monitor.start_monitoring()
            
            # Classify intent - use enriched command
            intent = await self.intent_classifier.analyze_intent(enriched_command)
            self.logger.debug(f"Classified intent: {intent.task_type}")
            
            # Create task from intent with enriched command
            task = Task.from_intent(intent, enriched_command)
            
            # Check if we have any agents, if not spawn one
            available_agents = self.agent_registry.get_available_agents()
            if not available_agents:
                self.logger.info("No agents available, spawning a backend agent for the task...")
                try:
                    await self._auto_spawn_agent(intent)
                except Exception as e:
                    self.logger.error(f"Failed to auto-spawn agent: {e}")
                    return TaskResult(
                        task_id=task.id,
                        agent_id="none",
                        status="failed",
                        output="",
                        error=f"No agents available and failed to spawn: {e}"
                    )
            
            # Route to appropriate agent
            routing_decision = await self.command_router.route_command(command, context or {})
            
            if not routing_decision.primary_agent:
                return TaskResult(
                    task_id=task.id,
                    agent_id="none",
                    status="failed",
                    output="",
                    error="No suitable agent found for command"
                )
            
            # Execute with best agent
            best_agent = routing_decision.primary_agent
            agent_instance = self.agent_registry.get_agent_by_id(best_agent.id)
            
            if not agent_instance:
                return TaskResult(
                    task_id=task.id,
                    agent_id=best_agent.id,
                    status="failed",
                    output="",
                    error="Agent instance not found"
                )
            
            # Register agent with monitor if monitoring is enabled
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                monitor = self.coordination_engine.swarm_monitor
                from agentic.core.swarm_monitor_unified import AgentStatus
                
                # Determine a meaningful role based on agent type and task
                if "test" in command.lower():
                    role = "test_runner"
                elif "claude" in best_agent.agent_config.agent_type.value.lower():
                    role = "analyzer"
                elif "frontend" in best_agent.agent_config.agent_type.value.lower():
                    role = "frontend_dev"
                elif "backend" in best_agent.agent_config.agent_type.value.lower():
                    role = "backend_dev"
                else:
                    role = best_agent.agent_config.agent_type.value.replace("_", " ").title()
                
                # Register agent with monitor
                monitor.register_agent(
                    agent_id=best_agent.id,
                    agent_name=best_agent.agent_config.name,
                    agent_type=best_agent.agent_config.agent_type.value,
                    role=role
                )
                monitor.update_agent_status(best_agent.id, AgentStatus.INITIALIZING)
                
                # Set monitor reference in agent for status updates
                if hasattr(agent_instance, 'set_monitor'):
                    agent_instance.set_monitor(monitor, best_agent.id)
            
            # Set up simple progress monitor if we have a status updater but no full monitoring
            simple_monitor = None
            if not should_monitor and status_updater:
                from agentic.core.simple_progress import SimpleProgressMonitor
                simple_monitor = SimpleProgressMonitor(status_updater)
                simple_monitor.start_monitoring(best_agent.id, best_agent.agent_config.name)
                
                # Give agent a reference to update progress
                if hasattr(agent_instance, 'set_progress_monitor'):
                    agent_instance.set_progress_monitor(simple_monitor)
            
            # Register task and execute
            await self.shared_memory.register_task(task)
            
            # Update monitor - starting task
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                monitor = self.coordination_engine.swarm_monitor
                monitor.update_agent_status(best_agent.id, AgentStatus.SETTING_UP)
                
                # Use appropriate method based on monitor type
                if hasattr(monitor, 'start_agent_task'):
                    monitor.start_agent_task(best_agent.id, task.id, task.command[:50] + "...")
                else:
                    monitor.start_task(best_agent.id, task.id, task.command[:50] + "...")
            
            result = await agent_instance.execute_task(task)
            
            # Update monitor - task completed
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                monitor = self.coordination_engine.swarm_monitor
                is_success = result.status == "completed"
                
                # Use appropriate method based on monitor type
                if hasattr(monitor, 'complete_agent_task'):
                    monitor.complete_agent_task(best_agent.id, task.id, is_success)
                else:
                    monitor.complete_task(best_agent.id, task.id, is_success)
                
                if is_success:
                    monitor.update_agent_status(best_agent.id, AgentStatus.COMPLETED)
                else:
                    monitor.update_agent_status(best_agent.id, AgentStatus.FAILED)
            
            await self.shared_memory.complete_task(task.id, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return TaskResult(
                task_id=task.id if 'task' in locals() else "unknown",
                agent_id="orchestrator",
                status="failed",
                output="",
                error=str(e)
            )
        finally:
            # Stop monitoring if it was started
            if should_monitor and hasattr(self, 'coordination_engine') and hasattr(self.coordination_engine, 'swarm_monitor'):
                await self.coordination_engine.swarm_monitor.stop_monitoring()
            
            # Stop simple monitor if it was started
            if 'simple_monitor' in locals() and simple_monitor:
                await simple_monitor.stop_monitoring() 