"""
Autonomous Agent Execution

This module enables agents to work naturally without over-processing or constraining their capabilities.
The focus is on orchestration, not reimplementation.
"""

import asyncio
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from agentic.models.task import Task, TaskResult
from agentic.models.agent import AgentSession
from agentic.utils.logging import LoggerMixin


class ExecutionMode(Enum):
    """How the agent should execute the task"""
    AUTONOMOUS = "autonomous"      # Agent handles everything independently
    INTERACTIVE = "interactive"    # Agent can ask questions, we pass through
    SUPERVISED = "supervised"      # We monitor but don't interfere


@dataclass
class AgentInteraction:
    """Represents an interaction request from an agent"""
    agent_id: str
    question: str
    context: Dict[str, Any]
    response_callback: Callable


class AutonomousExecutor(LoggerMixin):
    """Enables natural agent execution without unnecessary constraints"""
    
    def __init__(self, agent_registry=None):
        super().__init__()
        self.agent_registry = agent_registry
        self.active_interactions: Dict[str, AgentInteraction] = {}
        self.user_input_handler: Optional[Callable] = None
    
    def set_user_input_handler(self, handler: Callable):
        """Set the handler for user input requests"""
        self.user_input_handler = handler
    
    async def execute_with_natural_flow(self, agent: AgentSession, task: Task, 
                                       mode: ExecutionMode = ExecutionMode.AUTONOMOUS) -> TaskResult:
        """Execute task allowing agent to work naturally"""
        
        agent_type = agent.agent_config.agent_type.value
        self.logger.info(f"Executing task naturally with {agent_type} in {mode.value} mode")
        
        # Let the agent work in its natural way
        if agent_type == "claude_code":
            return await self._execute_claude_naturally(agent, task, mode)
        elif agent_type.startswith("aider"):
            return await self._execute_aider_naturally(agent, task, mode)
        else:
            # Fallback to standard execution
            return await self._execute_standard(agent, task)
    
    async def _execute_claude_naturally(self, agent: AgentSession, task: Task, 
                                       mode: ExecutionMode) -> TaskResult:
        """Let Claude Code work naturally without constraints"""
        
        # Get the actual agent instance
        if not self.agent_registry:
            self.logger.error("No agent registry provided")
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status="failed",
                error="Agent registry not available"
            )
        
        agent_instance = self.agent_registry.get_agent_by_id(agent.id)
        
        if not agent_instance:
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status="failed",
                error="Agent instance not found"
            )
        
        # For test tasks, don't use print mode - let Claude iterate
        command_lower = task.command.lower()
        is_test_task = any(word in command_lower for word in ['test', 'tests', 'passing', 'fail'])
        
        if is_test_task:
            self.logger.info("Test task detected - enabling Claude's natural iteration")
            # Remove any execution mode override
            if hasattr(agent_instance, '_determine_execution_mode'):
                original_determine = agent_instance._determine_execution_mode
                agent_instance._determine_execution_mode = lambda t: "interactive"
        
        # For interactive mode, set up input handler
        if mode == ExecutionMode.INTERACTIVE and self.user_input_handler:
            agent_instance.set_input_handler(self.user_input_handler)
        
        try:
            # Execute without artificial constraints
            result = await agent_instance.execute_task(task)
            
            # Claude naturally handles test iteration, so we trust the result
            if is_test_task and result.status == "completed":
                self.logger.info("Claude completed test task with natural iteration")
            
            return result
            
        finally:
            # Restore original execution mode determination
            if is_test_task and 'original_determine' in locals():
                agent_instance._determine_execution_mode = original_determine
    
    async def _execute_aider_naturally(self, agent: AgentSession, task: Task,
                                      mode: ExecutionMode) -> TaskResult:
        """Let Aider work naturally without over-constraining"""
        
        # Get the actual agent instance
        if not self.agent_registry:
            self.logger.error("No agent registry provided")
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status="failed",
                error="Agent registry not available"
            )
        
        agent_instance = self.agent_registry.get_agent_by_id(agent.id)
        
        if not agent_instance:
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status="failed",
                error="Agent instance not found"
            )
        
        # For interactive mode, remove --exit flag
        if mode == ExecutionMode.INTERACTIVE:
            self.logger.info("Enabling Aider interactive mode")
            if hasattr(agent_instance, '_build_enhanced_aider_command'):
                original_build = agent_instance._build_enhanced_aider_command
                
                async def build_without_exit(task, target_files):
                    cmd = await original_build(task, target_files)
                    # Remove --exit flag to allow interaction
                    if "--exit" in cmd:
                        cmd.remove("--exit")
                    # Remove --yes-always for interactive mode
                    if "--yes-always" in cmd:
                        cmd.remove("--yes-always")
                    return cmd
                
                agent_instance._build_enhanced_aider_command = build_without_exit
        
        # Don't over-process the command
        # Trust Aider to understand and execute naturally
        return await agent_instance.execute_task(task)
    
    async def _execute_standard(self, agent: AgentSession, task: Task) -> TaskResult:
        """Standard execution for other agent types"""
        if not self.agent_registry:
            self.logger.error("No agent registry provided")
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status="failed",
                error="Agent registry not available"
            )
        
        agent_instance = self.agent_registry.get_agent_by_id(agent.id)
        
        if not agent_instance:
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status="failed",
                error="Agent instance not found"
            )
        
        return await agent_instance.execute_task(task)
    
    def determine_execution_mode(self, command: str) -> ExecutionMode:
        """Determine the best execution mode based on the command"""
        command_lower = command.lower()
        
        # Interactive mode for exploratory or unclear commands
        interactive_patterns = [
            "help me", "work with me", "assist", "guide",
            "what should", "how do i", "can you explain"
        ]
        
        if any(pattern in command_lower for pattern in interactive_patterns):
            return ExecutionMode.INTERACTIVE
        
        # Supervised mode for critical operations
        supervised_patterns = [
            "production", "deploy", "migration", "delete all",
            "drop database", "rm -rf"
        ]
        
        if any(pattern in command_lower for pattern in supervised_patterns):
            return ExecutionMode.SUPERVISED
        
        # Default to autonomous for most tasks
        return ExecutionMode.AUTONOMOUS
    
    async def handle_agent_question(self, agent_id: str, question: str, 
                                   context: Dict[str, Any]) -> Optional[str]:
        """Handle a question from an agent"""
        if not self.user_input_handler:
            self.logger.warning(f"No input handler for agent question: {question}")
            return None
        
        # Create interaction record
        interaction = AgentInteraction(
            agent_id=agent_id,
            question=question,
            context=context,
            response_callback=None  # Set by agent
        )
        
        self.active_interactions[agent_id] = interaction
        
        try:
            # Get user response
            response = await self.user_input_handler(question)
            return response
        finally:
            # Clean up interaction
            if agent_id in self.active_interactions:
                del self.active_interactions[agent_id]