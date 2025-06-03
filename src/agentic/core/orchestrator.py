"""
Main Orchestrator for coordinating multiple AI agents
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from agentic.core.agent_registry import AgentRegistry
from agentic.core.command_router import CommandRouter, RoutingDecision
from agentic.core.project_analyzer import ProjectAnalyzer
from agentic.models.agent import AgentSession
from agentic.models.config import AgenticConfig
from agentic.models.project import ProjectStructure
from agentic.models.task import Task, TaskResult
from agentic.utils.logging import LoggerMixin


class Orchestrator(LoggerMixin):
    """
    Main orchestrator that coordinates multiple AI agents for development tasks
    """
    
    def __init__(self, config: AgenticConfig):
        super().__init__()
        self.config = config
        self.workspace_path = config.workspace_path
        
        # Core components
        self.agent_registry = AgentRegistry(self.workspace_path)
        self.command_router = CommandRouter(self.agent_registry)
        self.project_analyzer = ProjectAnalyzer(self.workspace_path)
        
        # State
        self.project_structure: Optional[ProjectStructure] = None
        self.active_sessions: List[AgentSession] = []
        self.task_history: List[Task] = []
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator and auto-spawn agents"""
        try:
            self.logger.info("Initializing Agentic orchestrator...")
            
            # Analyze project structure
            self.logger.info("Analyzing project structure...")
            self.project_structure = await self.project_analyzer.analyze()
            
            # Auto-spawn appropriate agents based on project structure
            self.logger.info("Auto-spawning agents based on project analysis...")
            self.active_sessions = await self.agent_registry.auto_spawn_agents(self.project_structure)
            
            if not self.active_sessions:
                self.logger.warning("No agents were spawned successfully")
                return False
            
            agent_names = [session.agent_config.name for session in self.active_sessions]
            self.logger.info(f"Successfully spawned {len(self.active_sessions)} agents: {', '.join(agent_names)}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    async def execute_command(self, command: str, context: Optional[dict] = None) -> TaskResult:
        """Execute a command using the appropriate agent(s)"""
        if not self.is_initialized:
            return TaskResult(
                task_id="uninitialized",
                agent_id="orchestrator",
                success=False,
                output="",
                error="Orchestrator not initialized. Run 'agentic init' first."
            )
        
        try:
            self.logger.info(f"Executing command: {command[:100]}...")
            
            # Route the command to appropriate agents
            routing_decision = await self.command_router.route_command(command, context)
            
            # Create execution plan
            task = await self.command_router.create_execution_plan(command, routing_decision)
            self.task_history.append(task)
            
            # Execute the task
            if routing_decision.requires_coordination:
                result = await self._execute_coordinated_task(task, routing_decision)
            else:
                result = await self._execute_single_agent_task(task, routing_decision)
            
            # Log the result
            if result.success:
                self.logger.info(f"Command executed successfully")
            else:
                self.logger.error(f"Command execution failed: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command execution error: {e}")
            return TaskResult(
                task_id="error",
                agent_id="orchestrator",
                success=False,
                output="",
                error=str(e)
            )
    
    async def _execute_single_agent_task(self, task: Task, routing_decision: RoutingDecision) -> TaskResult:
        """Execute a task using a single agent"""
        primary_agent = self.agent_registry.get_agent_by_id(routing_decision.primary_agent.id)
        
        if not primary_agent:
            return TaskResult(
                task_id=task.id,
                agent_id="orchestrator",
                success=False,
                output="",
                error="Primary agent not found"
            )
        
        # Mark agent as busy
        routing_decision.primary_agent.mark_busy(task.id)
        
        try:
            # Execute the task
            result = await primary_agent.execute_task(task)
            
            # Update task status
            if result.success:
                task.mark_completed(result)
            else:
                task.mark_failed(result.error)
            
            return result
            
        finally:
            # Mark agent as available again
            routing_decision.primary_agent.mark_available()
    
    async def _execute_coordinated_task(self, task: Task, routing_decision: RoutingDecision) -> TaskResult:
        """Execute a task requiring coordination between multiple agents"""
        self.logger.info(f"Executing coordinated task with {len(routing_decision.all_agents)} agents")
        
        try:
            # Phase 1: Reasoning and analysis (if reasoning agent available)
            reasoning_result = None
            if routing_decision.reasoning_agent:
                self.logger.info("Phase 1: Reasoning and analysis")
                reasoning_agent = self.agent_registry.get_agent_by_id(routing_decision.reasoning_agent.id)
                if reasoning_agent:
                    reasoning_result = await reasoning_agent.execute_task(task)
                    if not reasoning_result.success:
                        self.logger.warning("Reasoning phase failed, but continuing...")
            
            # Phase 2: Primary agent execution
            self.logger.info("Phase 2: Primary agent execution")
            primary_agent = self.agent_registry.get_agent_by_id(routing_decision.primary_agent.id)
            
            if not primary_agent:
                return TaskResult(
                    task_id=task.id,
                    agent_id="orchestrator",
                    success=False,
                    output="",
                    error="Primary agent not found"
                )
            
            # Mark agents as busy
            for agent_session in routing_decision.all_agents:
                agent_session.mark_busy(task.id)
            
            # Prepare enhanced task with reasoning context
            enhanced_task = task
            if reasoning_result and reasoning_result.success:
                # Add reasoning context to task command
                enhanced_command = f"{task.command}\n\nReasoning guidance:\n{reasoning_result.output}"
                enhanced_task = Task(
                    command=enhanced_command,
                    task_type=task.task_type,
                    complexity_score=task.complexity_score,
                    estimated_duration=task.estimated_duration,
                    affected_areas=task.affected_areas,
                    requires_reasoning=task.requires_reasoning,
                    requires_coordination=task.requires_coordination
                )
                enhanced_task.assigned_agent_id = task.assigned_agent_id
            
            # Execute primary task
            primary_result = await primary_agent.execute_task(enhanced_task)
            
            # Phase 3: Supporting agent tasks (if any)
            supporting_results = []
            if routing_decision.supporting_agents and primary_result.success:
                self.logger.info(f"Phase 3: Executing {len(routing_decision.supporting_agents)} supporting tasks")
                
                # Create tasks for supporting agents
                supporting_tasks = []
                for i, agent_session in enumerate(routing_decision.supporting_agents):
                    # Create focused task for each supporting agent
                    supporting_task = self._create_supporting_task(task, agent_session, primary_result)
                    supporting_tasks.append((agent_session, supporting_task))
                
                # Execute supporting tasks in parallel
                supporting_results = await self._execute_supporting_tasks(supporting_tasks)
            
            # Phase 4: Combine results
            combined_result = self._combine_coordination_results(
                task, primary_result, supporting_results, reasoning_result
            )
            
            # Update task status
            if combined_result.success:
                task.mark_completed(combined_result)
            else:
                task.mark_failed(combined_result.error)
            
            return combined_result
            
        except Exception as e:
            error_msg = f"Coordinated task execution failed: {e}"
            self.logger.error(error_msg)
            
            return TaskResult(
                task_id=task.id,
                agent_id="orchestrator",
                success=False,
                output="",
                error=error_msg
            )
        
        finally:
            # Mark all agents as available again
            for agent_session in routing_decision.all_agents:
                agent_session.mark_available()
    
    def _create_supporting_task(self, original_task: Task, agent_session: AgentSession, 
                               primary_result: TaskResult) -> Task:
        """Create a focused task for a supporting agent"""
        
        # Customize the task based on agent type and focus areas
        focus_context = f"Focus on {', '.join(agent_session.agent_config.focus_areas)} aspects"
        
        supporting_command = f"""
{focus_context} of this task:

Original request: {original_task.command}

Primary agent results: {primary_result.output[:300]}...

Please provide {agent_session.agent_config.name}-specific contributions to complete this task.
        """.strip()
        
        supporting_task = Task(
            command=supporting_command,
            task_type=original_task.task_type,
            complexity_score=original_task.complexity_score * 0.6,  # Supporting tasks are generally simpler
            estimated_duration=original_task.estimated_duration // 2,  # Supporting tasks take less time
            affected_areas=original_task.affected_areas,
            requires_reasoning=False,  # Supporting tasks don't need additional reasoning
            requires_coordination=False  # Supporting tasks are focused
        )
        
        supporting_task.assigned_agent_id = agent_session.id
        return supporting_task
    
    async def _execute_supporting_tasks(self, supporting_tasks: List[tuple]) -> List[TaskResult]:
        """Execute supporting tasks in parallel"""
        async def execute_single_supporting_task(agent_session: AgentSession, task: Task) -> TaskResult:
            agent = self.agent_registry.get_agent_by_id(agent_session.id)
            if agent:
                return await agent.execute_task(task)
            else:
                return TaskResult(
                    task_id=task.id,
                    agent_id=agent_session.id,
                    success=False,
                    output="",
                    error="Agent not found"
                )
        
        # Execute all supporting tasks concurrently
        tasks = [
            execute_single_supporting_task(agent_session, task)
            for agent_session, task in supporting_tasks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed TaskResults
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task_id="error",
                    agent_id="unknown",
                    success=False,
                    output="",
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _combine_coordination_results(self, task: Task, primary_result: TaskResult,
                                     supporting_results: List[TaskResult],
                                     reasoning_result: Optional[TaskResult]) -> TaskResult:
        """Combine results from coordinated execution"""
        
        # Build combined output
        output_parts = []
        
        # Add reasoning context if available
        if reasoning_result and reasoning_result.success:
            output_parts.append("🧠 REASONING & ANALYSIS")
            output_parts.append(reasoning_result.output)
            output_parts.append("")
        
        # Add primary result
        output_parts.append("🎯 PRIMARY IMPLEMENTATION")
        output_parts.append(primary_result.output)
        output_parts.append("")
        
        # Add supporting results
        if supporting_results:
            successful_supporting = [r for r in supporting_results if r.success]
            if successful_supporting:
                output_parts.append("🤝 SUPPORTING CONTRIBUTIONS")
                for i, result in enumerate(successful_supporting):
                    output_parts.append(f"Support {i+1}:")
                    output_parts.append(result.output)
                    output_parts.append("")
        
        # Determine overall success
        overall_success = primary_result.success
        
        # Collect any errors
        errors = []
        if not primary_result.success:
            errors.append(f"Primary: {primary_result.error}")
        
        for result in supporting_results:
            if not result.success:
                errors.append(f"Supporting: {result.error}")
        
        combined_output = "\n".join(output_parts)
        combined_error = "; ".join(errors) if errors else ""
        
        return TaskResult(
            task_id=task.id,
            agent_id="orchestrator",
            success=overall_success,
            output=combined_output,
            error=combined_error
        )
    
    async def get_agent_status(self) -> Dict[str, dict]:
        """Get status of all agents"""
        return await self.agent_registry.get_agent_status()
    
    async def spawn_agent(self, agent_type: str, name: str, focus_areas: List[str]) -> bool:
        """Manually spawn a new agent"""
        try:
            from agentic.models.agent import AgentConfig, AgentType
            
            # Convert string to AgentType enum
            try:
                enum_type = AgentType(agent_type.upper())
            except ValueError:
                self.logger.error(f"Invalid agent type: {agent_type}")
                return False
            
            config = AgentConfig(
                agent_type=enum_type,
                name=name,
                workspace_path=self.workspace_path,
                focus_areas=focus_areas,
                ai_model_config={"model": "claude-3-5-sonnet"}
            )
            
            session = await self.agent_registry.spawn_agent(config)
            if session.status == "active":
                self.active_sessions.append(session)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to spawn agent: {e}")
            return False
    
    async def terminate_agent(self, agent_name: str) -> bool:
        """Terminate a specific agent by name"""
        for session in self.active_sessions:
            if session.agent_config.name == agent_name:
                success = await self.agent_registry.terminate_agent(session.id)
                if success:
                    self.active_sessions.remove(session)
                return success
        
        self.logger.warning(f"Agent '{agent_name}' not found")
        return False
    
    async def terminate_all_agents(self) -> int:
        """Terminate all active agents"""
        count = await self.agent_registry.terminate_all_agents()
        self.active_sessions.clear()
        return count
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all components"""
        health_status = {
            "orchestrator": True,
            "project_analyzer": True,
            "agent_registry": len(self.active_sessions) > 0,
            "command_router": True,
        }
        
        # Check individual agents
        agent_health = await self.agent_registry.health_check_all()
        health_status.update(agent_health)
        
        return health_status
    
    @property
    def is_ready(self) -> bool:
        """Check if orchestrator is ready to execute commands"""
        return self.is_initialized and len(self.active_sessions) > 0
    
    @property
    def agent_count(self) -> int:
        """Get count of active agents"""
        return len(self.active_sessions) 