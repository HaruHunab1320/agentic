"""
Command Router for routing commands to appropriate agents
"""

from __future__ import annotations

from typing import List, Optional

from agentic.core.agent_registry import AgentRegistry
from agentic.core.intent_classifier import IntentClassifier
from agentic.models.agent import AgentSession, AgentType
from agentic.models.task import Task, TaskIntent, TaskType
from agentic.utils.logging import LoggerMixin


class RoutingDecision:
    """Represents a routing decision"""
    
    def __init__(self, primary_agent: AgentSession, supporting_agents: List[AgentSession] = None,
                 requires_coordination: bool = False, reasoning_agent: Optional[AgentSession] = None):
        self.primary_agent = primary_agent
        self.supporting_agents = supporting_agents or []
        self.requires_coordination = requires_coordination
        self.reasoning_agent = reasoning_agent
    
    @property
    def all_agents(self) -> List[AgentSession]:
        """Get all agents involved in this routing decision"""
        agents = [self.primary_agent] + self.supporting_agents
        if self.reasoning_agent and self.reasoning_agent not in agents:
            agents.append(self.reasoning_agent)
        return agents


class CommandRouter(LoggerMixin):
    """Routes commands to appropriate agents based on intent analysis"""
    
    def __init__(self, agent_registry: AgentRegistry):
        super().__init__()
        self.agent_registry = agent_registry
        self.intent_classifier = IntentClassifier()
    
    async def route_command(self, command: str, context: Optional[dict] = None) -> RoutingDecision:
        """Route a command to appropriate agents"""
        self.logger.info(f"Routing command: {command[:100]}...")
        
        # Analyze intent
        intent = await self.intent_classifier.analyze_intent(command)
        self.logger.debug(f"Intent analysis: {intent}")
        
        # Find appropriate agents
        routing_decision = await self._select_agents(intent, context or {})
        
        self.logger.info(f"Routed to primary agent: {routing_decision.primary_agent.agent_config.name}")
        if routing_decision.supporting_agents:
            support_names = [agent.agent_config.name for agent in routing_decision.supporting_agents]
            self.logger.info(f"Supporting agents: {', '.join(support_names)}")
        
        return routing_decision
    
    async def _select_agents(self, intent: TaskIntent, context: dict) -> RoutingDecision:
        """Select appropriate agents based on intent"""
        
        # Get available agents
        available_agents = self.agent_registry.get_available_agents()
        
        if not available_agents:
            raise RuntimeError("No available agents found")
        
        # Find reasoning agent if needed
        reasoning_agent = None
        if intent.requires_reasoning:
            reasoning_agents = self.agent_registry.get_agents_by_type(AgentType.CLAUDE_CODE)
            reasoning_agent = reasoning_agents[0] if reasoning_agents else None
        
        # Find primary agent based on affected areas and task type
        primary_agent = await self._select_primary_agent(intent, available_agents)
        
        # Find supporting agents if coordination is required
        supporting_agents = []
        if intent.requires_coordination:
            supporting_agents = await self._select_supporting_agents(
                intent, primary_agent, available_agents, reasoning_agent
            )
        
        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            requires_coordination=intent.requires_coordination,
            reasoning_agent=reasoning_agent
        )
    
    async def _select_primary_agent(self, intent: TaskIntent, available_agents: List[AgentSession]) -> AgentSession:
        """Select the primary agent for the task"""
        
        # For reasoning-heavy tasks, prefer Claude Code
        if intent.requires_reasoning and intent.task_type in [TaskType.DEBUG, TaskType.EXPLAIN]:
            claude_agents = [a for a in available_agents if a.agent_config.agent_type == AgentType.CLAUDE_CODE]
            if claude_agents:
                return claude_agents[0]
        
        # For area-specific tasks, prefer specialized agents
        if len(intent.affected_areas) == 1:
            area = intent.affected_areas[0]
            
            if area == "frontend":
                frontend_agents = [a for a in available_agents if a.agent_config.agent_type == AgentType.AIDER_FRONTEND]
                if frontend_agents:
                    return frontend_agents[0]
            
            elif area == "backend":
                backend_agents = [a for a in available_agents if a.agent_config.agent_type == AgentType.AIDER_BACKEND]
                if backend_agents:
                    return backend_agents[0]
            
            elif area == "testing":
                testing_agents = [a for a in available_agents if a.agent_config.agent_type == AgentType.AIDER_TESTING]
                if testing_agents:
                    return testing_agents[0]
        
        # For test-related tasks, prefer testing agent
        if intent.task_type == TaskType.TEST:
            testing_agents = [a for a in available_agents if a.agent_config.agent_type == AgentType.AIDER_TESTING]
            if testing_agents:
                return testing_agents[0]
        
        # Fallback logic based on task type
        if intent.task_type in [TaskType.DEBUG, TaskType.EXPLAIN]:
            # Prefer reasoning agents for debugging and explanation
            claude_agents = [a for a in available_agents if a.agent_config.agent_type == AgentType.CLAUDE_CODE]
            if claude_agents:
                return claude_agents[0]
        
        # Default: find best match by capabilities
        return self._find_best_capability_match(intent, available_agents)
    
    async def _select_supporting_agents(self, intent: TaskIntent, primary_agent: AgentSession,
                                       available_agents: List[AgentSession], 
                                       reasoning_agent: Optional[AgentSession]) -> List[AgentSession]:
        """Select supporting agents for coordination"""
        supporting_agents = []
        
        # Don't include primary agent or reasoning agent in supporting agents
        excluded_agent_ids = {primary_agent.id}
        if reasoning_agent:
            excluded_agent_ids.add(reasoning_agent.id)
        
        candidate_agents = [a for a in available_agents if a.id not in excluded_agent_ids]
        
        # For multi-area tasks, include agents from other affected areas
        for area in intent.affected_areas:
            if area == "frontend" and primary_agent.agent_config.agent_type != AgentType.AIDER_FRONTEND:
                frontend_agents = [a for a in candidate_agents if a.agent_config.agent_type == AgentType.AIDER_FRONTEND]
                if frontend_agents:
                    supporting_agents.append(frontend_agents[0])
            
            elif area == "backend" and primary_agent.agent_config.agent_type != AgentType.AIDER_BACKEND:
                backend_agents = [a for a in candidate_agents if a.agent_config.agent_type == AgentType.AIDER_BACKEND]
                if backend_agents:
                    supporting_agents.append(backend_agents[0])
            
            elif area == "testing" and primary_agent.agent_config.agent_type != AgentType.AIDER_TESTING:
                testing_agents = [a for a in candidate_agents if a.agent_config.agent_type == AgentType.AIDER_TESTING]
                if testing_agents:
                    supporting_agents.append(testing_agents[0])
        
        # For high complexity tasks, include additional agents
        if intent.complexity_score > 0.8:
            # Add reasoning agent if not already included
            if not reasoning_agent and primary_agent.agent_config.agent_type != AgentType.CLAUDE_CODE:
                claude_agents = [a for a in candidate_agents if a.agent_config.agent_type == AgentType.CLAUDE_CODE]
                if claude_agents:
                    supporting_agents.append(claude_agents[0])
        
        # Remove duplicates while preserving order using agent IDs
        seen_ids = set()
        unique_supporting = []
        for agent in supporting_agents:
            if agent.id not in seen_ids:
                seen_ids.add(agent.id)
                unique_supporting.append(agent)
        
        return unique_supporting
    
    def _find_best_capability_match(self, intent: TaskIntent, available_agents: List[AgentSession]) -> AgentSession:
        """Find agent with best capability match"""
        best_agent = None
        best_score = -1
        
        for agent in available_agents:
            score = self._calculate_capability_score(intent, agent)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent or available_agents[0]  # Fallback to first available
    
    def _calculate_capability_score(self, intent: TaskIntent, agent: AgentSession) -> float:
        """Calculate how well an agent matches the task intent"""
        score = 0.0
        
        # Check area overlap
        agent_areas = set(agent.agent_config.focus_areas)
        intent_areas = set(intent.affected_areas)
        area_overlap = len(agent_areas & intent_areas)
        score += area_overlap * 0.5
        
        # Bonus for agent type alignment
        if intent.task_type == TaskType.DEBUG:
            if agent.agent_config.agent_type == AgentType.CLAUDE_CODE:
                score += 0.3  # Reasoning agents good for debugging
        elif intent.task_type == TaskType.TEST:
            if agent.agent_config.agent_type == AgentType.AIDER_TESTING:
                score += 0.4  # Testing agents best for tests
        elif intent.task_type == TaskType.IMPLEMENT:
            if agent.agent_config.agent_type in [AgentType.AIDER_FRONTEND, AgentType.AIDER_BACKEND]:
                score += 0.3  # Implementation agents good for building
        
        # Bonus for specialized agents in their domain
        if "frontend" in intent.affected_areas and agent.agent_config.agent_type == AgentType.AIDER_FRONTEND:
            score += 0.4
        if "backend" in intent.affected_areas and agent.agent_config.agent_type == AgentType.AIDER_BACKEND:
            score += 0.4
        if "testing" in intent.affected_areas and agent.agent_config.agent_type == AgentType.AIDER_TESTING:
            score += 0.4
        
        return score
    
    async def create_execution_plan(self, command: str, routing_decision: RoutingDecision) -> Task:
        """Create an execution plan for the routed command"""
        intent = await self.intent_classifier.analyze_intent(command)
        
        # Create task
        task = Task(
            command=command,
            task_type=intent.task_type,
            complexity_score=intent.complexity_score,
            estimated_duration=intent.estimated_duration,
            affected_areas=intent.affected_areas,
            requires_reasoning=intent.requires_reasoning,
            requires_coordination=intent.requires_coordination
        )
        
        # Set primary agent
        task.assigned_agent_id = routing_decision.primary_agent.id
        
        # Add coordination plan if needed
        if routing_decision.requires_coordination:
            coordination_steps = []
            
            # Step 1: Primary agent analysis
            coordination_steps.append(f"Primary analysis by {routing_decision.primary_agent.agent_config.name}")
            
            # Step 2: Reasoning if needed
            if routing_decision.reasoning_agent:
                coordination_steps.append(f"Reasoning and guidance by {routing_decision.reasoning_agent.agent_config.name}")
            
            # Step 3: Supporting agent actions
            for agent in routing_decision.supporting_agents:
                coordination_steps.append(f"Support from {agent.agent_config.name}")
            
            # Step 4: Final coordination
            coordination_steps.append("Final coordination and integration")
            
            task.execution_plan.coordination_steps = coordination_steps
        
        return task 