"""Agent escalation system for hierarchical communication."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio

from ..models.agent import Agent, AgentType, Discovery, DiscoveryType
from ..models.task import Task, TaskType
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EscalationRequest:
    """Request for escalation to a higher-level agent."""
    from_agent_id: str
    from_agent_type: str
    to_agent_type: str
    reason: str
    context: Dict[str, Any]
    original_task: Optional[Task] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class AgentEscalationManager:
    """Manages escalation between agents in the hierarchy."""
    
    def __init__(self, agent_registry):
        self.agent_registry = agent_registry
        self.escalation_queue: List[EscalationRequest] = []
        self.escalation_history: List[EscalationRequest] = []
        
        # Define escalation paths
        self.escalation_hierarchy = {
            # Aider agents escalate to domain architects
            "aider_frontend": "frontend-architect",
            "aider_backend": "backend-architect", 
            "aider_testing": "backend-architect",
            "aider_devops": "devops-architect",
            
            # Domain architects escalate to chief architect
            "frontend-architect": "chief-architect",
            "backend-architect": "chief-architect",
            "devops-architect": "chief-architect",
            "testing-architect": "chief-architect",
            
            # Claude Code generic escalates to Gemini
            "claude_code": "chief-architect"
        }
    
    async def request_escalation(self, from_agent: Agent, reason: str, 
                                context: Dict[str, Any], task: Optional[Task] = None) -> Any:
        """Request escalation from one agent to a higher level."""
        
        # Determine target agent
        from_agent_name = getattr(from_agent.session, 'agent_config', {}).get('name', '')
        from_agent_type = from_agent.get_capabilities().agent_type
        
        # Try name-based escalation first, then type-based
        to_agent = self.escalation_hierarchy.get(from_agent_name) or \
                   self.escalation_hierarchy.get(from_agent_type)
        
        if not to_agent:
            logger.warning(f"No escalation path for agent {from_agent_name}/{from_agent_type}")
            return None
        
        escalation = EscalationRequest(
            from_agent_id=from_agent.id,
            from_agent_type=from_agent_type,
            to_agent_type=to_agent,
            reason=reason,
            context=context,
            original_task=task
        )
        
        logger.info(f"Escalation requested: {from_agent_type} → {to_agent} - {reason}")
        
        # Process escalation
        return await self._process_escalation(escalation)
    
    async def _process_escalation(self, escalation: EscalationRequest) -> Any:
        """Process an escalation request."""
        self.escalation_history.append(escalation)
        
        # Find target agent
        target_agent = self._find_agent_by_name(escalation.to_agent_type)
        
        if not target_agent:
            logger.error(f"Target agent {escalation.to_agent_type} not found for escalation")
            return None
        
        # Create escalation task
        escalation_task = self._create_escalation_task(escalation)
        
        # Execute with target agent
        result = await target_agent.execute_task(escalation_task)
        
        # Report back as discovery
        if result and result.output:
            discovery = Discovery(
                type=DiscoveryType.INTEGRATION_POINT,
                description=f"Escalation response: {result.output[:200]}...",
                context={
                    "escalation_from": escalation.from_agent_type,
                    "escalation_to": escalation.to_agent_type,
                    "reason": escalation.reason
                },
                severity="info",
                agent_name=escalation.to_agent_type
            )
            
            # If original agent has discovery callback, report back
            original_agent = self.agent_registry.get_agent_by_id(escalation.from_agent_id)
            if original_agent and original_agent.discovery_callback:
                await original_agent.discovery_callback(discovery)
        
        return result
    
    def _find_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        """Find an agent by name."""
        for session_id, session in self.agent_registry.active_sessions.items():
            if session.agent_config.name == agent_name:
                return self.agent_registry.get_agent_by_id(session_id)
        return None
    
    def _create_escalation_task(self, escalation: EscalationRequest) -> Task:
        """Create a task for the escalated agent."""
        # Build comprehensive context
        context_parts = [
            f"**Escalation from {escalation.from_agent_type}**",
            f"**Reason**: {escalation.reason}",
            ""
        ]
        
        if escalation.original_task:
            context_parts.extend([
                "**Original Task**:",
                f"- Description: {escalation.original_task.description}",
                f"- Type: {escalation.original_task.type}",
                ""
            ])
        
        if escalation.context:
            context_parts.extend([
                "**Additional Context**:",
                *[f"- {k}: {v}" for k, v in escalation.context.items()],
                ""
            ])
        
        # Determine task type based on escalation reason
        task_type = self._determine_task_type(escalation.reason)
        
        prompt = "\n".join(context_parts) + "\n" + escalation.reason
        
        return Task(
            type=task_type,
            description=f"Escalation: {escalation.reason[:100]}...",
            prompt=prompt,
            context={
                "escalation": True,
                "from_agent": escalation.from_agent_type,
                **escalation.context
            }
        )
    
    def _determine_task_type(self, reason: str) -> TaskType:
        """Determine appropriate task type based on escalation reason."""
        reason_lower = reason.lower()
        
        if any(word in reason_lower for word in ["research", "find", "search", "latest"]):
            return TaskType.RESEARCH
        elif any(word in reason_lower for word in ["analyze", "understand", "explain"]):
            return TaskType.ANALYSIS
        elif any(word in reason_lower for word in ["design", "architect", "structure"]):
            return TaskType.DESIGN
        elif any(word in reason_lower for word in ["review", "check", "validate"]):
            return TaskType.CODE_REVIEW
        else:
            return TaskType.CONSULTATION
    
    async def bulk_escalate(self, escalations: List[EscalationRequest]) -> List[Any]:
        """Process multiple escalations in parallel."""
        tasks = [self._process_escalation(esc) for esc in escalations]
        return await asyncio.gather(*tasks)
    
    def get_escalation_stats(self) -> Dict[str, Any]:
        """Get statistics about escalations."""
        stats = {
            "total_escalations": len(self.escalation_history),
            "by_source": {},
            "by_target": {},
            "by_reason_keywords": {}
        }
        
        for esc in self.escalation_history:
            # Count by source
            stats["by_source"][esc.from_agent_type] = \
                stats["by_source"].get(esc.from_agent_type, 0) + 1
            
            # Count by target
            stats["by_target"][esc.to_agent_type] = \
                stats["by_target"].get(esc.to_agent_type, 0) + 1
            
            # Extract reason keywords
            keywords = ["research", "clarification", "architecture", "cross-domain", "context"]
            for keyword in keywords:
                if keyword in esc.reason.lower():
                    stats["by_reason_keywords"][keyword] = \
                        stats["by_reason_keywords"].get(keyword, 0) + 1
        
        return stats


class EscalationMixin:
    """Mixin for agents to add escalation capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.escalation_manager = None
    
    async def escalate(self, reason: str, context: Optional[Dict[str, Any]] = None,
                      task: Optional[Task] = None) -> Any:
        """Escalate to a higher-level agent."""
        if not self.escalation_manager:
            logger.warning("No escalation manager configured")
            return None
        
        return await self.escalation_manager.request_escalation(
            from_agent=self,
            reason=reason,
            context=context or {},
            task=task
        )
    
    async def needs_clarification(self, question: str, context: Dict[str, Any]) -> Any:
        """Request clarification from higher-level agent."""
        return await self.escalate(
            reason=f"Need clarification: {question}",
            context=context
        )
    
    async def needs_research(self, topic: str, context: Dict[str, Any]) -> Any:
        """Request research from chief architect."""
        return await self.escalate(
            reason=f"Need research on: {topic}",
            context=context
        )
    
    async def needs_architectural_decision(self, decision: str, options: List[str],
                                         context: Dict[str, Any]) -> Any:
        """Request architectural decision from higher level."""
        return await self.escalate(
            reason=f"Need architectural decision: {decision}",
            context={
                "options": options,
                **context
            }
        )