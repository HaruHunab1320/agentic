"""
Inter-Agent Communication System

Leverages Claude Code's native memory features for agent-to-agent context sharing
and coordination. Provides a unified interface for agents to share findings,
decisions, and context with each other.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from agentic.models.agent import AgentSession
from agentic.utils.logging import LoggerMixin


class InterAgentMessage:
    """Represents a message between agents"""
    
    def __init__(
        self,
        from_agent: str,
        to_agent: Optional[str],  # None means broadcast to all
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = f"msg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.message_type = message_type
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
        self.read = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for storage"""
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "read": self.read
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterAgentMessage':
        """Create message from dictionary"""
        msg = cls(
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            message_type=data["message_type"],
            content=data["content"],
            metadata=data.get("metadata", {})
        )
        msg.id = data["id"]
        msg.timestamp = datetime.fromisoformat(data["timestamp"])
        msg.read = data.get("read", False)
        return msg


class InterAgentCommunicationHub(LoggerMixin):
    """
    Hub for inter-agent communication using Claude Code's memory system
    """
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path
        self.memory_file = workspace_path / "AGENT_COMMUNICATION.md"
        self.agents: Dict[str, AgentSession] = {}
        
        # Message types
        self.MESSAGE_TYPES = {
            "context_share": "Share context/findings with other agents",
            "task_handoff": "Hand off a task to another agent",
            "question": "Ask question to specific agent or all agents",
            "decision": "Inform about important decision made",
            "completion": "Notify about task completion",
            "error": "Report error that other agents should know about",
            "discovery": "Share important discovery or insight"
        }
    
    def register_agent(self, agent_session: AgentSession) -> None:
        """Register an agent for communication"""
        self.agents[agent_session.id] = agent_session
        self.logger.info(f"Registered agent {agent_session.agent_config.name} for inter-agent communication")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent {agent_id} from inter-agent communication")
    
    async def send_message(
        self,
        from_agent_id: str,
        message_type: str,
        content: str,
        to_agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a message between agents"""
        
        # Validate message type
        if message_type not in self.MESSAGE_TYPES:
            raise ValueError(f"Invalid message type: {message_type}")
        
        # Create message
        message = InterAgentMessage(
            from_agent=from_agent_id,
            to_agent=to_agent_id,
            message_type=message_type,
            content=content,
            metadata=metadata
        )
        
        # Store message in memory system
        await self._store_message_in_memory(message)
        
        # Notify target agent(s) if they have Claude Code capabilities
        await self._notify_agents(message)
        
        self.logger.info(f"Message sent from {from_agent_id} to {to_agent_id or 'all'}: {message_type}")
        return message.id
    
    async def _store_message_in_memory(self, message: InterAgentMessage) -> None:
        """Store message in Claude Code's memory system"""
        try:
            # Ensure memory file exists
            if not self.memory_file.exists():
                await self._initialize_memory_file()
            
            # Read current content
            current_content = self.memory_file.read_text()
            
            # Format message for memory
            message_section = f"""
## Message: {message.id}
**From:** {message.from_agent}  
**To:** {message.to_agent or "All Agents"}  
**Type:** {message.message_type}  
**Time:** {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{message.content}

{f"**Metadata:** {json.dumps(message.metadata, indent=2)}" if message.metadata else ""}

---
"""
            
            # Insert at the top of the communication log section
            if "## Communication Log" in current_content:
                parts = current_content.split("## Communication Log")
                updated_content = parts[0] + "## Communication Log" + message_section + parts[1] if len(parts) > 1 else parts[0] + "## Communication Log" + message_section
            else:
                updated_content = current_content + "\n\n## Communication Log" + message_section
            
            # Write back to file
            self.memory_file.write_text(updated_content)
            
            # Update Claude Code memory using CLI
            await self._update_claude_memory(message)
            
        except Exception as e:
            self.logger.error(f"Failed to store message in memory: {e}")
    
    async def _initialize_memory_file(self) -> None:
        """Initialize the inter-agent communication memory file"""
        initial_content = """# Inter-Agent Communication Memory

This file contains shared context and communication between Agentic agents.
It leverages Claude Code's memory system for persistent agent coordination.

## Current Context
- Multiple agents are working on this project
- This file serves as the communication hub
- Agents can share findings, ask questions, and coordinate tasks

## Active Agents
(Updated dynamically as agents join/leave)

## Communication Log
(Recent messages appear here)

---

**Note:** This file is managed by Agentic's inter-agent communication system.
"""
        self.memory_file.write_text(initial_content)
        self.logger.info(f"Initialized inter-agent communication memory at {self.memory_file}")
    
    async def _update_claude_memory(self, message: InterAgentMessage) -> None:
        """Update Claude Code's memory using the CLI"""
        try:
            # Use Claude Code's memory feature to store key context
            memory_update = f"# Agent Communication: {message.message_type} from {message.from_agent}"
            
            if message.message_type == "context_share":
                memory_update += f"\n{message.content[:200]}..."
            elif message.message_type == "decision":
                memory_update += f"\nDecision: {message.content[:150]}..."
            elif message.message_type == "discovery":
                memory_update += f"\nDiscovery: {message.content[:150]}..."
            
            # Use Claude CLI to update memory
            cmd = ["claude", "-p", f"/memory add {memory_update}"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_path)
            )
            
            await process.communicate()
            self.logger.debug(f"Updated Claude memory with message {message.id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update Claude memory: {e}")
    
    async def _notify_agents(self, message: InterAgentMessage) -> None:
        """Notify target agents about new message"""
        target_agents = []
        
        if message.to_agent:
            # Specific agent
            if message.to_agent in self.agents:
                target_agents.append(self.agents[message.to_agent])
        else:
            # Broadcast to all agents except sender
            target_agents = [
                agent for agent_id, agent in self.agents.items() 
                if agent_id != message.from_agent
            ]
        
        # Notify Claude Code agents using shared context
        for agent_session in target_agents:
            try:
                # Get the agent instance
                pass
                
                # Check if it's a Claude Code agent
                if hasattr(agent_session, 'agent_config') and agent_session.agent_config.agent_type.value == "claude_code":
                    # Find the actual agent instance (this is simplified - in practice we'd need a registry)
                    context_summary = f"New {message.message_type} from {message.from_agent}: {message.content[:100]}..."
                    
                    # This would be called on the actual agent instance
                    # agent_instance.set_shared_context(context_summary)
                    
                    self.logger.debug(f"Notified agent {agent_session.id} about message {message.id}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to notify agent {agent_session.id}: {e}")
    
    async def get_messages_for_agent(
        self, 
        agent_id: str,
        unread_only: bool = False,
        message_type: Optional[str] = None
    ) -> List[InterAgentMessage]:
        """Get messages for a specific agent"""
        try:
            if not self.memory_file.exists():
                return []
            
            # For now, parse from memory file (could be optimized with proper storage)
            content = self.memory_file.read_text()
            
            # Extract messages (simplified parsing)
            messages = []
            # This is a simplified implementation - in practice we'd use proper parsing
            # or store messages in a structured format
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to get messages for agent {agent_id}: {e}")
            return []
    
    async def get_shared_context(self) -> str:
        """Get current shared context summary for new agents"""
        try:
            if not self.memory_file.exists():
                return "No shared context available."
            
            content = self.memory_file.read_text()
            
            # Extract current context section
            if "## Current Context" in content:
                parts = content.split("## Current Context")[1].split("##")[0]
                return parts.strip()
            
            return "No current context available."
            
        except Exception as e:
            self.logger.error(f"Failed to get shared context: {e}")
            return "Error retrieving shared context."
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            "active_agents": len(self.agents),
            "agent_names": [agent.agent_config.name for agent in self.agents.values()],
            "memory_file_exists": self.memory_file.exists(),
            "memory_file_path": str(self.memory_file)
        }