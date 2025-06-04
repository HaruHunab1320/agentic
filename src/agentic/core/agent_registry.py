"""
Agent Registry for managing multiple agent instances
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type

from agentic.models.agent import Agent, AgentConfig, AgentSession, AgentType, AgentCapability
from agentic.models.project import ProjectStructure
from agentic.utils.logging import LoggerMixin


class AgentRegistry(LoggerMixin):
    """Manages multiple agent instances and their lifecycle"""
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path
        self.agent_types: Dict[AgentType, Type[Agent]] = {}
        self.active_sessions: Dict[str, AgentSession] = {}
        self.agents: Dict[str, Agent] = {}
        
        # Register built-in agent types
        self._register_builtin_agents()
    
    def _register_builtin_agents(self) -> None:
        """Register built-in agent types"""
        from agentic.agents.aider_agents import (
            AiderFrontendAgent,
            AiderBackendAgent, 
            AiderTestingAgent
        )
        from agentic.agents.claude_code_agent import ClaudeCodeAgent
        
        self.agent_types[AgentType.AIDER_FRONTEND] = AiderFrontendAgent
        self.agent_types[AgentType.AIDER_BACKEND] = AiderBackendAgent
        self.agent_types[AgentType.AIDER_TESTING] = AiderTestingAgent
        self.agent_types[AgentType.CLAUDE_CODE] = ClaudeCodeAgent
    
    def register_agent_type(self, agent_type: AgentType, agent_class: Type[Agent]) -> None:
        """Register a new agent type"""
        self.logger.info(f"Registering agent type: {agent_type}")
        self.agent_types[agent_type] = agent_class
    
    async def spawn_agent(self, config: AgentConfig) -> AgentSession:
        """Spawn a new agent instance"""
        self.logger.info(f"Spawning agent: {config.name} ({config.agent_type})")
        
        if config.agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {config.agent_type}")
        
        agent_class = self.agent_types[config.agent_type]
        agent = agent_class(config)
        
        session = AgentSession(
            agent_config=config,
            workspace=config.workspace_path,
            status="starting"
        )
        
        try:
            success = await agent.start()
            if success:
                session.mark_active()
                self.active_sessions[session.id] = session
                self.agents[session.id] = agent
                agent.session = session
                self.logger.info(f"Agent {config.name} started successfully")
            else:
                session.mark_error("Failed to start agent")
                self.logger.error(f"Failed to start agent {config.name}")
        except Exception as e:
            session.mark_error(str(e))
            self.logger.error(f"Error starting agent {config.name}: {e}")
        
        return session
    
    async def terminate_agent(self, session_id: str) -> bool:
        """Terminate an agent session"""
        if session_id not in self.agents:
            self.logger.warning(f"Attempt to terminate unknown agent session: {session_id}")
            return False
        
        agent = self.agents[session_id]
        session = self.active_sessions[session_id]
        
        self.logger.info(f"Terminating agent: {session.agent_config.name}")
        
        try:
            await agent.stop()
            del self.agents[session_id]
            del self.active_sessions[session_id]
            self.logger.info(f"Agent {session.agent_config.name} terminated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error terminating agent {session.agent_config.name}: {e}")
            return False
    
    async def terminate_all_agents(self) -> int:
        """Terminate all active agents"""
        self.logger.info("Terminating all active agents")
        terminated_count = 0
        
        # Create a copy of session IDs to avoid modifying dict during iteration
        session_ids = list(self.active_sessions.keys())
        
        for session_id in session_ids:
            if await self.terminate_agent(session_id):
                terminated_count += 1
        
        self.logger.info(f"Terminated {terminated_count} agents")
        return terminated_count
    
    async def initialize(self) -> None:
        """Initialize the agent registry"""
        self.logger.info("Initializing agent registry...")
        # Registry initialization is mostly done in __init__
        # This method exists for compatibility with orchestrator
        self.logger.info("Agent registry initialized")
    
    def get_all_agents(self) -> List[AgentSession]:
        """Get all active agent sessions"""
        return list(self.active_sessions.values())
    
    async def stop_agent(self, session_id: str) -> bool:
        """Stop an agent (alias for terminate_agent for compatibility)"""
        return await self.terminate_agent(session_id)
    
    def get_agents_by_capability(self, capability: str) -> List[AgentSession]:
        """Find agents with specific capability"""
        matching_agents = []
        
        for session in self.active_sessions.values():
            if session.status == "active" and capability in session.agent_config.focus_areas:
                matching_agents.append(session)
        
        return matching_agents
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentSession]:
        """Find agents of specific type"""
        return [
            session for session in self.active_sessions.values()
            if session.agent_config.agent_type == agent_type and session.status == "active"
        ]
    
    def get_available_agents(self) -> List[AgentSession]:
        """Get all available agents (active and not busy)"""
        return [
            session for session in self.active_sessions.values()
            if session.is_available
        ]
    
    def get_agent_by_id(self, session_id: str) -> Optional[Agent]:
        """Get agent instance by session ID"""
        return self.agents.get(session_id)
    
    def get_session_by_id(self, session_id: str) -> Optional[AgentSession]:
        """Get agent session by ID"""
        return self.active_sessions.get(session_id)
    
    async def get_agent_status(self) -> Dict[str, Dict]:
        """Get status of all agents"""
        status = {}
        
        for session_id, session in self.active_sessions.items():
            agent = self.agents.get(session_id)
            try:
                is_healthy = await agent.health_check() if agent else False
            except Exception:
                is_healthy = False
            
            status[session_id] = {
                "name": session.agent_config.name,
                "type": session.agent_config.agent_type,
                "status": session.status,
                "focus_areas": session.agent_config.focus_areas,
                "current_task": session.current_task,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "is_healthy": is_healthy
            }
        
        return status
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all agents"""
        health_status = {}
        
        for session_id, agent in self.agents.items():
            try:
                health_status[session_id] = await agent.health_check()
            except Exception as e:
                self.logger.error(f"Health check failed for agent {session_id}: {e}")
                health_status[session_id] = False
        
        return health_status
    
    async def auto_spawn_agents(self, project_structure: ProjectStructure) -> List[AgentSession]:
        """Automatically spawn appropriate agents based on project structure"""
        self.logger.info("Auto-spawning agents based on project structure")
        spawned_sessions = []
        
        # Always spawn a reasoning agent (Claude Code)
        reasoning_config = AgentConfig(
            agent_type=AgentType.CLAUDE_CODE,
            name="reasoning",
            workspace_path=self.workspace_path,
            focus_areas=["debugging", "analysis", "explanation", "reasoning"],
            ai_model_config={"model": "claude-3-5-sonnet"}
        )
        
        reasoning_session = await self.spawn_agent(reasoning_config)
        if reasoning_session.status == "active":
            spawned_sessions.append(reasoning_session)
        
        # Spawn frontend agent for web projects
        web_frameworks = {"react", "vue", "angular", "svelte", "nextjs", "nuxt"}
        if any(fw.lower() in web_frameworks for fw in project_structure.tech_stack.frameworks):
            frontend_config = AgentConfig(
                agent_type=AgentType.AIDER_FRONTEND,
                name="frontend",
                workspace_path=self.workspace_path,
                focus_areas=["frontend", "components", "ui", "styling"],
                ai_model_config={"model": "claude-3-5-sonnet"}
            )
            
            frontend_session = await self.spawn_agent(frontend_config)
            if frontend_session.status == "active":
                spawned_sessions.append(frontend_session)
        
        # Spawn backend agent for server-side projects
        backend_languages = {"python", "javascript", "typescript", "go", "rust", "java"}
        backend_frameworks = {"fastapi", "django", "flask", "express", "nestjs", "gin", "actix", "spring"}
        
        has_backend = (
            any(lang.lower() in backend_languages for lang in project_structure.tech_stack.languages) or
            any(fw.lower() in backend_frameworks for fw in project_structure.tech_stack.frameworks)
        )
        
        if has_backend:
            backend_config = AgentConfig(
                agent_type=AgentType.AIDER_BACKEND,
                name="backend",
                workspace_path=self.workspace_path,
                focus_areas=["backend", "api", "database", "server"],
                ai_model_config={"model": "claude-3-5-sonnet"}
            )
            
            backend_session = await self.spawn_agent(backend_config)
            if backend_session.status == "active":
                spawned_sessions.append(backend_session)
        
        # Spawn testing agent if tests exist
        if project_structure.tech_stack.testing_frameworks or project_structure.test_directories:
            testing_config = AgentConfig(
                agent_type=AgentType.AIDER_TESTING,
                name="testing",
                workspace_path=self.workspace_path,
                focus_areas=["testing", "tests", "qa", "quality"],
                ai_model_config={"model": "claude-3-5-sonnet"}
            )
            
            testing_session = await self.spawn_agent(testing_config)
            if testing_session.status == "active":
                spawned_sessions.append(testing_session)
        
        self.logger.info(f"Auto-spawned {len(spawned_sessions)} agents")
        return spawned_sessions
    
    @property
    def active_agent_count(self) -> int:
        """Get count of active agents"""
        return len([s for s in self.active_sessions.values() if s.status == "active"])
    
    @property  
    def available_agent_count(self) -> int:
        """Get count of available agents"""
        return len(self.get_available_agents()) 