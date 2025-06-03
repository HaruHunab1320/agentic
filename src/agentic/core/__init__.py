"""
Core functionality for Agentic

This package contains the core orchestration and analysis components.
"""

from agentic.core.project_analyzer import ProjectAnalyzer
from agentic.core.orchestrator import Orchestrator
from agentic.core.agent_registry import AgentRegistry
from agentic.core.command_router import CommandRouter, RoutingDecision
from agentic.core.intent_classifier import IntentClassifier

__all__ = [
    "ProjectAnalyzer",
    "Orchestrator",
    "AgentRegistry",
    "CommandRouter",
    "RoutingDecision",
    "IntentClassifier",
] 