"""
Intent Classifier for analyzing user commands and determining routing
"""

from __future__ import annotations

import re
from typing import List, Set

from agentic.models.task import TaskIntent, TaskType


class IntentClassifier:
    """Analyzes user commands to determine intent and routing"""
    
    def __init__(self):
        # Keywords for different task types
        self.debug_keywords = {
            "debug", "fix", "error", "bug", "issue", "problem", "crash", "fail", 
            "broken", "wrong", "incorrect", "troubleshoot", "diagnose"
        }
        
        self.implement_keywords = {
            "add", "create", "build", "implement", "make", "develop", "write",
            "generate", "construct", "produce", "establish", "setup", "install"
        }
        
        self.refactor_keywords = {
            "refactor", "reorganize", "restructure", "cleanup", "optimize",
            "improve", "simplify", "modernize", "rewrite", "rework"
        }
        
        self.explain_keywords = {
            "explain", "describe", "what", "how", "why", "tell", "show",
            "clarify", "detail", "analyze", "understand", "meaning"
        }
        
        self.test_keywords = {
            "test", "spec", "unit test", "integration test", "e2e", "testing",
            "coverage", "verify", "validate", "check", "assert", "tests", 
            "passing", "failing", "pass", "fail", "run tests", "test suite",
            "pytest", "jest", "mocha", "jasmine"
        }
        
        # Keywords for different areas
        self.frontend_keywords = {
            "component", "ui", "frontend", "react", "vue", "angular", "css", 
            "style", "page", "form", "button", "modal", "layout", "design",
            "interface", "user", "client", "browser", "dom", "jsx", "tsx"
        }
        
        self.backend_keywords = {
            "api", "backend", "server", "database", "endpoint", "service", 
            "auth", "authentication", "authorization", "middleware", "route",
            "controller", "model", "query", "sql", "orm", "rest", "graphql"
        }
        
        self.testing_keywords = {
            "test", "spec", "unit", "integration", "e2e", "coverage", "mock",
            "stub", "fixture", "assertion", "verify", "validate", "qa"
        }
        
        self.devops_keywords = {
            "deploy", "deployment", "docker", "container", "ci", "cd", "pipeline",
            "build", "infrastructure", "k8s", "kubernetes", "terraform", "aws"
        }
        
        # Complexity indicators
        self.high_complexity_keywords = {
            "entire", "all", "throughout", "across", "multiple", "complex",
            "architecture", "system", "integration", "workflow", "algorithm"
        }
        
        self.coordination_keywords = {
            "across", "throughout", "all", "entire", "multiple", "both",
            "frontend and backend", "full stack", "end to end", "complete",
            "comprehensive", "system", "application", "platform", "solution"
        }
    
    async def analyze_intent(self, command: str) -> TaskIntent:
        """Analyze command to determine intent"""
        command_lower = command.lower()
        command_words = set(re.findall(r'\b\w+\b', command_lower))
        
        # Determine task type
        task_type = self._classify_task_type(command_words)
        
        # Analyze complexity
        complexity_score = self._calculate_complexity(command, command_words)
        
        # Determine affected areas
        affected_areas = self._identify_affected_areas(command_words)
        
        # Check if reasoning is required
        requires_reasoning = self._requires_reasoning(task_type, complexity_score, command_words)
        
        # Check if coordination is required - pass the full command for phrase matching
        requires_coordination = self._requires_coordination_v2(command_lower, command_words, affected_areas)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(task_type, complexity_score, len(command_words))
        
        return TaskIntent(
            task_type=task_type,
            complexity_score=complexity_score,
            estimated_duration=estimated_duration,
            affected_areas=affected_areas,
            requires_reasoning=requires_reasoning,
            requires_coordination=requires_coordination
        )
    
    def _classify_task_type(self, command_words: Set[str]) -> TaskType:
        """Classify the type of task based on keywords"""
        # Count matches for each task type
        debug_matches = len(command_words & self.debug_keywords)
        implement_matches = len(command_words & self.implement_keywords)
        refactor_matches = len(command_words & self.refactor_keywords)
        explain_matches = len(command_words & self.explain_keywords)
        test_matches = len(command_words & self.test_keywords)
        
        # Find the task type with most matches
        task_scores = {
            TaskType.DEBUG: debug_matches,
            TaskType.IMPLEMENT: implement_matches,
            TaskType.REFACTOR: refactor_matches,
            TaskType.EXPLAIN: explain_matches,
            TaskType.TEST: test_matches
        }
        
        # Return the task type with highest score, default to IMPLEMENT
        max_score = max(task_scores.values())
        if max_score == 0:
            return TaskType.IMPLEMENT
        
        for task_type, score in task_scores.items():
            if score == max_score:
                return task_type
        
        return TaskType.IMPLEMENT
    
    def _calculate_complexity(self, command: str, command_words: Set[str]) -> float:
        """Calculate complexity score from 0.0 to 1.0"""
        complexity_factors = []
        
        # Length factor (longer commands tend to be more complex)
        word_count = len(command_words)
        length_factor = min(word_count / 20.0, 1.0)  # Normalize to 20 words = 1.0
        complexity_factors.append(length_factor * 0.3)
        
        # High complexity keywords
        complexity_keywords_found = len(command_words & self.high_complexity_keywords)
        keyword_factor = min(complexity_keywords_found / 3.0, 1.0)  # 3+ keywords = 1.0
        complexity_factors.append(keyword_factor * 0.4)
        
        # Multiple areas mentioned increases complexity
        area_keywords = (
            (command_words & self.frontend_keywords) |
            (command_words & self.backend_keywords) |
            (command_words & self.testing_keywords) |
            (command_words & self.devops_keywords)
        )
        area_factor = min(len(area_keywords) / 5.0, 1.0)  # 5+ area keywords = 1.0
        complexity_factors.append(area_factor * 0.3)
        
        return min(sum(complexity_factors), 1.0)
    
    def _identify_affected_areas(self, command_words: Set[str]) -> List[str]:
        """Identify which areas of codebase are affected"""
        areas = []
        
        # Check each area
        if command_words & self.frontend_keywords:
            areas.append("frontend")
        
        if command_words & self.backend_keywords:
            areas.append("backend")
        
        if command_words & self.testing_keywords:
            areas.append("testing")
        
        if command_words & self.devops_keywords:
            areas.append("devops")
        
        # Default to general if no specific area detected
        return areas if areas else ["general"]
    
    def _requires_reasoning(self, task_type: TaskType, complexity_score: float, 
                           command_words: Set[str]) -> bool:
        """Determine if task requires reasoning capabilities"""
        # Always require reasoning for debug and explain tasks
        if task_type in [TaskType.DEBUG, TaskType.EXPLAIN]:
            return True
        
        # High complexity tasks need reasoning
        if complexity_score > 0.7:
            return True
        
        # Tasks with analysis keywords need reasoning
        analysis_keywords = {"analyze", "understand", "why", "how", "cause", "reason"}
        if command_words & analysis_keywords:
            return True
        
        return False
    
    def _requires_coordination(self, command_words: Set[str], affected_areas: List[str]) -> bool:
        """Determine if task requires coordination between agents"""
        # Multiple areas affected requires coordination
        if len(affected_areas) > 1:
            return True
        
        # Specific coordination keywords
        if command_words & self.coordination_keywords:
            return True
        
        # Check for multi-word coordination phrases
        command_text = ' '.join(command_words)
        multi_word_phrases = [
            "production ready", "production-ready", "sniper bot", "trading bot",
            "full stack", "end to end", "with tests", "including tests",
            "real time", "real-time", "with frontend", "with backend",
            "complete application", "complete system", "build me a"
        ]
        
        for phrase in multi_word_phrases:
            if phrase in command_text:
                return True
        
        return False
    
    def _requires_coordination_v2(self, command_lower: str, command_words: Set[str], affected_areas: List[str]) -> bool:
        """Determine if task requires coordination between agents based on complexity indicators"""
        # Multiple areas affected requires coordination
        if len(affected_areas) > 1:
            return True
        
        # Specific coordination keywords
        if command_words & self.coordination_keywords:
            return True
        
        # Check for multi-word coordination phrases that indicate complexity
        multi_word_phrases = [
            "full stack", "full-stack", "end to end", "end-to-end",
            "with tests", "including tests", "and tests", "with documentation",
            "complete application", "complete system", "entire system",
            "production ready", "production-ready", "enterprise grade"
        ]
        
        for phrase in multi_word_phrases:
            if phrase in command_lower:
                return True
        
        # Heuristic: Complex creation tasks with multiple components
        creation_words = {"build", "create", "implement", "develop", "make", "construct"}
        complexity_modifiers = {"complete", "full", "entire", "comprehensive", "production", "enterprise", "scalable", "distributed"}
        scope_words = {"application", "system", "platform", "service", "solution", "project", "bot", "tool", "utility"}
        
        has_creation = bool(command_words & creation_words)
        has_modifier = bool(command_words & complexity_modifiers)
        has_scope = bool(command_words & scope_words)
        
        # If command has creation + modifier + scope, it likely needs multiple agents
        if has_creation and has_modifier and has_scope:
            return True
        
        # If command mentions multiple technical components
        technical_components = 0
        component_categories = [
            {"api", "backend", "server", "database", "auth", "authentication"},
            {"ui", "frontend", "interface", "dashboard", "client"},
            {"test", "testing", "tests", "qa", "quality"},
            {"deploy", "deployment", "docker", "kubernetes", "ci", "cd"},
            {"docs", "documentation", "readme", "guide"}
        ]
        
        for category in component_categories:
            if command_words & category:
                technical_components += 1
        
        # Multiple technical components suggest coordination needed
        if technical_components >= 2:
            return True
        
        return False
    
    def _estimate_duration(self, task_type: TaskType, complexity_score: float, 
                          word_count: int) -> int:
        """Estimate task duration in minutes"""
        # Base duration by task type
        base_durations = {
            TaskType.EXPLAIN: 2,      # Usually quick explanations
            TaskType.DEBUG: 10,       # Can vary widely, medium base
            TaskType.IMPLEMENT: 15,   # Usually the longest
            TaskType.REFACTOR: 12,    # Medium duration
            TaskType.TEST: 8,         # Usually focused and quick
        }
        
        base_duration = base_durations.get(task_type, 10)
        
        # Adjust for complexity
        complexity_multiplier = 1.0 + (complexity_score * 2.0)  # 1.0 to 3.0x
        
        # Adjust for command length (more words = more detail = longer task)
        length_multiplier = 1.0 + (word_count / 50.0)  # Small adjustment for length
        
        estimated = int(base_duration * complexity_multiplier * length_multiplier)
        
        # Clamp to reasonable bounds
        return max(1, min(estimated, 120))  # 1 minute to 2 hours 