"""
Enhanced Task Analyzer for Intelligent Agent Selection
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from agentic.utils.logging import LoggerMixin


@dataclass
class TaskAnalysis:
    """Detailed analysis of a task for agent selection"""
    command: str
    file_count_estimate: int
    complexity_score: float  # 0.0 to 1.0
    requires_creativity: bool
    follows_pattern: bool
    ambiguity_level: float  # 0.0 to 1.0
    bulk_operation: bool
    suggested_agent: str
    reasoning: List[str]
    complexity_indicators: List[str]
    operation_type: str  # create, modify, analyze, refactor, etc.


class TaskAnalyzer(LoggerMixin):
    """Analyzes tasks to determine optimal agent selection"""
    
    def __init__(self):
        super().__init__()
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize command patterns for analysis"""
        self.command_patterns = {
            # Claude-optimal patterns
            'claude_preferred': [
                (r'explain\s+(?:how|why|what)', 'explanation_needed'),
                (r'analyze\s+(?:the|this|my)', 'analysis_required'),
                (r'design\s+(?:a|an|the)', 'design_task'),
                (r'architect(?:ure)?\s+', 'architecture_task'),
                (r'review\s+(?:this|the|my)', 'code_review'),
                (r'debug\s+(?:why|this)', 'complex_debugging'),
                (r'optimize\s+the\s+algorithm', 'algorithm_optimization'),
                (r'suggest\s+improvements', 'improvement_suggestions'),
                (r'complex\s+logic', 'complex_logic'),
                (r'algorithm\s+for', 'algorithm_design'),
                (r'understand\s+', 'comprehension_task'),
                (r'best\s+(?:way|approach|practice)', 'best_practice_advice'),
            ],
            
            # Aider-optimal patterns  
            'aider_preferred': [
                (r'create\s+\d+\s+files', 'bulk_file_creation'),
                (r'add\s+.*\s+to\s+all', 'bulk_modification'),
                (r'update\s+every', 'bulk_update'),
                (r'implement\s+across', 'multi_file_implementation'),
                (r'bulk\s+', 'bulk_operation'),
                (r'rename\s+.*\s+to', 'refactoring_operation'),
                (r'move\s+files', 'file_reorganization'),
                (r'reorganize\s+', 'project_reorganization'),
                (r'apply\s+.*\s+pattern', 'pattern_application'),
                (r'scaffold\s+', 'scaffolding_task'),
                (r'boilerplate\s+', 'boilerplate_creation'),
            ],
            
            # Hybrid patterns (benefit from both)
            'hybrid_preferred': [
                (r'complete\s+application', 'full_application'),
                (r'full[\s-]stack', 'full_stack_feature'),
                (r'entire\s+feature', 'complete_feature'),
                (r'end[\s-]to[\s-]end', 'e2e_implementation'),
                (r'comprehensive\s+', 'comprehensive_task'),
                (r'(?:build|create)\s+.*\s+with\s+tests', 'feature_with_tests'),
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = [
            'algorithm', 'optimize', 'performance', 'security', 'authenticate',
            'authorize', 'encrypt', 'compress', 'cache', 'distribute',
            'scale', 'concurrent', 'parallel', 'async', 'synchronize',
            'transaction', 'migrate', 'transform', 'parse', 'compile'
        ]
        
        # Creativity indicators
        self.creativity_indicators = [
            'design', 'architect', 'create', 'build', 'develop', 'implement',
            'innovative', 'novel', 'new way', 'better approach', 'improve',
            'enhance', 'reimagine', 'rethink', 'from scratch'
        ]
        
        # Pattern-following indicators
        self.pattern_indicators = [
            'similar to', 'like the', 'same as', 'follow the pattern',
            'copy from', 'based on', 'match the style', 'consistent with',
            'use existing', 'follow convention'
        ]
    
    async def analyze_task(self, command: str, project_context: Optional[Dict] = None) -> TaskAnalysis:
        """Perform comprehensive task analysis"""
        command_lower = command.lower()
        
        # Basic analysis
        file_count = self._estimate_file_count(command_lower, project_context)
        complexity = self._calculate_complexity(command_lower)
        creativity = self._requires_creativity(command_lower)
        pattern_following = self._follows_pattern(command_lower)
        ambiguity = self._measure_ambiguity(command_lower)
        bulk_op = self._is_bulk_operation(command_lower)
        op_type = self._determine_operation_type(command_lower)
        
        # Determine suggested agent
        agent, reasoning = self._suggest_agent(
            command_lower, file_count, complexity, creativity, 
            pattern_following, ambiguity, bulk_op
        )
        
        # Find complexity indicators
        found_indicators = [
            indicator for indicator in self.complexity_indicators
            if indicator in command_lower
        ]
        
        return TaskAnalysis(
            command=command,
            file_count_estimate=file_count,
            complexity_score=complexity,
            requires_creativity=creativity,
            follows_pattern=pattern_following,
            ambiguity_level=ambiguity,
            bulk_operation=bulk_op,
            suggested_agent=agent,
            reasoning=reasoning,
            complexity_indicators=found_indicators,
            operation_type=op_type
        )
    
    def _estimate_file_count(self, command: str, project_context: Optional[Dict]) -> int:
        """Estimate how many files will be affected"""
        # Look for explicit numbers
        number_match = re.search(r'(\d+)\s*files?', command)
        if number_match:
            return int(number_match.group(1))
        
        # Keywords suggesting multiple files
        multi_file_keywords = [
            'all', 'every', 'across', 'throughout', 'multiple', 'several',
            'entire', 'whole', 'full', 'complete application', 'system'
        ]
        
        multi_file_count = sum(1 for keyword in multi_file_keywords if keyword in command)
        
        # Component keywords
        component_keywords = [
            'frontend', 'backend', 'api', 'database', 'ui', 'server',
            'client', 'tests', 'documentation', 'config', 'deployment'
        ]
        
        component_count = sum(1 for keyword in component_keywords if keyword in command)
        
        # Estimate based on keywords
        if multi_file_count >= 2 or component_count >= 3:
            return 10  # Large multi-file operation
        elif multi_file_count >= 1 or component_count >= 2:
            return 5   # Medium multi-file operation
        elif component_count >= 1:
            return 3   # Small multi-file operation
        else:
            return 1   # Single file operation
    
    def _calculate_complexity(self, command: str) -> float:
        """Calculate task complexity score (0.0 to 1.0)"""
        complexity_score = 0.0
        
        # Check for complexity indicators
        for indicator in self.complexity_indicators:
            if indicator in command:
                complexity_score += 0.15
        
        # Check for multiple requirements
        if ' and ' in command:
            complexity_score += 0.1 * command.count(' and ')
        
        # Long commands tend to be more complex
        word_count = len(command.split())
        if word_count > 30:
            complexity_score += 0.3
        elif word_count > 20:
            complexity_score += 0.2
        elif word_count > 10:
            complexity_score += 0.1
        
        # Architecture/design tasks are complex
        if any(keyword in command for keyword in ['architect', 'design', 'system', 'framework']):
            complexity_score += 0.3
        
        return min(complexity_score, 1.0)
    
    def _requires_creativity(self, command: str) -> bool:
        """Determine if task requires creative problem solving"""
        return any(indicator in command for indicator in self.creativity_indicators)
    
    def _follows_pattern(self, command: str) -> bool:
        """Determine if task follows existing patterns"""
        return any(indicator in command for indicator in self.pattern_indicators)
    
    def _measure_ambiguity(self, command: str) -> float:
        """Measure command ambiguity (0.0 = clear, 1.0 = very ambiguous)"""
        ambiguity_score = 0.0
        
        # Vague terms increase ambiguity
        vague_terms = [
            'somehow', 'maybe', 'probably', 'might', 'could', 'should',
            'better', 'improve', 'enhance', 'fix', 'update', 'change'
        ]
        
        for term in vague_terms:
            if term in command:
                ambiguity_score += 0.15
        
        # Questions increase ambiguity
        if '?' in command:
            ambiguity_score += 0.2
        
        # Short commands without specifics are ambiguous
        word_count = len(command.split())
        if word_count < 5:
            ambiguity_score += 0.3
        
        # Missing details
        if not any(char in command for char in ['.', '/', '_', '-']):
            # No file paths or specific names
            ambiguity_score += 0.1
        
        return min(ambiguity_score, 1.0)
    
    def _is_bulk_operation(self, command: str) -> bool:
        """Determine if this is a bulk operation"""
        bulk_keywords = [
            'all', 'every', 'bulk', 'multiple', 'batch', 'across',
            'throughout', 'update all', 'change all', 'modify all'
        ]
        return any(keyword in command for keyword in bulk_keywords)
    
    def _determine_operation_type(self, command: str) -> str:
        """Determine the primary operation type"""
        operation_patterns = [
            (r'^create|^add|^implement|^build', 'create'),
            (r'^modify|^update|^change|^edit', 'modify'),
            (r'^analyze|^explain|^understand', 'analyze'),
            (r'^refactor|^reorganize|^restructure', 'refactor'),
            (r'^test|^verify|^validate', 'test'),
            (r'^debug|^fix|^solve', 'debug'),
            (r'^optimize|^improve|^enhance', 'optimize'),
            (r'^document|^comment', 'document'),
            (r'^deploy|^release|^publish', 'deploy')
        ]
        
        for pattern, op_type in operation_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return op_type
        
        return 'general'
    
    def _suggest_agent(self, command: str, file_count: int, complexity: float,
                      creativity: bool, pattern_following: bool, ambiguity: float,
                      bulk_op: bool) -> Tuple[str, List[str]]:
        """Suggest the optimal agent based on analysis"""
        reasoning = []
        
        # Check for explicit pattern matches
        for pattern, indicator in self.command_patterns['claude_preferred']:
            if re.search(pattern, command):
                reasoning.append(f"Claude pattern match: {indicator}")
                return 'claude_code', reasoning
        
        for pattern, indicator in self.command_patterns['aider_preferred']:
            if re.search(pattern, command):
                reasoning.append(f"Aider pattern match: {indicator}")
                return self._select_aider_variant(command), reasoning
        
        # Heuristic-based selection
        if ambiguity > 0.7:
            reasoning.append(f"High ambiguity ({ambiguity:.2f}) - Claude better at understanding unclear requests")
            return 'claude_code', reasoning
        
        if complexity > 0.7 and creativity:
            reasoning.append(f"High complexity ({complexity:.2f}) with creativity needed")
            return 'claude_code', reasoning
        
        if file_count > 3 or bulk_op:
            reasoning.append(f"Multi-file operation ({file_count} files) - Aider more efficient")
            return self._select_aider_variant(command), reasoning
        
        if pattern_following and not creativity:
            reasoning.append("Following existing patterns - Aider efficient at pattern replication")
            return self._select_aider_variant(command), reasoning
        
        # Hybrid approach for medium complexity
        if file_count >= 2 and complexity > 0.5:
            reasoning.append("Medium complexity multi-file task - consider hybrid approach")
            return 'hybrid', reasoning
        
        # Default based on operation type
        op_type = self._determine_operation_type(command)
        if op_type in ['analyze', 'debug', 'optimize']:
            reasoning.append(f"Operation type '{op_type}' - Claude excels at reasoning tasks")
            return 'claude_code', reasoning
        else:
            reasoning.append(f"Operation type '{op_type}' - standard implementation task")
            return self._select_aider_variant(command), reasoning
    
    def _select_aider_variant(self, command: str) -> str:
        """Select the appropriate Aider variant"""
        if any(keyword in command for keyword in ['frontend', 'ui', 'react', 'component', 'css']):
            return 'aider_frontend'
        elif any(keyword in command for keyword in ['test', 'spec', 'coverage', 'jest', 'pytest']):
            return 'aider_testing'
        else:
            return 'aider_backend'