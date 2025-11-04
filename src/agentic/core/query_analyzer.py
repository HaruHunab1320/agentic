"""
Query Analyzer for Intelligent Command Routing

Determines whether a query should be handled by a single agent
or requires coordinated multi-agent analysis.
"""

from dataclasses import dataclass
from typing import List, Optional
import re

from agentic.utils.logging import LoggerMixin


@dataclass
class QueryAnalysis:
    """Analysis result for a user query"""
    query_type: str  # "question", "analysis", "implementation", "refactoring"
    complexity: str  # "simple", "moderate", "complex"
    scope: str  # "single_file", "module", "cross_cutting", "system_wide"
    requires_code_generation: bool
    requires_multi_perspective: bool
    suggested_approach: str  # "single_agent", "coordinated_analysis", "multi_agent_implementation"
    suggested_agent: Optional[str] = None
    reasoning: List[str] = None
    

class QueryAnalyzer(LoggerMixin):
    """Analyzes queries to determine optimal execution strategy"""
    
    def __init__(self):
        super().__init__()
        
        # Query patterns for different types
        self.question_patterns = [
            r'\b(what|where|when|why|how|who|which)\b',
            r'\b(explain|describe|show|tell me about|tell me what)\b',
            r'\b(does|is|are|can|could|should|would)\b.*\?',
            r'\?$'  # Ends with question mark
        ]
        
        self.analysis_patterns = [
            r'\b(analyze|review|assess|evaluate|examine)\b',
            r'\b(architecture|design|structure|flow|dependencies)\b',
            r'\b(performance|security|quality|issues)\b'
        ]
        
        self.implementation_patterns = [
            r'\b(create|build|implement|add|write|generate)\b',
            r'\b(feature|component|function|class|module)\b',
            r'\b(test|tests|testing)\b.*\b(for|to)\b'
        ]
        
        self.refactoring_patterns = [
            r'\b(refactor|improve|optimize|clean up|reorganize)\b',
            r'\b(rename|move|extract|inline)\b',
            r'\b(fix|repair|resolve|debug)\b'
        ]
        
        # Scope indicators
        self.single_file_indicators = [
            r'\b(this file|this function|this class|this method)\b',
            r'\b(in \w+\.\w+)\b',  # "in file.ext"
            r'[\w/]+\.\w+',  # File path
            r'\b(specific|particular|single)\b'
        ]
        
        self.cross_cutting_indicators = [
            r'\b(across|throughout|all|entire|whole)\b',
            r'\b(system|application|codebase|project)\b',
            r'\b(everywhere|anywhere|multiple)\b',
            r'\b(integration|interaction|communication)\b'
        ]
        
        self.multi_perspective_indicators = [
            r'\b(frontend and backend|full stack|end-to-end)\b',
            r'\b(different perspectives|various aspects|multiple angles)\b',
            r'\b(comprehensive|complete|thorough)\b',
            r'\b(how.*work together|interconnect|integrate)\b'
        ]
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a query to determine execution strategy"""
        query_lower = query.lower()
        
        # Determine query type
        query_type = self._determine_query_type(query_lower)
        
        # Determine scope
        scope = self._determine_scope(query_lower)
        
        # Determine complexity
        complexity = self._determine_complexity(query_lower, query_type, scope)
        
        # Check if code generation is needed
        requires_code_generation = self._requires_code_generation(query_lower, query_type)
        
        # Check if multi-perspective analysis is beneficial
        requires_multi_perspective = self._requires_multi_perspective(query_lower, scope)
        
        # Determine suggested approach
        suggested_approach, suggested_agent, reasoning = self._determine_approach(
            query_type, complexity, scope, requires_code_generation, requires_multi_perspective, query_lower
        )
        
        return QueryAnalysis(
            query_type=query_type,
            complexity=complexity,
            scope=scope,
            requires_code_generation=requires_code_generation,
            requires_multi_perspective=requires_multi_perspective,
            suggested_approach=suggested_approach,
            suggested_agent=suggested_agent,
            reasoning=reasoning or []
        )
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query"""
        # Check if this is a follow-up implementation request
        implementation_followup_phrases = [
            "now implement", "can you now implement", "please implement",
            "go ahead", "proceed with", "let's implement",
            "the missing logic", "complete the", "build that",
            "implement it", "create it", "build it"
        ]
        
        if any(phrase in query for phrase in implementation_followup_phrases):
            return "implementation"
        
        # Check if query contains analysis context (enriched command)
        if "Based on the following analysis:" in query and any(word in query for word in ['implement', 'build', 'create', 'complete']):
            return "implementation"
        
        # Check for questions first if they contain question indicators
        if any(re.search(pattern, query) for pattern in self.question_patterns):
            # Common question patterns that should remain questions
            if any(phrase in query for phrase in [
                "tell me what", "can you tell me", "what does", "what is", 
                "how does", "why does", "explain", "describe"
            ]):
                return "question"
            # Check if it also has implementation keywords
            elif any(re.search(pattern, query) for pattern in self.implementation_patterns):
                return "implementation"
            else:
                return "question"
        # Check patterns in order of specificity
        elif any(re.search(pattern, query) for pattern in self.implementation_patterns):
            return "implementation"
        elif any(re.search(pattern, query) for pattern in self.refactoring_patterns):
            return "refactoring"
        elif any(re.search(pattern, query) for pattern in self.analysis_patterns):
            return "analysis"
        else:
            # Default based on context
            if any(word in query for word in ['create', 'build', 'add']):
                return "implementation"
            else:
                return "question"
    
    def _determine_scope(self, query: str) -> str:
        """Determine the scope of the query"""
        # Check for explicit file references
        if any(re.search(pattern, query) for pattern in self.single_file_indicators):
            return "single_file"
        
        # Check for cross-cutting concerns
        if any(re.search(pattern, query) for pattern in self.cross_cutting_indicators):
            return "system_wide"
        
        # Check for module-level keywords
        if any(word in query for word in ['module', 'package', 'component', 'service']):
            return "module"
        
        # Default based on query type indicators
        if 'how' in query and any(word in query for word in ['works', 'flow', 'process']):
            return "cross_cutting"
        
        return "module"  # Default
    
    def _determine_complexity(self, query: str, query_type: str, scope: str) -> str:
        """Determine query complexity"""
        # Simple queries
        if query_type == "question" and scope == "single_file":
            return "simple"
        
        # Complex queries
        if scope == "system_wide" or (query_type == "analysis" and scope == "cross_cutting"):
            return "complex"
        
        # Implementation complexity
        if query_type == "implementation":
            if any(word in query for word in ['simple', 'basic', 'small']):
                return "simple"
            elif any(word in query for word in ['complete', 'full', 'comprehensive']):
                return "complex"
        
        return "moderate"
    
    def _requires_code_generation(self, query: str, query_type: str) -> bool:
        """Check if query requires code generation"""
        # Questions about what to build don't require code generation, just analysis
        if "tell me what" in query.lower() and "build" in query.lower():
            return False
            
        if query_type in ["implementation", "refactoring"]:
            return True
        
        # Some analysis might need example code
        if query_type == "analysis" and any(word in query for word in ['example', 'demonstrate', 'show how']):
            return True
        
        return False
    
    def _requires_multi_perspective(self, query: str, scope: str) -> bool:
        """Check if query benefits from multiple perspectives"""
        if scope in ["cross_cutting", "system_wide"]:
            return True
        
        if any(re.search(pattern, query) for pattern in self.multi_perspective_indicators):
            return True
        
        return False
    
    def _determine_approach(self, query_type: str, complexity: str, scope: str, 
                          requires_code_generation: bool, requires_multi_perspective: bool,
                          query: str = "") -> tuple[str, Optional[str], List[str]]:
        """Determine the suggested approach and agent"""
        reasoning = []
        
        # Questions - prefer Claude Code for exploration
        if query_type == "question":
            reasoning.append("Question requires code exploration and analysis")
            reasoning.append("Claude Code can freely explore the codebase to find answers")
            # Always use Claude for questions that need exploration
            if any(word in query.lower() for word in ['test', 'tests', 'skipped', 'find', 'explore', 'search']):
                reasoning.append("Query involves finding specific code elements")
            return "single_agent", "claude_code", reasoning
        
        # Complex analysis - consider coordinated approach
        if query_type == "analysis" and (complexity == "complex" or requires_multi_perspective):
            reasoning.append("Complex analysis benefits from multiple perspectives")
            reasoning.append("Different agents can analyze different aspects in parallel")
            return "coordinated_analysis", None, reasoning
        
        # Implementation tasks
        if query_type == "implementation":
            # Check if this is a follow-up implementation with context
            if "Based on the following analysis:" in query:
                reasoning.append("Implementation request includes analysis context")
                reasoning.append("Multiple specialized agents needed based on analysis")
                return "multi_agent_implementation", None, reasoning
            
            # Check for follow-up phrases that suggest multi-agent work
            follow_up_indicators = ["now implement", "complete the", "missing logic", "implement it"]
            if any(indicator in query.lower() for indicator in follow_up_indicators):
                reasoning.append("Follow-up implementation likely requires multiple components")
                reasoning.append("Multi-agent approach ensures comprehensive implementation")
                return "multi_agent_implementation", None, reasoning
            
            if complexity == "simple" or scope == "single_file":
                reasoning.append("Simple implementation can be handled by one agent")
                reasoning.append("Aider is efficient for focused code generation")
                return "single_agent", "aider", reasoning
            else:
                reasoning.append("Complex implementation benefits from specialized agents")
                reasoning.append("Multiple agents can work on different components")
                return "multi_agent_implementation", None, reasoning
        
        # Refactoring
        if query_type == "refactoring":
            if scope == "single_file":
                reasoning.append("Single file refactoring is straightforward")
                return "single_agent", "aider", reasoning
            else:
                reasoning.append("Cross-file refactoring needs coordination")
                return "coordinated_analysis", None, reasoning
        
        # Default fallback
        reasoning.append("Query can be handled efficiently by a single agent")
        return "single_agent", "claude_code", reasoning


def format_query_analysis(analysis: QueryAnalysis) -> str:
    """Format query analysis for display"""
    lines = [
        f"Query Type: {analysis.query_type}",
        f"Complexity: {analysis.complexity}",
        f"Scope: {analysis.scope}",
        f"Requires Code Generation: {'Yes' if analysis.requires_code_generation else 'No'}",
        f"Multi-Perspective Beneficial: {'Yes' if analysis.requires_multi_perspective else 'No'}",
        f"Suggested Approach: {analysis.suggested_approach}"
    ]
    
    if analysis.suggested_agent:
        lines.append(f"Suggested Agent: {analysis.suggested_agent}")
    
    if analysis.reasoning:
        lines.append("\nReasoning:")
        for reason in analysis.reasoning:
            lines.append(f"  • {reason}")
    
    return "\n".join(lines)