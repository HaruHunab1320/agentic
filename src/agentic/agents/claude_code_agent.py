"""
Claude Code Agent for reasoning and analysis tasks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from agentic.models.agent import Agent, AgentConfig, AgentType
from agentic.models.task import Task, TaskResult, TaskType
from agentic.utils.logging import LoggerMixin


class ClaudeCodeAgent(Agent):
    """Claude Code agent for reasoning, analysis, and explanation tasks"""
    
    def __init__(self, config: AgentConfig):
        # Ensure config is set for Claude Code
        if config.agent_type != AgentType.CLAUDE_CODE:
            config.agent_type = AgentType.CLAUDE_CODE
        if not config.focus_areas:
            config.focus_areas = ["reasoning", "analysis", "debugging", "explanation"]
        
        super().__init__(config)
        self.context_cache: Dict[str, str] = {}
    
    async def start(self) -> bool:
        """Start the Claude Code agent"""
        try:
            self.logger.info(f"Starting Claude Code agent: {self.config.name}")
            
            # Claude Code doesn't require external processes like Aider
            # Just verify configuration and prepare for reasoning tasks
            if not self._validate_config():
                return False
            
            self.logger.info(f"Claude Code agent {self.config.name} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Claude Code agent {self.config.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the Claude Code agent"""
        try:
            # Clear context cache
            self.context_cache.clear()
            
            self.logger.info(f"Claude Code agent {self.config.name} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping Claude Code agent {self.config.name}: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate agent configuration"""
        if not self.config.workspace_path or not self.config.workspace_path.exists():
            self.logger.error("Invalid workspace path")
            return False
        
        return True
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a reasoning/analysis task"""
        self.logger.info(f"Executing reasoning task: {task.command[:100]}...")
        
        try:
            # Different handling based on task type
            if task.task_type == TaskType.DEBUG:
                result = await self._handle_debug_task(task)
            elif task.task_type == TaskType.EXPLAIN:
                result = await self._handle_explain_task(task)
            elif task.task_type == TaskType.IMPLEMENT:
                result = await self._handle_implementation_guidance(task)
            elif task.task_type == TaskType.REFACTOR:
                result = await self._handle_refactor_guidance(task)
            else:
                result = await self._handle_general_analysis(task)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                agent_id=self.session.id if self.session else "unknown",
                success=False,
                output="",
                error=str(e)
            )
    
    async def _handle_debug_task(self, task: Task) -> TaskResult:
        """Handle debugging and troubleshooting tasks"""
        analysis_steps = []
        
        # Step 1: Analyze the problem
        analysis_steps.append("🔍 PROBLEM ANALYSIS")
        analysis_steps.append(f"Command: {task.command}")
        analysis_steps.append("")
        
        # Step 2: Gather context
        context = await self._gather_debugging_context(task)
        analysis_steps.append("📋 CONTEXT GATHERED")
        analysis_steps.extend(context)
        analysis_steps.append("")
        
        # Step 3: Identify potential issues
        potential_issues = await self._identify_potential_issues(task, context)
        analysis_steps.append("⚠️ POTENTIAL ISSUES IDENTIFIED")
        analysis_steps.extend(potential_issues)
        analysis_steps.append("")
        
        # Step 4: Provide debugging steps
        debugging_steps = await self._generate_debugging_steps(task, potential_issues)
        analysis_steps.append("🛠️ DEBUGGING RECOMMENDATIONS")
        analysis_steps.extend(debugging_steps)
        
        output = "\n".join(analysis_steps)
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.session.id if self.session else "unknown",
            success=True,
            output=output,
            error=""
        )
    
    async def _handle_explain_task(self, task: Task) -> TaskResult:
        """Handle explanation and analysis tasks"""
        explanation_parts = []
        
        # Step 1: Understand what needs explanation
        explanation_parts.append("📖 EXPLANATION REQUEST")
        explanation_parts.append(f"Query: {task.command}")
        explanation_parts.append("")
        
        # Step 2: Provide context
        context = await self._gather_explanation_context(task)
        explanation_parts.append("🎯 CONTEXT")
        explanation_parts.extend(context)
        explanation_parts.append("")
        
        # Step 3: Detailed explanation
        detailed_explanation = await self._generate_detailed_explanation(task)
        explanation_parts.append("💡 DETAILED EXPLANATION")
        explanation_parts.extend(detailed_explanation)
        explanation_parts.append("")
        
        # Step 4: Additional insights
        insights = await self._generate_insights(task)
        if insights:
            explanation_parts.append("🚀 ADDITIONAL INSIGHTS")
            explanation_parts.extend(insights)
        
        output = "\n".join(explanation_parts)
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.session.id if self.session else "unknown",
            success=True,
            output=output,
            error=""
        )
    
    async def _handle_implementation_guidance(self, task: Task) -> TaskResult:
        """Provide guidance for implementation tasks"""
        guidance_parts = []
        
        guidance_parts.append("🏗️ IMPLEMENTATION GUIDANCE")
        guidance_parts.append(f"Task: {task.command}")
        guidance_parts.append("")
        
        # Analysis phase
        analysis = await self._analyze_implementation_requirements(task)
        guidance_parts.append("📋 REQUIREMENTS ANALYSIS")
        guidance_parts.extend(analysis)
        guidance_parts.append("")
        
        # Architecture recommendations
        architecture = await self._recommend_architecture(task)
        guidance_parts.append("🏛️ ARCHITECTURE RECOMMENDATIONS")
        guidance_parts.extend(architecture)
        guidance_parts.append("")
        
        # Implementation plan
        plan = await self._create_implementation_plan(task)
        guidance_parts.append("📝 IMPLEMENTATION PLAN")
        guidance_parts.extend(plan)
        
        output = "\n".join(guidance_parts)
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.session.id if self.session else "unknown",
            success=True,
            output=output,
            error=""
        )
    
    async def _handle_refactor_guidance(self, task: Task) -> TaskResult:
        """Provide guidance for refactoring tasks"""
        guidance_parts = []
        
        guidance_parts.append("♻️ REFACTORING GUIDANCE")
        guidance_parts.append(f"Task: {task.command}")
        guidance_parts.append("")
        
        # Current state analysis
        current_analysis = await self._analyze_current_state(task)
        guidance_parts.append("📊 CURRENT STATE ANALYSIS")
        guidance_parts.extend(current_analysis)
        guidance_parts.append("")
        
        # Refactoring opportunities
        opportunities = await self._identify_refactoring_opportunities(task)
        guidance_parts.append("💡 REFACTORING OPPORTUNITIES")
        guidance_parts.extend(opportunities)
        guidance_parts.append("")
        
        # Refactoring plan
        plan = await self._create_refactoring_plan(task)
        guidance_parts.append("📋 REFACTORING PLAN")
        guidance_parts.extend(plan)
        
        output = "\n".join(guidance_parts)
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.session.id if self.session else "unknown",
            success=True,
            output=output,
            error=""
        )
    
    async def _handle_general_analysis(self, task: Task) -> TaskResult:
        """Handle general analysis tasks"""
        analysis_parts = []
        
        analysis_parts.append("🔬 GENERAL ANALYSIS")
        analysis_parts.append(f"Task: {task.command}")
        analysis_parts.append("")
        
        # Context gathering
        context = await self._gather_general_context(task)
        analysis_parts.append("📋 CONTEXT")
        analysis_parts.extend(context)
        analysis_parts.append("")
        
        # Analysis
        analysis = await self._perform_general_analysis(task)
        analysis_parts.append("🎯 ANALYSIS")
        analysis_parts.extend(analysis)
        
        output = "\n".join(analysis_parts)
        
        return TaskResult(
            task_id=task.id,
            agent_id=self.session.id if self.session else "unknown",
            success=True,
            output=output,
            error=""
        )
    
    # Context gathering methods
    async def _gather_debugging_context(self, task: Task) -> List[str]:
        """Gather context for debugging tasks"""
        context = []
        
        # Look for error patterns in the command
        if any(word in task.command.lower() for word in ["error", "exception", "crash", "fail"]):
            context.append("- Error/exception situation detected")
        
        if any(word in task.command.lower() for word in ["slow", "performance", "timeout"]):
            context.append("- Performance issue detected")
        
        # Check affected areas
        for area in task.affected_areas:
            context.append(f"- Affects {area} components")
        
        return context
    
    async def _gather_explanation_context(self, task: Task) -> List[str]:
        """Gather context for explanation tasks"""
        context = []
        
        # Identify what type of explanation is needed
        if any(word in task.command.lower() for word in ["how", "work", "function"]):
            context.append("- Functional explanation requested")
        
        if any(word in task.command.lower() for word in ["why", "reason", "purpose"]):
            context.append("- Reasoning/purpose explanation requested")
        
        if any(word in task.command.lower() for word in ["what", "describe"]):
            context.append("- Descriptive explanation requested")
        
        return context
    
    async def _gather_general_context(self, task: Task) -> List[str]:
        """Gather general context"""
        context = []
        context.append(f"- Task complexity: {task.complexity_score:.1f}/1.0")
        context.append(f"- Estimated duration: {task.estimated_duration} minutes")
        context.append(f"- Affected areas: {', '.join(task.affected_areas)}")
        
        return context
    
    # Analysis methods
    async def _identify_potential_issues(self, task: Task, context: List[str]) -> List[str]:
        """Identify potential issues for debugging"""
        issues = []
        
        # Common patterns based on keywords
        command_lower = task.command.lower()
        
        if "import" in command_lower or "module" in command_lower:
            issues.append("- Possible import/module resolution issues")
            issues.append("- Check Python path and virtual environment")
            issues.append("- Verify package installation")
        
        if "api" in command_lower or "request" in command_lower:
            issues.append("- Possible API/network connectivity issues")
            issues.append("- Check API endpoints and authentication")
            issues.append("- Verify request format and headers")
        
        if "database" in command_lower or "sql" in command_lower:
            issues.append("- Possible database connection issues")
            issues.append("- Check database credentials and connectivity")
            issues.append("- Verify query syntax and permissions")
        
        return issues
    
    async def _generate_debugging_steps(self, task: Task, potential_issues: List[str]) -> List[str]:
        """Generate debugging steps"""
        steps = []
        
        steps.append("1. 🔍 Reproduce the issue consistently")
        steps.append("2. 📝 Check logs and error messages")
        steps.append("3. 🧪 Isolate the problem area")
        steps.append("4. 🔧 Test potential fixes incrementally")
        steps.append("5. ✅ Verify the fix works")
        
        return steps
    
    async def _generate_detailed_explanation(self, task: Task) -> List[str]:
        """Generate detailed explanation"""
        explanation = []
        
        # This would be enhanced with actual code analysis
        explanation.append("Based on the query, here's a detailed explanation:")
        explanation.append("")
        explanation.append("The system works by following these key principles:")
        explanation.append("- Modular design with clear separation of concerns")
        explanation.append("- Event-driven architecture for loose coupling")
        explanation.append("- Async/await patterns for concurrent operations")
        explanation.append("- Type safety through comprehensive annotations")
        
        return explanation
    
    async def _generate_insights(self, task: Task) -> List[str]:
        """Generate additional insights"""
        insights = []
        
        if "performance" in task.command.lower():
            insights.append("- Consider caching for frequently accessed data")
            insights.append("- Use connection pooling for database operations")
            insights.append("- Implement lazy loading where appropriate")
        
        if "security" in task.command.lower():
            insights.append("- Always validate and sanitize user input")
            insights.append("- Use parameterized queries to prevent SQL injection")
            insights.append("- Implement proper authentication and authorization")
        
        return insights
    
    # Implementation guidance methods
    async def _analyze_implementation_requirements(self, task: Task) -> List[str]:
        """Analyze implementation requirements"""
        analysis = []
        
        analysis.append("Key requirements identified:")
        
        # Extract requirements from command
        command_lower = task.command.lower()
        
        if "api" in command_lower:
            analysis.append("- API endpoint implementation required")
        if "database" in command_lower:
            analysis.append("- Database integration needed")
        if "frontend" in command_lower or "ui" in command_lower:
            analysis.append("- User interface components required")
        if "test" in command_lower:
            analysis.append("- Testing implementation needed")
        
        return analysis
    
    async def _recommend_architecture(self, task: Task) -> List[str]:
        """Recommend architecture approach"""
        recommendations = []
        
        recommendations.append("Recommended architectural patterns:")
        recommendations.append("- Use dependency injection for loose coupling")
        recommendations.append("- Implement proper error handling and logging")
        recommendations.append("- Follow SOLID principles")
        recommendations.append("- Use async patterns for I/O operations")
        
        return recommendations
    
    async def _create_implementation_plan(self, task: Task) -> List[str]:
        """Create implementation plan"""
        plan = []
        
        plan.append("Suggested implementation order:")
        plan.append("1. 📋 Define data models and interfaces")
        plan.append("2. 🏗️ Implement core business logic")
        plan.append("3. 🔌 Add external integrations")
        plan.append("4. 🎨 Create user interface")
        plan.append("5. 🧪 Add comprehensive tests")
        plan.append("6. 📚 Document the implementation")
        
        return plan
    
    # Refactoring guidance methods
    async def _analyze_current_state(self, task: Task) -> List[str]:
        """Analyze current state for refactoring"""
        analysis = []
        
        analysis.append("Current state assessment:")
        analysis.append("- Code complexity and maintainability")
        analysis.append("- Performance bottlenecks")
        analysis.append("- Technical debt areas")
        analysis.append("- Test coverage gaps")
        
        return analysis
    
    async def _identify_refactoring_opportunities(self, task: Task) -> List[str]:
        """Identify refactoring opportunities"""
        opportunities = []
        
        opportunities.append("Potential improvements:")
        opportunities.append("- Extract common functionality into utilities")
        opportunities.append("- Simplify complex conditional logic")
        opportunities.append("- Improve naming conventions")
        opportunities.append("- Reduce code duplication")
        opportunities.append("- Enhance error handling")
        
        return opportunities
    
    async def _create_refactoring_plan(self, task: Task) -> List[str]:
        """Create refactoring plan"""
        plan = []
        
        plan.append("Refactoring steps:")
        plan.append("1. 🛡️ Ensure comprehensive test coverage")
        plan.append("2. 🧹 Start with small, safe refactorings")
        plan.append("3. 📦 Extract reusable components")
        plan.append("4. 🔄 Simplify complex methods")
        plan.append("5. ✅ Verify tests still pass")
        plan.append("6. 📝 Update documentation")
        
        return plan
    
    async def _perform_general_analysis(self, task: Task) -> List[str]:
        """Perform general analysis"""
        analysis = []
        
        analysis.append("Analysis results:")
        analysis.append("- Task appears to be well-scoped")
        analysis.append("- Consider breaking down if complexity is high")
        analysis.append("- Ensure proper testing strategy")
        analysis.append("- Plan for error handling and edge cases")
        
        return analysis
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy"""
        # Claude Code agent is always healthy if properly configured
        return self._validate_config() 