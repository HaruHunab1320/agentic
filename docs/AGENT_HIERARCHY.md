# Agent Hierarchy & Architecture

## Overview

The Agentic framework uses a hierarchical multi-agent architecture that leverages each agent's unique strengths in a coordinated system.

## Agent Hierarchy

### 1. **Gemini - Chief Architect & Knowledge Hub** 🏗️
**Role**: System-wide architecture, research, and knowledge coordination

**Responsibilities**:
- **Codebase-wide architecture** decisions using 1M token context
- **Research** latest best practices and technologies
- **Visual analysis** of architecture diagrams and specifications
- **Knowledge hub** - answers questions from other agents
- **Cross-domain coordination** - understands how all parts connect
- **Technology selection** - researches and recommends tools/frameworks

**When Gemini Gets Involved**:
- Any agent needs clarification or additional context
- Architecture decisions affecting multiple domains
- Need to understand the "big picture"
- Research on best practices or new technologies
- Processing visual specifications or documentation
- Codebase-wide refactoring decisions

### 2. **Claude Code - Domain Architects** 🎯
**Role**: Deep reasoning and architecture within specific domains

**Domain Specializations**:
- **Frontend Architect**: Component design, state management, UI/UX decisions
- **Backend Architect**: API design, data modeling, service architecture  
- **DevOps Architect**: Infrastructure, deployment, monitoring strategies
- **Testing Architect**: Test strategies, coverage planning, QA processes

**Responsibilities**:
- **Domain-specific architecture** within their area
- **Complex debugging** within their domain
- **Algorithm design** and optimization
- **Code quality reviews** for their domain
- **Mentoring** Aider agents in their domain

**Escalation to Gemini**:
- Cross-domain architectural decisions
- Need broader context beyond their domain
- Require research on new technologies
- Conflicting requirements between domains

### 3. **Aider Agents - Implementation Specialists** 🔨
**Role**: Efficient implementation following architectural guidance

**Specializations**:
- **Aider Frontend**: React/Vue/Angular implementation
- **Aider Backend**: API endpoints, services, data layer
- **Aider Testing**: Test implementation, coverage
- **Aider DevOps**: Configuration, scripts, automation

**Responsibilities**:
- **Implementation** following patterns
- **Bulk operations** across files
- **Refactoring** based on guidelines
- **Code generation** from specifications

**Escalation Path**:
- To Claude Code: Complex logic or unclear requirements
- To Gemini: Need broader context or research

## Communication Flow

### Hierarchical Escalation
```
Aider Agents → Claude Domain Architects → Gemini Chief Architect
     ↓                    ↓                        ↓
Implementation      Domain Decisions        System Architecture
```

### Knowledge Flow
```
Gemini (Research & Analysis)
    ↓
Claude Code (Domain Interpretation)
    ↓
Aider (Implementation)
```

## Example Workflows

### Workflow 1: New Feature Implementation
```python
# 1. Gemini analyzes requirements and overall impact
coordinator.assign_task(
    agent_type="gemini",
    task="Analyze PDF requirements and assess impact on system architecture"
)

# 2. Claude architects design domain-specific solutions
coordinator.assign_tasks([
    {
        "agent": "claude_frontend_architect",
        "task": "Design component architecture based on Gemini's analysis"
    },
    {
        "agent": "claude_backend_architect", 
        "task": "Design API structure based on Gemini's analysis"
    }
])

# 3. Aider agents implement
coordinator.assign_tasks([
    {
        "agent": "aider_frontend",
        "task": "Implement components following Claude's design"
    },
    {
        "agent": "aider_backend",
        "task": "Implement APIs following Claude's design"
    }
])
```

### Workflow 2: Performance Optimization
```python
# 1. Gemini researches and analyzes
coordinator.assign_task(
    agent_type="gemini",
    task="Research latest performance optimization techniques for our tech stack and analyze our codebase for bottlenecks"
)

# 2. Claude architects create optimization plan
coordinator.assign_task(
    agent_type="claude_backend_architect",
    task="Design caching strategy based on Gemini's findings"
)

# 3. Aider implements optimizations
coordinator.assign_task(
    agent_type="aider_backend",
    task="Implement caching layer following the design"
)
```

### Workflow 3: Agent Confusion Resolution
```python
# Aider is confused about implementation
aider_agent.escalate_to_claude(
    "Unclear how to implement authentication across microservices"
)

# Claude needs broader context
claude_architect.escalate_to_gemini(
    "Need to understand how other systems handle distributed authentication"
)

# Gemini researches and provides guidance
gemini.research_and_advise(
    "Research distributed authentication patterns and recommend approach"
)
```

## Agent Selection Logic

### Primary Assignment Rules

1. **Start with Gemini when**:
   - Task involves visual inputs (PDFs, images, diagrams)
   - Requires research or current information
   - Needs codebase-wide understanding
   - Architecture affecting multiple domains
   - Other agents request clarification

2. **Start with Claude Code when**:
   - Domain-specific architecture needed
   - Complex logic or algorithms
   - Debugging within a domain
   - Code quality review
   - Design patterns within domain

3. **Start with Aider when**:
   - Clear implementation requirements
   - Pattern-based changes
   - Bulk operations
   - Following established patterns

### Escalation Triggers

**Aider → Claude Code**:
- Implementation requires architectural decision
- Logic too complex for pattern matching
- Unclear requirements need interpretation

**Claude Code → Gemini**:
- Needs information beyond domain
- Requires research on best practices
- Cross-domain architectural impact
- Needs visual specification analysis

**Any Agent → Gemini**:
- "I need more context about..."
- "What's the best practice for..."
- "How does this affect other systems..."
- "Can you research..."

## Configuration Example

```python
# Hierarchical agent configuration
AGENT_HIERARCHY = {
    "chief_architect": {
        "type": "gemini",
        "capabilities": ["research", "visual_analysis", "system_architecture"],
        "context_window": 1000000
    },
    "domain_architects": {
        "frontend": {
            "type": "claude_code",
            "specialization": "frontend",
            "escalation": "chief_architect"
        },
        "backend": {
            "type": "claude_code", 
            "specialization": "backend",
            "escalation": "chief_architect"
        },
        "devops": {
            "type": "claude_code",
            "specialization": "devops",
            "escalation": "chief_architect"
        }
    },
    "implementers": {
        "frontend": {
            "type": "aider_frontend",
            "escalation": "domain_architects.frontend"
        },
        "backend": {
            "type": "aider_backend",
            "escalation": "domain_architects.backend"
        },
        "testing": {
            "type": "aider_testing",
            "escalation": "domain_architects.backend"
        }
    }
}
```

## Benefits of This Hierarchy

1. **Optimal Context Usage**:
   - Gemini's 1M tokens for system-wide understanding
   - Claude's 200k for domain-specific depth
   - Aider's focused context for implementation

2. **Clear Responsibilities**:
   - Each agent knows their role and boundaries
   - Clear escalation paths prevent confusion
   - Specialization improves quality

3. **Knowledge Sharing**:
   - Gemini serves as central knowledge repository
   - Information flows down through hierarchy
   - Agents can query up for clarification

4. **Cost Optimization**:
   - Expensive operations (Gemini research) used strategically
   - Routine implementation handled by efficient Aider
   - Claude used for complex domain-specific tasks

## Implementation Guidelines

### For Coordinators
1. Start with appropriate level based on task
2. Monitor for escalation triggers
3. Facilitate knowledge transfer between levels
4. Track which agents need which information

### For Agents
1. Know your boundaries and strengths
2. Escalate when outside your domain
3. Document decisions for knowledge sharing
4. Request clarification when needed

This hierarchical approach maximizes each agent's strengths while providing clear communication paths and optimal resource utilization.