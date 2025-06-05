# Agentic Project - Claude Code Memory Template

## Project Overview
This is the Agentic multi-agent development framework that enables sophisticated AI-powered development workflows through Claude Code integration.

## Coding Standards
- **Python Style**: Follow PEP 8 and the repository's coding standards defined in `.cursorrules`
- **Type Hints**: Always use type hints for function parameters and return values
- **Documentation**: Use docstrings for all public classes and functions
- **Error Handling**: Implement proper exception handling with specific exception types
- **Testing**: Write tests for all new features using pytest

## Architecture Principles
- **Modular Design**: Keep agents, models, and utilities in separate modules
- **Dependency Injection**: Use constructor injection for agent dependencies  
- **Async First**: Use async/await patterns for all I/O operations
- **Pydantic Models**: Use Pydantic for data validation and serialization
- **Agent Patterns**: Follow the established agent interface patterns

## Development Workflow
- **Git Flow**: Use feature branches and meaningful commit messages
- **Testing**: Run `pytest` before committing changes
- **Type Checking**: Use `mypy` for static type checking
- **Code Formatting**: Use `black` and `isort` for consistent formatting
- **Pre-commit**: Use pre-commit hooks for automated quality checks

## Agentic-Specific Guidelines
- **Agent Design**: New agents should inherit from the base `Agent` class
- **Task Handling**: Use the `Task` and `TaskResult` models for all agent interactions
- **Configuration**: Use `AgentConfig` for agent setup and configuration
- **Capabilities**: Define clear `AgentCapability` for each agent type
- **Session Management**: Use `AgentSession` for state tracking

## Claude Code Integration
- **Enhanced Features**: Leverage memory, session persistence, and extended thinking
- **Tool Selection**: Use appropriate tool permissions based on task type
- **Memory Management**: Use project-specific memory for coding standards
- **Session Continuity**: Use session IDs for complex multi-turn tasks

## Performance Considerations
- **Async Operations**: Use asyncio for concurrent task execution
- **Resource Management**: Properly cleanup agent sessions and temporary files
- **Memory Usage**: Monitor memory usage for long-running agent sessions
- **Token Limits**: Be mindful of Claude's context token limits

## Security Guidelines
- **Input Validation**: Validate all external inputs using Pydantic models
- **Tool Permissions**: Use minimal necessary tool permissions for each task
- **Environment Variables**: Use environment variables for sensitive configuration
- **Error Information**: Don't expose sensitive information in error messages

## Team Collaboration
- **Code Reviews**: All changes require code review before merging
- **Documentation**: Update relevant documentation when adding features
- **Testing**: Maintain high test coverage for all core functionality
- **Communication**: Use clear commit messages and PR descriptions 