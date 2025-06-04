Contributing to Agentic
=======================

We welcome contributions to Agentic! This guide will help you get started with contributing to the project.

Code of Conduct
---------------

By participating in this project, you agree to abide by our `Code of Conduct <https://github.com/agentic-ai/agentic/blob/main/CODE_OF_CONDUCT.md>`_.

Getting Started
---------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork and Clone**

   .. code-block:: bash

      # Fork the repository on GitHub, then clone your fork
      git clone https://github.com/your-username/agentic.git
      cd agentic

2. **Set Up Python Environment**

   .. code-block:: bash

      # Using pyenv (recommended)
      pyenv install 3.11.0
      pyenv local 3.11.0

      # Create virtual environment
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**

   .. code-block:: bash

      # Install development dependencies
      pip install -e ".[dev]"

      # Install pre-commit hooks
      pre-commit install

4. **Verify Installation**

   .. code-block:: bash

      # Run tests to ensure everything works
      pytest

      # Run linting
      black --check .
      isort --check-only .
      mypy src/

Development Workflow
--------------------

Branch Management
~~~~~~~~~~~~~~~~~

We use a Git flow-based workflow:

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation branches

.. code-block:: bash

   # Create a feature branch
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name

   # Make changes and commit
   git add .
   git commit -m "Add your feature description"

   # Push and create pull request
   git push origin feature/your-feature-name

Commit Message Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

We follow the `Conventional Commits <https://www.conventionalcommits.org/>`_ specification:

.. code-block:: text

   <type>[optional scope]: <description>

   [optional body]

   [optional footer(s)]

**Types:**

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semi-colons, etc)
- `refactor`: Code changes that neither fix a bug nor add a feature
- `perf`: Performance improvements
- `test`: Adding missing tests or correcting existing tests
- `chore`: Other changes that don't modify src or test files

**Examples:**

.. code-block:: text

   feat(agents): add support for custom agent plugins

   fix(cli): resolve issue with config file parsing
   
   docs(api): add examples to agent manager documentation
   
   test(core): add unit tests for production stability module

Code Standards
--------------

Python Standards
~~~~~~~~~~~~~~~~

We follow strict Python standards as defined in our `.cursorrules` file:

**Type Annotations**

.. code-block:: python

   # Required for all function signatures
   def process_request(self, request: AgentRequest) -> AgentResponse:
       """Process an agent request and return response."""
       pass

   # Required for class attributes
   class Agent:
       name: str
       capabilities: List[str]
       status: AgentStatus

**Pydantic Models**

.. code-block:: python

   # Prefer Pydantic for data validation
   from pydantic import BaseModel, Field

   class AgentConfig(BaseModel):
       name: str = Field(..., description="Agent name")
       max_tokens: int = Field(4000, ge=1, le=8000)
       temperature: float = Field(0.7, ge=0.0, le=2.0)

**Error Handling**

.. code-block:: python

   # Use specific exception types
   try:
       result = await dangerous_operation()
   except ValidationError as e:
       logger.error("Validation failed", error=str(e))
       raise AgentValidationError(f"Invalid input: {e}")
   except TimeoutError as e:
       logger.warning("Operation timed out", timeout=timeout)
       raise AgentTimeoutError("Request timed out")

**Async/Await Patterns**

.. code-block:: python

   # Use async/await consistently
   async def process_multiple_requests(
       self, 
       requests: List[AgentRequest]
   ) -> List[AgentResponse]:
       tasks = [self.process_request(req) for req in requests]
       return await asyncio.gather(*tasks)

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

**Docstrings**

Use Google-style docstrings for all public functions and classes:

.. code-block:: python

   def calculate_quality_score(
       self, 
       test_coverage: float,
       complexity_score: float,
       security_score: float
   ) -> float:
       """Calculate overall quality score for code changes.
       
       Args:
           test_coverage: Test coverage percentage (0.0-1.0)
           complexity_score: Code complexity score (0.0-1.0)
           security_score: Security analysis score (0.0-1.0)
           
       Returns:
           Overall quality score (0.0-1.0)
           
       Raises:
           ValueError: If any score is outside valid range
           
       Example:
           >>> calculator = QualityCalculator()
           >>> score = calculator.calculate_quality_score(0.85, 0.7, 0.9)
           >>> print(f"Quality score: {score:.2f}")
           Quality score: 0.82
       """
       if not all(0.0 <= score <= 1.0 for score in [test_coverage, complexity_score, security_score]):
           raise ValueError("All scores must be between 0.0 and 1.0")
       
       return (test_coverage * 0.4 + complexity_score * 0.3 + security_score * 0.3)

Testing Standards
-----------------

Testing Philosophy
~~~~~~~~~~~~~~~~~~

- **100% test coverage** for critical components
- **Integration tests** for multi-component interactions
- **Property-based testing** using Hypothesis for complex logic
- **Performance benchmarks** for critical paths

Test Structure
~~~~~~~~~~~~~~

.. code-block:: python

   # tests/core/test_agent_manager.py
   import pytest
   from unittest.mock import AsyncMock, Mock
   from agentic.core.agent_manager import AgentManager
   from agentic.models.requests import AgentRequest

   class TestAgentManager:
       @pytest.fixture
       async def agent_manager(self):
           """Create an agent manager for testing."""
           manager = AgentManager()
           await manager.initialize()
           return manager

       @pytest.fixture
       def sample_request(self):
           """Create a sample agent request."""
           return AgentRequest(
               content="Generate a Python function",
               context={"language": "python"},
               timeout=300
           )

       async def test_route_request_success(
           self, 
           agent_manager: AgentManager,
           sample_request: AgentRequest
       ):
           """Test successful request routing."""
           # Given
           mock_agent = AsyncMock()
           mock_agent.process_request.return_value = AgentResponse(
               success=True,
               content="Generated function"
           )
           agent_manager.agents["python-expert"] = mock_agent

           # When
           response = await agent_manager.route_request(sample_request)

           # Then
           assert response.success
           assert "Generated function" in response.content
           mock_agent.process_request.assert_called_once_with(sample_request)

       @pytest.mark.parametrize("timeout", [1, 10, 300])
       async def test_request_timeout_handling(
           self,
           agent_manager: AgentManager,
           timeout: int
       ):
           """Test request timeout handling with various timeouts."""
           request = AgentRequest(
               content="Long running task",
               timeout=timeout
           )
           
           # Test implementation here

       @pytest.mark.integration
       async def test_full_workflow_integration(self, agent_manager: AgentManager):
           """Test complete workflow from request to response."""
           # Integration test implementation

**Property-Based Testing**

.. code-block:: python

   from hypothesis import given, strategies as st

   class TestQualityCalculator:
       @given(
           test_coverage=st.floats(min_value=0.0, max_value=1.0),
           complexity_score=st.floats(min_value=0.0, max_value=1.0),
           security_score=st.floats(min_value=0.0, max_value=1.0)
       )
       def test_quality_score_always_in_range(
           self,
           test_coverage: float,
           complexity_score: float,
           security_score: float
       ):
           """Quality score should always be between 0.0 and 1.0."""
           calculator = QualityCalculator()
           score = calculator.calculate_quality_score(
               test_coverage, complexity_score, security_score
           )
           assert 0.0 <= score <= 1.0

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=src/agentic --cov-report=html

   # Run specific test categories
   pytest -m "not integration"  # Skip integration tests
   pytest -m "integration"      # Run only integration tests

   # Run performance benchmarks
   pytest tests/benchmarks/

Security Considerations
-----------------------

Security Guidelines
~~~~~~~~~~~~~~~~~~~

- **Input Validation**: Validate all user inputs using Pydantic models
- **No Eval/Exec**: Never use `eval()` or `exec()` functions
- **Secure File Operations**: Use safe file operations with proper checks
- **Environment Variables**: Store sensitive data in environment variables
- **Audit Logging**: Log security-relevant events

**Security Review Checklist**

Before submitting a PR, ensure:

- [ ] All user inputs are validated
- [ ] No hardcoded secrets or credentials
- [ ] File operations use safe patterns
- [ ] Error messages don't leak sensitive information
- [ ] External API calls use proper authentication
- [ ] Dependencies are up-to-date and secure

Performance Guidelines
----------------------

Performance Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Async/Await**: Use async patterns for I/O-bound operations
- **Caching**: Implement caching for expensive operations
- **Lazy Loading**: Load resources only when needed
- **Connection Pooling**: Reuse connections to external services
- **Memory Management**: Be conscious of memory usage in loops

**Performance Testing**

.. code-block:: python

   # tests/benchmarks/test_agent_performance.py
   import pytest
   import time
   from agentic.core.agent_manager import AgentManager

   class TestAgentPerformance:
       @pytest.mark.benchmark
       def test_request_processing_speed(self, benchmark):
           """Benchmark request processing speed."""
           def process_request():
               # Setup and execution code
               pass
           
           result = benchmark(process_request)
           assert result is not None

       @pytest.mark.performance  
       async def test_concurrent_request_handling(self):
           """Test handling of concurrent requests."""
           manager = AgentManager()
           
           # Create 100 concurrent requests
           requests = [create_test_request() for _ in range(100)]
           
           start_time = time.time()
           responses = await asyncio.gather(*[
               manager.route_request(req) for req in requests
           ])
           duration = time.time() - start_time
           
           assert all(r.success for r in responses)
           assert duration < 30  # Should complete within 30 seconds

Plugin Development
------------------

Creating Custom Agents
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # plugins/custom_agent.py
   from agentic.core.base_agent import BaseAgent
   from agentic.models.requests import AgentRequest, AgentResponse

   class CustomAgent(BaseAgent):
       """Custom agent for domain-specific tasks."""
       
       capabilities = [
           "custom-task-1",
           "custom-task-2"
       ]
       
       def __init__(self, name: str = "custom-agent"):
           super().__init__(name)
           self.specialized_tool = SpecializedTool()
       
       async def process_request(self, request: AgentRequest) -> AgentResponse:
           """Process custom agent requests."""
           try:
               # Custom processing logic
               result = await self.specialized_tool.process(request.content)
               
               return AgentResponse(
                   success=True,
                   content=result,
                   metadata={"agent": self.name, "tool": "specialized"}
               )
           except Exception as e:
               return AgentResponse(
                   success=False,
                   error=str(e)
               )
       
       async def cleanup(self) -> None:
           """Cleanup agent resources."""
           await self.specialized_tool.close()

**Plugin Registration**

.. code-block:: python

   # plugins/__init__.py
   from agentic.core.plugin_manager import register_plugin
   from .custom_agent import CustomAgent

   def register_plugins():
       """Register all plugins in this package."""
       register_plugin("custom-agent", CustomAgent)

Creating Custom Tools
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # plugins/tools/specialized_tool.py
   from agentic.core.base_tool import BaseTool

   class SpecializedTool(BaseTool):
       """Tool for specialized domain operations."""
       
       def __init__(self):
           self.api_client = None
       
       async def initialize(self) -> bool:
           """Initialize tool resources."""
           self.api_client = await create_api_client()
           return True
       
       async def process(self, content: str) -> str:
           """Process content using specialized logic."""
           # Implementation here
           pass
       
       async def close(self) -> None:
           """Close tool resources."""
           if self.api_client:
               await self.api_client.close()

Submitting Contributions
------------------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. **Create Feature Branch**

   .. code-block:: bash

      git checkout develop
      git pull origin develop
      git checkout -b feature/your-feature-name

2. **Make Changes**
   
   - Follow coding standards
   - Add comprehensive tests
   - Update documentation
   - Add type annotations

3. **Run Quality Checks**

   .. code-block:: bash

      # Format code
      black .
      isort .

      # Type checking
      mypy src/

      # Run tests
      pytest --cov=src/agentic

      # Security check
      bandit -r src/

4. **Commit Changes**

   .. code-block:: bash

      git add .
      git commit -m "feat(agents): add custom agent support"

5. **Push and Create PR**

   .. code-block:: bash

      git push origin feature/your-feature-name

6. **Create Pull Request**
   
   - Use descriptive title and description
   - Reference related issues
   - Include screenshots for UI changes
   - Request reviews from maintainers

Pull Request Template
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes made.

   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing completed

   ## Quality Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-review of code completed
   - [ ] Comments added for complex logic
   - [ ] Documentation updated
   - [ ] Tests pass locally
   - [ ] Type annotations added

   ## Breaking Changes
   List any breaking changes and migration steps.

   ## Screenshots
   Add screenshots for UI changes.

Code Review Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**For Authors:**

- Keep PRs focused and small
- Provide clear descriptions
- Respond promptly to feedback
- Be open to suggestions

**For Reviewers:**

- Be constructive and respectful
- Focus on code quality and correctness
- Suggest improvements clearly
- Approve when standards are met

Release Process
---------------

Version Management
~~~~~~~~~~~~~~~~~~

We use semantic versioning (SemVer):

- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

**Pre-release versions:**

- `1.0.0-alpha.1`: Alpha release
- `1.0.0-beta.1`: Beta release
- `1.0.0-rc.1`: Release candidate

Changelog Management
~~~~~~~~~~~~~~~~~~~~

We automatically generate changelogs from commit messages. Ensure your commits follow the conventional commit format for proper categorization.

Community and Support
----------------------

Getting Help
~~~~~~~~~~~~

- **Documentation**: Check this documentation first
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our Discord community
- **Email**: Contact maintainers directly for security issues

Reporting Bugs
~~~~~~~~~~~~~~

When reporting bugs, include:

- Python version and OS
- Agentic version
- Complete error traceback
- Minimal reproduction case
- Expected vs actual behavior

Requesting Features
~~~~~~~~~~~~~~~~~~~

For feature requests:

- Search existing feature requests
- Describe the use case clearly
- Explain why the feature is valuable
- Consider proposing an implementation approach

Recognition
-----------

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project documentation
- Social media announcements

Thank you for contributing to Agentic! Your contributions help make multi-agent AI development accessible to everyone. 