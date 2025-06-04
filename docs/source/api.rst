API Reference
=============

The Agentic API provides programmatic access to all multi-agent functionality. This reference covers the complete Python API for integrating Agentic into your applications.

Core Classes
------------

AgenticClient
~~~~~~~~~~~~~

Main client for interacting with Agentic programmatically.

.. autoclass:: agentic.core.client.AgenticClient
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic import AgenticClient
   
   # Initialize client
   client = AgenticClient(config_path="~/.agentic/config.yaml")
   
   # Execute natural language request
   result = await client.execute("Add user authentication to my FastAPI app")
   
   # Get project analysis
   analysis = await client.analyze_project()

Agent Management
~~~~~~~~~~~~~~~~

AgentManager
^^^^^^^^^^^^

Manages AI agents and their capabilities.

.. autoclass:: agentic.agents.manager.AgentManager
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic.agents import AgentManager
   
   # Initialize agent manager
   agent_manager = AgentManager()
   
   # Get available agents
   agents = await agent_manager.list_agents()
   
   # Create custom agent
   custom_agent = await agent_manager.create_agent(
       name="security-expert",
       specialization="security",
       model="gpt-4"
   )
   
   # Execute task with specific agent
   result = await agent_manager.execute_with_agent(
       agent_name="security-expert",
       task="Review this code for vulnerabilities",
       context={"files": ["src/auth.py"]}
   )

BaseAgent
^^^^^^^^^

Base class for all AI agents.

.. autoclass:: agentic.agents.base.BaseAgent
   :members:
   :undoc-members:
   :show-inheritance:

**Creating Custom Agents:**

.. code-block:: python

   from agentic.agents.base import BaseAgent
   from agentic.models.request import AgentRequest
   from agentic.models.response import AgentResponse
   
   class CustomSecurityAgent(BaseAgent):
       """Custom security-focused agent."""
       
       def __init__(self):
           super().__init__(
               name="security-expert",
               description="Specialized in security code review",
               capabilities=["security-review", "vulnerability-scan"]
           )
       
       async def process_request(self, request: AgentRequest) -> AgentResponse:
           """Process security-specific requests."""
           # Custom security analysis logic
           vulnerabilities = await self._scan_vulnerabilities(request.content)
           
           return AgentResponse(
               success=True,
               content=f"Security analysis complete. Found {len(vulnerabilities)} issues.",
               metadata={"vulnerabilities": vulnerabilities}
           )

Project Management
~~~~~~~~~~~~~~~~~~

ProjectAnalyzer
^^^^^^^^^^^^^^^

Analyzes project structure and provides insights.

.. autoclass:: agentic.core.analyzer.ProjectAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic.core.analyzer import ProjectAnalyzer
   
   # Initialize analyzer
   analyzer = ProjectAnalyzer(project_path="./my-project")
   
   # Analyze project structure
   analysis = await analyzer.analyze()
   
   # Get specific insights
   security_insights = await analyzer.get_security_insights()
   performance_insights = await analyzer.get_performance_insights()
   
   # Generate recommendations
   recommendations = await analyzer.generate_recommendations()

CodeProcessor
^^^^^^^^^^^^^

Processes and modifies code files.

.. autoclass:: agentic.core.processor.CodeProcessor
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic.core.processor import CodeProcessor
   
   # Initialize processor
   processor = CodeProcessor()
   
   # Process single file
   result = await processor.process_file(
       file_path="src/main.py",
       instruction="Add type hints to all functions"
   )
   
   # Process multiple files
   batch_result = await processor.process_batch(
       file_patterns=["src/*.py"],
       instruction="Add docstrings to all classes"
   )

Configuration
~~~~~~~~~~~~~

Config
^^^^^^

Configuration management for Agentic.

.. autoclass:: agentic.core.config.Config
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic.core.config import Config
   
   # Load configuration
   config = Config.load_from_file("~/.agentic/config.yaml")
   
   # Access configuration values
   api_key = config.get("api.openai.api_key")
   timeout = config.get("agents.default_timeout", default=300)
   
   # Update configuration
   config.set("agents.max_retries", 5)
   config.save()

Data Models
-----------

Request/Response Models
~~~~~~~~~~~~~~~~~~~~~~~

AgentRequest
^^^^^^^^^^^^

Request model for agent interactions.

.. autoclass:: agentic.models.request.AgentRequest
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from agentic.models.request import AgentRequest
   
   request = AgentRequest(
       content="Add user authentication",
       context={
           "files": ["src/main.py", "src/auth.py"],
           "project_type": "fastapi",
           "requirements": ["JWT", "SQLAlchemy"]
       },
       agent_preferences=["python-expert", "security-expert"]
   )

AgentResponse
^^^^^^^^^^^^^

Response model from agent interactions.

.. autoclass:: agentic.models.response.AgentResponse
   :members:
   :undoc-members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from agentic.models.response import AgentResponse
   
   response = AgentResponse(
       success=True,
       content="Authentication system implemented successfully",
       changes=[
           {
               "file": "src/auth.py",
               "action": "created",
               "lines_added": 150
           }
       ],
       metadata={
           "agent_used": "python-expert",
           "duration_seconds": 45.2,
           "confidence": 0.95
       }
   )

Project Models
~~~~~~~~~~~~~~

ProjectInfo
^^^^^^^^^^^

Information about a project structure.

.. autoclass:: agentic.models.project.ProjectInfo
   :members:
   :undoc-members:
   :show-inheritance:

Analysis
^^^^^^^^

Project analysis results.

.. autoclass:: agentic.models.analysis.Analysis
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

File Operations
~~~~~~~~~~~~~~~

FileManager
^^^^^^^^^^^

Safe file operations with backup support.

.. autoclass:: agentic.utils.file_manager.FileManager
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic.utils.file_manager import FileManager
   
   # Initialize with backup enabled
   file_manager = FileManager(backup_enabled=True)
   
   # Safe file write with automatic backup
   await file_manager.write_file(
       path="src/important.py",
       content="# Updated code here",
       create_backup=True
   )
   
   # Restore from backup if needed
   await file_manager.restore_backup("src/important.py")

Git Integration
~~~~~~~~~~~~~~~

GitManager
^^^^^^^^^^

Git operations and repository management.

.. autoclass:: agentic.utils.git_manager.GitManager
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic.utils.git_manager import GitManager
   
   # Initialize git manager
   git_manager = GitManager(repo_path="./my-project")
   
   # Create feature branch
   await git_manager.create_branch(
       branch_name="feature/user-auth",
       base_branch="main"
   )
   
   # Auto-commit changes
   await git_manager.auto_commit(
       message="Add user authentication system",
       include_all=True
   )

Error Handling
--------------

Exception Classes
~~~~~~~~~~~~~~~~~

AgenticError
^^^^^^^^^^^^

Base exception for all Agentic errors.

.. autoclass:: agentic.exceptions.AgenticError
   :members:
   :undoc-members:
   :show-inheritance:

AgentExecutionError
^^^^^^^^^^^^^^^^^^^

Errors during agent execution.

.. autoclass:: agentic.exceptions.AgentExecutionError
   :members:
   :undoc-members:
   :show-inheritance:

ConfigurationError
^^^^^^^^^^^^^^^^^^

Configuration-related errors.

.. autoclass:: agentic.exceptions.ConfigurationError
   :members:
   :undoc-members:
   :show-inheritance:

**Example Error Handling:**

.. code-block:: python

   from agentic import AgenticClient
   from agentic.exceptions import AgentExecutionError, ConfigurationError
   
   client = AgenticClient()
   
   try:
       result = await client.execute("Complex task")
   except AgentExecutionError as e:
       print(f"Agent execution failed: {e}")
       print(f"Agent: {e.agent_name}")
       print(f"Error code: {e.error_code}")
   except ConfigurationError as e:
       print(f"Configuration error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Async Context Managers
----------------------

Many Agentic operations support async context managers for proper resource management:

.. code-block:: python

   # Agent execution with context
   async with client.agent_context("python-expert") as agent:
       result1 = await agent.execute("Add logging")
       result2 = await agent.execute("Add error handling")
   
   # File processing with context
   async with FileManager() as fm:
       await fm.write_file("test.py", "print('hello')")
       content = await fm.read_file("test.py")
   
   # Git operations with context
   async with GitManager("./project") as git:
       await git.create_branch("feature/new")
       await git.commit_changes("Add new feature")

Plugin System
-------------

Plugin Development
~~~~~~~~~~~~~~~~~~

BasePlugin
^^^^^^^^^^

Base class for developing Agentic plugins.

.. autoclass:: agentic.plugins.base.BasePlugin
   :members:
   :undoc-members:
   :show-inheritance:

**Creating a Plugin:**

.. code-block:: python

   from agentic.plugins.base import BasePlugin
   from agentic.models.request import AgentRequest
   from agentic.models.response import AgentResponse
   
   class DockerPlugin(BasePlugin):
       """Plugin for Docker operations."""
       
       def __init__(self):
           super().__init__(
               name="docker",
               version="1.0.0",
               description="Docker container management",
               capabilities=["dockerfile-generation", "container-deployment"]
           )
       
       async def handle_request(self, request: AgentRequest) -> AgentResponse:
           """Handle Docker-related requests."""
           if "dockerfile" in request.content.lower():
               dockerfile_content = await self._generate_dockerfile(request)
               return AgentResponse(
                   success=True,
                   content=dockerfile_content,
                   metadata={"plugin": "docker", "type": "dockerfile"}
               )
           
           return AgentResponse(
               success=False,
               content="Unsupported Docker operation"
           )

PluginManager
^^^^^^^^^^^^^

Manages installed plugins.

.. autoclass:: agentic.plugins.manager.PluginManager
   :members:
   :undoc-members:
   :show-inheritance:

**Plugin Usage:**

.. code-block:: python

   from agentic.plugins.manager import PluginManager
   
   # Initialize plugin manager
   plugin_manager = PluginManager()
   
   # Load plugins
   await plugin_manager.load_plugins()
   
   # Install plugin
   await plugin_manager.install_plugin("agentic-docker")
   
   # Use plugin
   result = await plugin_manager.execute_with_plugin(
       plugin_name="docker",
       request="Generate Dockerfile for Python FastAPI app"
   )

Quality Assurance
-----------------

QualityAssuranceManager
~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive testing and quality assurance.

.. autoclass:: agentic.core.quality_assurance.QualityAssuranceManager
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic.core.quality_assurance import QualityAssuranceManager
   from pathlib import Path
   
   # Initialize QA manager
   qa_manager = QualityAssuranceManager(source_dirs=[Path("src")])
   
   # Run comprehensive testing
   results = await qa_manager.run_comprehensive_testing()
   
   # Check specific quality metrics
   coverage = results["coverage_analysis"]["coverage_percent"]
   test_success_rate = results["test_execution"]["metrics"]["success_rate"]
   security_score = results["security_testing"]["overall_score"]

Production Stability
--------------------

ProductionStabilityManager
~~~~~~~~~~~~~~~~~~~~~~~~~~

Production monitoring and stability management.

.. autoclass:: agentic.core.production_stability.ProductionStabilityManager
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from agentic.core.production_stability import ProductionStabilityManager
   
   # Initialize stability manager
   stability_manager = ProductionStabilityManager()
   
   # Start monitoring
   await stability_manager.initialize()
   
   # Use circuit breaker for external calls
   async with stability_manager.circuit_breaker_call("external_api"):
       # Make external API call
       response = await external_api.call()
   
   # Get health metrics
   metrics = await stability_manager.get_health_metrics()
   
   # Shutdown gracefully
   await stability_manager.shutdown()

Advanced Usage Examples
-----------------------

Complete Integration Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from pathlib import Path
   from agentic import AgenticClient
   from agentic.core.quality_assurance import QualityAssuranceManager
   from agentic.core.production_stability import ProductionStabilityManager
   
   async def main():
       """Complete Agentic integration example."""
       
       # Initialize components
       client = AgenticClient(config_path="~/.agentic/config.yaml")
       qa_manager = QualityAssuranceManager(source_dirs=[Path("src")])
       stability_manager = ProductionStabilityManager()
       
       try:
           # Start production monitoring
           await stability_manager.initialize()
           
           # Execute development task
           result = await client.execute(
               "Add comprehensive logging to all API endpoints"
           )
           
           if result.success:
               # Run quality assurance
               qa_results = await qa_manager.run_comprehensive_testing()
               
               if qa_results["overall_metrics"]["quality_level"] == "EXCELLENT":
                   # Auto-commit if quality is high
                   await client.git_manager.auto_commit(
                       message="Add comprehensive logging with QA validation"
                   )
                   print("Changes committed successfully!")
               else:
                   print("Quality checks failed, manual review required")
           
       finally:
           # Cleanup
           await stability_manager.shutdown()
   
   if __name__ == "__main__":
       asyncio.run(main())

Batch Processing Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def batch_code_improvement():
       """Batch process multiple files for code improvement."""
       
       client = AgenticClient()
       
       tasks = [
           {
               "files": ["src/models/*.py"],
               "instruction": "Add comprehensive type hints and docstrings"
           },
           {
               "files": ["src/api/*.py"],
               "instruction": "Add input validation and error handling"
           },
           {
               "files": ["tests/*.py"],
               "instruction": "Increase test coverage to >95%"
           }
       ]
       
       results = []
       for task in tasks:
           result = await client.process_batch(
               file_patterns=task["files"],
               instruction=task["instruction"]
           )
           results.append(result)
       
       # Generate summary report
       total_files = sum(len(r.changes) for r in results)
       print(f"Processed {total_files} files successfully")
       
       return results

Custom Agent Pipeline
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from agentic.agents.manager import AgentManager
   from agentic.models.request import AgentRequest
   
   async def security_review_pipeline(file_path: str):
       """Custom security review pipeline using multiple agents."""
       
       agent_manager = AgentManager()
       
       # Stage 1: Code analysis
       analysis_request = AgentRequest(
           content=f"Analyze {file_path} for potential security issues",
           context={"stage": "analysis", "file": file_path}
       )
       
       analysis_result = await agent_manager.execute_with_agent(
           agent_name="security-expert",
           request=analysis_request
       )
       
       # Stage 2: Vulnerability scanning
       if analysis_result.success:
           scan_request = AgentRequest(
               content=f"Perform detailed vulnerability scan of {file_path}",
               context={
                   "stage": "scanning",
                   "previous_analysis": analysis_result.metadata
               }
           )
           
           scan_result = await agent_manager.execute_with_agent(
               agent_name="vulnerability-scanner",
               request=scan_request
           )
           
           # Stage 3: Generate fixes
           if scan_result.metadata.get("vulnerabilities_found", 0) > 0:
               fix_request = AgentRequest(
                   content=f"Generate security fixes for {file_path}",
                   context={
                       "vulnerabilities": scan_result.metadata["vulnerabilities"],
                       "fix_mode": "conservative"
                   }
               )
               
               fix_result = await agent_manager.execute_with_agent(
                   agent_name="security-fixer",
                   request=fix_request
               )
               
               return {
                   "analysis": analysis_result,
                   "scan": scan_result,
                   "fixes": fix_result,
                   "security_score": fix_result.metadata.get("security_score", 0)
               }
       
       return {"analysis": analysis_result, "scan": scan_result}

Type Hints and Annotations
--------------------------

Agentic uses comprehensive type hints throughout the codebase. Here are the key type definitions:

.. code-block:: python

   from typing import Dict, List, Optional, Union, Any, Callable, Awaitable
   from pathlib import Path
   
   # Common type aliases used in Agentic
   FilePath = Union[str, Path]
   ConfigDict = Dict[str, Any]
   AgentCapabilities = List[str]
   FileContent = str
   ErrorCallback = Callable[[Exception], Awaitable[None]]
   
   # Generic response type
   from typing import TypeVar, Generic
   T = TypeVar('T')
   
   class Response(Generic[T]):
       success: bool
       data: T
       error: Optional[str] = None

API Reference Quick Links
-------------------------

* :class:`agentic.core.client.AgenticClient` - Main client interface
* :class:`agentic.agents.manager.AgentManager` - Agent management
* :class:`agentic.core.analyzer.ProjectAnalyzer` - Project analysis
* :class:`agentic.core.processor.CodeProcessor` - Code processing
* :class:`agentic.utils.file_manager.FileManager` - File operations
* :class:`agentic.utils.git_manager.GitManager` - Git integration
* :class:`agentic.plugins.manager.PluginManager` - Plugin system
* :class:`agentic.core.quality_assurance.QualityAssuranceManager` - Quality assurance
* :class:`agentic.core.production_stability.ProductionStabilityManager` - Production stability

For more examples and tutorials, see the :doc:`quickstart` guide and check out our `GitHub repository <https://github.com/agentic-ai/agentic>`_. 