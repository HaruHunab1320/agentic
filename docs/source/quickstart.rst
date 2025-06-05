Quick Start Guide
==================

This guide will help you get started with Agentic in minutes.

Prerequisites
-------------

Before you begin, ensure you have:

* **Python 3.9+** installed
* **Aider** installed and configured (see :doc:`installation` for details)
* **Claude Code** installed and authenticated (see :doc:`installation` for details)
* **API keys** for your preferred AI provider:

  - `Anthropic API Key <https://console.anthropic.com/>`_ (for Claude models)
  - `OpenAI API Key <https://platform.openai.com/api-keys>`_ (for GPT models)
  - `DeepSeek API Key <https://platform.deepseek.com/>`_ (for DeepSeek models)

**Quick Prerequisites Setup**

If you haven't set up the prerequisites yet:

.. code-block:: bash

   # 1. Install Aider
   python -m pip install aider-install
   aider-install
   
   # 2. Install Claude Code (requires Node.js)
   npm install -g @anthropic-ai/claude-code
   
   # 3. Install Agentic
   pip install agentic

For complete installation instructions, see the :doc:`installation` guide.

First Steps
-----------

1. **Initialize a Project**

   .. code-block:: bash

      # Navigate to your project directory
      cd /path/to/your/project

      # Initialize Agentic
      agentic init

   This creates a ``.agentic/`` directory with configuration files.

2. **Verify Installation**

   .. code-block:: bash

      # Check version
      agentic --version

      # View available commands
      agentic --help

Basic Usage
-----------

**Simple Code Generation**

.. code-block:: bash

   # Generate a Python function
   agentic "Create a function to calculate fibonacci numbers"

   # Add features to existing code  
   agentic "Add error handling to the user authentication module"

   # Refactor code
   agentic "Refactor the database connection code to use connection pooling"

**Project Analysis**

.. code-block:: bash

   # Analyze project structure
   agentic analyze

   # Get project insights
   agentic insights

   # Check code quality
   agentic quality-check

**Multi-Agent Coordination**

.. code-block:: bash

   # Complex task requiring multiple agents
   agentic "Add a REST API with authentication, database models, and tests"

   # Architecture planning
   agentic "Design a microservices architecture for this monolith"

Configuration
-------------

**Basic Configuration**

Create or edit ``.agentic/config.yaml``:

.. code-block:: yaml

   # Agent preferences
   agents:
     preferred_models:
       - "gpt-4"
       - "claude-3-sonnet"
     timeout: 300
     max_retries: 3

   # Project settings
   project:
     auto_commit: false
     backup_enabled: true
     excluded_files:
       - "*.pyc"
       - ".git/*"
       - "node_modules/*"

**API Configuration**

Set environment variables:

.. code-block:: bash

   # OpenAI
   export OPENAI_API_KEY="your-openai-api-key"

   # Anthropic
   export ANTHROPIC_API_KEY="your-anthropic-api-key"

Or add to config file:

.. code-block:: yaml

   api:
     openai:
       api_key: "your-openai-api-key"
       model: "gpt-4"
     anthropic:
       api_key: "your-anthropic-api-key"
       model: "claude-3-sonnet"

Common Workflows
----------------

**Web Development**

.. code-block:: bash

   # Create a FastAPI application
   agentic "Create a FastAPI app with user authentication and database models"

   # Add frontend
   agentic "Add a React frontend with authentication forms"

   # Add testing
   agentic "Add comprehensive tests for the API endpoints"

**Data Science**

.. code-block:: bash

   # Data analysis pipeline
   agentic "Create a data pipeline to analyze sales data from CSV files"

   # Machine learning model
   agentic "Implement a classification model with feature engineering"

   # Visualization
   agentic "Add interactive visualizations using plotly"

**DevOps & Automation**

.. code-block:: bash

   # CI/CD pipeline
   agentic "Add GitHub Actions workflow for testing and deployment"

   # Docker setup
   agentic "Add Docker configuration with multi-stage builds"

   # Monitoring
   agentic "Add application monitoring with health checks and metrics"

Advanced Features
-----------------

**Custom Agents**

Create specialized agents for your workflow:

.. code-block:: bash

   # Create a custom agent configuration
   agentic agent create --name "security-expert" --specialization "security"

   # Use the custom agent
   agentic --agent security-expert "Review this code for security vulnerabilities"

**Batch Processing**

Process multiple files or tasks:

.. code-block:: bash

   # Process multiple files
   agentic batch --input "*.py" "Add type hints to all functions"

   # Run multiple commands
   agentic batch --commands commands.txt

**Integration with Git**

.. code-block:: bash

   # Auto-commit changes
   agentic --auto-commit "Add user registration feature"

   # Create feature branch
   agentic branch "feature/user-auth" "Implement user authentication system"

Tips and Best Practices
------------------------

**Writing Effective Prompts**

1. **Be Specific**
   
   ❌ Bad: "Fix the code"
   
   ✅ Good: "Fix the memory leak in the image processing function"

2. **Provide Context**
   
   ❌ Bad: "Add authentication"
   
   ✅ Good: "Add JWT-based authentication to the FastAPI application with user registration and login endpoints"

3. **Specify Requirements**
   
   ❌ Bad: "Add tests"
   
   ✅ Good: "Add pytest tests with >90% coverage for the user service module"

**Project Organization**

1. **Use .agentic/ignore**
   
   .. code-block:: text
   
      # Add files to ignore
      *.log
      .env
      __pycache__/
      .pytest_cache/

2. **Configure File Patterns**
   
   .. code-block:: yaml
   
      project:
        include_patterns:
          - "*.py"
          - "*.js"
          - "*.md"
        exclude_patterns:
          - "*/migrations/*"
          - "*/vendor/*"

**Performance Optimization**

1. **Use Focused Commands**
   
   .. code-block:: bash
   
      # Target specific files
      agentic --files "src/auth.py" "Add rate limiting"
   
      # Limit scope
      agentic --scope "tests/" "Update all test fixtures"

2. **Enable Caching**
   
   .. code-block:: yaml
   
      cache:
        enabled: true
        ttl: 3600  # 1 hour

Troubleshooting
---------------

**Common Issues**

1. **API Rate Limits**
   
   .. code-block:: bash
   
      # Add delays between requests
      agentic --delay 2 "Large refactoring task"

2. **Large Projects**
   
   .. code-block:: bash
   
      # Process in chunks
      agentic --chunk-size 10 "Update all modules"

3. **Permission Issues**
   
   .. code-block:: bash
   
      # Run with appropriate permissions
      sudo agentic "Modify system configuration files"

**Getting Help**

* Use ``agentic --help`` for command help
* Check logs: ``~/.agentic/logs/``
* Enable debug mode: ``agentic --debug``
* Join our community: https://discord.gg/agentic

Examples
--------

**Example 1: Building a Blog API**

.. code-block:: bash

   # Initialize project
   mkdir blog-api && cd blog-api
   agentic init

   # Create the API structure
   agentic "Create a FastAPI blog application with:
   - Post model with title, content, author, created_at
   - CRUD endpoints for posts
   - SQLAlchemy database integration
   - Pydantic schemas for validation
   - Basic error handling"

   # Add authentication
   agentic "Add JWT authentication with:
   - User model and registration
   - Login endpoint
   - Protected routes for post creation
   - Token validation middleware"

   # Add tests
   agentic "Add comprehensive pytest tests with:
   - Test fixtures for database and auth
   - Unit tests for all endpoints
   - Integration tests for user flows
   - >90% code coverage"

**Example 2: Data Analysis Pipeline**

.. code-block:: bash

   # Create analysis project
   mkdir sales-analysis && cd sales-analysis
   agentic init

   # Build the pipeline
   agentic "Create a sales data analysis pipeline with:
   - CSV data ingestion with pandas
   - Data cleaning and validation
   - Statistical analysis and trends
   - Interactive visualizations with plotly
   - Export functionality to PDF reports"

   # Add automation
   agentic "Add automated reporting with:
   - Scheduled data processing
   - Email notifications for reports
   - Error handling and logging
   - Configuration management"

Next Steps
----------

Now that you're familiar with the basics:

1. Read the :doc:`cli` reference for detailed command documentation
2. Explore :doc:`api` for programmatic usage
3. Check out :doc:`architecture` to understand how Agentic works
4. See example projects in our `GitHub repository <https://github.com/agentic-ai/agentic/tree/main/examples>`_

Happy coding with Agentic! 🚀 