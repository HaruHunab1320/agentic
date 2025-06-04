Frequently Asked Questions
===========================

This document addresses common questions about Agentic, from basic usage to advanced configuration and troubleshooting.

General Questions
-----------------

What is Agentic?
~~~~~~~~~~~~~~~~

Agentic is a multi-agent AI development framework that orchestrates specialized AI agents to handle complex software development tasks. Instead of relying on a single AI model, Agentic coordinates multiple specialized agents (Python expert, security specialist, frontend developer, etc.) to provide comprehensive development assistance.

How is Agentic different from other AI coding tools?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-Agent Architecture**: Unlike single-model tools, Agentic uses specialized agents for different tasks, providing more accurate and context-aware assistance.

**Production-Ready**: Built-in quality assurance, security scanning, and production monitoring make it suitable for enterprise environments.

**Plugin System**: Extensible architecture allows custom agents and tools for domain-specific requirements.

**Quality-First**: Integrated testing, code review, and quality metrics ensure high-quality output.

What programming languages does Agentic support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agentic currently focuses on Python development but includes agents for:

- **Python**: Full support including FastAPI, Django, Flask, data science
- **Frontend**: React, Vue.js, HTML/CSS, TypeScript
- **DevOps**: Docker, CI/CD, deployment scripts
- **Databases**: SQL, NoSQL, migrations, schema design

Additional language support can be added through the plugin system.

Installation and Setup
----------------------

What are the system requirements?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum Requirements:**
- Python 3.9 or higher
- 2GB RAM
- 1GB disk space
- Internet connection for AI providers

**Recommended:**
- Python 3.11+
- 4GB RAM  
- 2GB disk space
- SSD storage for better performance

How do I get API keys for AI providers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**OpenAI:**

1. Visit https://platform.openai.com/
2. Create an account and verify your phone number
3. Navigate to API Keys section
4. Generate a new API key
5. Set environment variable: ``export OPENAI_API_KEY="your-key"``

**Anthropic (Claude):**

1. Visit https://console.anthropic.com/
2. Create an account
3. Request API access (may require approval)
4. Generate API key in the dashboard
5. Set environment variable: ``export ANTHROPIC_API_KEY="your-key"``

**Local Models:**

For privacy-conscious setups, you can use local models like Ollama:

.. code-block:: bash

   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Download a model
   ollama pull codellama:13b
   
   # Configure Agentic to use local model
   agentic config set ai.provider ollama
   agentic config set ai.model codellama:13b

Can I use Agentic without API keys?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Agentic supports local AI models through:

- **Ollama**: For running models locally
- **OpenAI-compatible APIs**: For self-hosted models
- **HuggingFace Transformers**: For local inference

See the :doc:`installation` guide for local setup instructions.

Why is the first run slow?
~~~~~~~~~~~~~~~~~~~~~~~~~~

The first run may be slower due to:

- **Model Downloads**: Local models need to be downloaded
- **Dependency Installation**: Additional packages may be installed
- **Cache Warming**: Response caches are empty initially
- **Project Analysis**: Initial project analysis takes time

Subsequent runs will be significantly faster due to caching.

Usage and Configuration
-----------------------

How do I initialize a project?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Navigate to your project directory
   cd /path/to/your/project
   
   # Initialize Agentic
   agentic init
   
   # This creates .agentic/ directory with configuration

What configuration options are available?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agentic uses hierarchical configuration:

.. code-block:: yaml

   # .agentic/config.yaml
   ai:
     provider: openai  # openai, anthropic, ollama, local
     model: gpt-4      # Model name
     temperature: 0.7  # Creativity level (0.0-2.0)
     max_tokens: 4000  # Maximum response length
   
   agents:
     python_expert:
       enabled: true
       max_concurrent: 2
     security_specialist:
       enabled: true
       scan_on_generation: true
   
   quality:
     min_test_coverage: 0.8
     enable_type_checking: true
     enable_linting: true
   
   safety:
     backup_before_changes: true
     require_confirmation: false
     max_files_per_operation: 20

How do I customize agent behavior?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Per-Agent Configuration:**

.. code-block:: yaml

   # .agentic/agents/python_expert.yaml
   name: "Python Expert"
   temperature: 0.3  # Lower temperature for more consistent code
   max_tokens: 6000
   
   preferences:
     - "Use type hints for all functions"
     - "Follow PEP 8 style guidelines"
     - "Prefer Pydantic for data validation"
     - "Write comprehensive docstrings"
   
   capabilities:
     - code-generation
     - refactoring
     - testing
     - documentation

**Global Agent Settings:**

.. code-block:: bash

   # Enable/disable specific agents
   agentic config set agents.frontend_developer.enabled false
   
   # Set concurrency limits
   agentic config set agents.max_concurrent_total 4
   
   # Configure timeouts
   agentic config set agents.default_timeout 300

How do I handle large projects?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large projects (>1000 files):

**Configure Resource Limits:**

.. code-block:: yaml

   # .agentic/config.yaml
   resources:
     max_files_analyzed: 500
     analysis_timeout: 600
     memory_limit_gb: 4
   
   performance:
     enable_parallel_analysis: true
     chunk_size: 100
     use_incremental_analysis: true

**Use Selective Analysis:**

.. code-block:: bash

   # Analyze specific directories
   agentic analyze --include "src/" --exclude "tests/"
   
   # Focus on specific file types
   agentic analyze --pattern "*.py" --exclude "migrations/"

**Incremental Processing:**

.. code-block:: bash

   # Process changes since last commit
   agentic "Review changes since last commit"
   
   # Focus on specific modules
   agentic "Refactor the authentication module"

Can I run Agentic in CI/CD pipelines?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Agentic works well in automated environments:

**GitHub Actions:**

.. code-block:: yaml

   # .github/workflows/agentic-review.yml
   name: Agentic Code Review
   on:
     pull_request:
       types: [opened, synchronize]
   
   jobs:
     review:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Setup Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.11'
         - name: Install Agentic
           run: pip install agentic
         - name: Review PR
           run: |
             agentic review-pr --pr-number ${{ github.event.number }}
           env:
             OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
             GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

**GitLab CI:**

.. code-block:: yaml

   # .gitlab-ci.yml
   agentic-review:
     stage: review
     image: python:3.11
     script:
       - pip install agentic
       - agentic review-mr --mr-id $CI_MERGE_REQUEST_IID
     only:
       - merge_requests

Agent-Specific Questions
------------------------

How do I get better code quality?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Enable Quality Checks:**

.. code-block:: bash

   # Enable all quality features
   agentic config set quality.enable_all true
   
   # Set quality thresholds
   agentic config set quality.min_test_coverage 0.9
   agentic config set quality.max_complexity 10
   agentic config set quality.require_type_hints true

**Use Quality-Focused Prompts:**

.. code-block:: bash

   # Request specific quality improvements
   agentic "Add comprehensive type hints to all functions"
   agentic "Increase test coverage to 95%"
   agentic "Refactor complex functions to reduce cyclomatic complexity"

How does the security agent work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The security specialist agent:

- **Static Analysis**: Scans code for common vulnerabilities
- **Dependency Checking**: Identifies vulnerable dependencies
- **Security Best Practices**: Suggests security improvements
- **Compliance Scanning**: Checks against security standards

**Enable Security Scanning:**

.. code-block:: bash

   # Enable automatic security scanning
   agentic config set security.auto_scan true
   agentic config set security.scan_dependencies true
   
   # Request security review
   agentic "Perform a comprehensive security audit"

Can agents work together on complex tasks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Agentic automatically coordinates agents for complex tasks:

**Example: Adding Authentication**

.. code-block:: bash

   agentic "Add JWT authentication to my FastAPI app"

This request triggers:

1. **Python Expert**: Creates authentication logic and models
2. **Security Specialist**: Reviews for security vulnerabilities
3. **Testing Agent**: Generates comprehensive tests
4. **Documentation Agent**: Updates API documentation

**Manual Agent Coordination:**

.. code-block:: bash

   # Request specific agent collaboration
   agentic "Have the Python expert create a REST API, then the security specialist should review it"

How do I handle agent conflicts?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When agents provide conflicting suggestions:

**Priority System**: Agents have built-in priorities:
1. Security Specialist (highest priority for security issues)
2. Quality Assurance (for quality standards)
3. Python Expert (for code correctness)
4. Other specialists

**Manual Resolution:**

.. code-block:: bash

   # Choose specific agent recommendations
   agentic resolve-conflict --prefer security
   
   # Request clarification
   agentic "The security and performance agents disagree on caching strategy. Please provide a balanced approach."

Performance and Optimization
----------------------------

Why are responses slow?
~~~~~~~~~~~~~~~~~~~~~~

Common causes and solutions:

**Large Projects:**
- Enable incremental analysis
- Use selective file patterns
- Increase resource limits

**Network Issues:**
- Check API provider status
- Configure request timeouts
- Use local models for better reliability

**Resource Constraints:**
- Increase memory limits
- Reduce concurrent agent count
- Enable garbage collection

How can I improve performance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Caching:**

.. code-block:: bash

   # Enable response caching
   agentic config set cache.enabled true
   agentic config set cache.ttl 3600  # 1 hour
   
   # Pre-warm cache for common requests
   agentic "Analyze project structure" --cache-only

**Parallel Processing:**

.. code-block:: yaml

   # .agentic/config.yaml
   performance:
     max_concurrent_agents: 4
     enable_parallel_file_processing: true
     chunk_size: 50

**Resource Optimization:**

.. code-block:: bash

   # Monitor resource usage
   agentic status --detailed
   
   # Cleanup unused resources
   agentic cleanup --force

How much does it cost to run Agentic?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Costs depend on your AI provider and usage:

**OpenAI GPT-4:**
- ~$0.03-0.06 per 1K tokens
- Typical request: 2000-4000 tokens
- Cost per request: ~$0.06-0.24

**Cost Optimization:**

.. code-block:: bash

   # Use more efficient models
   agentic config set ai.model gpt-3.5-turbo  # Cheaper option
   
   # Enable smart caching
   agentic config set cache.smart_caching true
   
   # Set token limits
   agentic config set ai.max_tokens 2000

**Local Models (Free):**

.. code-block:: bash

   # Use local models for zero cost
   agentic config set ai.provider ollama
   agentic config set ai.model codellama:13b

Troubleshooting
---------------

Common Error Messages
~~~~~~~~~~~~~~~~~~~~

**"No API key provided"**

.. code-block:: bash

   # Set API key
   export OPENAI_API_KEY="your-key-here"
   
   # Or in config file
   agentic config set ai.api_key "your-key-here"

**"Project not initialized"**

.. code-block:: bash

   # Initialize the project
   agentic init
   
   # Or specify project path
   agentic init --project-path /path/to/project

**"Agent failed to respond"**

.. code-block:: bash

   # Check agent status
   agentic agents status
   
   # Restart failed agents
   agentic agents restart --agent python-expert
   
   # Check logs
   agentic logs --agent python-expert --tail 50

**"Permission denied" errors**

.. code-block:: bash

   # Fix file permissions
   chmod +x .agentic/scripts/*
   
   # Run with appropriate permissions
   sudo agentic --allow-root

How do I enable debug mode?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Enable verbose logging
   agentic --verbose "your request"
   
   # Enable debug mode
   agentic config set logging.level debug
   
   # View detailed logs
   agentic logs --level debug --tail 100

How do I reset configuration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Reset all configuration
   agentic config reset
   
   # Reset specific sections
   agentic config reset --section agents
   agentic config reset --section ai
   
   # Backup before reset
   agentic config backup --file config-backup.yaml

What if Agentic is stuck or unresponsive?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check system status
   agentic status
   
   # Kill stuck processes
   agentic kill --force
   
   # Restart all services
   agentic restart
   
   # Clean temporary files
   agentic cleanup --temp-files

How do I update Agentic?
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Update to latest version
   pip install --upgrade agentic
   
   # Check version
   agentic --version
   
   # Migrate configuration if needed
   agentic migrate-config

Data and Privacy
----------------

What data does Agentic send to AI providers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agentic sends:

- **Code Content**: The files you're working with
- **Context Information**: Project structure, file relationships
- **User Requests**: Your natural language instructions
- **Configuration**: Relevant settings for processing

**What is NOT sent:**
- Personal information
- Credentials or API keys
- Files outside your project
- System information

How can I protect sensitive data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Local Models:**

.. code-block:: bash

   # Configure local AI provider
   agentic config set ai.provider ollama
   agentic config set ai.model codellama:13b

**Data Filtering:**

.. code-block:: yaml

   # .agentic/config.yaml
   privacy:
     exclude_patterns:
       - "*.env"
       - "secrets/*"
       - "*.key"
       - "config/database.yaml"
     
     filter_sensitive_data: true
     anonymize_variable_names: true

**Project-Specific Exclusions:**

.. code-block:: bash

   # Add to .agentic/ignore
   echo "secrets/" >> .agentic/ignore
   echo "*.env" >> .agentic/ignore
   echo "config/production.yaml" >> .agentic/ignore

Can I run Agentic completely offline?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, with local models:

.. code-block:: bash

   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Download models
   ollama pull codellama:13b
   ollama pull mistral:7b
   
   # Configure for offline use
   agentic config set ai.provider ollama
   agentic config set network.offline_mode true

Where are my project files stored?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Project Data**: Stored in `.agentic/` directory within your project
- **Global Config**: `~/.agentic/config.yaml`
- **Cache**: `~/.agentic/cache/`
- **Logs**: `~/.agentic/logs/`

**Backup Important Data:**

.. code-block:: bash

   # Backup project configuration
   cp -r .agentic/ .agentic-backup/
   
   # Backup global configuration
   agentic config backup --file ~/.agentic/backup.yaml

Integration and Ecosystem
-------------------------

Can I integrate Agentic with my IDE?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**VS Code Extension:**

.. code-block:: bash

   # Install VS Code extension (when available)
   code --install-extension agentic.agentic-vscode

**Command Palette Integration:**

.. code-block:: json

   // .vscode/tasks.json
   {
     "version": "2.0.0",
     "tasks": [
       {
         "label": "Agentic: Analyze File",
         "type": "shell",
         "command": "agentic",
         "args": ["analyze", "${file}"],
         "group": "build"
       }
     ]
   }

**JetBrains IDEs:**

.. code-block:: bash

   # External tools configuration
   # Program: agentic
   # Arguments: "analyze current file"
   # Working directory: $ProjectFileDir$

How does Agentic work with Git?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Git Integration:**

.. code-block:: bash

   # Analyze changes before commit
   agentic review-changes
   
   # Generate commit messages
   agentic generate-commit-message
   
   # Review pull requests
   agentic review-pr --pr-number 123

**Git Hooks:**

.. code-block:: bash

   # Pre-commit hook
   echo '#!/bin/bash\nagentic review-changes --exit-on-issues' > .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit

Can I use Agentic with Docker?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Docker Image:**

.. code-block:: bash

   # Run Agentic in Docker
   docker run -v $(pwd):/workspace agentic/agentic:latest "analyze project"
   
   # With API keys
   docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd):/workspace agentic/agentic:latest

**Docker Compose:**

.. code-block:: yaml

   # docker-compose.yml
   version: '3.8'
   services:
     agentic:
       image: agentic/agentic:latest
       volumes:
         - .:/workspace
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

Enterprise and Team Usage
-------------------------

Can multiple team members use the same configuration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Shared Configuration:**

.. code-block:: bash

   # Version control .agentic/config.yaml
   git add .agentic/config.yaml
   git commit -m "Add shared Agentic configuration"

**Team-Specific Settings:**

.. code-block:: yaml

   # .agentic/team-config.yaml
   team:
     name: "Development Team"
     standards:
       - "Follow company coding standards"
       - "Require security review for authentication code"
       - "Maintain 90% test coverage"
   
   agents:
     shared_preferences:
       - "Use company-approved libraries only"
       - "Follow established architecture patterns"

How do I set up Agentic for enterprise use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Centralized Configuration:**

.. code-block:: bash

   # Set enterprise defaults
   agentic config set-global enterprise.enabled true
   agentic config set-global enterprise.policy_file /etc/agentic/policy.yaml
   
   # Deploy to team
   agentic deploy-config --team development --environment production

**Custom Policies:**

.. code-block:: yaml

   # /etc/agentic/policy.yaml
   security:
     required_scans: ["vulnerability", "dependency", "secret"]
     blocked_patterns: ["eval(", "exec(", "subprocess.shell"]
   
   quality:
     min_test_coverage: 0.85
     required_documentation: true
     max_complexity: 15

What about compliance and auditing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Audit Logging:**

.. code-block:: bash

   # Enable comprehensive audit logging
   agentic config set audit.enabled true
   agentic config set audit.log_level detailed
   
   # Export audit logs
   agentic audit export --format json --start-date 2024-01-01

**Compliance Reports:**

.. code-block:: bash

   # Generate compliance report
   agentic compliance report --standard SOC2
   agentic compliance report --standard PCI-DSS

Getting Help and Support
------------------------

Where can I get help?
~~~~~~~~~~~~~~~~~~~~

**Documentation:**
- Official docs: https://docs.agentic.ai
- API reference: https://api.agentic.ai
- Examples: https://github.com/agentic-ai/examples

**Community:**
- GitHub Discussions: https://github.com/agentic-ai/agentic/discussions
- Discord: https://discord.gg/agentic
- Stack Overflow: Tag your questions with `agentic`

**Professional Support:**
- Enterprise support: support@agentic.ai
- Training and consulting: consulting@agentic.ai

How do I report bugs?
~~~~~~~~~~~~~~~~~~~~

**Before Reporting:**

1. Check existing issues: https://github.com/agentic-ai/agentic/issues
2. Update to latest version: `pip install --upgrade agentic`
3. Try with minimal configuration
4. Gather debug information: `agentic debug-info`

**Bug Report Template:**

.. code-block:: markdown

   **Environment:**
   - Agentic version: 
   - Python version:
   - Operating system:
   - AI provider:

   **Expected Behavior:**
   What you expected to happen.

   **Actual Behavior:**
   What actually happened.

   **Steps to Reproduce:**
   1. Run command X
   2. See error Y

   **Error Output:**
   ```
   Paste full error traceback here
   ```

   **Additional Context:**
   Any other relevant information.

How can I contribute to Agentic?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See our :doc:`contributing` guide for detailed information on:

- Setting up development environment
- Code standards and testing
- Submitting pull requests
- Creating plugins and extensions

**Quick Start:**

.. code-block:: bash

   # Fork and clone
   git clone https://github.com/your-username/agentic.git
   cd agentic
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Run tests
   pytest
   
   # Make your changes and submit a PR

We welcome contributions of all types: bug fixes, features, documentation, examples, and community support! 