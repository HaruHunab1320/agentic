Command Line Interface
======================

The Agentic CLI provides a unified interface for multi-agent AI development workflows. This reference covers all available commands, options, and usage patterns.

Quick Reference
---------------

.. code-block:: bash

   # Basic usage
   agentic [OPTIONS] COMMAND [ARGS]...
   agentic [OPTIONS] "NATURAL_LANGUAGE_REQUEST"

   # Common commands
   agentic --help                    # Show help
   agentic --version                 # Show version
   agentic init                      # Initialize project
   agentic "Add user authentication" # Natural language request

Global Options
--------------

These options are available for all commands:

.. code-block:: bash

   -v, --verbose          Enable verbose output
   -q, --quiet           Suppress non-essential output
   --debug               Enable debug mode with detailed logging
   --config PATH         Path to configuration file
   --log-level LEVEL     Set logging level (DEBUG, INFO, WARNING, ERROR)
   --no-color            Disable colored output
   --profile             Enable performance profiling
   --help                Show help message
   --version             Show version information

Core Commands
-------------

init
~~~~

Initialize Agentic in a project directory.

.. code-block:: bash

   agentic init [OPTIONS] [DIRECTORY]

**Options:**

* ``--template NAME``: Use a project template (fastapi, react, python-package)
* ``--force``: Overwrite existing configuration
* ``--minimal``: Create minimal configuration
* ``--interactive``: Interactive setup wizard

**Examples:**

.. code-block:: bash

   # Initialize in current directory
   agentic init

   # Initialize with FastAPI template
   agentic init --template fastapi

   # Interactive setup
   agentic init --interactive

Natural Language Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute natural language requests directly:

.. code-block:: bash

   agentic [OPTIONS] "REQUEST_TEXT"

**Options:**

* ``--agent NAME``: Use specific agent
* ``--model NAME``: Use specific AI model
* ``--files PATTERN``: Target specific files
* ``--scope DIRECTORY``: Limit scope to directory
* ``--dry-run``: Show what would be done without executing
* ``--auto-commit``: Automatically commit changes
* ``--backup``: Create backup before changes

**Examples:**

.. code-block:: bash

   # Basic request
   agentic "Add logging to all functions"

   # With specific agent
   agentic --agent python-expert "Optimize this algorithm"

   # Target specific files
   agentic --files "*.py" "Add type hints"

   # Dry run mode
   agentic --dry-run "Refactor authentication module"

Project Management
------------------

analyze
~~~~~~~

Analyze project structure and provide insights.

.. code-block:: bash

   agentic analyze [OPTIONS]

**Options:**

* ``--format FORMAT``: Output format (text, json, html)
* ``--depth LEVEL``: Analysis depth (1-5)
* ``--include-metrics``: Include code quality metrics
* ``--output FILE``: Save analysis to file

**Examples:**

.. code-block:: bash

   # Basic analysis
   agentic analyze

   # Detailed analysis with metrics
   agentic analyze --depth 5 --include-metrics

   # Save to file
   agentic analyze --format json --output analysis.json

status
~~~~~~

Show current project status and active changes.

.. code-block:: bash

   agentic status [OPTIONS]

**Options:**

* ``--verbose``: Show detailed status
* ``--changes-only``: Show only files with changes
* ``--branch``: Show Git branch information

insights
~~~~~~~~

Generate project insights and recommendations.

.. code-block:: bash

   agentic insights [OPTIONS]

**Options:**

* ``--category TYPE``: Focus on specific insights (performance, security, structure)
* ``--priority LEVEL``: Filter by priority (low, medium, high, critical)
* ``--format FORMAT``: Output format

Agent Management
----------------

agent
~~~~~

Manage AI agents and their configurations.

.. code-block:: bash

   agentic agent SUBCOMMAND [OPTIONS]

**Subcommands:**

* ``list``: List available agents
* ``create``: Create custom agent
* ``remove``: Remove custom agent
* ``info``: Show agent information
* ``test``: Test agent functionality

**Examples:**

.. code-block:: bash

   # List all agents
   agentic agent list

   # Create custom agent
   agentic agent create --name "security-expert" --specialization "security"

   # Get agent info
   agentic agent info python-expert

   # Test agent
   agentic agent test --agent python-expert

models
~~~~~~

Manage AI models and provider configurations.

.. code-block:: bash

   agentic models SUBCOMMAND [OPTIONS]

**Subcommands:**

* ``list``: List available models
* ``test``: Test model connectivity
* ``benchmark``: Benchmark model performance

**Examples:**

.. code-block:: bash

   # List models
   agentic models list

   # Test OpenAI connection
   agentic models test --provider openai

   # Benchmark models
   agentic models benchmark --task "code-generation"

Batch Operations
----------------

batch
~~~~~

Execute operations on multiple files or run multiple commands.

.. code-block:: bash

   agentic batch [OPTIONS]

**Options:**

* ``--input PATTERN``: Input file pattern
* ``--commands FILE``: File containing commands to run
* ``--parallel N``: Number of parallel operations
* ``--continue-on-error``: Continue processing after errors
* ``--output-dir DIR``: Directory for output files

**Examples:**

.. code-block:: bash

   # Process multiple files
   agentic batch --input "src/*.py" "Add docstrings to all functions"

   # Run commands from file
   agentic batch --commands batch_commands.txt

   # Parallel processing
   agentic batch --parallel 4 --input "*.js" "Convert to TypeScript"

Git Integration
---------------

branch
~~~~~~

Create and manage feature branches with AI assistance.

.. code-block:: bash

   agentic branch [OPTIONS] BRANCH_NAME "TASK_DESCRIPTION"

**Options:**

* ``--base BRANCH``: Base branch (default: main)
* ``--push``: Push branch after creation
* ``--pull-request``: Create pull request

**Examples:**

.. code-block:: bash

   # Create feature branch
   agentic branch feature/auth "Implement user authentication"

   # Create and push branch
   agentic branch --push feature/api "Add REST API endpoints"

commit
~~~~~~

Generate commit messages and commit changes.

.. code-block:: bash

   agentic commit [OPTIONS] [MESSAGE]

**Options:**

* ``--all``: Commit all changes (git add -A)
* ``--generate``: Generate commit message automatically
* ``--conventional``: Use conventional commit format

**Examples:**

.. code-block:: bash

   # Auto-generate commit message
   agentic commit --generate

   # Commit with generated conventional message
   agentic commit --all --conventional

Quality & Testing
-----------------

quality-check
~~~~~~~~~~~~~

Run comprehensive code quality checks.

.. code-block:: bash

   agentic quality-check [OPTIONS]

**Options:**

* ``--fix``: Automatically fix issues where possible
* ``--strict``: Use strict quality standards
* ``--report FILE``: Generate quality report
* ``--format FORMAT``: Report format (text, json, html)

**Examples:**

.. code-block:: bash

   # Basic quality check
   agentic quality-check

   # Fix issues automatically
   agentic quality-check --fix

   # Generate detailed report
   agentic quality-check --report quality_report.html --format html

test
~~~~

Generate and run tests with AI assistance.

.. code-block:: bash

   agentic test [OPTIONS] [PATTERN]

**Options:**

* ``--generate``: Generate missing tests
* ``--coverage MIN``: Minimum coverage requirement
* ``--type TYPE``: Test type (unit, integration, e2e)
* ``--update-snapshots``: Update test snapshots

**Examples:**

.. code-block:: bash

   # Run tests
   agentic test

   # Generate tests for specific files
   agentic test --generate "src/auth.py"

   # Require 90% coverage
   agentic test --coverage 90

Configuration
-------------

config
~~~~~~

Manage Agentic configuration.

.. code-block:: bash

   agentic config SUBCOMMAND [OPTIONS]

**Subcommands:**

* ``show``: Show current configuration
* ``set``: Set configuration value
* ``unset``: Remove configuration value
* ``validate``: Validate configuration

**Examples:**

.. code-block:: bash

   # Show configuration
   agentic config show

   # Set API key
   agentic config set api.openai.api_key "your-key"

   # Validate configuration
   agentic config validate

Advanced Usage
--------------

Plugin System
~~~~~~~~~~~~~

.. code-block:: bash

   # List plugins
   agentic plugins list

   # Install plugin
   agentic plugins install agentic-docker

   # Enable plugin
   agentic plugins enable docker

Debugging & Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Debug mode
   agentic --debug "Complex task"

   # Profile performance
   agentic --profile "Large refactoring"

   # View logs
   agentic logs --tail 100

   # System status
   agentic system-status

Custom Scripts
~~~~~~~~~~~~~~

.. code-block:: bash

   # Run custom script
   agentic run script.py

   # Execute workflow
   agentic workflow run deployment

Environment Variables
---------------------

Agentic recognizes these environment variables:

.. code-block:: bash

   # API Keys
   OPENAI_API_KEY              # OpenAI API key
   ANTHROPIC_API_KEY           # Anthropic API key
   GOOGLE_API_KEY              # Google AI API key

   # Configuration
   AGENTIC_CONFIG_DIR          # Configuration directory
   AGENTIC_LOG_LEVEL           # Default log level
   AGENTIC_CACHE_DIR           # Cache directory
   AGENTIC_PLUGINS_DIR         # Plugins directory

   # Behavior
   AGENTIC_AUTO_COMMIT         # Auto-commit changes (true/false)
   AGENTIC_BACKUP_ENABLED      # Enable backups (true/false)
   AGENTIC_PARALLEL_LIMIT      # Max parallel operations

Exit Codes
----------

Agentic uses standard exit codes:

* ``0``: Success
* ``1``: General error
* ``2``: Configuration error
* ``3``: API error
* ``4``: Permission error
* ``5``: Network error
* ``130``: Interrupted by user (Ctrl+C)

Examples
--------

**Complete Workflow Example**

.. code-block:: bash

   # Initialize project
   cd my-project
   agentic init --template fastapi

   # Analyze current state
   agentic analyze --include-metrics

   # Add features
   agentic "Add user authentication with JWT tokens"
   agentic "Add input validation for all endpoints"
   agentic "Add comprehensive logging"

   # Generate tests
   agentic test --generate --coverage 90

   # Quality check
   agentic quality-check --fix

   # Commit changes
   agentic commit --generate --conventional

**Batch Processing Example**

.. code-block:: bash

   # Create batch command file
   echo 'Add type hints to all functions' > commands.txt
   echo 'Add docstrings to all classes' >> commands.txt
   echo 'Add error handling to all API endpoints' >> commands.txt

   # Execute batch
   agentic batch --commands commands.txt --parallel 3

**Custom Agent Example**

.. code-block:: bash

   # Create security-focused agent
   agentic agent create --name "security-expert" \
     --specialization "security" \
     --model "gpt-4" \
     --temperature 0.1

   # Use custom agent
   agentic --agent security-expert "Review this code for vulnerabilities"

Shell Completion
----------------

Enable shell completion for better CLI experience:

**Bash:**

.. code-block:: bash

   # Add to ~/.bashrc
   eval "$(_AGENTIC_COMPLETE=bash_source agentic)"

**Zsh:**

.. code-block:: bash

   # Add to ~/.zshrc
   eval "$(_AGENTIC_COMPLETE=zsh_source agentic)"

**Fish:**

.. code-block:: bash

   # Add to ~/.config/fish/config.fish
   eval (env _AGENTIC_COMPLETE=fish_source agentic)

Getting Help
------------

* Use ``agentic --help`` for general help
* Use ``agentic COMMAND --help`` for command-specific help
* Check the logs: ``~/.agentic/logs/agentic.log``
* Join our community: https://discord.gg/agentic
* Report issues: https://github.com/agentic-ai/agentic/issues 