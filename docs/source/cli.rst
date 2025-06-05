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
   agentic status                    # Show agent status
   agentic analyze                   # Analyze project
   agentic exec "Add authentication" # Execute natural language request
   agentic interactive               # Start interactive mode

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
   --help                Show help message
   --version             Show version information

Core Commands
-------------

init
~~~~

Initialize Agentic in a project directory.

.. code-block:: bash

   agentic init [OPTIONS]

**Options:**

* ``--force``: Overwrite existing configuration

**Examples:**

.. code-block:: bash

   # Initialize in current directory
   agentic init

   # Force re-initialization
   agentic init --force

**What it does:**

* Creates `.agentic/` directory with configuration
* Analyzes project structure
* Sets up agent registry
* Initializes orchestrator

status
~~~~~~

Show current status of agents and system.

.. code-block:: bash

   agentic status [OPTIONS]

**Options:**

* ``--format FORMAT``: Output format (table, json, simple)

**Examples:**

.. code-block:: bash

   # Show status table
   agentic status

   # JSON output
   agentic status --format json

   # Simple text output
   agentic status --format simple

analyze
~~~~~~~

Analyze project structure and provide insights.

.. code-block:: bash

   agentic analyze [OPTIONS]

**Options:**

* ``--output FORMAT``: Output format (table, json, yaml)

**Examples:**

.. code-block:: bash

   # Basic analysis
   agentic analyze

   # JSON output
   agentic analyze --output json

   # YAML output  
   agentic analyze --output yaml

exec
~~~~

Execute commands using natural language.

.. code-block:: bash

   agentic exec [OPTIONS] COMMAND...

**Options:**

* ``--agent NAME``: Route to specific agent
* ``--context TEXT``: Additional context for the command

**Examples:**

.. code-block:: bash

   # Basic command execution
   agentic exec "Add user authentication to the API"

   # With specific agent
   agentic exec --agent claude-code "Debug the login function"

   # With additional context
   agentic exec --context "This is a FastAPI project" "Add error handling"

spawn
~~~~~

Manually spawn specific agents.

.. code-block:: bash

   agentic spawn [OPTIONS] AGENT_TYPE

**Available Agent Types:**

* ``claude-code``: Claude Code CLI agent for development tasks
* ``aider-frontend``: Frontend development with Aider
* ``aider-backend``: Backend development with Aider
* ``aider-testing``: Testing and QA with Aider

**Examples:**

.. code-block:: bash

   # Spawn Claude Code agent
   agentic spawn claude-code

   # Spawn Aider frontend agent
   agentic spawn aider-frontend

interactive
~~~~~~~~~~~

Start interactive mode for conversational development.

.. code-block:: bash

   agentic interactive [OPTIONS]

**Features:**

* Real-time agent status monitoring
* Command history and suggestions
* Multi-pane interface with Rich TUI
* Agent health monitoring
* Live execution feedback

**Interactive Commands:**

* ``status`` - Show agent status
* ``clear`` - Clear output
* ``help`` - Show help
* ``exit/quit/q`` - Exit interactive mode

**Examples:**

.. code-block:: bash

   # Start interactive mode
   agentic interactive

Natural Language Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also execute natural language requests directly:

.. code-block:: bash

   agentic [OPTIONS] "REQUEST_TEXT"

This is equivalent to ``agentic exec "REQUEST_TEXT"`` but more convenient for quick commands.

**Examples:**

.. code-block:: bash

   # Direct natural language command
   agentic "Add logging to all functions"

   # With options
   agentic --agent claude-code "Optimize this algorithm"

IDE Integration Commands
------------------------

The CLI also includes IDE integration commands under the ``ide`` subcommand group:

.. code-block:: bash

   agentic ide init           # Initialize IDE integrations
   agentic ide status         # Show IDE integration status  
   agentic ide command TEXT   # Execute IDE command
   agentic ide edit FILE      # Edit file through IDE integration

Configuration
-------------

Agentic uses configuration files in the ``.agentic/`` directory:

* ``config.yml`` - Main configuration
* ``agents/`` - Agent-specific configurations
* ``logs/`` - Log files

**Sample Configuration:**

.. code-block:: yaml

   workspace_path: "/path/to/project"
   log_level: "INFO"
   
   agents:
     claude_code:
       enabled: true
       model: "claude-3-sonnet"
     
     aider:
       enabled: true
       focus_areas: ["frontend", "backend", "testing"]

Environment Variables
---------------------

Key environment variables:

* ``AGENTIC_CONFIG_PATH`` - Path to configuration file
* ``AGENTIC_LOG_LEVEL`` - Default log level
* ``ANTHROPIC_API_KEY`` - Anthropic API key (optional enhancement)
* ``OPENAI_API_KEY`` - OpenAI API key (optional)

Authentication Setup
--------------------

Agentic requires authentication for AI providers:

1. **Claude Code CLI** (Primary):
   
   .. code-block:: bash
   
      # Requires Claude Pro subscription
      # Authentication handled through browser

2. **API Keys** (Enhancement):
   
   .. code-block:: bash
   
      export ANTHROPIC_API_KEY="your-key"
      export OPENAI_API_KEY="your-key"

See ``AUTHENTICATION_SETUP.md`` for detailed setup instructions.

Troubleshooting
---------------

**Common Issues:**

* **"No available agents found"**: Run ``agentic init`` first
* **"Claude Code CLI not found"**: Install with ``npm install -g @anthropic-ai/claude-code``
* **Authentication errors**: Check Claude Pro subscription and API keys

**Debug Mode:**

.. code-block:: bash

   agentic --debug exec "your command"

**Verbose Output:**

.. code-block:: bash

   agentic --verbose status

**Log Files:**

Check ``.agentic/logs/`` for detailed execution logs.

Performance Tips
----------------

* Use ``--quiet`` for scripting to reduce output
* Use ``agentic interactive`` for multiple operations
* Use specific agents (``--agent``) when you know the task type
* Check ``agentic status`` before running commands 