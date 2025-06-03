Welcome to Agentic's documentation!
====================================

**Agentic** is a multi-agent AI development orchestrator that coordinates specialized AI agents from a single CLI interface.

Key Features
------------

* **Unified CLI**: Single interface for multiple AI agents
* **Intelligent Routing**: Automatically routes tasks to the most appropriate agent
* **Conflict Resolution**: Manages conflicts across multi-file changes
* **Project Analysis**: Deep understanding of codebase structure and dependencies

Quick Start
-----------

.. code-block:: bash

   # Install Agentic
   pip install agentic

   # Initialize in your project
   agentic init

   # Execute a command
   agentic "Add user authentication to the API"

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   cli
   api
   architecture
   contributing

API Reference
=============

.. autosummary::
   :toctree: _autosummary
   :recursive:

   agentic

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 