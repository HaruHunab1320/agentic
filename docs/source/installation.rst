Installation Guide
==================

System Requirements
-------------------

**Python Version**
    Agentic requires Python 3.9 or higher.

**Operating Systems**
    * Linux (Ubuntu 20.04+, CentOS 8+, Debian 11+)
    * macOS (10.15+)
    * Windows (10+)

**Hardware Requirements**
    * Minimum: 2GB RAM, 1GB disk space
    * Recommended: 4GB RAM, 2GB disk space

Installation Methods
--------------------

PyPI Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install the latest stable version
   pip install agentic

   # Install with development dependencies
   pip install "agentic[dev]"

   # Install specific version
   pip install agentic==0.1.0

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/agentic-ai/agentic.git
   cd agentic

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode
   pip install -e ".[dev]"

Docker Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Pull the latest image
   docker pull agentic/agentic:latest

   # Run Agentic in a container
   docker run -v $(pwd):/workspace agentic/agentic:latest

Using Poetry
~~~~~~~~~~~~

.. code-block:: bash

   # Add to existing project
   poetry add agentic

   # Install development dependencies
   poetry add agentic --group dev

Virtual Environment Setup
--------------------------

**Using venv (Recommended)**

.. code-block:: bash

   # Create virtual environment
   python -m venv agentic-env
   
   # Activate on Linux/macOS
   source agentic-env/bin/activate
   
   # Activate on Windows
   agentic-env\Scripts\activate

**Using conda**

.. code-block:: bash

   # Create conda environment
   conda create -n agentic python=3.11
   conda activate agentic
   
   # Install via pip
   pip install agentic

Verification
------------

Verify your installation by running:

.. code-block:: bash

   # Check version
   agentic --version
   
   # Run help
   agentic --help
   
   # Test basic functionality
   agentic init --help

Expected output:

.. code-block:: text

   Agentic v0.1.0
   Multi-agent AI development workflows from a single CLI

Configuration
-------------

**Environment Variables**

.. code-block:: bash

   # Set API keys (optional)
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Set configuration directory
   export AGENTIC_CONFIG_DIR="~/.agentic"

**Configuration File**

Create ``~/.agentic/config.yaml``:

.. code-block:: yaml

   # API Configuration
   api:
     openai:
       api_key: "your-openai-key"
       model: "gpt-4"
     anthropic:
       api_key: "your-anthropic-key"
       model: "claude-3-sonnet"
   
   # Agent Configuration
   agents:
     default_timeout: 300
     max_retries: 3
   
   # Project Settings
   project:
     auto_commit: false
     backup_enabled: true

Troubleshooting
---------------

**Common Issues**

1. **Permission Errors**
   
   .. code-block:: bash
   
      # Install in user directory
      pip install --user agentic

2. **Python Version Issues**
   
   .. code-block:: bash
   
      # Check Python version
      python --version
      
      # Use specific Python version
      python3.11 -m pip install agentic

3. **Network Issues**
   
   .. code-block:: bash
   
      # Use different index
      pip install -i https://pypi.org/simple/ agentic
      
      # Increase timeout
      pip install --timeout 300 agentic

4. **Virtual Environment Issues**
   
   .. code-block:: bash
   
      # Recreate virtual environment
      rm -rf venv
      python -m venv venv
      source venv/bin/activate
      pip install agentic

**Platform-Specific Notes**

**macOS**
   If you encounter SSL certificate issues:
   
   .. code-block:: bash
   
      # Update certificates
      /Applications/Python\ 3.x/Install\ Certificates.command

**Windows**
   Use Command Prompt or PowerShell as Administrator for installation.

**Linux**
   Ensure you have ``python3-dev`` and ``build-essential`` installed:
   
   .. code-block:: bash
   
      sudo apt-get install python3-dev build-essential

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/agentic-ai/agentic/issues>`_
2. Read the `FAQ <faq.html>`_
3. Join our `Discord Community <https://discord.gg/agentic>`_
4. Contact support at support@agentic.ai

Next Steps
----------

After installation, proceed to the :doc:`quickstart` guide to begin using Agentic. 