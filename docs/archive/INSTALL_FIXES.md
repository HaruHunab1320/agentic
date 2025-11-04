# Installing the Fixed Version

The fixes we've made are in the development code, but your CLI is using the previously installed version. Here's how to update:

## Quick Install (Recommended)

From the agentic directory, run:

```bash
# Install in editable mode (uses live code)
pip install -e .
```

This will:
- Use the current development code with all fixes
- Automatically pick up any future changes
- Keep your `agentic` command working

## Alternative: Clean Install

If you prefer a clean installation:

```bash
# Uninstall current version
pip uninstall agentic -y

# Install from current directory
pip install .
```

## Verify Installation

After installing, verify the fixes are active:

```bash
# Check version
agentic --version

# Test with a simple command
agentic
> create a test file
```

## What You Should See

With the fixes installed:

1. **Better Error Messages**: Instead of "Unknown error", you'll see specific errors like:
   - "No agent available to execute this task. Agent spawn failed."
   - "Claude Code not authenticated. Run 'claude' manually to authenticate."

2. **Proper Task Results**: Errors will be properly displayed in the chat interface

3. **Enhanced Logging**: Debug information will be available in logs

## For Claude Code Users

If using Claude Code agents, ensure authentication:

```bash
# Check Claude status
python diagnose_claude_auth.py

# If not authenticated
claude
```

## Development Mode

For active development, always use editable install:

```bash
pip install -e .
```

This ensures you're always using the latest code without reinstalling.