#!/usr/bin/env python3
"""Test the new execution summary display"""

import asyncio
import os
import sys
from pathlib import Path

# Set automated mode
os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.chat_interface import ChatInterface


async def test_summary_display():
    """Test the new summary display with multi-agent execution"""
    print("=== Testing New Summary Display ===\n")
    
    # Initialize chat interface
    chat = ChatInterface(Path.cwd(), debug=False)
    
    if not await chat.initialize():
        print("Failed to initialize chat interface")
        return
    
    print("\n1. Testing single agent execution with Claude Code...")
    await chat._execute_natural_command("create a simple hello.txt file with 'Hello World' content")
    
    print("\n\n2. Testing multi-agent execution...")
    await chat._execute_natural_command("@all create a simple React component with tests")
    
    print("\n\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_summary_display())