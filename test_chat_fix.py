#!/usr/bin/env python3
"""
Test script to verify the chat interface JSON output fix
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.chat_interface import ChatInterface


async def test_chat_interface():
    """Test the chat interface with a Claude analysis task"""
    print("Testing chat interface JSON output fix...")
    
    # Create chat interface
    workspace = Path.cwd()
    chat = ChatInterface(workspace_path=workspace, debug=False)
    
    # Test with an analysis command
    test_command = "analyze the orchestrator.py file and tell me about its main functions"
    
    print(f"\nExecuting: {test_command}")
    print("-" * 80)
    
    # Process the command
    await chat.process_input(test_command)
    
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_chat_interface())