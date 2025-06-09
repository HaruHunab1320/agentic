#!/usr/bin/env python3
"""Test script to verify coordination engine fixes"""

import asyncio
import os
import sys
from pathlib import Path

# Set automated mode to avoid interactive prompts
os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig


async def test_single_agent_execution():
    """Test single agent execution with swarm monitor"""
    print("Testing single agent execution...")
    
    # Initialize orchestrator
    config = AgenticConfig.load_or_create(Path.cwd())
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    # Test command
    command = "create a simple test.py file that prints hello world"
    
    try:
        print(f"\nExecuting: {command}")
        result = await orchestrator.execute_command(command)
        
        print("\n--- Execution Result ---")
        print(f"Success: {result.success}")
        print(f"Status: {result.status}")
        if result.output:
            print(f"Output: {result.output[:200]}...")
        if result.error:
            print(f"Error: {result.error}")
        
        return result.success
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("=== Testing Coordination Engine Fixes ===\n")
    
    # Test single agent execution
    success = await test_single_agent_execution()
    
    if success:
        print("\n✅ Test passed!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())