#!/usr/bin/env python3
"""Test script to verify swarm monitor displays properly"""

import asyncio
import os
import sys
from pathlib import Path

# Set automated mode to avoid interactive prompts
os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.cli import cli
import typer
from unittest.mock import MagicMock, AsyncMock


async def test_monitor_with_real_task():
    """Test that the swarm monitor displays properly during execution"""
    print("Testing swarm monitor with real task...")
    
    # Create a mock context
    ctx = MagicMock()
    ctx.ensure_object = MagicMock(return_value={})
    
    try:
        # Use the CLI's chat command to test
        await cli.callback(ctx)
        
        # Now execute a command that takes time
        command = "Create a comprehensive test suite for a calculator module with at least 10 test cases"
        
        # Import the functions we need
        from agentic.cli import _execute_single_command
        
        print(f"\nExecuting command: {command}")
        print("Watch the swarm monitor display...\n")
        
        # Execute the command
        result = await _execute_single_command(command, ctx.obj)
        
        if result and hasattr(result, 'success'):
            print(f"\n✅ Execution completed - Success: {result.success}")
        else:
            print(f"\n✅ Execution completed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function"""
    print("=== Testing Swarm Monitor Display ===\n")
    
    await test_monitor_with_real_task()


if __name__ == "__main__":
    asyncio.run(main())