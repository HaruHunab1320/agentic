#!/usr/bin/env python3
"""Test error handling in coordination engine"""

import asyncio
import os
import sys
from pathlib import Path

# Set automated mode to trigger error
os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig


async def test_error_messages():
    """Test that proper error messages are shown"""
    print("=== Testing Error Message Display ===\n")
    
    # Initialize orchestrator
    config = AgenticConfig.load_or_create(Path.cwd())
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    # Test command that will fail if Claude is not authenticated
    command = "run the tests and ensure they all pass"
    
    try:
        print(f"Executing: {command}\n")
        result = await orchestrator.execute_command(command)
        
        print("\n--- Result ---")
        print(f"Success: {result.success}")
        print(f"Status: {result.status}")
        
        if hasattr(result, 'task_results'):
            print("\nTask Results:")
            for task_id, task_result in result.task_results.items():
                print(f"  Task {task_id[:8]}...")
                print(f"    Status: {task_result.status}")
                if task_result.error:
                    print(f"    Error: {task_result.error}")
        
        if hasattr(result, 'error'):
            print(f"\nDirect error: {result.error}")
            
    except Exception as e:
        print(f"\nException caught: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_error_messages())