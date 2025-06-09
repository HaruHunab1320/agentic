#!/usr/bin/env python3
"""Simple test to see exact error"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Set automated mode
os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig


async def test_execution():
    """Test simple execution to see exact error"""
    print("=== Testing Simple Execution ===\n")
    
    try:
        # Initialize orchestrator
        config = AgenticConfig.load_or_create(Path.cwd())
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        
        # Simple test command
        command = "create a hello.txt file with 'Hello World' content"
        
        print(f"Executing: {command}\n")
        
        # Execute command
        result = await orchestrator.execute_command(command)
        
        print(f"\nResult type: {type(result)}")
        print(f"Result status: {result.status}")
        print(f"Result success: {result.success}")
        
        if hasattr(result, 'task_results'):
            print(f"\nTask results count: {len(result.task_results)}")
            for task_id, task_result in result.task_results.items():
                print(f"\nTask {task_id[:8]}...")
                print(f"  Status: {task_result.status}")
                print(f"  Error: {task_result.error}")
                if task_result.output:
                    print(f"  Output: {task_result.output[:100]}...")
        
        if hasattr(result, 'coordination_log'):
            print(f"\nCoordination log entries: {len(result.coordination_log)}")
            for entry in result.coordination_log:
                if entry.get('type') == 'execution_error':
                    print(f"\nExecution error found:")
                    print(f"  Error: {entry.get('error')}")
                    if 'traceback' in entry:
                        print(f"  Traceback:\n{entry['traceback']}")
                        
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_execution())