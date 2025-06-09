#!/usr/bin/env python3
"""Test swarm monitor display during task execution"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Set automated mode to avoid interactive prompts
os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig


async def test_monitor_display():
    """Test that swarm monitor displays properly"""
    print("=== Testing Swarm Monitor Display ===\n")
    
    # Initialize orchestrator
    config = AgenticConfig.load_or_create(Path.cwd())
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    # Test with a longer-running task
    command = """Create a Python module called calculator.py with the following functions:
    - add(a, b): returns sum of a and b
    - subtract(a, b): returns a minus b
    - multiply(a, b): returns product of a and b
    - divide(a, b): returns a divided by b with error handling for division by zero
    Include docstrings and type hints for all functions."""
    
    print(f"Executing task...\n")
    print("You should see the swarm monitor display below:")
    print("-" * 60)
    
    try:
        # Give time for monitor to start
        start_time = time.time()
        
        # Execute command
        result = await orchestrator.execute_command(command)
        
        execution_time = time.time() - start_time
        
        print("\n" + "-" * 60)
        print(f"\nExecution completed in {execution_time:.1f} seconds")
        print(f"Success: {result.success}")
        print(f"Status: {result.status}")
        
        if result.output:
            print(f"\nOutput preview:")
            print(result.output[:300] + "..." if len(result.output) > 300 else result.output)
            
        # Check if calculator.py was created
        if Path("calculator.py").exists():
            print("\n✅ calculator.py was created successfully!")
            # Clean up
            Path("calculator.py").unlink()
        else:
            print("\n❌ calculator.py was not created")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_monitor_display())