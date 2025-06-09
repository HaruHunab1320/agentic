#!/usr/bin/env python3
"""Test single agent execution with swarm monitor"""

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


async def test_single_agent_with_monitor():
    """Test single agent execution with monitoring enabled"""
    print("=== Testing Single Agent with Swarm Monitor ===\n")
    
    # Initialize orchestrator
    config = AgenticConfig.load_or_create(Path.cwd())
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    # Test with a task that takes some time
    command = """Write a Python function that checks if a string is a palindrome. 
    Include comprehensive tests and edge cases."""
    
    print(f"Executing task: {command[:50]}...\n")
    print("Monitor should display below:")
    print("-" * 80)
    
    try:
        start_time = time.time()
        
        # Execute with monitoring enabled (default)
        context = {"enable_monitoring": True}
        result = await orchestrator.execute_command(command, context)
        
        execution_time = time.time() - start_time
        
        print("\n" + "-" * 80)
        print(f"\nExecution completed in {execution_time:.1f} seconds")
        print(f"Success: {result.success}")
        print(f"Status: {result.status}")
        
        if result.success:
            print("\n✅ Task completed successfully!")
            
            # Parse and display the output properly
            if hasattr(result, 'output') and result.output:
                output = result.output
                
                # Check if it's Claude Code JSON output
                if isinstance(output, str) and output.strip().startswith('[{'):
                    try:
                        import json
                        # Parse the JSON array
                        messages = json.loads(output)
                        
                        # Extract assistant messages
                        print("\n📝 Claude's Response:")
                        print("-" * 40)
                        
                        for msg in messages:
                            if msg.get('type') == 'assistant':
                                content = msg.get('content', '')
                                if content:
                                    print(content)
                                    print()
                        
                        # Show usage stats if available
                        for msg in messages:
                            if msg.get('type') == 'system' and msg.get('subtype') == 'usage':
                                usage = msg.get('usage', {})
                                if usage:
                                    print("\n📊 Usage Statistics:")
                                    print(f"  Total tokens: {usage.get('totalTokens', 0):,}")
                                    if 'cacheReadTokens' in usage:
                                        print(f"  Cache tokens: {usage.get('cacheReadTokens', 0):,}")
                    except json.JSONDecodeError:
                        # Not JSON, print as-is
                        print(f"\n📝 Output:\n{output}")
                else:
                    # Not Claude Code format, print normally
                    print(f"\n📝 Output:\n{output}")
        else:
            print(f"\n❌ Task failed: {result.error}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_single_agent_with_monitor())