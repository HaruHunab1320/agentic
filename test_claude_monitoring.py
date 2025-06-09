#!/usr/bin/env python3
"""Test script to verify Claude Code's real-time activity monitoring"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.agents.claude_code_agent import ClaudeCodeAgent
from agentic.models.task import Task
from agentic.core.swarm_monitor import SwarmMonitor, AgentStatus


async def test_claude_monitoring():
    """Test Claude Code agent with real-time monitoring"""
    print("Starting Claude Code monitoring test...")
    
    # Create monitor
    monitor = SwarmMonitor(use_alternate_screen=True)
    
    # Create Claude Code agent
    agent = ClaudeCodeAgent(
        name="test-claude",
        workspace_path=Path.cwd(),
        ai_model_config={"model": "sonnet"},
        mode="print"
    )
    
    # Register agent with monitor
    agent_id = "claude-test-1"
    monitor.register_agent(
        agent_id=agent_id,
        agent_name="Claude Test Agent",
        agent_type="claude_code",
        role="Testing real-time monitoring"
    )
    
    # Set monitor on agent
    agent._monitor = monitor
    agent._monitor_agent_id = agent_id
    
    # Create a test task
    task = Task(
        id="test-task-1",
        type="analysis",
        priority="high",
        command="Find all Python files in the src directory that contain 'monitor' in their name, then analyze their purpose"
    )
    
    # Start monitoring display
    monitor_task = asyncio.create_task(monitor.run())
    
    try:
        # Execute task
        print("Executing task...")
        result = await agent.execute_task(task)
        
        # Show result
        print(f"\nTask Result:")
        print(f"Status: {result.status}")
        print(f"Output: {result.output[:500]}..." if result.output else "No output")
        
        # Keep monitor running for a bit to see final state
        await asyncio.sleep(3)
        
    finally:
        # Stop monitor
        await monitor.stop()
        await monitor_task
    
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_claude_monitoring())