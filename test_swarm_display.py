#!/usr/bin/env python3
"""Test if the swarm monitor display is working correctly"""

import asyncio
import sys
import os

# Set automated mode
os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from agentic.core.swarm_monitor_enhanced import SwarmMonitorEnhanced, AgentStatus


async def test_swarm_display():
    """Test the swarm monitor display"""
    print("Testing swarm monitor display...")
    print("You should see a clean grid display that updates in place.")
    print("Starting in 2 seconds...\n")
    await asyncio.sleep(2)
    
    # Create monitor
    monitor = SwarmMonitorEnhanced(use_alternate_screen=True)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Register some fake agents
    for i in range(3):
        monitor.register_agent(
            agent_id=f"agent_{i}",
            agent_name=f"Agent {i}",
            agent_type="test_agent",
            role=f"role_{i}"
        )
        await asyncio.sleep(0.5)
    
    # Update their statuses
    for i in range(3):
        monitor.update_agent_status(f"agent_{i}", AgentStatus.EXECUTING)
        monitor.update_agent_activity(f"agent_{i}", f"Processing task {i+1}")
        await asyncio.sleep(1)
    
    # Show some progress
    for j in range(5):
        for i in range(3):
            monitor.update_agent_activity(f"agent_{i}", f"Step {j+1} of 5")
        await asyncio.sleep(1)
    
    # Complete agents
    for i in range(3):
        monitor.update_agent_status(f"agent_{i}", AgentStatus.COMPLETED)
        await asyncio.sleep(0.5)
    
    # Stop monitoring
    await monitor.stop_monitoring()
    
    print("\nTest complete! Did you see a clean grid display?")


if __name__ == "__main__":
    asyncio.run(test_swarm_display())