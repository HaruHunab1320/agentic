#!/usr/bin/env python3
"""
Direct test of the enhanced swarm monitor to show grid layout
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agentic.core.swarm_monitor_enhanced import SwarmMonitorEnhanced, AgentStatus


async def demo_enhanced_monitor():
    """Demo the enhanced swarm monitor with simulated agent activities"""
    
    monitor = SwarmMonitorEnhanced(use_alternate_screen=True)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Register 6 agents like in the reference image
    agents = [
        ("agent_1", "Architect", "claude_code", "architect"),
        ("agent_2", "Backend Developer", "aider_backend", "backend_developer"),
        ("agent_3", "Frontend Developer", "aider_frontend", "frontend_developer"),
        ("agent_4", "ML Engineer", "claude_code", "ml_engineer"),
        ("agent_5", "QA Engineer", "claude_code", "qa_engineer"),
        ("agent_6", "DevOps Engineer", "aider_backend", "devops_engineer")
    ]
    
    for agent_id, name, agent_type, role in agents:
        monitor.register_agent(agent_id, name, agent_type, role)
    
    # Set task analysis
    monitor.update_task_analysis(
        total_files=10,
        complexity=0.70,
        suggested_agents=["aider_frontend", "aider_backend", "claude_code"]
    )
    
    # Simulate agent tasks
    agent_tasks = [
        ("agent_1", [("t1", "Design the complete architecture"), ("t2", "Create technical specs")]),
        ("agent_2", [("t3", "Implement backend API"), ("t4", "Set up database schema")]),
        ("agent_3", [("t5", "Build monitoring dashboard"), ("t6", "Create UI components")]),
        ("agent_4", [("t7", "Implement ML model"), ("t8", "Create training pipeline")]),
        ("agent_5", [("t9", "Create test scenarios"), ("t10", "Write integration tests")]),
        ("agent_6", [("t11", "Set up CI/CD"), ("t12", "Configure deployment")])
    ]
    
    # Set tasks for each agent
    for agent_id, tasks in agent_tasks:
        monitor.set_agent_tasks(agent_id, tasks)
    
    # Simulate execution
    print("Starting simulated agent execution...")
    print("Press Ctrl+C to stop")
    
    try:
        # Phase 1: Initialize all agents
        for agent_id, _, _, _ in agents:
            monitor.update_agent_status(agent_id, AgentStatus.INITIALIZING)
        await asyncio.sleep(2)
        
        # Phase 2: Start first tasks
        for i, (agent_id, tasks) in enumerate(agent_tasks):
            monitor.update_agent_status(agent_id, AgentStatus.EXECUTING)
            if tasks:
                monitor.start_agent_task(agent_id, tasks[0][0], tasks[0][1])
                
                # Simulate different activities
                if i == 0:  # Architect
                    monitor.update_agent_activity(agent_id, "Analyzing project structure...")
                elif i == 1:  # Backend
                    monitor.update_agent_activity(agent_id, "Setting up Express.js server...")
                elif i == 2:  # Frontend  
                    monitor.update_agent_activity(agent_id, "Creating React components...")
                elif i == 3:  # ML
                    monitor.update_agent_activity(agent_id, "Loading training data...")
                elif i == 4:  # QA
                    monitor.update_agent_activity(agent_id, "Writing test specifications...")
                elif i == 5:  # DevOps
                    monitor.update_agent_activity(agent_id, "Configuring Docker containers...")
        
        await asyncio.sleep(3)
        
        # Phase 3: Update activities
        activities = [
            ("agent_1", "Creating ARCHITECTURE.md file..."),
            ("agent_2", "Implementing user endpoints..."),
            ("agent_3", "Building chart components..."),
            ("agent_4", "Training neural network..."),
            ("agent_5", "Running unit tests..."),
            ("agent_6", "Building Docker images...")
        ]
        
        for agent_id, activity in activities:
            monitor.update_agent_activity(agent_id, activity)
        
        await asyncio.sleep(3)
        
        # Phase 4: Complete first tasks, start second
        for i, (agent_id, tasks) in enumerate(agent_tasks):
            if tasks:
                # Complete first task
                monitor.complete_agent_task(agent_id, tasks[0][0], success=True)
                
                # Start second task if available
                if len(tasks) > 1:
                    monitor.start_agent_task(agent_id, tasks[1][0], tasks[1][1])
                    
                    # Update activity
                    if i == 0:
                        monitor.update_agent_activity(agent_id, "Writing API specifications...")
                    elif i == 1:
                        monitor.update_agent_activity(agent_id, "Creating database migrations...")
                    elif i == 2:
                        monitor.update_agent_activity(agent_id, "Styling components...")
                    elif i == 3:
                        monitor.update_agent_activity(agent_id, "Optimizing model parameters...")
                    elif i == 4:
                        monitor.update_agent_activity(agent_id, "Running integration tests...")
                    elif i == 5:
                        monitor.update_agent_activity(agent_id, "Setting up GitHub Actions...")
        
        await asyncio.sleep(3)
        
        # Phase 5: More progress
        for agent_id, _ in agents:
            # Add some file modifications
            monitor.add_file_modified(agent_id, f"src/{agent_id}_output.js")
        
        await asyncio.sleep(3)
        
        # Phase 6: Complete all tasks
        for agent_id, tasks in agent_tasks:
            if len(tasks) > 1:
                monitor.complete_agent_task(agent_id, tasks[1][0], success=True)
            monitor.update_agent_status(agent_id, AgentStatus.COMPLETED)
            monitor.update_agent_activity(agent_id, "All tasks completed successfully!")
        
        await asyncio.sleep(5)
        
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
    
    # Stop monitoring
    await monitor.stop_monitoring()
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(demo_enhanced_monitor())