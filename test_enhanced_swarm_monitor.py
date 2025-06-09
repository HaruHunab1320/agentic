#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced swarm monitor with grid layout
and proper task-based progress tracking
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig
from agentic.models.task import Task
from agentic.core.coordination_engine import ExecutionPlan
from datetime import datetime


async def run_multi_agent_demo():
    """Run a multi-agent task to demonstrate the enhanced swarm monitor"""
    print("🚀 Starting Enhanced Swarm Monitor Demo...")
    
    # Load configuration
    config = AgenticConfig.load_or_create(Path.cwd())
    
    # Initialize orchestrator
    orchestrator = Orchestrator(config)
    
    # Initialize
    success = await orchestrator.initialize()
    if not success:
        print("❌ Failed to initialize orchestrator")
        return
    
    print("✅ Orchestrator initialized")
    
    # Create a multi-agent task that will show the grid monitor
    commands = [
        "create a React component for a user dashboard with charts",
        "build the backend API endpoints for user analytics data", 
        "write comprehensive tests for the dashboard component",
        "implement the self-learning ML model for user behavior prediction",
        "create QA test scenarios for the entire feature",
        "set up the production deployment pipeline"
    ]
    
    # Build tasks with proper agent hints
    tasks = []
    
    # Task 1: Frontend dashboard (Architect)
    task1 = Task(
        id="task_001",
        command=commands[0],
        task_type="implement",
        complexity_score=0.7,
        affected_areas=["frontend", "ui", "components"],
        coordination_context={"role": "architect"}
    )
    task1.agent_type_hint = "claude_code"
    tasks.append(task1)
    
    # Task 2: Backend API (Backend Developer)
    task2 = Task(
        id="task_002", 
        command=commands[1],
        task_type="implement",
        complexity_score=0.6,
        affected_areas=["backend", "api"],
        coordination_context={"role": "backend_developer"}
    )
    task2.agent_type_hint = "aider_backend"
    tasks.append(task2)
    
    # Task 3: Frontend tests (Frontend Developer)
    task3 = Task(
        id="task_003",
        command=commands[2],
        task_type="test",
        complexity_score=0.5,
        affected_areas=["frontend", "testing"],
        coordination_context={"role": "frontend_developer"}
    )
    task3.agent_type_hint = "aider_frontend"
    tasks.append(task3)
    
    # Task 4: ML implementation (ML Engineer)
    task4 = Task(
        id="task_004",
        command=commands[3],
        task_type="implement",
        complexity_score=0.8,
        affected_areas=["ml", "backend"],
        coordination_context={"role": "ml_engineer"}
    )
    task4.agent_type_hint = "claude_code"
    tasks.append(task4)
    
    # Task 5: QA scenarios (QA Engineer)
    task5 = Task(
        id="task_005",
        command=commands[4],
        task_type="test",
        complexity_score=0.5,
        affected_areas=["testing", "qa"],
        coordination_context={"role": "qa_engineer"}
    )
    task5.agent_type_hint = "claude_code"
    tasks.append(task5)
    
    # Task 6: DevOps pipeline (DevOps Engineer)
    task6 = Task(
        id="task_006",
        command=commands[5],
        task_type="implement",
        complexity_score=0.6,
        affected_areas=["devops", "deployment"],
        coordination_context={"role": "devops_engineer"}
    )
    task6.agent_type_hint = "aider_backend"
    tasks.append(task6)
    
    # Create execution plan
    plan = ExecutionPlan(
        id=f"demo_plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        tasks=tasks,
        parallel_groups=None,  # Let coordination engine figure it out
        estimated_duration=600
    )
    
    print(f"\n📋 Created execution plan with {len(tasks)} tasks")
    print("🎯 Tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"   {i}. {task.command[:60]}...")
    
    print("\n🚀 Starting multi-agent execution...")
    print("👀 Watch the swarm monitor for real-time progress!\n")
    
    # Execute the plan
    try:
        result = await orchestrator.execute_execution_plan(plan)
        
        print(f"\n✅ Execution completed!")
        print(f"   Status: {result.status}")
        print(f"   Completed tasks: {len(result.completed_tasks)}")
        print(f"   Failed tasks: {len(result.failed_tasks)}")
        print(f"   Total duration: {result.total_duration:.1f}s")
        
        # Show task results
        if result.task_results:
            print("\n📊 Task Results:")
            for task_id, task_result in result.task_results.items():
                status_icon = "✅" if task_result.status == "completed" else "❌"
                print(f"   {status_icon} {task_id}: {task_result.status}")
                if task_result.error:
                    print(f"      Error: {task_result.error[:100]}...")
                    
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Shutdown
    await orchestrator.shutdown()
    print("\n👋 Demo completed!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_multi_agent_demo())