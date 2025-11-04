#!/usr/bin/env python3
"""Test script for Gemini agent integration."""

import asyncio
import os
from pathlib import Path

from agentic.agents.gemini_agent import GeminiAgent
from agentic.models.agent import AgentConfig, AgentType, Task, TaskType


async def test_gemini_agent():
    """Test basic Gemini agent functionality."""
    
    # Configure the agent
    config = AgentConfig(
        agent_type=AgentType.GEMINI,
        name="test-gemini",
        workspace_path=Path.cwd(),
        focus_areas=["multimodal", "research", "documentation"],
        ai_model_config={"model": "gemini-2.5-pro"}
    )
    
    # Create agent instance
    agent = GeminiAgent(config)
    
    print("Starting Gemini agent...")
    started = await agent.start()
    
    if not started:
        print("Failed to start Gemini agent")
        print("Make sure you have:")
        print("1. Node.js 18+ installed")
        print("2. Run: npm install -g @google/gemini-cli")
        print("3. Set GEMINI_API_KEY environment variable")
        return
    
    print("Gemini agent started successfully!")
    
    # Test capabilities
    capabilities = agent.get_capabilities()
    print(f"\nAgent capabilities:")
    print(f"- Max context tokens: {capabilities.max_context_tokens:,}")
    print(f"- Multimodal: {capabilities.multimodal_capability}")
    print(f"- Web browsing: {capabilities.web_browsing_capability}")
    print(f"- Specializations: {', '.join(capabilities.specializations)}")
    
    # Create a test task
    task = Task(
        type=TaskType.FEATURE,
        description="Research the latest best practices for Python async programming",
        prompt="Please research and summarize the latest best practices for Python async programming in 2024. Use Google Search to find current information.",
        context={"use_search": True}
    )
    
    print(f"\nExecuting task: {task.description}")
    result = await agent.execute_task(task)
    
    print(f"\nTask result:")
    print(f"- Status: {result.status}")
    print(f"- Execution time: {result.execution_time:.2f}s")
    if result.output:
        print(f"- Output preview: {result.output[:200]}...")
    if result.error:
        print(f"- Error: {result.error}")
    
    # Test multimodal capability
    print("\nTesting multimodal capability...")
    multimodal_task = Task(
        type=TaskType.ANALYSIS,
        description="Analyze a system architecture diagram",
        prompt="If you had an architecture diagram image, describe what capabilities you would need to analyze it effectively.",
        context={}
    )
    
    multimodal_result = await agent.execute_task(multimodal_task)
    print(f"Multimodal test status: {multimodal_result.status}")
    
    # Stop the agent
    print("\nStopping Gemini agent...")
    await agent.stop()
    print("Agent stopped.")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set. You'll need to authenticate interactively.")
    
    asyncio.run(test_gemini_agent())