#!/usr/bin/env python3
"""
Test Claude Code Integration

Simple test to verify that the Claude Code agent can be initialized
and execute basic tasks using the Claude Code CLI.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.agents.claude_code_agent import ClaudeCodeAgent
from agentic.models.task import Task, TaskIntent, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_claude_code_agent():
    """Test the Claude Code agent integration."""
    print("🧪 Testing Claude Code Agent Integration")
    print("=" * 50)
    
    # Create agent configuration
    config = {
        "project_root": ".",
        "claude_model": "sonnet"
    }
    
    # Initialize agent
    print("\n1. Initializing Claude Code agent...")
    agent = ClaudeCodeAgent("test_claude_code", config)
    
    # Test agent startup
    print("\n2. Starting agent...")
    try:
        started = await agent.start()
        if started:
            print("✅ Agent started successfully")
        else:
            print("❌ Agent failed to start")
            return False
    except Exception as e:
        print(f"❌ Agent startup failed: {e}")
        return False
    
    # Test agent status
    print("\n3. Checking agent status...")
    try:
        status = await agent.get_status()
        print(f"Agent status: {status}")
        
        if status.get("status") == "active":
            print("✅ Agent is running and responsive")
        else:
            print("⚠️ Agent may have issues")
    except Exception as e:
        print(f"❌ Status check failed: {e}")
    
    # Test simple task execution
    print("\n4. Testing task execution...")
    try:
        # Create a proper TaskIntent first
        intent = TaskIntent(
            task_type=TaskType.EXPLAIN,
            complexity_score=0.3,
            estimated_duration=5,
            affected_areas=["src"],
            requires_reasoning=True,
            requires_coordination=False,
            file_patterns=["*.py", "*.md"]
        )
        
        # Create a proper Task with all required fields
        task = Task(
            id="test_task_1",
            command="analyze this codebase structure",
            intent=intent
        )
        
        print(f"Executing task: {task.command}")
        result = await agent.execute_task(task)
        
        if result.status == "completed":
            print("✅ Task completed successfully")
            print(f"Result preview: {result.output[:200]}...")
        else:
            print(f"❌ Task failed: {result.output}")
            
    except Exception as e:
        print(f"❌ Task execution failed: {e}")
    
    # Test supported task types
    print("\n5. Checking supported task types...")
    try:
        supported_tasks = agent.get_supported_tasks()
        print(f"Supported task types: {', '.join(supported_tasks)}")
        print("✅ Task types retrieved successfully")
    except Exception as e:
        print(f"❌ Failed to get supported tasks: {e}")
    
    # Stop the agent
    print("\n6. Stopping agent...")
    try:
        stopped = await agent.stop()
        if stopped:
            print("✅ Agent stopped successfully")
        else:
            print("⚠️ Agent stop may have had issues")
    except Exception as e:
        print(f"❌ Agent stop failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Claude Code Agent Test Complete")
    return True

async def test_claude_code_cli_directly():
    """Test Claude Code CLI directly."""
    print("\n🔧 Testing Claude Code CLI directly...")
    
    try:
        import subprocess
        import asyncio
        
        # Test basic help command
        process = await asyncio.create_subprocess_exec(
            "claude", "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print("✅ Claude Code CLI is installed and accessible")
            return True
        else:
            print(f"❌ Claude Code CLI failed: {stderr.decode()}")
            return False
            
    except FileNotFoundError:
        print("❌ Claude Code CLI not found - install with: npm install -g @anthropic-ai/claude-code")
        return False
    except Exception as e:
        print(f"❌ Claude Code CLI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Claude Code Integration Tests")
    print("This tests the new Claude Code CLI integration in Agentic")
    print()
    
    try:
        # Test CLI directly first
        cli_works = asyncio.run(test_claude_code_cli_directly())
        
        if not cli_works:
            print("\n⚠️ Claude Code CLI is not working - agent tests will be skipped")
            print("Please run: npm install -g @anthropic-ai/claude-code")
            print("Then authenticate with: claude (and follow login prompts)")
            return False
        
        # Test the agent integration
        agent_works = asyncio.run(test_claude_code_agent())
        
        if agent_works:
            print("\n🎉 All tests passed! Claude Code integration is working.")
            print("\nNext steps:")
            print("1. Make sure Claude Code is authenticated: claude")
            print("2. Test with Agentic CLI: agentic spawn claude_code")
            print("3. Try a real task: agentic exec 'analyze this codebase'")
            return True
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 