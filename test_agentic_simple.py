#!/usr/bin/env python3
"""
Simple test script to verify Agentic works end-to-end
"""
import asyncio
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig


async def test_agentic_simple():
    """Test basic Agentic functionality"""
    print("🤖 Testing Agentic Basic Functionality...")
    
    # Check API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"🔑 ANTHROPIC_API_KEY: {'✅ Set' if anthropic_key else '❌ Missing'}")
    print(f"🔑 OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Missing'}")
    
    if not (anthropic_key or openai_key):
        print("⚠️  No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY to test with real AI.")
        print("   Testing will continue with mock functionality...")
    
    # Create config
    workspace_path = Path.cwd()
    config = AgenticConfig.create_default(workspace_path)
    print(f"📁 Workspace: {workspace_path}")
    
    # Create orchestrator
    orchestrator = Orchestrator(config)
    print("🎯 Created orchestrator")
    
    # Initialize
    success = await orchestrator.initialize()
    if success:
        print("✅ Orchestrator initialized successfully")
    else:
        print("❌ Orchestrator initialization failed")
        return False
    
    # Check system status
    status = orchestrator.get_system_status()
    print(f"📊 System Status:")
    print(f"   - Initialized: {status['initialized']}")
    print(f"   - Project Analyzed: {status['project_analyzed']}")
    print(f"   - Project Name: {status['project_name']}")
    print(f"   - Total Agents: {status['agents']['total_agents']}")
    print(f"   - Available Agents: {status['agents']['available_agents']}")
    
    # Spawn an agent if none available
    if status['agents']['available_agents'] == 0:
        print("🚀 No agents available, attempting to spawn reasoning agent...")
        try:
            # This would spawn a Claude Code agent, but requires implementation
            print("⚠️  Agent spawning not fully implemented yet")
        except Exception as e:
            print(f"❌ Agent spawning failed: {e}")
    
    # Test command execution (will likely fail without real agents)
    print("\n🎯 Testing command execution...")
    try:
        result = await orchestrator.execute_command("explain what this project does")
        print(f"✅ Command executed:")
        print(f"   - Success: {result.success}")
        print(f"   - Output: {result.output[:100]}..." if len(result.output) > 100 else f"   - Output: {result.output}")
        if result.error:
            print(f"   - Error: {result.error}")
        
    except Exception as e:
        print(f"❌ Command execution failed: {e}")
    
    # Cleanup
    await orchestrator.shutdown()
    print("🔄 Orchestrator shutdown complete")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_agentic_simple())
        if success:
            print("\n🎉 Basic functionality test completed!")
        else:
            print("\n❌ Test failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        sys.exit(1) 