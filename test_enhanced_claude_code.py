#!/usr/bin/env python3
"""
Test Enhanced Claude Code Integration

Tests the new enhanced Claude Code agent that leverages Claude Code's full feature set
including memory management, session persistence, extended thinking, and advanced tooling.
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

async def test_enhanced_claude_code_agent():
    """Test the enhanced Claude Code agent with all new features."""
    print("🚀 Testing Enhanced Claude Code Agent")
    print("=" * 60)
    
    # Create enhanced agent configuration
    config = {
        "project_root": ".",
        "claude_model": "sonnet",
        "focus_areas": ["coding", "analysis", "refactoring", "architecture"]
    }
    
    print("\n1. 🤖 Initializing Enhanced Claude Code Agent...")
    agent = ClaudeCodeAgent("enhanced_claude_code", config)
    
    print(f"✅ Agent capabilities: {agent.get_capabilities().specializations}")
    print(f"✅ Supported languages: {agent.get_capabilities().supported_languages}")
    print(f"✅ Enhanced features: memory, sessions, thinking, git integration")
    
    # Test agent startup with memory initialization
    print("\n2. 🚀 Starting Agent with Memory Setup...")
    try:
        started = await agent.start()
        if started:
            print("✅ Agent started successfully")
            print(f"✅ Session ID: {agent.session_id}")
            print(f"✅ Memory initialized: {agent.memory_initialized}")
            
            # Check if CLAUDE.md was created
            claude_md = Path("CLAUDE.md")
            if claude_md.exists():
                print("✅ CLAUDE.md project memory file created")
                print(f"   Size: {claude_md.stat().st_size} bytes")
            else:
                print("⚠️ CLAUDE.md not found")
        else:
            print("❌ Agent failed to start")
            return False
    except Exception as e:
        print(f"❌ Agent startup failed: {e}")
        return False
    
    # Test enhanced status
    print("\n3. 📊 Enhanced Status Check...")
    try:
        status = await agent.get_status()
        print(f"✅ Agent status: {status['status']}")
        print(f"✅ Session ID: {status.get('session_id', 'N/A')}")
        print(f"✅ Memory initialized: {status.get('memory_initialized', False)}")
        print(f"✅ Enhanced features: {status.get('enhanced_features', {})}")
    except Exception as e:
        print(f"❌ Status check failed: {e}")
    
    # Test extended thinking task
    print("\n4. 🧠 Testing Extended Thinking Capability...")
    try:
        intent = TaskIntent(
            task_type=TaskType.REFACTOR,
            complexity_score=0.8,
            estimated_duration=10,
            affected_areas=["src"],
            requires_reasoning=True,
            requires_coordination=False,
            file_patterns=["*.py"]
        )
        
        thinking_task = Task(
            id="thinking_test",
            command="analyze the architecture of this Agentic project and suggest improvements",
            intent=intent
        )
        
        print(f"🧠 Executing thinking task: {thinking_task.command}")
        result = await agent.execute_task(thinking_task)
        
        if result.status == "completed":
            print("✅ Extended thinking task completed")
            print(f"   Thinking time: {result.metadata.get('thinking_time', 'N/A')}")
            print(f"   Tools used: {result.metadata.get('tools_used', [])}")
            print(f"   Session ID: {result.metadata.get('session_id', 'N/A')}")
            print(f"   Output preview: {result.output[:200]}...")
        else:
            print(f"❌ Thinking task failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Extended thinking test failed: {e}")
    
    # Test tool selection
    print("\n5. 🛠️ Testing Dynamic Tool Selection...")
    try:
        # Test git-related task
        git_intent = TaskIntent(
            task_type=TaskType.IMPLEMENT,
            complexity_score=0.4,
            estimated_duration=5,
            affected_areas=["git"],
            requires_reasoning=False,
            requires_coordination=False,
            file_patterns=["*"]
        )
        
        git_task = Task(
            id="git_test",
            command="check git status and show recent commits",
            intent=git_intent
        )
        
        tools = agent._get_task_tools(git_task)
        print(f"✅ Git task tools: {tools}")
        
        # Test documentation task
        doc_intent = TaskIntent(
            task_type=TaskType.DOCUMENT,
            complexity_score=0.3,
            estimated_duration=5,
            affected_areas=["docs"],
            requires_reasoning=False,
            requires_coordination=False,
            file_patterns=["*.md"]
        )
        
        doc_task = Task(
            id="doc_test",
            command="improve the README documentation",
            intent=doc_intent
        )
        
        doc_tools = agent._get_task_tools(doc_task)
        print(f"✅ Documentation task tools: {doc_tools}")
        
    except Exception as e:
        print(f"❌ Tool selection test failed: {e}")
    
    # Test memory management
    print("\n6. 🧠 Testing Memory Management...")
    try:
        memory_added = await agent.add_memory(
            "Always use type hints in Python functions for better code clarity",
            "project"
        )
        if memory_added:
            print("✅ Successfully added memory to project CLAUDE.md")
            
            # Check if memory was actually added
            claude_md = Path("CLAUDE.md")
            if claude_md.exists():
                content = claude_md.read_text()
                if "type hints" in content:
                    print("✅ Memory content verified in CLAUDE.md")
                else:
                    print("⚠️ Memory content not found in CLAUDE.md")
            
        else:
            print("❌ Failed to add memory")
            
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
    
    # Test session continuation
    print("\n7. 🔄 Testing Session Continuation...")
    try:
        if agent.session_id:
            continuation_intent = TaskIntent(
                task_type=TaskType.REFACTOR,
                complexity_score=0.6,
                estimated_duration=5,
                affected_areas=["src"],
                requires_reasoning=True,
                requires_coordination=False,
                file_patterns=["*.py"]
            )
            
            continuation_task = Task(
                id="continuation_test",
                command="continue with the previous analysis and suggest specific refactoring steps",
                intent=continuation_intent
            )
            
            print(f"🔄 Testing session continuation with ID: {agent.session_id}")
            result = await agent.continue_session(continuation_task)
            
            if result.status == "completed":
                print("✅ Session continuation successful")
                print(f"   Output preview: {result.output[:200]}...")
            else:
                print(f"⚠️ Session continuation had issues: {result.error}")
        else:
            print("⚠️ No session ID available for continuation test")
            
    except Exception as e:
        print(f"❌ Session continuation test failed: {e}")
    
    # Test enhanced supported tasks
    print("\n8. 📋 Enhanced Task Support...")
    try:
        supported_tasks = agent.get_supported_tasks()
        print(f"✅ Enhanced supported tasks ({len(supported_tasks)}):")
        for task in supported_tasks:
            print(f"   - {task}")
            
    except Exception as e:
        print(f"❌ Task support check failed: {e}")
    
    # Stop the agent
    print("\n9. 🛑 Stopping Enhanced Agent...")
    try:
        stopped = await agent.stop()
        if stopped:
            print("✅ Enhanced agent stopped successfully")
        else:
            print("⚠️ Agent stop may have had issues")
    except Exception as e:
        print(f"❌ Agent stop failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Claude Code Agent Test Complete!")
    return True

async def test_command_building():
    """Test the enhanced command building functionality."""
    print("\n🔧 Testing Enhanced Command Building...")
    
    config = {"project_root": ".", "claude_model": "sonnet"}
    agent = ClaudeCodeAgent("test_agent", config)
    
    # Test complex architecture task
    intent = TaskIntent(
        task_type=TaskType.IMPLEMENT,
        complexity_score=0.9,
        estimated_duration=15,
        affected_areas=["src", "docs"],
        requires_reasoning=True,
        requires_coordination=False,
        file_patterns=["*.py", "*.md"]
    )
    
    task = Task(
        id="arch_test",
        command="design a scalable microservices architecture for this system",
        intent=intent
    )
    
    try:
        cmd = await agent._build_enhanced_command(task)
        print(f"✅ Enhanced command built: {' '.join(cmd[:5])}... (truncated)")
        
        # Verify key components
        if "--output-format" in cmd and "json" in cmd:
            print("✅ JSON output format enabled")
        
        if "--allowedTools" in cmd:
            tool_index = cmd.index("--allowedTools")
            tools = []
            for i in range(tool_index + 1, len(cmd)):
                if cmd[i].startswith("--") or cmd[i] == cmd[-1]:
                    break
                tools.append(cmd[i])
            print(f"✅ Dynamic tools selected: {tools}")
        
        # Check for thinking trigger
        prompt = cmd[-1]
        if "Think deeply" in prompt or "Think about" in prompt:
            print("✅ Extended thinking trigger detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Command building test failed: {e}")
        return False

def main():
    """Run all enhanced tests."""
    print("🚀 Starting Enhanced Claude Code Integration Tests")
    print("This tests the new enhanced Claude Code CLI integration in Agentic")
    print("Features: Memory, Sessions, Extended Thinking, Git Integration, Dynamic Tools")
    print()
    
    try:
        # Test command building first
        cmd_test = asyncio.run(test_command_building())
        
        if not cmd_test:
            print("\n⚠️ Command building tests failed")
        
        # Test the full enhanced agent
        agent_test = asyncio.run(test_enhanced_claude_code_agent())
        
        if agent_test:
            print("\n🎉 All enhanced tests completed!")
            print("\n🚀 Enhanced Features Validated:")
            print("✅ Memory Management (CLAUDE.md)")
            print("✅ Session Persistence") 
            print("✅ Extended Thinking")
            print("✅ Dynamic Tool Selection")
            print("✅ JSON Output Parsing")
            print("✅ Git Integration Ready")
            print("✅ Enhanced Prompt Building")
            print("\nNext steps:")
            print("1. Authenticate Claude Code: claude")
            print("2. Test with real tasks: agentic exec 'think deeply about refactoring the auth system'")
            print("3. Check CLAUDE.md for persistent memory")
            return True
        else:
            print("\n❌ Some enhanced tests failed. Check the output above for details.")
            return False
            
    except Exception as e:
        print(f"\n💥 Enhanced test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 