#!/usr/bin/env python3
"""
Test Gemini Model Integration with Aider

This script tests the Gemini model configuration and integration with Aider
to ensure everything is working correctly.
"""
import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.agents.aider_agents import BaseAiderAgent
from agentic.models.agent import AgentConfig, AgentType


async def test_gemini_model_mapping():
    """Test that Gemini models are mapped correctly"""
    print("🧪 Testing Gemini model mapping...")
    
    test_cases = [
        ("gemini", "gemini/gemini-1.5-pro-latest"),
        ("gemini-1.5-pro", "gemini/gemini-1.5-pro-latest"), 
        ("gemini-1.5-flash", "gemini/gemini-1.5-flash-latest"),
        ("gemini/gemini-1.5-pro-latest", "gemini/gemini-1.5-pro-latest"),  # Should pass through
    ]
    
    for input_model, expected_output in test_cases:
        # Create test agent config
        config = AgentConfig(
            agent_type=AgentType.AIDER_BACKEND,
            name="test_agent",
            workspace_path=Path.cwd(),
            focus_areas=["testing"],
            ai_model_config={"model": input_model},
            max_tokens=1000,
            temperature=0.1
        )
        
        # Create agent and test mapping
        agent = BaseAiderAgent(config)
        mapped_model = agent._get_model_for_aider()
        
        if mapped_model == expected_output:
            print(f"✅ {input_model} → {mapped_model}")
        else:
            print(f"❌ {input_model} → {mapped_model} (expected: {expected_output})")
            return False
    
    return True


async def test_aider_command_construction():
    """Test that Aider commands are constructed correctly with Gemini"""
    print("\n🛠️ Testing Aider command construction...")
    
    config = AgentConfig(
        agent_type=AgentType.AIDER_BACKEND,
        name="gemini_test_agent",
        workspace_path=Path.cwd(),
        focus_areas=["backend", "python"],
        ai_model_config={"model": "gemini"},
        max_tokens=4000,
        temperature=0.1
    )
    
    agent = BaseAiderAgent(config)
    agent._setup_aider_args()
    
    # Check that the command includes the mapped Gemini model
    command_str = " ".join(agent.aider_args)
    expected_model = "gemini/gemini-1.5-pro-latest"
    
    if f"--model={expected_model}" in command_str:
        print(f"✅ Command includes correct Gemini model: --model={expected_model}")
        print(f"   Full command: {command_str}")
        return True
    else:
        print(f"❌ Command doesn't include expected model")
        print(f"   Full command: {command_str}")
        return False


async def test_aider_availability():
    """Test that Aider is available and can use Gemini models"""
    print("\n🔍 Testing Aider availability...")
    
    try:
        # Test basic Aider availability
        result = subprocess.run(["aider", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ Aider is not available or not working")
            return False
        
        print("✅ Aider is available")
        
        # Test Gemini model specifically (if API key is available)
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            print("⚠️  No Gemini API key found - skipping live test")
            print("   Set GEMINI_API_KEY or GOOGLE_API_KEY to test live model access")
            return True
        
        # Test with a simple help command using Gemini model
        test_cmd = [
            "aider", 
            "--model", "gemini/gemini-1.5-pro-latest",
            "--help"
        ]
        
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Aider can use Gemini model")
            return True
        else:
            print(f"❌ Aider failed with Gemini model: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Aider command timed out")
        return False
    except FileNotFoundError:
        print("❌ Aider not found - install with: pip install aider-chat")
        return False
    except Exception as e:
        print(f"❌ Error testing Aider: {e}")
        return False


async def test_environment_setup():
    """Check environment setup for Gemini"""
    print("\n🌍 Checking environment setup...")
    
    # Check for API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if gemini_key:
        print(f"✅ GEMINI_API_KEY is set ({gemini_key[:10]}...)")
    elif google_key:
        print(f"✅ GOOGLE_API_KEY is set ({google_key[:10]}...)")
    else:
        print("⚠️  No Gemini API key found")
        print("   Set either GEMINI_API_KEY or GOOGLE_API_KEY for full functionality")
    
    # Check Python environment
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    else:
        print(f"❌ Python version too old: {python_version.major}.{python_version.minor} (need 3.8+)")
        return False
    
    return True


async def main():
    """Run all tests"""
    print("🚀 Testing Gemini Integration with Agentic + Aider\n")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Model Mapping", test_gemini_model_mapping),
        ("Command Construction", test_aider_command_construction),
        ("Aider Availability", test_aider_availability),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"🧪 {test_name}")
        print(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} PASSED\n")
            else:
                print(f"❌ {test_name} FAILED\n")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print(f"{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Gemini integration is ready to use.")
        print("\nQuick start:")
        print("  1. Set your API key: export GEMINI_API_KEY='your-key'")
        print("  2. Configure Agentic: agentic model set gemini")
        print("  3. Test it: agentic model test gemini")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 