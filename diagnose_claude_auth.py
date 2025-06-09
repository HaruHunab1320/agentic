#!/usr/bin/env python3
"""Diagnose Claude Code authentication issues"""

import asyncio
import subprocess
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.agents.claude_code_agent import ClaudeCodeAgent
from agentic.models.agent import AgentConfig, AgentType


async def check_claude_cli():
    """Check if Claude CLI is installed and accessible"""
    print("1. Checking Claude CLI installation...")
    
    try:
        result = subprocess.run(["which", "claude"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Claude CLI found at: {result.stdout.strip()}")
        else:
            print("❌ Claude CLI not found in PATH")
            return False
    except Exception as e:
        print(f"❌ Error checking Claude CLI: {e}")
        return False
    
    return True


async def check_claude_version():
    """Check Claude CLI version"""
    print("\n2. Checking Claude CLI version...")
    
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Claude version: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Claude version check failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Claude version check timed out - may need authentication")
        return False
    except Exception as e:
        print(f"❌ Error checking Claude version: {e}")
        return False


async def check_claude_auth_status():
    """Check Claude authentication status"""
    print("\n3. Checking Claude authentication status...")
    
    # Import the check method
    from agentic.agents.claude_code_agent import ClaudeCodeAgent
    
    # Create a temporary agent to use its auth check
    config = AgentConfig(
        agent_type=AgentType.CLAUDE_CODE,
        name="test_agent",
        workspace_path=Path.cwd(),
        focus_areas=["test"]
    )
    
    agent = ClaudeCodeAgent(config)
    
    try:
        auth_status = await agent._check_authentication_status()
        print(f"Authentication status: {auth_status}")
        
        if auth_status == "authenticated":
            print("✅ Claude is authenticated")
            return True
        elif auth_status == "needs_browser_auth":
            print("⚠️  Claude needs browser authentication")
            print("   Run 'claude' manually to complete authentication")
            return False
        elif auth_status == "needs_setup":
            print("⚠️  Claude needs initial setup")
            print("   Run 'claude' manually to complete setup")
            return False
        else:
            print(f"❓ Unknown status: {auth_status}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking auth status: {e}")
        return False


async def test_agent_spawn():
    """Test spawning a Claude Code agent"""
    print("\n4. Testing Claude Code agent spawn...")
    
    # Set automated mode to skip interactive prompts
    os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'
    
    config = AgentConfig(
        agent_type=AgentType.CLAUDE_CODE,
        name="test_claude_agent",
        workspace_path=Path.cwd(),
        focus_areas=["test"],
        ai_model_config={"model": "sonnet"}
    )
    
    agent = ClaudeCodeAgent(config)
    
    try:
        success = await agent.start()
        if success:
            print("✅ Claude Code agent started successfully")
            await agent.stop()
            return True
        else:
            print("❌ Claude Code agent failed to start")
            return False
    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all diagnostics"""
    print("=== Claude Code Authentication Diagnostics ===\n")
    
    # Run checks
    cli_ok = await check_claude_cli()
    if not cli_ok:
        print("\n⚠️  Claude CLI not installed. Install with:")
        print("   npm install -g @anthropic-ai/claude-code")
        return
    
    version_ok = await check_claude_version()
    auth_ok = await check_claude_auth_status()
    
    if not auth_ok:
        print("\n⚠️  Claude is not authenticated. Please run:")
        print("   claude")
        print("   And complete the authentication flow")
        return
    
    spawn_ok = await test_agent_spawn()
    
    print("\n=== Summary ===")
    print(f"CLI Installed: {'✅' if cli_ok else '❌'}")
    print(f"Version Check: {'✅' if version_ok else '❌'}")
    print(f"Authenticated: {'✅' if auth_ok else '❌'}")
    print(f"Agent Spawn:   {'✅' if spawn_ok else '❌'}")
    
    if all([cli_ok, version_ok, auth_ok, spawn_ok]):
        print("\n✅ All checks passed! Claude Code should work properly.")
    else:
        print("\n❌ Some checks failed. Please address the issues above.")


if __name__ == "__main__":
    asyncio.run(main())