#!/usr/bin/env python3
"""
Test script to demonstrate Aider's file exploration capabilities
with different invocation methods.
"""

import subprocess
import os
import sys

def test_aider_with_exit():
    """Test Aider with --exit flag (current approach)"""
    print("\n=== Testing Aider WITH --exit flag ===")
    
    cmd = [
        "aider",
        "--yes-always",
        "--no-git",
        "--no-auto-commits",
        "--model", "gemini/gemini-1.5-flash-latest",
        "--message", "Please examine the project structure and list all Python files in the src/agentic/agents directory. Show me what files exist there.",
        "--exit"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"\nReturn code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    

def test_aider_without_exit():
    """Test Aider without --exit flag but with input"""
    print("\n\n=== Testing Aider WITHOUT --exit flag (interactive simulation) ===")
    
    cmd = [
        "aider", 
        "--yes-always",
        "--no-git",
        "--no-auto-commits",
        "--model", "gemini/gemini-1.5-flash-latest",
    ]
    
    # Simulate interactive usage by providing commands via stdin
    input_commands = """Please examine the project structure and list all Python files in the src/agentic/agents directory. Show me what files exist there.
/exit
"""
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Input: {repr(input_commands)}")
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=input_commands)
    
    print(f"\nReturn code: {process.returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")


def test_aider_with_read_flag():
    """Test Aider with --read flag to give it file context"""
    print("\n\n=== Testing Aider WITH --read flag ===")
    
    cmd = [
        "aider",
        "--yes-always", 
        "--no-git",
        "--no-auto-commits",
        "--model", "gemini/gemini-1.5-flash-latest",
        "--read", "src/agentic/agents/*.py",  # Give it read access to files
        "--message", "Please examine the Python files I've given you read access to and summarize what agents are available.",
        "--exit"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"\nReturn code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")


def main():
    """Run all tests"""
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Ensure we have GEMINI_API_KEY
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Please set it to run these tests.")
        sys.exit(1)
    
    print("Testing Aider's file exploration capabilities...")
    print("Working directory:", os.getcwd())
    
    # Test 1: With --exit (current approach)
    test_aider_with_exit()
    
    # Test 2: Without --exit (interactive)
    # Commented out as it may hang
    # test_aider_without_exit()
    
    # Test 3: With --read flag
    test_aider_with_read_flag()


if __name__ == "__main__":
    main()