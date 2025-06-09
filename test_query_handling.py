#!/usr/bin/env python3
"""Test query handling with various question types"""

import asyncio
import os
from pathlib import Path

# Set automated mode
os.environ['AGENTIC_AUTOMATED_MODE'] = 'true'

# Add the src directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.query_analyzer import QueryAnalyzer, format_query_analysis


async def test_query_analysis():
    """Test the query analyzer with various types of queries"""
    print("=== Testing Query Analysis ===\n")
    
    analyzer = QueryAnalyzer()
    
    # Test queries
    queries = [
        # Simple questions - should use single agent
        "What does the orchestrator.py file do?",
        "Where is authentication handled in the system?",
        "How does the shared memory work?",
        
        # Complex analysis - should use coordinated approach
        "How does the authentication flow work across the entire system?",
        "Analyze the architecture and design patterns used in this codebase",
        "What are the performance bottlenecks in the application?",
        
        # Implementation tasks
        "Create a simple React button component",
        "Build a complete REST API with authentication",
        
        # Refactoring tasks
        "Refactor the user service to use async/await",
        "Clean up and optimize the database queries"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 60)
        
        analysis = await analyzer.analyze_query(query)
        print(format_query_analysis(analysis))
        print()


async def test_live_queries():
    """Test actual query execution through chat interface"""
    print("\n\n=== Testing Live Query Execution ===\n")
    
    from agentic.core.chat_interface import ChatInterface
    
    # Initialize chat interface
    chat = ChatInterface(Path.cwd(), debug=False)
    
    if not await chat.initialize():
        print("Failed to initialize chat interface")
        return
    
    # Test different query types
    test_queries = [
        # Simple question - should use single Claude agent
        ("Simple question", "What is the purpose of the coordination engine?"),
        
        # Complex analysis - should trigger multi-agent coordination
        ("Complex analysis", "How does the multi-agent coordination system work across the architecture?"),
        
        # Simple implementation - single agent
        ("Simple implementation", "Create a hello.txt file with 'Hello World' content"),
        
        # Complex implementation - multi-agent
        ("Complex implementation", "@all Create a simple todo list React component with tests")
    ]
    
    for query_type, query in test_queries:
        print(f"\n\n{'='*60}")
        print(f"Testing: {query_type}")
        print(f"Query: {query}")
        print('='*60)
        
        await chat._execute_natural_command(query)
        
        # Brief pause between tests
        await asyncio.sleep(2)


async def main():
    """Run all tests"""
    # First test the analyzer itself
    await test_query_analysis()
    
    # Then test live execution
    response = input("\nRun live execution tests? (y/n): ")
    if response.lower() == 'y':
        await test_live_queries()
    
    print("\n\nTest complete!")


if __name__ == "__main__":
    asyncio.run(main())