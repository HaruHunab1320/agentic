#!/usr/bin/env python3
"""
Test script to verify language context awareness in the Agentic system
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.core.project_indexer import ProjectIndexer
from agentic.core.language_selector import LanguageSelector
from agentic.models.task import Task, TaskIntent, TaskType, ProjectContext


async def test_project_indexing():
    """Test project indexing capabilities"""
    print("\n=== Testing Project Indexing ===")
    
    workspace = Path.cwd()
    indexer = ProjectIndexer(workspace)
    
    print(f"Indexing workspace: {workspace}")
    index = await indexer.index_project()
    
    print(f"\nProject Statistics:")
    print(f"  Total files: {index.total_files}")
    print(f"  Total size: {index.total_size:,} bytes")
    print(f"\nLanguage distribution:")
    for lang, count in sorted(index.language_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {lang}: {count} files")
    
    print(f"\nDetected frameworks:")
    for framework, files in index.framework_indicators.items():
        print(f"  {framework}: {len(files)} indicator files")
    
    # Get project context
    context = indexer.get_project_context()
    print(f"\nProject Context:")
    print(f"  Primary language: {context.primary_language}")
    print(f"  Primary framework: {context.framework}")
    print(f"  Package manager: {context.package_manager}")
    print(f"  Project type: {context.project_type}")
    
    return context


def test_language_detection():
    """Test language detection from commands"""
    print("\n=== Testing Language Detection ===")
    
    selector = LanguageSelector()
    
    test_commands = [
        "create a react component for user authentication",
        "build a FastAPI backend with user management",
        "implement a todo app",
        "create a vue.js dashboard",
        "build a django REST API",
        "create a mobile app with react native",
        "implement a CLI tool in go",
        "create a rust web server"
    ]
    
    for command in test_commands:
        detected_lang = selector.detect_language_from_command(command)
        detected_fw = selector.detect_framework_from_command(command)
        print(f"\nCommand: '{command}'")
        print(f"  Detected language: {detected_lang}")
        print(f"  Detected framework: {detected_fw}")


def test_language_inference(project_context: ProjectContext):
    """Test language inference with project context"""
    print("\n=== Testing Language Inference ===")
    
    selector = LanguageSelector()
    
    test_commands = [
        "create a new component",  # Should use project language
        "add a new API endpoint",  # Should use project language
        "create a react component",  # Explicit language
        "build a vue dashboard"  # Different framework
    ]
    
    for command in test_commands:
        lang, fw = selector.infer_language_from_context(command, project_context)
        print(f"\nCommand: '{command}'")
        print(f"  Inferred language: {lang}")
        print(f"  Inferred framework: {fw}")


def test_clarification_prompts(project_context: ProjectContext):
    """Test clarification prompt generation"""
    print("\n=== Testing Clarification Prompts ===")
    
    selector = LanguageSelector()
    
    ambiguous_commands = [
        "create a todo app",
        "build a web application",
        "create an API",
        "implement a dashboard"
    ]
    
    for command in ambiguous_commands:
        prompt = selector.get_clarification_prompt(command, project_context)
        if prompt:
            print(f"\nCommand: '{command}'")
            print("Clarification needed:")
            print(prompt)


def test_task_with_context():
    """Test creating a task with language context"""
    print("\n=== Testing Task with Language Context ===")
    
    # Create a mock task intent
    intent = TaskIntent(
        task_type=TaskType.IMPLEMENT,
        complexity_score=0.5,
        estimated_duration=30,
        affected_areas=["frontend"],
        requires_reasoning=True,
        requires_coordination=False
    )
    
    # Create task
    task = Task.from_intent(intent, "create a user profile component")
    
    # Create project context
    context = ProjectContext(
        primary_language="typescript",
        languages={"typescript", "javascript", "python"},
        framework="react",
        frameworks={"react", "express"},
        package_manager="npm",
        project_type="web"
    )
    
    # Apply context to task
    task.project_context = context
    task.target_language = context.primary_language
    task.target_framework = context.framework
    
    print(f"Task: {task.command}")
    print(f"Target language: {task.target_language}")
    print(f"Target framework: {task.target_framework}")
    print(f"Project type: {task.project_context.project_type}")


async def main():
    """Run all tests"""
    print("Testing Agentic Language Context Awareness System")
    print("=" * 50)
    
    # Test project indexing
    project_context = await test_project_indexing()
    
    # Test language detection
    test_language_detection()
    
    # Test language inference
    test_language_inference(project_context)
    
    # Test clarification prompts
    test_clarification_prompts(project_context)
    
    # Test task creation with context
    test_task_with_context()
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())