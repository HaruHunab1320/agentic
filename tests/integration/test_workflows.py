# Integration tests for major workflows
import pytest
import asyncio
import tempfile
from pathlib import Path
import time
from typing import Dict, Any

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig
from agentic.models.task import Task, TaskResult
from agentic.models.agent import AgentType


class TestMajorWorkflows:
    """Integration tests for major multi-agent workflows"""
 
    @pytest.fixture
    async def test_project_path(self):
        """Create a test project environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            
            # Create basic project structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()
            (project_path / "package.json").write_text('{"name": "test-project", "version": "1.0.0"}')
            
            yield project_path

    @pytest.fixture
    async def orchestrator(self, test_project_path):
        """Create orchestrator for testing"""
        config = AgenticConfig(
            workspace_name="test_workspace",
            workspace_path=test_project_path,
            claude_api_key="test-key",
            aider_config={
                "default_model": "gemini/gemini-2.5-pro-preview-06-05",
                "model_settings": {}
            }
        )
        orchestrator = Orchestrator(config)
        await orchestrator.initialize(test_project_path)
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.integration
    async def test_basic_command_execution(self, orchestrator):
        """Test basic command execution through orchestrator"""
        command = "Create a simple hello world function"
        
        start_time = time.time()
        result = await orchestrator.execute_command(command)
        execution_time = time.time() - start_time
        
        # Verify basic execution
        assert isinstance(result, TaskResult)
        assert execution_time < 60  # Should complete within 1 minute
        
        # Check that we got some kind of response
        assert result.task_id is not None

    @pytest.mark.integration
    async def test_orchestrator_initialization(self, test_project_path):
        """Test orchestrator initialization with project"""
        config = AgenticConfig(
            workspace_name="test_workspace",
            workspace_path=test_project_path,
            claude_api_key="test-key"
        )
        orchestrator = Orchestrator(config)
        
        # Initialize with project
        await orchestrator.initialize(test_project_path)
        
        # Verify initialization
        status = orchestrator.get_system_status()
        assert status["initialized"] is True
        assert status["project_analyzed"] is True
        
        await orchestrator.shutdown()

    @pytest.mark.integration
    async def test_agent_status_monitoring(self, orchestrator):
        """Test agent status monitoring"""
        # Get agent status as a dict
        agent_status = await orchestrator.get_agent_status()
        
        # This returns a dict of agent_id -> agent_info
        assert isinstance(agent_status, dict)
        
        # Get system status for agent counts
        system_status = orchestrator.get_system_status()
        assert "agents" in system_status
        assert "total_agents" in system_status["agents"]
        assert "available_agents" in system_status["agents"]
        
        # Should have non-negative agent counts
        assert system_status["agents"]["total_agents"] >= 0
        assert system_status["agents"]["available_agents"] >= 0

    @pytest.mark.integration
    async def test_multiple_commands_execution(self, orchestrator):
        """Test execution of multiple commands in sequence"""
        commands = [
            "Create a basic README file",
            "Add project structure documentation", 
            "Create a simple test file"
        ]
        
        results = []
        for command in commands:
            result = await orchestrator.execute_command(command)
            results.append(result)
            # Small delay between commands
            await asyncio.sleep(0.1)
        
        # Verify all commands were processed
        assert len(results) == len(commands)
        for result in results:
            assert isinstance(result, TaskResult)
            assert result.task_id is not None

    @pytest.mark.integration
    async def test_system_status_tracking(self, orchestrator):
        """Test system status tracking during operations"""
        initial_status = orchestrator.get_system_status()
        
        # Execute a command
        await orchestrator.execute_command("Create a simple function")
        
        final_status = orchestrator.get_system_status()
        
        # Verify status tracking
        assert initial_status["initialized"] == final_status["initialized"]
        assert initial_status["project_analyzed"] == final_status["project_analyzed"]


class TestWorkflowPerformance:
    """Performance tests for workflows"""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for performance testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)
            config = AgenticConfig(
                workspace_name="test_workspace",
                workspace_path=test_path,
                claude_api_key="test-key"
            )
            orchestrator = Orchestrator(config)
            await orchestrator.initialize()
            yield orchestrator
            await orchestrator.shutdown()

    @pytest.mark.integration
    @pytest.mark.performance
    async def test_concurrent_command_performance(self, orchestrator):
        """Test performance with concurrent commands"""
        commands = [f"Create component {i}" for i in range(5)]
        
        start_time = time.time()
        tasks = [orchestrator.execute_command(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Verify performance requirements
        assert execution_time < 120  # Should complete within 2 minutes
        assert len(results) == len(commands)
        
        # Check that most commands succeeded (allow for some failures in test env)
        successful_results = [r for r in results if isinstance(r, TaskResult)]
        assert len(successful_results) >= len(commands) // 2

    @pytest.mark.integration
    @pytest.mark.performance  
    async def test_orchestrator_response_time(self, orchestrator):
        """Test orchestrator response time for simple commands"""
        simple_commands = [
            "Hello world",
            "Create variable x",
            "Add comment to code"
        ]
        
        response_times = []
        for command in simple_commands:
            start_time = time.time()
            await orchestrator.execute_command(command)
            response_time = time.time() - start_time
            response_times.append(response_time)
        
        # Verify reasonable response times
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 30  # Average under 30 seconds
        assert max(response_times) < 60  # No single command over 1 minute


class TestWorkflowResilience:
    """Resilience tests for workflows"""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for resilience testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)
            config = AgenticConfig(
                workspace_name="test_workspace",
                workspace_path=test_path,
                claude_api_key="test-key"
            )
            orchestrator = Orchestrator(config)
            await orchestrator.initialize()
            yield orchestrator
            await orchestrator.shutdown()

    @pytest.mark.integration
    async def test_invalid_command_handling(self, orchestrator):
        """Test handling of invalid commands"""
        invalid_commands = [
            "",  # Empty command
            "nonexistent_command_xyz_123",  # Non-existent command
            "€∂ƒ˙∆˚¬",  # Special characters
        ]
        
        for command in invalid_commands:
            result = await orchestrator.execute_command(command)
            # Should handle gracefully without crashing
            assert isinstance(result, TaskResult)
            assert result.task_id is not None

    @pytest.mark.integration
    async def test_system_recovery_after_errors(self, orchestrator):
        """Test system recovery after encountering errors"""
        # Execute some invalid commands
        await orchestrator.execute_command("invalid_command_that_should_fail")
        await orchestrator.execute_command("")
        
        # System should still be functional
        status = orchestrator.get_system_status()
        assert status["initialized"] is True
        
        # Should be able to execute valid commands
        result = await orchestrator.execute_command("Create a simple test")
        assert isinstance(result, TaskResult)

    @pytest.mark.integration
    async def test_orchestrator_shutdown_and_restart(self):
        """Test orchestrator shutdown and restart"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)
            config = AgenticConfig(
                workspace_name="test_workspace",
                workspace_path=test_path,
                claude_api_key="test-key"
            )
            
            # First orchestrator instance
            orchestrator1 = Orchestrator(config)
            await orchestrator1.initialize()
            
            status1 = orchestrator1.get_system_status()
            assert status1["initialized"] is True
            
            # Shutdown first instance
            await orchestrator1.shutdown()
            
            # Create new instance
            orchestrator2 = Orchestrator(config)
            await orchestrator2.initialize()
            
            status2 = orchestrator2.get_system_status()
            assert status2["initialized"] is True
            
            await orchestrator2.shutdown() 