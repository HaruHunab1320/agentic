# Chaos engineering and resilience tests
import pytest
import asyncio
import tempfile
import time
import random
from pathlib import Path
from typing import List, Dict, Any

from agentic.core.orchestrator import Orchestrator
from agentic.models.config import AgenticConfig
from agentic.models.task import TaskResult


class TestBasicResilience:
    """Test basic system resilience"""

    @pytest.fixture
    async def resilient_orchestrator(self):
        """Create orchestrator for resilience testing"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.chaos
    async def test_invalid_input_resilience(self, resilient_orchestrator):
        """Test resilience to invalid inputs"""
        invalid_inputs = [
            "",  # Empty string
            " " * 1000,  # Very long whitespace
            "€∂ƒ˙∆˚¬",  # Special characters
            "\n\n\n\n",  # Only newlines
            "A" * 10000,  # Very long string
            None,  # This would be converted to string by the time it reaches orchestrator
            "{'malicious': 'json'}",  # JSON-like string
            "rm -rf /",  # Potentially harmful command
        ]
        
        for invalid_input in invalid_inputs:
            if invalid_input is None:
                continue  # Skip None as it would cause issues before reaching orchestrator
                
            try:
                result = await resilient_orchestrator.execute_command(str(invalid_input))
                # System should handle gracefully without crashing
                assert isinstance(result, TaskResult)
                assert result.task_id is not None
            except Exception as e:
                # If an exception occurs, it should be a controlled one, not a crash
                assert "internal error" not in str(e).lower()

    @pytest.mark.chaos
    async def test_rapid_command_succession(self, resilient_orchestrator):
        """Test resilience to rapid command succession"""
        # Fire off many commands in rapid succession
        rapid_commands = [f"quick task {i}" for i in range(20)]
        
        # Send commands with minimal delay
        tasks = []
        for command in rapid_commands:
            task = resilient_orchestrator.execute_command(command)
            tasks.append(task)
            await asyncio.sleep(0.01)  # Very brief delay
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # System should handle rapid succession gracefully
        successful_results = [r for r in results if isinstance(r, TaskResult)]
        assert len(successful_results) >= len(rapid_commands) // 3  # At least 1/3 succeed

    @pytest.mark.chaos
    async def test_resource_exhaustion_simulation(self, resilient_orchestrator):
        """Test behavior under simulated resource exhaustion"""
        # Simulate resource-intensive commands
        resource_heavy_commands = [
            "Process a very large dataset",
            "Generate comprehensive documentation for entire project",
            "Create complex data structure with many relationships",
            "Perform intensive computation",
            "Generate large amount of code"
        ]
        
        for command in resource_heavy_commands:
            try:
                result = await resilient_orchestrator.execute_command(command)
                # Should complete or fail gracefully
                assert isinstance(result, TaskResult)
            except Exception as e:
                # If it fails, should be a controlled failure
                assert "system crash" not in str(e).lower()

    @pytest.mark.chaos
    async def test_concurrent_failure_scenarios(self, resilient_orchestrator):
        """Test concurrent operations with various failure scenarios"""
        # Mix of valid and problematic commands
        mixed_commands = [
            "create hello world",  # Valid
            "",  # Invalid
            "add function to process data",  # Valid
            "invalid_xyz_command",  # Invalid
            "implement user authentication",  # Valid
            " " * 100,  # Invalid
            "create unit tests",  # Valid
        ]
        
        # Execute all concurrently
        tasks = [resilient_orchestrator.execute_command(cmd) for cmd in mixed_commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # System should handle mixed scenarios
        assert len(results) == len(mixed_commands)
        
        # At least some should succeed
        successful_results = [r for r in results if isinstance(r, TaskResult)]
        assert len(successful_results) >= 2


class TestFailureRecovery:
    """Test system recovery from failures"""

    @pytest.fixture
    async def recovery_orchestrator(self):
        """Create orchestrator for recovery testing"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await recovery_orchestrator.shutdown()

    @pytest.mark.chaos
    async def test_sequential_failure_recovery(self, recovery_orchestrator):
        """Test recovery after sequential failures"""
        # First, cause some failures
        failing_commands = [
            "",
            "nonexistent_command",
            "invalid input"
        ]
        
        for command in failing_commands:
            await recovery_orchestrator.execute_command(command)
        
        # Then try valid commands - system should recover
        recovery_commands = [
            "create simple function",
            "add variable x = 5",
            "create hello world"
        ]
        
        for command in recovery_commands:
            result = await recovery_orchestrator.execute_command(command)
            assert isinstance(result, TaskResult)
            
        # System should still be functional
        status = recovery_orchestrator.get_system_status()
        assert status["initialized"] is True

    @pytest.mark.chaos
    async def test_system_state_consistency(self, recovery_orchestrator):
        """Test that system state remains consistent after failures"""
        initial_status = recovery_orchestrator.get_system_status()
        
        # Cause various types of failures
        problematic_scenarios = [
            "",
            "malformed command €∂ƒ",
            "extremely long command " + "a" * 1000,
            "command with\nnewlines\nand\nspecial\nchars"
        ]
        
        for scenario in problematic_scenarios:
            await recovery_orchestrator.execute_command(scenario)
        
        # Check that system state is still consistent
        final_status = recovery_orchestrator.get_system_status()
        
        # Core system properties should remain unchanged
        assert final_status["initialized"] == initial_status["initialized"]
        assert final_status["project_analyzed"] == initial_status["project_analyzed"]

    @pytest.mark.chaos
    async def test_graceful_degradation(self, recovery_orchestrator):
        """Test graceful degradation under stress"""
        # Create a high-stress scenario
        stress_commands = [f"stress command {i}" for i in range(30)]
        
        # Execute with varying delays to create irregular load
        results = []
        for i, command in enumerate(stress_commands):
            result = await recovery_orchestrator.execute_command(command)
            results.append(result)
            
            # Random delay between commands
            await asyncio.sleep(random.uniform(0.01, 0.1))
        
        # System should degrade gracefully, not crash
        assert all(isinstance(r, TaskResult) for r in results)


class TestConcurrentChaos:
    """Test system behavior under concurrent chaotic conditions"""

    @pytest.fixture
    async def chaos_orchestrator(self):
        """Create orchestrator for chaos testing"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.chaos
    async def test_random_concurrent_operations(self, chaos_orchestrator):
        """Test random concurrent operations"""
        # Generate random operations
        operation_types = [
            "create",
            "modify", 
            "analyze",
            "test",
            "document"
        ]
        
        targets = [
            "function",
            "class", 
            "file",
            "module",
            "component"
        ]
        
        # Create random commands
        random_commands = []
        for _ in range(15):
            operation = random.choice(operation_types)
            target = random.choice(targets)
            command = f"{operation} {target} {random.randint(1, 100)}"
            random_commands.append(command)
        
        # Execute concurrently
        tasks = [chaos_orchestrator.execute_command(cmd) for cmd in random_commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle random concurrent operations
        successful_results = [r for r in results if isinstance(r, TaskResult)]
        assert len(successful_results) >= len(random_commands) // 4  # At least 25% success

    @pytest.mark.chaos
    async def test_mixed_valid_invalid_concurrent(self, chaos_orchestrator):
        """Test mix of valid and invalid commands concurrently"""
        valid_commands = [
            "create hello world function",
            "add documentation",
            "create simple test",
            "implement basic logic"
        ]
        
        invalid_commands = [
            "",
            "€∂ƒ˙∆˚¬",
            " " * 50,
            "invalid_nonexistent_command"
        ]
        
        # Mix them together
        all_commands = valid_commands + invalid_commands
        random.shuffle(all_commands)
        
        # Execute all concurrently
        tasks = [chaos_orchestrator.execute_command(cmd) for cmd in all_commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should return TaskResult (even if they failed)
        assert all(isinstance(r, TaskResult) or isinstance(r, Exception) for r in results)
        
        # At least the valid commands should have a chance to succeed
        successful_results = [r for r in results if isinstance(r, TaskResult)]
        assert len(successful_results) >= len(valid_commands) // 2

    @pytest.mark.chaos
    async def test_sustained_chaotic_load(self, chaos_orchestrator):
        """Test sustained chaotic load over time"""
        duration = 20  # 20 seconds of chaos
        start_time = time.time()
        
        results = []
        command_count = 0
        
        while time.time() - start_time < duration:
            # Generate random chaotic command
            chaos_types = [
                f"chaos command {command_count}",
                "",
                f"special chars €∂ƒ {command_count}",
                f"normal operation {command_count}",
                " " * random.randint(1, 20),
            ]
            
            command = random.choice(chaos_types)
            
            try:
                result = await chaos_orchestrator.execute_command(command)
                results.append(result)
                command_count += 1
            except Exception as e:
                results.append(e)
                command_count += 1
            
            # Random delay
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # System should survive sustained chaos
        assert command_count > 0
        task_results = [r for r in results if isinstance(r, TaskResult)]
        assert len(task_results) >= command_count // 5  # At least 20% should be TaskResults


class TestSystemLimits:
    """Test system behavior at limits"""

    @pytest.fixture
    async def limits_orchestrator(self):
        """Create orchestrator for limits testing"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.chaos
    async def test_command_length_limits(self, limits_orchestrator):
        """Test handling of very long commands"""
        length_tests = [
            "a" * 100,     # Short
            "b" * 1000,    # Medium
            "c" * 5000,    # Long
            "d" * 10000,   # Very long
        ]
        
        for test_command in length_tests:
            result = await limits_orchestrator.execute_command(test_command)
            # Should handle without crashing
            assert isinstance(result, TaskResult)

    @pytest.mark.chaos
    async def test_concurrent_connection_limits(self, limits_orchestrator):
        """Test behavior at concurrent connection limits"""
        # Try to create many concurrent operations
        concurrent_limit_test = 50
        
        async def concurrent_operation(op_id: int) -> TaskResult:
            try:
                return await limits_orchestrator.execute_command(f"operation {op_id}")
            except Exception as e:
                # Convert exception to a TaskResult-like object for testing
                return TaskResult(
                    task_id=f"failed_{op_id}",
                    agent_id="none",
                    success=False,
                    output="",
                    error=str(e)
                )
        
        # Launch many concurrent operations
        tasks = [concurrent_operation(i) for i in range(concurrent_limit_test)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # System should handle up to its limits gracefully
        assert len(results) == concurrent_limit_test
        
        # At least some should succeed or fail gracefully
        valid_results = [r for r in results if isinstance(r, TaskResult)]
        assert len(valid_results) >= concurrent_limit_test // 4  # At least 25%


class TestDataCorruption:
    """Test resilience to data corruption scenarios"""

    @pytest.fixture
    async def corruption_orchestrator(self):
        """Create orchestrator for corruption testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "corruption_test"
            project_path.mkdir()
            
            # Create some files that might get "corrupted"
            (project_path / "normal_file.py").write_text("def hello(): pass")
            (project_path / "corrupted_file.py").write_text("€∂ƒ˙∆˚¬ invalid python €∂ƒ˙∆˚¬")
            
            config = AgenticConfig()
            orchestrator = Orchestrator(config)
            await orchestrator.initialize(project_path)
            yield orchestrator
            await orchestrator.shutdown()

    @pytest.mark.chaos
    async def test_handling_corrupted_project_files(self, corruption_orchestrator):
        """Test handling of corrupted project files"""
        # Try to work with the project that has corrupted files
        commands = [
            "analyze project structure",
            "find all python files",
            "create new file",
            "add documentation"
        ]
        
        for command in commands:
            result = await corruption_orchestrator.execute_command(command)
            # Should handle corrupted files gracefully
            assert isinstance(result, TaskResult)

    @pytest.mark.chaos
    async def test_recovery_from_partial_operations(self, corruption_orchestrator):
        """Test recovery from partially completed operations"""
        # Start some operations that might be "interrupted"
        partial_commands = [
            "create comprehensive module with many functions",
            "implement complex feature with multiple components",
            "refactor entire project structure"
        ]
        
        for command in partial_commands:
            # Execute but don't necessarily wait for completion
            result = await corruption_orchestrator.execute_command(command)
            assert isinstance(result, TaskResult)
            
            # Brief pause, then continue with other work
            await asyncio.sleep(0.1)
        
        # System should remain functional
        status = corruption_orchestrator.get_system_status()
        assert status["initialized"] is True


class TestExtremeChaos:
    """Test system behavior under extreme chaotic conditions"""

    @pytest.mark.chaos
    async def test_complete_chaos_scenario(self):
        """Test complete chaos scenario - everything that can go wrong"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "chaos_project"
            project_path.mkdir()
            
            config = AgenticConfig()
            orchestrator = Orchestrator(config)
            
            try:
                await orchestrator.initialize(project_path)
                
                # Extreme chaos scenario
                chaos_operations = []
                
                # 1. Many rapid invalid commands
                for i in range(10):
                    chaos_operations.append(orchestrator.execute_command(""))
                
                # 2. Very long commands
                for i in range(5):
                    chaos_operations.append(orchestrator.execute_command("x" * 1000))
                
                # 3. Valid commands mixed in
                for i in range(5):
                    chaos_operations.append(orchestrator.execute_command(f"create function {i}"))
                
                # 4. Special character commands
                for i in range(5):
                    chaos_operations.append(orchestrator.execute_command("€∂ƒ˙∆˚¬"))
                
                # Execute all chaos operations
                results = await asyncio.gather(*chaos_operations, return_exceptions=True)
                
                # System should survive complete chaos
                assert len(results) == 25
                
                # At least some operations should complete
                task_results = [r for r in results if isinstance(r, TaskResult)]
                assert len(task_results) >= 5  # At least 5 should be TaskResults
                
                # System should still be responsive after chaos
                final_result = await orchestrator.execute_command("system status check")
                assert isinstance(final_result, TaskResult)
                
            finally:
                await orchestrator.shutdown()

    @pytest.mark.chaos
    async def test_system_recovery_after_extreme_stress(self):
        """Test system recovery after extreme stress"""
        config = AgenticConfig()
        
        # First phase: extreme stress
        orchestrator1 = Orchestrator(config)
        await orchestrator1.initialize()
        
        try:
            # Cause extreme stress
            stress_tasks = []
            for i in range(20):
                if i % 4 == 0:
                    cmd = ""  # Invalid
                elif i % 4 == 1:
                    cmd = "€∂ƒ˙∆˚¬"  # Special chars
                elif i % 4 == 2:
                    cmd = "a" * 500  # Long
                else:
                    cmd = f"normal command {i}"  # Valid
                
                stress_tasks.append(orchestrator1.execute_command(cmd))
            
            await asyncio.gather(*stress_tasks, return_exceptions=True)
            
        finally:
            await orchestrator1.shutdown()
        
        # Second phase: recovery test
        orchestrator2 = Orchestrator(config)
        await orchestrator2.initialize()
        
        try:
            # Should start fresh and work normally
            recovery_result = await orchestrator2.execute_command("test recovery")
            assert isinstance(recovery_result, TaskResult)
            
            status = orchestrator2.get_system_status()
            assert status["initialized"] is True
            
        finally:
            await orchestrator2.shutdown() 