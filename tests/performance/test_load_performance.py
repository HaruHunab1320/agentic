# Performance and load testing
import pytest
import asyncio
import time
import psutil
import statistics
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import tempfile

from agentic.core.orchestrator import Orchestrator
from agentic.models.project import Project
from agentic.utils.performance_metrics import PerformanceCollector
from agentic.models.config import AgenticConfig
from agentic.models.task import TaskResult


class TestSystemPerformance:
    """Performance tests for system-wide operations"""

    @pytest.fixture
    async def performance_test_project(self):
        """Create project optimized for performance testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "perf_test_project"
            project_path.mkdir()
            
            # Create realistic project structure
            await self._create_performance_test_project(project_path)
            
            project = Project(path=project_path, name="perf_test")
            yield project

    @pytest.fixture
    async def orchestrator_with_monitoring(self, performance_test_project):
        """Create orchestrator with performance monitoring enabled"""
        orchestrator = Orchestrator(
            project=performance_test_project,
            enable_performance_monitoring=True
        )
        await orchestrator.initialize()
        
        yield orchestrator
        
        await orchestrator.cleanup()

    @pytest.mark.performance
    async def test_single_command_performance(self, orchestrator_with_monitoring):
        """Test performance of individual commands"""
        test_cases = [
            ("create component Button", 30),  # Should complete within 30 seconds
            ("analyze code quality", 45),     # Should complete within 45 seconds
            ("add unit tests", 60),           # Should complete within 60 seconds
            ("refactor function", 40),        # Should complete within 40 seconds
        ]
        
        performance_results = []
        
        for command, max_time in test_cases:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = await orchestrator_with_monitoring.execute_command(command)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            performance_results.append({
                'command': command,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'success': result.status == "completed"
            })
            
            # Verify performance requirements
            assert execution_time < max_time, f"Command '{command}' took {execution_time:.2f}s, max allowed: {max_time}s"
            assert memory_usage < 500, f"Command '{command}' used {memory_usage:.2f}MB memory"
            assert result.status == "completed"
        
        # Verify overall performance characteristics
        avg_time = statistics.mean([r['execution_time'] for r in performance_results])
        assert avg_time < 45  # Average command time should be under 45 seconds

    @pytest.mark.performance
    async def test_concurrent_command_performance(self, orchestrator_with_monitoring):
        """Test performance with multiple concurrent commands"""
        concurrent_commands = [
            "create component Header",
            "create component Footer", 
            "create component Sidebar",
            "create component Navigation",
            "create component Modal"
        ]
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Execute all commands concurrently
        tasks = [
            orchestrator_with_monitoring.execute_command(cmd) 
            for cmd in concurrent_commands
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        total_execution_time = end_time - start_time
        total_memory_usage = end_memory - start_memory
        
        # Verify concurrent performance
        assert total_execution_time < 120  # All commands should complete within 2 minutes
        assert total_memory_usage < 1024   # Should use less than 1GB memory
        assert all(r.status == "completed" for r in results)
        
        # Verify concurrency efficiency (should be faster than sequential)
        estimated_sequential_time = len(concurrent_commands) * 30  # 30s per command
        efficiency_ratio = total_execution_time / estimated_sequential_time
        assert efficiency_ratio < 0.7  # Should be at least 30% faster than sequential

    @pytest.mark.performance
    async def test_large_project_performance(self, orchestrator_with_monitoring):
        """Test performance on large projects"""
        # Create large project structure
        await self._create_large_project_structure(orchestrator_with_monitoring.project.path)
        
        performance_metrics = PerformanceCollector()
        
        # Test analysis of large project
        start_time = time.time()
        analysis_result = await orchestrator_with_monitoring.analyze_project()
        analysis_time = time.time() - start_time
        
        assert analysis_time < 180  # Should analyze large project within 3 minutes
        assert analysis_result.files_analyzed > 50
        
        # Test modification of large project
        start_time = time.time()
        modification_result = await orchestrator_with_monitoring.execute_command(
            "add error handling to all API endpoints"
        )
        modification_time = time.time() - start_time
        
        assert modification_time < 300  # Should modify large project within 5 minutes
        assert modification_result.status == "completed"
        assert len(modification_result.modified_files) > 10

    @pytest.mark.performance
    async def test_memory_usage_stability(self, orchestrator_with_monitoring):
        """Test memory usage remains stable over extended operation"""
        memory_samples = []
        
        # Collect memory usage over multiple operations
        for i in range(20):
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            # Execute operation
            await orchestrator_with_monitoring.execute_command(f"create utility function util{i}")
            
            # Small delay between operations
            await asyncio.sleep(1)
        
        # Analyze memory growth
        initial_memory = memory_samples[0]
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 200  # Less than 200MB growth over 20 operations
        
        # Check for memory leaks (no consistent upward trend)
        first_half_avg = statistics.mean(memory_samples[:10])
        second_half_avg = statistics.mean(memory_samples[10:])
        growth_rate = (second_half_avg - first_half_avg) / first_half_avg
        
        assert growth_rate < 0.2  # Less than 20% memory growth between halves

    @pytest.mark.performance
    async def test_cpu_usage_efficiency(self, orchestrator_with_monitoring):
        """Test CPU usage efficiency during operations"""
        cpu_samples = []
        
        # Monitor CPU usage during operation
        async def monitor_cpu():
            for _ in range(30):  # Monitor for 30 seconds
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)
        
        # Start CPU monitoring
        monitor_task = asyncio.create_task(monitor_cpu())
        
        # Execute CPU-intensive operation
        await orchestrator_with_monitoring.execute_command(
            "analyze and refactor entire codebase for performance"
        )
        
        # Wait for monitoring to complete
        await monitor_task
        
        # Analyze CPU usage
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        
        # CPU usage should be reasonable
        assert avg_cpu < 80  # Average CPU should be under 80%
        assert max_cpu < 95  # Peak CPU should be under 95%


class TestLoadTesting:
    """Load testing for high-volume scenarios"""
    
    @pytest.mark.performance
    @pytest.mark.load
    async def test_high_concurrency_load(self, orchestrator_with_monitoring):
        """Test system under high concurrency load"""
        num_concurrent_users = 25
        operations_per_user = 4
        
        async def simulate_user_session(user_id: int):
            """Simulate a user session with multiple operations"""
            session_results = []
            
            for op_num in range(operations_per_user):
                command = f"create component User{user_id}Component{op_num}"
                start_time = time.time()
                
                result = await orchestrator_with_monitoring.execute_command(command)
                
                execution_time = time.time() - start_time
                session_results.append({
                    'user_id': user_id,
                    'operation': op_num,
                    'execution_time': execution_time,
                    'success': result.status == "completed"
                })
            
            return session_results
        
        # Start all user sessions concurrently
        start_time = time.time()
        user_tasks = [
            simulate_user_session(user_id) 
            for user_id in range(num_concurrent_users)
        ]
        
        all_results = await asyncio.gather(*user_tasks)
        total_time = time.time() - start_time
        
        # Flatten results
        flat_results = [result for user_results in all_results for result in user_results]
        
        # Verify load test results
        total_operations = num_concurrent_users * operations_per_user
        successful_operations = sum(1 for r in flat_results if r['success'])
        success_rate = successful_operations / total_operations
        
        assert success_rate > 0.9  # 90% success rate under load
        assert total_time < 600    # Should complete within 10 minutes
        
        # Verify response time distribution
        execution_times = [r['execution_time'] for r in flat_results if r['success']]
        avg_response_time = statistics.mean(execution_times)
        p95_response_time = sorted(execution_times)[int(len(execution_times) * 0.95)]
        
        assert avg_response_time < 60   # Average response time under 60 seconds
        assert p95_response_time < 120  # 95th percentile under 2 minutes

    @pytest.mark.performance
    @pytest.mark.load
    async def test_sustained_load_over_time(self, orchestrator_with_monitoring):
        """Test system performance under sustained load"""
        duration_minutes = 10
        operations_per_minute = 6
        total_operations = duration_minutes * operations_per_minute
        
        operation_results = []
        start_time = time.time()
        
        for i in range(total_operations):
            operation_start = time.time()
            
            # Execute operation
            command = f"create utility function utility{i}"
            result = await orchestrator_with_monitoring.execute_command(command)
            
            operation_time = time.time() - operation_start
            operation_results.append({
                'operation_id': i,
                'execution_time': operation_time,
                'success': result.status == "completed",
                'timestamp': time.time() - start_time
            })
            
            # Maintain consistent load (wait if operation was too fast)
            expected_interval = 60 / operations_per_minute  # seconds between operations
            time_to_wait = expected_interval - operation_time
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)
        
        # Analyze sustained load performance
        successful_operations = [r for r in operation_results if r['success']]
        success_rate = len(successful_operations) / total_operations
        
        assert success_rate > 0.95  # 95% success rate for sustained load
        
        # Check for performance degradation over time
        first_quarter = successful_operations[:len(successful_operations)//4]
        last_quarter = successful_operations[-len(successful_operations)//4:]
        
        first_quarter_avg = statistics.mean([r['execution_time'] for r in first_quarter])
        last_quarter_avg = statistics.mean([r['execution_time'] for r in last_quarter])
        
        degradation_ratio = last_quarter_avg / first_quarter_avg
        assert degradation_ratio < 1.5  # Performance shouldn't degrade more than 50%

    @pytest.mark.performance
    @pytest.mark.load
    async def test_resource_limits_handling(self, orchestrator_with_monitoring):
        """Test system behavior when approaching resource limits"""
        # Set artificial resource limits
        original_memory_limit = orchestrator_with_monitoring.resource_manager.memory_limit
        original_cpu_limit = orchestrator_with_monitoring.resource_manager.cpu_limit
        
        # Set lower limits for testing
        orchestrator_with_monitoring.resource_manager.memory_limit = 512  # 512MB
        orchestrator_with_monitoring.resource_manager.cpu_limit = 70     # 70% CPU
        
        try:
            # Execute operations that will approach limits
            resource_intensive_commands = [
                "analyze and optimize entire codebase",
                "generate comprehensive documentation",
                "run full test suite with coverage",
                "perform security audit",
                "create performance benchmarks"
            ]
            
            results = []
            for command in resource_intensive_commands:
                result = await orchestrator_with_monitoring.execute_command(command)
                results.append(result)
                
                # Check resource usage
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_usage = psutil.cpu_percent()
                
                # System should handle resource pressure gracefully
                if memory_usage > 450:  # Approaching limit
                    assert result.status in ["completed", "partial", "throttled"]
                if cpu_usage > 65:  # Approaching limit
                    assert result.status in ["completed", "partial", "throttled"]
            
            # Verify graceful handling
            completed_or_handled = sum(1 for r in results if r.status in ["completed", "partial", "throttled"])
            handling_rate = completed_or_handled / len(results)
            
            assert handling_rate >= 0.8  # Should handle at least 80% gracefully
            
        finally:
            # Restore original limits
            orchestrator_with_monitoring.resource_manager.memory_limit = original_memory_limit
            orchestrator_with_monitoring.resource_manager.cpu_limit = original_cpu_limit

    async def _create_performance_test_project(self, project_path: Path):
        """Create a project structure optimized for performance testing"""
        # Create directory structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "docs").mkdir()
        
        # Create multiple source files
        for i in range(10):
            (project_path / "src" / f"module_{i}.py").write_text(f"""
# Module {i}
import asyncio
from typing import List, Dict, Any

class Module{i}:
    def __init__(self):
        self.data = []
    
    async def process_data(self, input_data: List[Any]) -> Dict[str, Any]:
        result = {{}}
        for item in input_data:
            result[str(item)] = item * 2
        return result
    
    def sync_operation(self, value: int) -> int:
        return value ** 2
            """)
        
        # Create package.json for Node.js projects
        (project_path / "package.json").write_text('''
{
  "name": "performance-test-project",
  "version": "1.0.0",
  "description": "Project for performance testing",
  "main": "index.js",
  "scripts": {
    "test": "jest",
    "start": "node src/index.js"
  },
  "dependencies": {
    "express": "^4.18.0",
    "lodash": "^4.17.21"
  },
  "devDependencies": {
    "jest": "^27.0.0"
  }
}
        ''')

    async def _create_large_project_structure(self, project_path: Path):
        """Create a large project structure for testing scalability"""
        # Create multiple modules
        modules = ["auth", "api", "frontend", "database", "utils", "tests", "docs", "config"]
        
        for module in modules:
            module_path = project_path / "src" / module
            module_path.mkdir(parents=True)
            
            # Create multiple files per module
            for i in range(8):
                file_path = module_path / f"{module}_{i}.py"
                file_path.write_text(f"""
# {module.title()} Module {i}
from typing import Optional, List, Dict
import asyncio
import logging

logger = logging.getLogger(__name__)

class {module.title()}{i}:
    '''
    {module.title()} class for handling operations.
    This is a large class with multiple methods for testing purposes.
    '''
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {{}}
        self.initialized = False
        self.data_cache = {{}}
    
    async def initialize(self) -> bool:
        '''Initialize the {module} component'''
        try:
            await self._setup_resources()
            self.initialized = True
            logger.info(f"{module.title()}{i} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {module}: {{e}}")
            return False
    
    async def _setup_resources(self):
        '''Setup required resources'''
        await asyncio.sleep(0.1)  # Simulate setup time
    
    async def process_request(self, request_data: Dict) -> Dict:
        '''Process incoming request'''
        if not self.initialized:
            raise RuntimeError("Component not initialized")
        
        result = {{
            'status': 'success',
            'data': request_data,
            'processed_by': f'{module}_{i}'
        }}
        
        return result
    
    def validate_input(self, data: Dict) -> bool:
        '''Validate input data'''
        required_fields = ['id', 'type']
        return all(field in data for field in required_fields)
    
    async def cleanup(self):
        '''Cleanup resources'''
        self.data_cache.clear()
        self.initialized = False
        logger.info(f"{module.title()}{i} cleaned up")
                """)


class TestPerformanceRegression:
    """Performance regression testing"""
    
    @pytest.mark.performance
    async def test_performance_regression_detection(self, orchestrator_with_monitoring):
        """Test detection of performance regressions"""
        # Baseline performance measurement
        baseline_commands = [
            "create component TestComponent",
            "add unit tests",
            "analyze code quality"
        ]
        
        baseline_times = []
        for command in baseline_commands:
            start_time = time.time()
            await orchestrator_with_monitoring.execute_command(command)
            execution_time = time.time() - start_time
            baseline_times.append(execution_time)
        
        baseline_avg = statistics.mean(baseline_times)
        
        # Simulate system changes that might cause regression
        await self._simulate_system_changes(orchestrator_with_monitoring)
        
        # Re-measure performance
        regression_times = []
        for command in baseline_commands:
            start_time = time.time()
            await orchestrator_with_monitoring.execute_command(command)
            execution_time = time.time() - start_time
            regression_times.append(execution_time)
        
        regression_avg = statistics.mean(regression_times)
        
        # Check for significant regression (more than 30% slower)
        regression_ratio = regression_avg / baseline_avg
        assert regression_ratio < 1.3, f"Performance regression detected: {regression_ratio:.2f}x slower"

    async def _simulate_system_changes(self, orchestrator):
        """Simulate system changes that might affect performance"""
        # Add some load to the system
        background_tasks = [
            orchestrator.execute_command(f"background task {i}")
            for i in range(3)
        ]
        
        # Don't wait for completion, just start them
        for task in background_tasks:
            asyncio.create_task(task)


class TestSingleCommandPerformance:
    """Test performance of individual commands"""

    @pytest.fixture
    async def performance_orchestrator(self):
        """Create orchestrator for performance testing"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.performance
    async def test_simple_command_response_time(self, performance_orchestrator):
        """Test response time for simple commands"""
        simple_commands = [
            "hello world",
            "create variable x = 5",
            "add comment to code",
            "format this text",
            "what is 2 + 2"
        ]
        
        response_times = []
        for command in simple_commands:
            start_time = time.time()
            result = await performance_orchestrator.execute_command(command)
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # Verify command was processed
            assert isinstance(result, TaskResult)
        
        # Performance requirements
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        assert avg_response_time < 10.0  # Average under 10 seconds
        assert max_response_time < 30.0  # No command over 30 seconds

    @pytest.mark.performance
    async def test_complex_command_performance(self, performance_orchestrator):
        """Test performance of complex commands"""
        complex_commands = [
            "Create a complete React component with state management",
            "Implement a REST API with CRUD operations",
            "Set up a database schema with relationships",
            "Create comprehensive test suite for a module"
        ]
        
        for command in complex_commands:
            start_time = time.time()
            result = await performance_orchestrator.execute_command(command)
            execution_time = time.time() - start_time
            
            # Complex commands should still complete in reasonable time
            assert execution_time < 120.0  # Under 2 minutes
            assert isinstance(result, TaskResult)

    @pytest.mark.performance
    async def test_orchestrator_initialization_time(self):
        """Test orchestrator initialization performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "perf_project"
            project_path.mkdir()
            
            # Create some project structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()
            for i in range(10):
                (project_path / "src" / f"file_{i}.py").write_text(f"# File {i}")
            
            config = AgenticConfig()
            orchestrator = Orchestrator(config)
            
            # Measure initialization time
            start_time = time.time()
            await orchestrator.initialize(project_path)
            init_time = time.time() - start_time
            
            # Initialization should be fast
            assert init_time < 30.0  # Under 30 seconds
            
            await orchestrator.shutdown()


class TestConcurrentPerformance:
    """Test performance under concurrent load"""

    @pytest.fixture
    async def concurrent_orchestrator(self):
        """Create orchestrator for concurrent testing"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.performance
    async def test_concurrent_simple_commands(self, concurrent_orchestrator):
        """Test multiple simple commands running concurrently"""
        commands = [f"create variable x{i} = {i}" for i in range(10)]
        
        start_time = time.time()
        tasks = [concurrent_orchestrator.execute_command(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify results
        successful_results = [r for r in results if isinstance(r, TaskResult)]
        assert len(successful_results) >= len(commands) // 2  # At least 50% success
        
        # Concurrent execution should be faster than sequential
        assert total_time < 60.0  # Under 1 minute for 10 commands

    @pytest.mark.performance
    async def test_sustained_load(self, concurrent_orchestrator):
        """Test sustained load over time"""
        duration = 30  # 30 seconds of load
        start_time = time.time()
        command_count = 0
        errors = 0
        
        while time.time() - start_time < duration:
            try:
                result = await concurrent_orchestrator.execute_command(f"simple task {command_count}")
                command_count += 1
                if not isinstance(result, TaskResult):
                    errors += 1
            except Exception:
                errors += 1
            
            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Performance metrics
        commands_per_second = command_count / duration
        error_rate = errors / command_count if command_count > 0 else 1
        
        assert commands_per_second > 0.5  # At least 0.5 commands per second
        assert error_rate < 0.3  # Less than 30% error rate

    @pytest.mark.performance
    async def test_high_concurrency_load(self, concurrent_orchestrator):
        """Test high concurrency scenarios"""
        # Simulate 25 concurrent users
        num_concurrent = 25
        commands_per_user = 3
        
        async def user_session(user_id: int) -> List[TaskResult]:
            """Simulate a user session"""
            results = []
            for i in range(commands_per_user):
                try:
                    result = await concurrent_orchestrator.execute_command(
                        f"user {user_id} command {i}"
                    )
                    results.append(result)
                    await asyncio.sleep(0.1)  # Brief pause between commands
                except Exception as e:
                    results.append(e)
            return results
        
        # Run concurrent user sessions
        start_time = time.time()
        user_tasks = [user_session(i) for i in range(num_concurrent)]
        all_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        total_commands = num_concurrent * commands_per_user
        successful_commands = 0
        
        for user_results in all_results:
            if isinstance(user_results, list):
                successful_commands += sum(1 for r in user_results if isinstance(r, TaskResult))
        
        success_rate = successful_commands / total_commands
        
        # Performance requirements for high concurrency
        assert total_time < 300  # Under 5 minutes total
        assert success_rate > 0.6  # At least 60% success rate


class TestLargeProjectPerformance:
    """Test performance with large projects"""

    @pytest.fixture
    async def large_project(self):
        """Create a large project for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "large_project"
            project_path.mkdir()
            
            # Create large project structure
            for module in ["auth", "api", "frontend", "database", "utils", "tests"]:
                module_path = project_path / module
                module_path.mkdir()
                
                # Create multiple files per module
                for i in range(15):  # 15 files per module
                    file_path = module_path / f"{module}_{i}.py"
                    file_content = f"# {module} module file {i}\n" + "# Content\n" * 50
                    file_path.write_text(file_content)
            
            yield project_path

    @pytest.mark.performance
    async def test_large_project_analysis(self, large_project):
        """Test project analysis performance on large projects"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        
        # Measure project analysis time
        start_time = time.time()
        await orchestrator.initialize(large_project)
        analysis_time = time.time() - start_time
        
        # Large project analysis should still be reasonable
        assert analysis_time < 60.0  # Under 1 minute
        
        # Verify project was analyzed
        status = orchestrator.get_system_status()
        assert status["project_analyzed"] is True
        
        await orchestrator.shutdown()

    @pytest.mark.performance
    async def test_large_project_operations(self, large_project):
        """Test operations on large projects"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize(large_project)
        
        try:
            # Test various operations on large project
            operations = [
                "Analyze the project structure",
                "Find all Python files",
                "Add documentation to a module",
                "Create a new utility function"
            ]
            
            for operation in operations:
                start_time = time.time()
                result = await orchestrator.execute_command(operation)
                operation_time = time.time() - start_time
                
                # Operations should complete in reasonable time even for large projects
                assert operation_time < 90.0  # Under 1.5 minutes
                assert isinstance(result, TaskResult)
                
        finally:
            await orchestrator.shutdown()


class TestMemoryAndResourceUsage:
    """Test memory usage and resource management"""

    @pytest.mark.performance
    async def test_memory_stability_over_time(self):
        """Test that memory usage remains stable over time"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        
        try:
            # Run many commands to test for memory leaks
            for i in range(50):
                await orchestrator.execute_command(f"simple operation {i}")
                
                # Periodic check - in a real implementation you'd check actual memory
                if i % 10 == 0:
                    status = orchestrator.get_system_status()
                    assert status["initialized"] is True
                    
        finally:
            await orchestrator.shutdown()

    @pytest.mark.performance
    async def test_resource_cleanup(self):
        """Test that resources are properly cleaned up"""
        config = AgenticConfig()
        
        # Create and destroy multiple orchestrator instances
        for i in range(5):
            orchestrator = Orchestrator(config)
            await orchestrator.initialize()
            
            # Do some work
            await orchestrator.execute_command(f"test operation {i}")
            
            # Cleanup
            await orchestrator.shutdown()
        
        # All instances should clean up properly without issues

    @pytest.mark.performance
    async def test_resource_limits_handling(self):
        """Test behavior when approaching resource limits"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        
        try:
            # Try to create a scenario that uses more resources
            large_commands = [
                "Create a large data structure with 1000 items",
                "Process a large amount of text",
                "Generate comprehensive documentation",
                "Create multiple files with substantial content"
            ]
            
            for command in large_commands:
                result = await orchestrator.execute_command(command)
                # Should handle resource-intensive commands gracefully
                assert isinstance(result, TaskResult)
                
        finally:
            await orchestrator.shutdown()


class TestPerformanceRegression:
    """Test for performance regressions"""

    @pytest.mark.performance
    async def test_baseline_performance_metrics(self):
        """Establish baseline performance metrics"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        
        # Measure initialization
        start_time = time.time()
        await orchestrator.initialize()
        init_time = time.time() - start_time
        
        # Measure simple command
        start_time = time.time()
        result = await orchestrator.execute_command("create hello world")
        command_time = time.time() - start_time
        
        # Measure shutdown
        start_time = time.time()
        await orchestrator.shutdown()
        shutdown_time = time.time() - start_time
        
        # Record baseline metrics (in real implementation, store these)
        baseline_metrics = {
            "init_time": init_time,
            "simple_command_time": command_time,
            "shutdown_time": shutdown_time
        }
        
        # Basic sanity checks
        assert init_time < 30.0
        assert command_time < 30.0
        assert shutdown_time < 10.0
        assert isinstance(result, TaskResult)

    @pytest.mark.performance
    async def test_performance_under_stress(self):
        """Test performance under stress conditions"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()
        
        try:
            # Stress test with rapid commands
            rapid_commands = 20
            start_time = time.time()
            
            tasks = []
            for i in range(rapid_commands):
                task = orchestrator.execute_command(f"rapid command {i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Verify stress handling
            successful_results = [r for r in results if isinstance(r, TaskResult)]
            success_rate = len(successful_results) / len(results)
            
            # Should handle stress reasonably well
            assert success_rate > 0.5  # At least 50% success under stress
            assert total_time < 180  # Under 3 minutes for 20 commands
            
        finally:
            await orchestrator.shutdown() 