"""
Comprehensive tests for swarm safety systems
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from agentic.core.change_tracker import ChangeTracker, ChangeType
from agentic.core.swarm_transaction import SwarmTransactionManager, TransactionPhase
from agentic.core.state_persistence import StatePersistenceManager, StateType
from agentic.core.error_recovery import ErrorRecoveryManager, ErrorCategory
from agentic.core.result_validation import ResultValidationManager, ValidationResult
from agentic.models.task import Task, TaskIntent, TaskType


class TestChangeTracker:
    """Test file change tracking and rollback"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def change_tracker(self, temp_workspace):
        """Create a change tracker instance"""
        return ChangeTracker(temp_workspace)
    
    async def test_track_file_creation(self, change_tracker, temp_workspace):
        """Test tracking creation of new files"""
        # Begin changeset
        changeset_id = change_tracker.begin_changeset(
            "Create test file",
            agent_id="test_agent"
        )
        
        # Track file creation
        test_file = temp_workspace / "test.py"
        change = change_tracker.track_file_change(
            changeset_id=changeset_id,
            file_path=test_file,
            new_content="print('Hello, world!')",
            agent_id="test_agent",
            task_id="task_1"
        )
        
        # Verify
        assert change.change_type == ChangeType.CREATE
        assert test_file.exists()
        assert test_file.read_text() == "print('Hello, world!')"
        
        # Commit changeset
        changeset = change_tracker.commit_changeset(changeset_id)
        assert changeset.committed
        assert len(changeset.changes) == 1
    
    async def test_track_file_modification(self, change_tracker, temp_workspace):
        """Test tracking modification of existing files"""
        # Create initial file
        test_file = temp_workspace / "existing.py"
        test_file.write_text("original content")
        
        # Begin changeset
        changeset_id = change_tracker.begin_changeset(
            "Modify test file",
            agent_id="test_agent"
        )
        
        # Track modification
        change = change_tracker.track_file_change(
            changeset_id=changeset_id,
            file_path=test_file,
            new_content="modified content",
            agent_id="test_agent",
            task_id="task_2"
        )
        
        # Verify
        assert change.change_type == ChangeType.MODIFY
        assert change.original_content == "original content"
        assert test_file.read_text() == "modified content"
    
    async def test_rollback_changes(self, change_tracker, temp_workspace):
        """Test rolling back changes"""
        # Create and modify files
        file1 = temp_workspace / "file1.py"
        file2 = temp_workspace / "file2.py"
        file2.write_text("original file2")
        
        changeset_id = change_tracker.begin_changeset(
            "Multiple changes",
            agent_id="test_agent"
        )
        
        # Make changes
        change_tracker.track_file_change(
            changeset_id, file1, "new file1", "test_agent", "task_1"
        )
        change_tracker.track_file_change(
            changeset_id, file2, "modified file2", "test_agent", "task_2"
        )
        
        # Verify changes applied
        assert file1.exists()
        assert file2.read_text() == "modified file2"
        
        # Rollback
        rolled_back = change_tracker.rollback_changeset(changeset_id)
        
        # Verify rollback
        assert len(rolled_back) == 2
        assert not file1.exists()  # Created file should be gone
        assert file2.read_text() == "original file2"  # Modified file restored
    
    async def test_concurrent_file_locking(self, change_tracker, temp_workspace):
        """Test that file locking prevents concurrent modifications"""
        test_file = temp_workspace / "locked.py"
        test_file.write_text("initial")
        
        # First changeset locks the file
        cs1 = change_tracker.begin_changeset("Change 1", "agent1")
        change_tracker.track_file_change(
            cs1, test_file, "agent1 change", "agent1", "task1"
        )
        
        # Second changeset should fail to modify same file
        cs2 = change_tracker.begin_changeset("Change 2", "agent2")
        
        with pytest.raises(RuntimeError, match="locked by another changeset"):
            change_tracker.track_file_change(
                cs2, test_file, "agent2 change", "agent2", "task2"
            )


class TestSwarmTransactionManager:
    """Test distributed transaction management"""
    
    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def transaction_manager(self, temp_workspace):
        change_tracker = ChangeTracker(temp_workspace)
        return SwarmTransactionManager(change_tracker)
    
    async def test_basic_transaction_flow(self, transaction_manager):
        """Test basic transaction lifecycle"""
        # Begin transaction
        agents = [
            {'agent_id': 'agent1', 'agent_type': 'frontend'},
            {'agent_id': 'agent2', 'agent_type': 'backend'}
        ]
        
        transaction = await transaction_manager.begin_transaction(
            "Test transaction",
            agents=agents
        )
        
        assert transaction.phase == TransactionPhase.PREPARING
        assert len(transaction.agents) == 2
        
        # Mark agents complete
        await transaction_manager.mark_agent_complete(transaction.id, 'agent1')
        await transaction_manager.mark_agent_complete(transaction.id, 'agent2')
        
        # Wait for transaction to complete
        await asyncio.sleep(0.1)
        
        # Verify transaction completed
        summary = transaction_manager.get_transaction_summary(transaction.id)
        assert summary['phase'] == TransactionPhase.COMPLETED
        assert summary['success'] == True
    
    async def test_transaction_rollback_on_failure(self, transaction_manager):
        """Test that transaction rolls back when an agent fails"""
        agents = [
            {'agent_id': 'agent1', 'agent_type': 'frontend'},
            {'agent_id': 'agent2', 'agent_type': 'backend'}
        ]
        
        transaction = await transaction_manager.begin_transaction(
            "Failing transaction",
            agents=agents,
            rollback_on_failure=True
        )
        
        # One agent succeeds, one fails
        await transaction_manager.mark_agent_complete(transaction.id, 'agent1')
        await transaction_manager.mark_agent_failed(
            transaction.id, 'agent2', "Network error"
        )
        
        # Wait for rollback
        await asyncio.sleep(0.1)
        
        # Verify rollback occurred
        summary = transaction_manager.get_transaction_summary(transaction.id)
        assert summary['phase'] == TransactionPhase.FAILED
        assert summary['success'] == False
    
    async def test_transaction_barriers(self, transaction_manager):
        """Test synchronization barriers between agents"""
        agents = [
            {'agent_id': 'agent1', 'agent_type': 'frontend'},
            {'agent_id': 'agent2', 'agent_type': 'backend'},
            {'agent_id': 'agent3', 'agent_type': 'database'}
        ]
        
        transaction = await transaction_manager.begin_transaction(
            "Barrier test",
            agents=agents
        )
        
        # Create barrier
        barrier = await transaction_manager.create_barrier(
            transaction.id,
            "api_design",
            required_agents=['agent1', 'agent2']
        )
        
        # Track barrier arrivals
        arrivals = []
        
        async def agent_work(agent_id, data):
            # Simulate work
            await asyncio.sleep(0.1)
            
            # Wait at barrier
            shared_data = await transaction_manager.wait_at_barrier(
                transaction.id,
                "api_design",
                agent_id,
                shared_data={agent_id: data}
            )
            
            arrivals.append((agent_id, shared_data))
        
        # Run agents concurrently
        await asyncio.gather(
            agent_work('agent1', 'frontend_api'),
            agent_work('agent2', 'backend_api')
        )
        
        # Verify both agents have each other's data
        assert len(arrivals) == 2
        for agent_id, shared_data in arrivals:
            assert 'agent1' in shared_data
            assert 'agent2' in shared_data
            assert shared_data['agent1'] == {'agent1': 'frontend_api'}
            assert shared_data['agent2'] == {'agent2': 'backend_api'}
    
    async def test_shared_context(self, transaction_manager):
        """Test sharing context between agents"""
        agents = [
            {'agent_id': 'agent1', 'agent_type': 'analyzer'},
            {'agent_id': 'agent2', 'agent_type': 'implementer'}
        ]
        
        transaction = await transaction_manager.begin_transaction(
            "Context sharing test",
            agents=agents
        )
        
        # Agent 1 shares API spec
        api_spec = {
            'endpoints': ['/users', '/posts'],
            'auth': 'JWT'
        }
        await transaction_manager.share_context(
            transaction.id,
            'api_spec',
            api_spec,
            agent_id='agent1'
        )
        
        # Agent 2 retrieves spec
        retrieved_spec = await transaction_manager.get_shared_context(
            transaction.id,
            'api_spec'
        )
        
        assert retrieved_spec == api_spec


class TestStatePersistence:
    """Test state persistence and recovery"""
    
    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def state_manager(self, temp_workspace):
        return StatePersistenceManager(temp_workspace)
    
    async def test_save_and_load_state(self, state_manager):
        """Test basic state persistence"""
        # Save execution state
        execution_data = {
            'tasks': ['task1', 'task2'],
            'status': 'running',
            'progress': 50
        }
        
        state_id = await state_manager.save_state(
            StateType.EXECUTION_CONTEXT,
            'exec_123',
            execution_data
        )
        
        # Load state
        loaded = await state_manager.load_state(
            StateType.EXECUTION_CONTEXT,
            'exec_123'
        )
        
        assert loaded is not None
        assert loaded.state_data == execution_data
        assert loaded.entity_id == 'exec_123'
    
    async def test_state_update_merge(self, state_manager):
        """Test merging state updates"""
        # Initial state
        await state_manager.save_state(
            StateType.AGENT_STATE,
            'agent_1',
            {'status': 'idle', 'tasks_completed': 0}
        )
        
        # Update with merge
        await state_manager.update_state(
            StateType.AGENT_STATE,
            'agent_1',
            {'status': 'busy', 'current_task': 'task_1'},
            merge=True
        )
        
        # Load and verify merge
        state = await state_manager.load_state(
            StateType.AGENT_STATE,
            'agent_1'
        )
        
        assert state.state_data['status'] == 'busy'
        assert state.state_data['tasks_completed'] == 0  # Preserved
        assert state.state_data['current_task'] == 'task_1'  # Added
    
    async def test_recovery_points(self, state_manager):
        """Test creating and restoring from recovery points"""
        execution_id = 'exec_456'
        
        # Save various states
        await state_manager.save_state(
            StateType.EXECUTION_CONTEXT,
            execution_id,
            {'status': 'running', 'phase': 'analysis'}
        )
        
        await state_manager.save_state(
            StateType.AGENT_STATE,
            f"{execution_id}:agent1",
            {'progress': 25}
        )
        
        await state_manager.save_state(
            StateType.TASK_PROGRESS,
            f"{execution_id}:task1",
            {'completed': False, 'output': 'partial'}
        )
        
        # Create recovery point
        recovery_id = await state_manager.create_recovery_point(
            execution_id,
            "Mid-execution checkpoint"
        )
        
        # Modify states after checkpoint
        await state_manager.save_state(
            StateType.EXECUTION_CONTEXT,
            execution_id,
            {'status': 'failed', 'error': 'Something went wrong'}
        )
        
        # Restore from recovery point
        restore_info = await state_manager.restore_from_recovery_point(recovery_id)
        
        # Verify restoration
        assert restore_info['execution_id'] == execution_id
        assert restore_info['states_restored']['execution_context'] == 1
        assert restore_info['states_restored']['agent_states'] == 1
        assert restore_info['states_restored']['task_progress'] == 1
        
        # Check restored state
        exec_state = await state_manager.load_state(
            StateType.EXECUTION_CONTEXT,
            execution_id
        )
        assert exec_state.state_data['status'] == 'running'  # Restored
        assert 'error' not in exec_state.state_data  # Error removed
    
    async def test_auto_checkpoint(self, state_manager):
        """Test automatic checkpointing"""
        execution_id = 'exec_789'
        
        # Start auto checkpoint with short interval
        await state_manager.start_auto_checkpoint(execution_id, interval_seconds=0.5)
        
        # Save some state
        await state_manager.save_state(
            StateType.EXECUTION_CONTEXT,
            execution_id,
            {'status': 'running'}
        )
        
        # Wait for auto checkpoint
        await asyncio.sleep(1)
        
        # Stop auto checkpoint
        state_manager.stop_auto_checkpoint(execution_id)
        
        # Verify checkpoint was created
        # (Would need to expose recovery points list in real implementation)
        stats = state_manager.get_statistics()
        assert stats['recovery_points']['count'] > 0


class TestErrorRecovery:
    """Test error recovery and retry strategies"""
    
    @pytest.fixture
    def error_manager(self):
        return ErrorRecoveryManager()
    
    async def test_error_categorization(self, error_manager):
        """Test automatic error categorization"""
        # Rate limit error
        rate_error = Exception("429 Too Many Requests")
        category = error_manager.categorize_error(rate_error)
        assert category == ErrorCategory.RATE_LIMIT
        
        # Network error
        network_error = ConnectionError("Connection refused")
        category = error_manager.categorize_error(network_error)
        assert category == ErrorCategory.NETWORK
        
        # Auth error
        auth_error = Exception("401 Authorization failed")
        category = error_manager.categorize_error(auth_error)
        assert category == ErrorCategory.AUTHENTICATION
    
    async def test_retry_with_backoff(self, error_manager):
        """Test exponential backoff retry"""
        call_count = 0
        call_times = []
        
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            call_times.append(asyncio.get_event_loop().time())
            
            # Fail first 2 times, succeed on 3rd
            if call_count < 3:
                raise Exception("Temporarily unavailable")
            return "success"
        
        start_time = asyncio.get_event_loop().time()
        result = await error_manager.execute_with_retry(
            operation=flaky_operation,
            operation_name="test_operation"
        )
        
        assert result == "success"
        assert call_count == 3
        
        # Verify backoff delays
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay2 > delay1  # Exponential backoff
    
    async def test_circuit_breaker(self, error_manager):
        """Test circuit breaker functionality"""
        failure_count = 0
        
        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            raise Exception("Service unavailable")
        
        # Make requests until circuit opens
        for i in range(10):
            try:
                await error_manager.execute_with_retry(
                    operation=failing_operation,
                    operation_name="circuit_test",
                    custom_retry_config=error_manager.get_retry_config(ErrorCategory.TRANSIENT)
                )
            except:
                pass
        
        # Check circuit breaker status
        status = error_manager.get_circuit_breaker_status()
        assert 'circuit_test' in status
        
        # Circuit should be open after failures
        breaker_status = status['circuit_test']
        assert breaker_status['failure_count'] > 0
    
    async def test_error_pattern_detection(self, error_manager):
        """Test detection of error patterns"""
        # Generate some errors
        for i in range(5):
            error_manager.create_error_context(
                Exception("Rate limit exceeded"),
                agent_id="agent1",
                operation="api_call"
            )
        
        for i in range(3):
            error_manager.create_error_context(
                ConnectionError("Timeout"),
                agent_id="agent2",
                operation="fetch_data"
            )
        
        # Analyze patterns
        patterns = error_manager.get_error_patterns()
        
        assert patterns['total_errors'] == 8
        assert ErrorCategory.RATE_LIMIT in patterns['errors_by_category']
        assert ErrorCategory.NETWORK in patterns['errors_by_category']
        assert 'api_call' in patterns['errors_by_operation']
    
    async def test_recovery_suggestions(self, error_manager):
        """Test that appropriate recovery suggestions are provided"""
        # Rate limit error
        rate_context = error_manager.create_error_context(
            Exception("429 Rate limit exceeded"),
            operation="api_request"
        )
        suggestions = error_manager.suggest_recovery_actions(rate_context)
        
        assert any("throttling" in s.lower() for s in suggestions)
        assert any("backoff" in s.lower() for s in suggestions)
        
        # Network error
        network_context = error_manager.create_error_context(
            ConnectionError("Connection timeout"),
            operation="external_service"
        )
        suggestions = error_manager.suggest_recovery_actions(network_context)
        
        assert any("connectivity" in s.lower() for s in suggestions)
        assert any("timeout" in s.lower() for s in suggestions)


class TestResultValidation:
    """Test code validation system"""
    
    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def validator(self, temp_workspace):
        return ResultValidationManager(temp_workspace)
    
    async def test_python_syntax_validation(self, validator, temp_workspace):
        """Test Python syntax validation"""
        # Valid Python
        valid_file = temp_workspace / "valid.py"
        valid_file.write_text("""
def hello():
    print("Hello, world!")
""")
        
        # Invalid Python
        invalid_file = temp_workspace / "invalid.py"
        invalid_file.write_text("""
def broken(
    print("Missing parenthesis")
""")
        
        # Validate files
        valid_suite = await validator.validate_file(valid_file)
        assert valid_suite.passed_checks > 0
        assert not valid_suite.has_errors
        
        invalid_suite = await validator.validate_file(invalid_file)
        assert invalid_suite.has_errors
        assert any(r.message.startswith("Syntax error") for r in invalid_suite.results)
    
    async def test_security_validation(self, validator, temp_workspace):
        """Test security issue detection"""
        # File with security issues
        insecure_file = temp_workspace / "insecure.py"
        insecure_file.write_text("""
API_KEY = "sk-1234567890abcdef"
password = "admin123"

def risky():
    user_input = input()
    eval(user_input)  # Security risk!
""")
        
        # Run security validation
        suite = await validator.validate_security([insecure_file])
        
        # Should detect hardcoded secrets and eval()
        assert suite.has_errors
        security_issues = [r for r in suite.results if r.validator_name == "security_validation"]
        
        assert any("API key" in r.message for r in security_issues)
        assert any("password" in r.message for r in security_issues)
        assert any("eval()" in r.message for r in security_issues)
    
    async def test_validation_report_formatting(self, validator, temp_workspace):
        """Test formatting of validation reports"""
        # Create test file
        test_file = temp_workspace / "test.py"
        test_file.write_text("""
def long_line_function():
    return "This is a very long line that exceeds the recommended character limit and should trigger a style warning in the validation"

password = "hardcoded"
""")
        
        # Validate
        file_suite = await validator.validate_file(test_file)
        security_suite = await validator.validate_security([test_file])
        
        # Format report
        report = validator.format_validation_report([file_suite, security_suite])
        
        # Verify report structure
        assert "# Validation Report" in report
        assert "## Summary" in report
        assert "Total checks:" in report
        assert "### Errors" in report or "### Warnings" in report


# Integration test combining all safety systems
class TestSwarmSafetyIntegration:
    """Test integration of all safety systems"""
    
    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    async def test_safe_multi_agent_execution(self, temp_workspace):
        """Test a complete safe multi-agent execution flow"""
        # Initialize all systems
        change_tracker = ChangeTracker(temp_workspace)
        transaction_manager = SwarmTransactionManager(change_tracker)
        state_manager = StatePersistenceManager(temp_workspace)
        error_manager = ErrorRecoveryManager()
        validator = ResultValidationManager(temp_workspace)
        
        # Create test files
        (temp_workspace / "main.py").write_text("# Original main")
        (temp_workspace / "api.py").write_text("# Original API")
        
        # Begin transaction
        agents = [
            {'agent_id': 'frontend_agent', 'agent_type': 'frontend'},
            {'agent_id': 'backend_agent', 'agent_type': 'backend'}
        ]
        
        transaction = await transaction_manager.begin_transaction(
            "Update application",
            agents=agents
        )
        
        # Save initial state
        await state_manager.save_state(
            StateType.EXECUTION_CONTEXT,
            transaction.id,
            {'status': 'started', 'agents': agents}
        )
        
        # Simulate agent work with change tracking
        
        # Frontend agent
        frontend_cs = change_tracker.begin_changeset(
            "Frontend updates",
            agent_id='frontend_agent'
        )
        
        await transaction_manager.register_agent_changeset(
            transaction.id,
            'frontend_agent',
            frontend_cs
        )
        
        change_tracker.track_file_change(
            frontend_cs,
            temp_workspace / "main.py",
            "# Updated main with UI",
            'frontend_agent',
            'task_1'
        )
        
        # Backend agent
        backend_cs = change_tracker.begin_changeset(
            "Backend updates",
            agent_id='backend_agent'
        )
        
        await transaction_manager.register_agent_changeset(
            transaction.id,
            'backend_agent',
            backend_cs
        )
        
        change_tracker.track_file_change(
            backend_cs,
            temp_workspace / "api.py",
            "# Updated API endpoints",
            'backend_agent',
            'task_2'
        )
        
        # Create checkpoint
        checkpoint_id = await state_manager.create_recovery_point(
            transaction.id,
            "Pre-validation checkpoint"
        )
        
        # Validate changes
        modified_files = [
            temp_workspace / "main.py",
            temp_workspace / "api.py"
        ]
        
        validation_suite = await validator.validate_files(modified_files)
        
        # Complete transaction
        if not validation_suite.has_errors:
            # Commit changesets
            change_tracker.commit_changeset(frontend_cs)
            change_tracker.commit_changeset(backend_cs)
            
            # Mark agents complete
            await transaction_manager.mark_agent_complete(
                transaction.id,
                'frontend_agent',
                outputs={'files': ['main.py']}
            )
            
            await transaction_manager.mark_agent_complete(
                transaction.id,
                'backend_agent',
                outputs={'files': ['api.py']}
            )
            
            # Wait for transaction completion
            await asyncio.sleep(0.1)
            
            # Verify success
            summary = transaction_manager.get_transaction_summary(transaction.id)
            assert summary['success'] == True
            assert summary['phase'] == TransactionPhase.COMPLETED
            
            # Verify files were updated
            assert "Updated main" in (temp_workspace / "main.py").read_text()
            assert "Updated API" in (temp_workspace / "api.py").read_text()
        
        else:
            # Rollback on validation failure
            await transaction_manager.mark_agent_failed(
                transaction.id,
                'frontend_agent',
                "Validation failed"
            )
            
            # Verify rollback
            await asyncio.sleep(0.1)
            assert "Original main" in (temp_workspace / "main.py").read_text()
            assert "Original API" in (temp_workspace / "api.py").read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])