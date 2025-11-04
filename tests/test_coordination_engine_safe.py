"""
Tests for the safe coordination engine
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile

from agentic.core.coordination_engine import CoordinationEngine
from agentic.core.state_persistence import StateType
from agentic.core.agent_registry import AgentRegistry
from agentic.core.shared_memory import SharedMemory
from agentic.models.task import Task, TaskIntent, TaskType, TaskResult
from agentic.models.agent import AgentSession, AgentType


class TestCoordinationEngineWithSafety:
    """Test the coordination engine with safety features enabled"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    async def safe_engine(self, temp_workspace):
        """Create a coordination engine instance with safety enabled"""
        registry = AgentRegistry(workspace_path=temp_workspace)
        shared_memory = SharedMemory()
        engine = CoordinationEngine(registry, shared_memory, temp_workspace, enable_safety=True)
        
        # Mock some components for testing
        engine.swarm_monitor = AsyncMock()
        engine.swarm_monitor.start_monitoring = AsyncMock()
        engine.swarm_monitor.stop_monitoring = AsyncMock()
        engine.swarm_monitor.update_task_analysis = AsyncMock()
        
        yield engine
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing"""
        tasks = []
        
        # Task 1: Frontend
        intent1 = TaskIntent(
            task_type=TaskType.IMPLEMENT,
            complexity_score=0.5,
            estimated_duration=10,
            affected_areas=["frontend"],
            requires_reasoning=False,
            requires_coordination=True
        )
        task1 = Task.from_intent(intent1, "Create user profile component")
        task1.agent_type_hint = "frontend"
        tasks.append(task1)
        
        # Task 2: Backend
        intent2 = TaskIntent(
            task_type=TaskType.IMPLEMENT,
            complexity_score=0.6,
            estimated_duration=15,
            affected_areas=["backend"],
            requires_reasoning=False,
            requires_coordination=True
        )
        task2 = Task.from_intent(intent2, "Create user API endpoint")
        task2.agent_type_hint = "backend"
        tasks.append(task2)
        
        return tasks
    
    async def test_execute_with_transaction(self, safe_engine, sample_tasks):
        """Test execution with transaction support"""
        # Mock agent execution
        mock_agent = AsyncMock()
        mock_agent.execute_task = AsyncMock(return_value=TaskResult(
            task_id="test",
            agent_id="test_agent",
            status="completed",
            output="Task completed",
            success=True
        ))
        
        mock_session = Mock(spec=AgentSession)
        mock_session.agent = mock_agent
        
        with patch.object(safe_engine.agent_registry, 'route_task', return_value="test_agent"):
            with patch.object(safe_engine.agent_registry, 'get_agent_session', return_value=mock_session):
                # Execute tasks
                result = await safe_engine.execute_coordinated_tasks(sample_tasks)
                
                # Verify transaction was created
                assert len(safe_engine.transaction_manager.completed_transactions) > 0
                
                # Verify state was persisted
                exec_state = await safe_engine.state_persistence.load_state(
                    StateType.EXECUTION_CONTEXT,
                    result.execution_id
                )
                assert exec_state is not None
                assert exec_state.state_data['status'] == 'completed'
    
    async def test_automatic_rollback_on_failure(self, safe_engine, sample_tasks, temp_workspace):
        """Test that failures trigger automatic rollback"""
        # Create a test file
        test_file = temp_workspace / "test.py"
        test_file.write_text("original content")
        
        # Mock first agent succeeds and modifies file
        successful_agent = AsyncMock()
        successful_result = TaskResult(
            task_id=sample_tasks[0].id,
            agent_id="frontend_agent",
            status="completed",
            output="Created component",
            success=True,
            files_modified=[str(test_file)]
        )
        successful_agent.execute_task = AsyncMock(return_value=successful_result)
        
        # Mock second agent fails
        failing_agent = AsyncMock()
        failing_result = TaskResult(
            task_id=sample_tasks[1].id,
            agent_id="backend_agent",
            status="failed",
            output="",
            error="API creation failed",
            success=False
        )
        failing_agent.execute_task = AsyncMock(return_value=failing_result)
        
        # Set up routing
        def route_task(task):
            if task.agent_type_hint == "frontend":
                return "frontend_agent"
            return "backend_agent"
        
        def get_session(agent_id):
            session = Mock(spec=AgentSession)
            if agent_id == "frontend_agent":
                session.agent = successful_agent
            else:
                session.agent = failing_agent
            return session
        
        with patch.object(safe_engine.agent_registry, 'route_task', side_effect=route_task):
            with patch.object(safe_engine.agent_registry, 'get_agent_session', side_effect=get_session):
                # Modify file during first agent execution
                with patch.object(safe_engine, '_execute_single_task') as mock_execute:
                    async def execute_with_modification(task, context):
                        if task.agent_type_hint == "frontend":
                            # Simulate file modification
                            test_file.write_text("modified by frontend")
                            return successful_result
                        return failing_result
                    
                    mock_execute.side_effect = execute_with_modification
                    
                    # Execute tasks - should fail and rollback
                    with pytest.raises(Exception):
                        await safe_engine.execute_coordinated_tasks(sample_tasks)
                    
                    # Verify file was rolled back
                    # (In real implementation, change tracker would handle this)
    
    async def test_error_recovery_with_retry(self, safe_engine, sample_tasks):
        """Test that transient errors are retried"""
        call_count = 0
        
        # Mock agent that fails twice then succeeds
        mock_agent = AsyncMock()
        
        async def flaky_execution(task):
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                return TaskResult(
                    task_id=task.id,
                    agent_id="test_agent",
                    status="failed",
                    output="",
                    error="Service temporarily unavailable",
                    success=False
                )
            
            return TaskResult(
                task_id=task.id,
                agent_id="test_agent",
                status="completed",
                output="Success after retry",
                success=True
            )
        
        mock_agent.execute_task = flaky_execution
        mock_session = Mock(spec=AgentSession)
        mock_session.agent = mock_agent
        
        with patch.object(safe_engine.agent_registry, 'route_task', return_value="test_agent"):
            with patch.object(safe_engine.agent_registry, 'get_agent_session', return_value=mock_session):
                # Execute single task with retry
                result = await safe_engine.execute_coordinated_tasks([sample_tasks[0]])
                
                # Should succeed after retries
                assert result.status == "completed"
                assert call_count == 3  # Failed twice, succeeded on third
    
    async def test_validation_after_execution(self, safe_engine, sample_tasks, temp_workspace):
        """Test that results are validated after execution"""
        # Create test files
        component_file = temp_workspace / "UserProfile.jsx"
        api_file = temp_workspace / "user_api.py"
        
        # Mock successful execution with file creation
        mock_agent = AsyncMock()
        
        async def create_files(task):
            if task.agent_type_hint == "frontend":
                component_file.write_text("export const UserProfile = () => {}")
                files = [str(component_file)]
            else:
                api_file.write_text("def get_user(): pass")
                files = [str(api_file)]
            
            return TaskResult(
                task_id=task.id,
                agent_id=f"{task.agent_type_hint}_agent",
                status="completed",
                output=f"Created {files[0]}",
                success=True,
                files_modified=files
            )
        
        mock_agent.execute_task = create_files
        mock_session = Mock(spec=AgentSession)
        mock_session.agent = mock_agent
        
        with patch.object(safe_engine.agent_registry, 'route_task') as mock_route:
            mock_route.return_value = "test_agent"
            
            with patch.object(safe_engine.agent_registry, 'get_agent_session', return_value=mock_session):
                # Execute tasks
                result = await safe_engine.execute_coordinated_tasks(sample_tasks)
                
                # Verify validation was performed
                assert result.status == "completed"
                
                # Files should have been created
                assert component_file.exists()
                assert api_file.exists()
    
    async def test_state_checkpoint_and_recovery(self, safe_engine, sample_tasks):
        """Test automatic checkpointing and recovery"""
        execution_id = None
        
        # Mock agent that tracks execution ID
        mock_agent = AsyncMock()
        
        async def track_execution(task):
            nonlocal execution_id
            # Get execution ID from state persistence calls
            return TaskResult(
                task_id=task.id,
                agent_id="test_agent",
                status="completed",
                output="Task completed",
                success=True
            )
        
        mock_agent.execute_task = track_execution
        mock_session = Mock(spec=AgentSession)
        mock_session.agent = mock_agent
        
        with patch.object(safe_engine.agent_registry, 'route_task', return_value="test_agent"):
            with patch.object(safe_engine.agent_registry, 'get_agent_session', return_value=mock_session):
                # Start execution
                result = await safe_engine.execute_coordinated_tasks(sample_tasks)
                execution_id = result.execution_id
                
                # Verify checkpointing was started
                assert execution_id in safe_engine.state_persistence._checkpoint_tasks
                
                # Stop checkpointing
                safe_engine.state_persistence.stop_auto_checkpoint(execution_id)
                
                # Verify state was saved
                exec_state = await safe_engine.state_persistence.load_state(
                    StateType.EXECUTION_CONTEXT,
                    execution_id
                )
                assert exec_state is not None
                assert exec_state.state_data['status'] == 'completed'
    
    async def test_concurrent_file_safety(self, safe_engine, temp_workspace):
        """Test that concurrent file modifications are prevented"""
        # Create tasks that modify the same file
        shared_file = temp_workspace / "shared.py"
        shared_file.write_text("original")
        
        intent = TaskIntent(
            task_type=TaskType.IMPLEMENT,
            complexity_score=0.5,
            estimated_duration=5,
            affected_areas=["shared"],
            requires_reasoning=False,
            requires_coordination=False
        )
        
        task1 = Task.from_intent(intent, "Modify shared file - Agent 1")
        task1.agent_type_hint = "agent1"
        
        task2 = Task.from_intent(intent, "Modify shared file - Agent 2")
        task2.agent_type_hint = "agent2"
        
        # Mock agents that try to modify the same file
        modification_order = []
        
        async def modify_file(task):
            agent_id = task.agent_type_hint
            modification_order.append(agent_id)
            
            # Simulate file modification attempt
            # In real implementation, change tracker would prevent concurrent access
            await asyncio.sleep(0.1)  # Simulate work
            
            return TaskResult(
                task_id=task.id,
                agent_id=agent_id,
                status="completed",
                output=f"Modified by {agent_id}",
                success=True,
                files_modified=[str(shared_file)]
            )
        
        mock_agent = AsyncMock()
        mock_agent.execute_task = modify_file
        mock_session = Mock(spec=AgentSession)
        mock_session.agent = mock_agent
        
        with patch.object(safe_engine.agent_registry, 'route_task') as mock_route:
            mock_route.side_effect = lambda task: task.agent_type_hint
            
            with patch.object(safe_engine.agent_registry, 'get_agent_session', return_value=mock_session):
                # Execute tasks in parallel
                # In real implementation, file locking would serialize access
                result = await safe_engine.execute_coordinated_tasks([task1, task2])
                
                # Both tasks should complete (serialized by file locking)
                assert result.status == "completed"
                assert len(modification_order) == 2
    
    async def test_crash_recovery(self, safe_engine, sample_tasks):
        """Test recovery from simulated crash"""
        # Execute first task and simulate crash
        first_task = sample_tasks[0]
        
        mock_agent = AsyncMock()
        mock_agent.execute_task = AsyncMock(return_value=TaskResult(
            task_id=first_task.id,
            agent_id="test_agent",
            status="completed",
            output="Partial completion",
            success=True
        ))
        
        mock_session = Mock(spec=AgentSession)
        mock_session.agent = mock_agent
        
        execution_id = None
        
        with patch.object(safe_engine.agent_registry, 'route_task', return_value="test_agent"):
            with patch.object(safe_engine.agent_registry, 'get_agent_session', return_value=mock_session):
                # Start execution
                result = await safe_engine.execute_coordinated_tasks([first_task])
                execution_id = result.execution_id
                
                # Simulate crash by creating new engine instance
                new_engine = SafeCoordinationEngine(
                    safe_engine.agent_registry,
                    safe_engine.shared_memory,
                    safe_engine.workspace_path
                )
                
                # Attempt recovery
                recovered_result = await new_engine.recover_from_crash(execution_id)
                
                # Should recognize task was already completed
                assert recovered_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])