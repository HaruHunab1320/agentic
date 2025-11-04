"""
Tests for the Intelligent Coordinator system
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from agentic.core.intelligent_coordinator import (
    IntelligentCoordinator, 
    AgentDiscovery,
    DiscoveryType,
    TaskPhase
)
from agentic.core.agent_registry import AgentRegistry
from agentic.core.shared_memory import SharedMemory
from agentic.models.config import AgenticConfig
from agentic.models.task import Task, TaskType, TaskIntent, TaskResult
from agentic.models.agent import AgentCapability, AgentType


class TestIntelligentCoordinator:
    """Test the intelligent coordinator functionality"""
    
    @pytest.fixture
    async def coordinator(self, tmp_path):
        """Create an intelligent coordinator instance"""
        config = AgenticConfig(
            workspace_name="test_workspace",
            workspace_path=tmp_path
        )
        agent_registry = AgentRegistry(config)
        shared_memory = SharedMemory()
        coordinator = IntelligentCoordinator(agent_registry, shared_memory)
        yield coordinator
    
    @pytest.fixture
    def mock_agent_registry(self):
        """Create a mock agent registry"""
        mock_registry = Mock(spec=AgentRegistry)
        
        # Mock agent session
        mock_session = Mock()
        mock_session.id = 'test-agent-1'
        mock_session.status = 'active'
        mock_session.error = None
        
        # Mock agent instance
        mock_agent = Mock()
        mock_agent.execute_task = AsyncMock(return_value=TaskResult(
            task_id='test-1',
            agent_id='test-agent-1',
            status='completed',
            output='Task completed successfully'
        ))
        mock_agent.set_discovery_callback = Mock()
        
        # Configure mock registry methods
        mock_registry.get_or_spawn_agent = AsyncMock(return_value=mock_session)
        mock_registry.get_agent_by_id = Mock(return_value=mock_agent)
        mock_registry.select_best_agent = Mock(return_value=AgentCapability(
            agent_type=AgentType.AIDER_TESTING,
            specializations=['testing'],
            supported_languages=['python', 'javascript']
        ))
        
        return mock_registry
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes properly"""
        assert coordinator is not None
        assert coordinator.agent_registry is not None
        assert coordinator.shared_memory is not None
        assert coordinator.current_phase == TaskPhase.EXPLORATION
    
    @pytest.mark.asyncio
    @patch('agentic.core.intelligent_coordinator.ProgressiveTaskGenerator.generate_initial_tasks')
    async def test_intelligent_execution(self, mock_generate_tasks, tmp_path):
        """Test intelligent execution with feedback loops"""
        config = AgenticConfig(
            workspace_name="test_workspace",
            workspace_path=tmp_path
        )
        
        mock_registry = Mock(spec=AgentRegistry)
        shared_memory = SharedMemory()
        
        # Mock task generation to return properly formed tasks
        intent = TaskIntent(
            task_type=TaskType.IMPLEMENT,
            complexity_score=0.5,
            estimated_duration=10
        )
        mock_task = Task(
            id='generated-task-1',
            command='Test task',
            intent=intent,
            coordination_context={'metadata': {'produces': {'test_complete'}}}
        )
        mock_generate_tasks.return_value = [mock_task]
        
        # Create coordinator with mocked registry
        coordinator = IntelligentCoordinator(mock_registry, shared_memory)
        
        # Mock agent behavior
        mock_session = Mock()
        mock_session.id = 'test-agent'
        mock_session.status = 'active'
        
        mock_agent = Mock()
        execution_count = 0
        
        async def mock_execute(*args, **kwargs):
            nonlocal execution_count
            execution_count += 1
            return TaskResult(
                task_id=f'task-{execution_count}',
                agent_id='test-agent',
                status='completed',
                output=f'Execution {execution_count} complete'
            )
        
        mock_agent.execute_task = mock_execute
        mock_agent.set_discovery_callback = Mock()
        
        mock_registry.get_or_spawn_agent = AsyncMock(return_value=mock_session)
        mock_registry.get_agent_by_id = Mock(return_value=mock_agent)
        mock_registry.select_best_agent = Mock(return_value=AgentCapability(
            agent_type=AgentType.AIDER_TESTING,
            specializations=['testing'],
            supported_languages=['python', 'javascript']
        ))
        
        # Execute with intelligence
        result = await coordinator.execute_with_intelligence("Test objective")
        
        assert result['objective'] == "Test objective"
        assert result['tasks_completed'] > 0
        assert 'duration_seconds' in result
        assert 'iterations' in result
    
    @pytest.mark.asyncio
    async def test_task_dependency_handling(self, coordinator):
        """Test that dependencies are properly managed"""
        # Add tasks with dependencies
        intent1 = TaskIntent(
            task_type=TaskType.IMPLEMENT,
            complexity_score=0.5,
            estimated_duration=10
        )
        task1 = Task(
            id='task-1',
            command='First task',
            intent=intent1,
            coordination_context={'metadata': {'produces': {'api_ready'}}}
        )
        
        intent2 = TaskIntent(
            task_type=TaskType.TEST,
            complexity_score=0.3,
            estimated_duration=5
        )
        task2 = Task(
            id='task-2',
            command='Second task',
            intent=intent2,
            coordination_context={'metadata': {'depends_on': {'api_ready'}}}
        )
        
        coordinator._add_task_to_graph(task1)
        coordinator._add_task_to_graph(task2)
        
        # Initially only task1 should be ready
        ready_tasks = coordinator.dependency_graph.get_ready_tasks()
        assert 'task-1' in ready_tasks
        assert 'task-2' not in ready_tasks
        
        # After completing task1, task2 should be ready
        coordinator.dependency_graph.mark_complete('task-1', {'api_ready'})
        ready_tasks = coordinator.dependency_graph.get_ready_tasks()
        assert 'task-2' in ready_tasks
    
    @pytest.mark.asyncio
    async def test_discovery_processing(self, coordinator):
        """Test that agent discoveries generate new tasks"""
        discovery = AgentDiscovery(
            agent_id='test-agent',
            discovery_type=DiscoveryType.API_READY,
            severity='info',
            context={'endpoint': '/api/users'},
            suggestions=['Generate tests for API'],
            affected_files=[Path('api.py')]
        )
        
        # Process discovery
        new_tasks = await coordinator.feedback_processor.process(discovery)
        
        # Should generate follow-up tasks
        assert len(new_tasks) > 0
        assert any('test' in task.command.lower() for task in new_tasks)
    
    @pytest.mark.asyncio
    async def test_phase_progression(self, coordinator):
        """Test phase progression logic"""
        # Start in exploration phase
        assert coordinator.current_phase == TaskPhase.EXPLORATION
        
        # Create and add a task that produces codebase_analysis
        intent = TaskIntent(
            task_type=TaskType.EXPLAIN,
            complexity_score=0.3,
            estimated_duration=15
        )
        task = Task(
            id='exploration-task',
            command='Analyze codebase',
            intent=intent,
            coordination_context={'metadata': {'produces': {'codebase_analysis'}}}
        )
        coordinator._add_task_to_graph(task)
        
        # Complete the task
        coordinator.dependency_graph.mark_complete('exploration-task', {'codebase_analysis'})
        
        # Should be ready to advance
        should_advance = await coordinator._should_advance_phase()
        assert should_advance
        
        # Get next phase
        next_phase = coordinator._get_next_phase()
        assert next_phase == TaskPhase.DESIGN


class TestCoordinatorErrorHandling:
    """Test error handling in the coordinator"""
    
    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, tmp_path):
        """Test handling of agent execution failures"""
        config = AgenticConfig(
            workspace_name="test_workspace",
            workspace_path=tmp_path
        )
        
        mock_registry = Mock(spec=AgentRegistry)
        mock_registry.workspace_path = tmp_path
        shared_memory = SharedMemory()
        coordinator = IntelligentCoordinator(mock_registry, shared_memory)
        
        # Mock failed agent spawn
        mock_session = Mock()
        mock_session.status = 'failed'
        mock_session.error = 'Agent spawn failed'
        
        mock_registry.get_or_spawn_agent = AsyncMock(return_value=mock_session)
        mock_registry.select_best_agent = Mock(return_value=AgentCapability(
            agent_type=AgentType.AIDER_TESTING,
            specializations=['testing'],
            supported_languages=['python', 'javascript']
        ))
        
        # Create a test task
        intent = TaskIntent(
            task_type=TaskType.TEST,
            complexity_score=0.5,
            estimated_duration=10
        )
        task = Task(id='test-task', command='Test command', intent=intent)
        coordinator.active_tasks[task.id] = task
        
        # Execute task (should handle failure gracefully)
        results = await coordinator._execute_tasks_with_feedback([task])
        
        assert 'test-task' in results
        result, discoveries = results['test-task']
        assert result.status == 'failed'
        assert 'Failed to spawn agent' in result.error
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, tmp_path):
        """Test handling of exceptions during execution"""
        config = AgenticConfig(
            workspace_name="test_workspace",
            workspace_path=tmp_path
        )
        
        mock_registry = Mock(spec=AgentRegistry)
        mock_registry.workspace_path = tmp_path
        shared_memory = SharedMemory()
        coordinator = IntelligentCoordinator(mock_registry, shared_memory)
        
        # Mock agent that throws exception
        mock_session = Mock()
        mock_session.status = 'active'
        
        mock_agent = Mock()
        mock_agent.execute_task = AsyncMock(side_effect=Exception("Execution error"))
        mock_agent.set_discovery_callback = Mock()
        
        mock_registry.get_or_spawn_agent = AsyncMock(return_value=mock_session)
        mock_registry.get_agent_by_id = Mock(return_value=mock_agent)
        mock_registry.select_best_agent = Mock(return_value=AgentCapability(
            agent_type=AgentType.AIDER_TESTING,
            specializations=['testing'],
            supported_languages=['python', 'javascript']
        ))
        
        # Create a test task
        intent = TaskIntent(
            task_type=TaskType.TEST,
            complexity_score=0.5,
            estimated_duration=10
        )
        task = Task(id='test-task', command='Test command', intent=intent)
        coordinator.active_tasks[task.id] = task
        
        # Execute task (should handle exception gracefully)
        results = await coordinator._execute_tasks_with_feedback([task])
        
        assert 'test-task' in results
        result, discoveries = results['test-task']
        assert result.status == 'failed'
        assert 'Execution error' in result.error