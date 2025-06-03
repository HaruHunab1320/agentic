"""
Tests for the SharedMemory system
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agentic.core.shared_memory import SharedMemory
from agentic.models.project import ProjectStructure, TechStack
from agentic.models.task import Task, TaskResult, TaskType, TaskIntent


@pytest.fixture
def shared_memory():
    """Create a SharedMemory instance for testing"""
    return SharedMemory()


@pytest.fixture
def sample_project_structure():
    """Create a sample project structure"""
    return ProjectStructure(
        root_path=Path("/test/project"),
        tech_stack=TechStack(
            languages=["python", "javascript"],
            frameworks=["fastapi", "react"],
            databases=["postgresql"],
            build_tools=["docker"]
        ),
        source_directories=[Path("/test/project/src")],
        test_directories=[Path("/test/project/tests")],
        config_files=[Path("/test/project/pyproject.toml")]
    )


@pytest.fixture
def sample_task():
    """Create a sample task"""
    intent = TaskIntent(
        task_type=TaskType.IMPLEMENT,
        complexity_score=0.7,
        estimated_duration=120,
        affected_areas=["backend", "security"],
        requires_reasoning=False,
        requires_coordination=True
    )
    
    return Task(
        command="Implement user authentication",
        intent=intent
    )


class TestSharedMemory:
    """Test cases for SharedMemory"""
    
    def test_project_structure_management(self, shared_memory, sample_project_structure):
        """Test project structure set/get"""
        # Initially should be None
        assert shared_memory.get_project_structure() is None
        
        # Set and retrieve
        shared_memory.set_project_structure(sample_project_structure)
        retrieved = shared_memory.get_project_structure()
        
        assert retrieved is not None
        assert retrieved.root_path == Path("/test/project")
        assert retrieved.tech_stack.languages == ["python", "javascript"]
    
    def test_recent_changes_tracking(self, shared_memory):
        """Test recent changes recording and retrieval"""
        # Add some changes
        files1 = [Path("/test/file1.py")]
        files2 = [Path("/test/file2.js")]
        
        shared_memory.add_recent_change("Updated user model", files1, "agent1")
        shared_memory.add_recent_change("Added API endpoint", files2, "agent2")
        
        # Get recent changes
        changes = shared_memory.get_recent_changes(limit=5)
        
        assert len(changes) == 2
        assert changes[0]["description"] == "Added API endpoint"  # Most recent first
        assert changes[1]["description"] == "Updated user model"
        assert changes[0]["agent_id"] == "agent2"
        assert changes[1]["agent_id"] == "agent1"
    
    def test_recent_changes_with_since_filter(self, shared_memory):
        """Test recent changes with time filter"""
        files = [Path("/test/file.py")]
        
        # Add change
        shared_memory.add_recent_change("Old change", files, "agent1")
        
        # Get changes since future time (should be empty)
        future_time = datetime.utcnow() + timedelta(minutes=1)
        changes = shared_memory.get_recent_changes(since=future_time)
        assert len(changes) == 0
        
        # Get changes since past time (should include our change)
        past_time = datetime.utcnow() - timedelta(minutes=1)
        changes = shared_memory.get_recent_changes(since=past_time)
        assert len(changes) == 1
    
    def test_recent_changes_limit(self, shared_memory):
        """Test recent changes respects history limit"""
        files = [Path("/test/file.py")]
        
        # Add more than the limit (100) changes
        for i in range(105):
            shared_memory.add_recent_change(f"Change {i}", files, "agent1")
        
        # Should only keep last 100
        assert len(shared_memory._recent_changes) == 100
        
        # Most recent should be the last one added
        changes = shared_memory.get_recent_changes(limit=1)
        assert changes[0]["description"] == "Change 104"
    
    @pytest.mark.asyncio
    async def test_inter_agent_messaging(self, shared_memory):
        """Test inter-agent messaging"""
        # Send messages
        await shared_memory.send_inter_agent_message(
            "agent1", "agent2", {"type": "status", "data": "working"}
        )
        await shared_memory.send_inter_agent_message(
            "agent3", "agent2", {"type": "request", "data": "help needed"}
        )
        
        # Get messages for agent2
        messages = await shared_memory.get_messages_for_agent("agent2")
        
        assert len(messages) == 2
        assert messages[0]["sender_id"] == "agent1"
        assert messages[1]["sender_id"] == "agent3"
        assert messages[0]["message"]["type"] == "status"
        assert messages[1]["message"]["type"] == "request"
        
        # Messages should be cleared after retrieval
        messages_again = await shared_memory.get_messages_for_agent("agent2")
        assert len(messages_again) == 0
    
    @pytest.mark.asyncio
    async def test_task_coordination(self, shared_memory, sample_task):
        """Test task registration and progress tracking"""
        # Register task
        await shared_memory.register_task(sample_task)
        
        # Check task progress
        progress = shared_memory.get_task_progress(sample_task.id)
        assert progress is not None
        assert progress["status"] == "registered"
        assert progress["progress"] == 0.0
        
        # Update progress
        await shared_memory.update_task_progress(sample_task.id, 0.5, "Half complete")
        progress = shared_memory.get_task_progress(sample_task.id)
        assert progress["progress"] == 0.5
        assert progress["message"] == "Half complete"
        
        # Complete task
        result = TaskResult(
            task_id=sample_task.id,
            agent_id="test_agent",
            status="completed",
            output="Task completed successfully"
        )
        await shared_memory.complete_task(sample_task.id, result)
        
        progress = shared_memory.get_task_progress(sample_task.id)
        assert progress["status"] == "completed"
        assert progress["progress"] == 1.0
        assert "completed_at" in progress
    
    @pytest.mark.asyncio
    async def test_file_access_coordination(self, shared_memory):
        """Test file lock coordination"""
        file_path = Path("/test/important_file.py")
        
        # Request lock
        success = await shared_memory.request_file_lock(file_path, "agent1")
        assert success is True
        assert shared_memory.is_file_locked(file_path)
        
        # Try to lock again with different agent (should fail)
        success = await shared_memory.request_file_lock(file_path, "agent2")
        assert success is False
        
        # Same agent can "re-lock" (should succeed)
        success = await shared_memory.request_file_lock(file_path, "agent1")
        assert success is True
        
        # Get file locks
        locks = shared_memory.get_file_locks()
        assert file_path in locks
        assert locks[file_path] == "agent1"
        
        # Release lock
        success = await shared_memory.release_file_lock(file_path, "agent1")
        assert success is True
        assert not shared_memory.is_file_locked(file_path)
        
        # Try to release already released lock
        success = await shared_memory.release_file_lock(file_path, "agent1")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_file_lock_wrong_agent_release(self, shared_memory):
        """Test that agents can't release locks held by others"""
        file_path = Path("/test/file.py")
        
        # Agent1 locks file
        await shared_memory.request_file_lock(file_path, "agent1")
        
        # Agent2 tries to release (should fail)
        success = await shared_memory.release_file_lock(file_path, "agent2")
        assert success is False
        
        # File should still be locked by agent1
        assert shared_memory.is_file_locked(file_path)
        locks = shared_memory.get_file_locks()
        assert locks[file_path] == "agent1"
    
    def test_agent_status_management(self, shared_memory):
        """Test agent status tracking"""
        # Update agent status
        status = {"state": "working", "task_count": 3}
        shared_memory.update_agent_status("agent1", status)
        
        # Get status
        retrieved_status = shared_memory.get_agent_status("agent1")
        assert retrieved_status is not None
        assert retrieved_status["state"] == "working"
        assert retrieved_status["task_count"] == 3
        assert "last_update" in retrieved_status
        
        # Get all agent statuses
        all_status = shared_memory.get_all_agent_status()
        assert "agent1" in all_status
        
        # Record heartbeat
        shared_memory.record_agent_heartbeat("agent1")
        shared_memory.record_agent_heartbeat("agent2")
        
        # Get stale agents (none should be stale yet)
        stale = shared_memory.get_stale_agents(threshold_minutes=1)
        assert len(stale) == 0
    
    def test_stale_agent_detection(self, shared_memory):
        """Test stale agent detection"""
        # Record heartbeats with backdated time
        old_time = datetime.utcnow() - timedelta(minutes=10)
        shared_memory._agent_heartbeats["agent1"] = old_time
        shared_memory._agent_heartbeats["agent2"] = datetime.utcnow()
        
        # Check for stale agents
        stale = shared_memory.get_stale_agents(threshold_minutes=5)
        assert "agent1" in stale
        assert "agent2" not in stale
    
    def test_cleanup_completed_tasks(self, shared_memory):
        """Test cleanup of old completed tasks"""
        # Add some task progress (simulating old completed tasks)
        old_time = datetime.utcnow() - timedelta(hours=25)
        recent_time = datetime.utcnow() - timedelta(hours=1)
        
        shared_memory._task_progress = {
            "old_task": {
                "status": "completed",
                "completed_at": old_time
            },
            "recent_task": {
                "status": "completed", 
                "completed_at": recent_time
            },
            "running_task": {
                "status": "running"
            }
        }
        
        # Cleanup old tasks
        cleaned_count = shared_memory.cleanup_completed_tasks(older_than_hours=24)
        
        assert cleaned_count == 1
        assert "old_task" not in shared_memory._task_progress
        assert "recent_task" in shared_memory._task_progress
        assert "running_task" in shared_memory._task_progress
    
    def test_cleanup_old_messages(self, shared_memory):
        """Test cleanup of old messages"""
        old_time = datetime.utcnow() - timedelta(hours=2)
        recent_time = datetime.utcnow() - timedelta(minutes=30)
        
        # Add old and recent messages
        shared_memory._message_queues = {
            "agent1": [
                {"timestamp": old_time, "message": "old"},
                {"timestamp": recent_time, "message": "recent"}
            ]
        }
        
        # Cleanup old messages
        cleaned_count = shared_memory.cleanup_old_messages(older_than_hours=1)
        
        assert cleaned_count == 1
        messages = shared_memory._message_queues["agent1"]
        assert len(messages) == 1
        assert messages[0]["message"] == "recent"
    
    @pytest.mark.asyncio
    async def test_cleanup_stale_file_locks(self, shared_memory):
        """Test cleanup of stale file locks"""
        file1 = Path("/test/file1.py")
        file2 = Path("/test/file2.py")
        
        # Set up stale and active agents
        old_time = datetime.utcnow() - timedelta(minutes=35)
        shared_memory._agent_heartbeats = {
            "stale_agent": old_time,
            "active_agent": datetime.utcnow()
        }
        
        # Set up file locks
        shared_memory._file_locks = {
            file1: "stale_agent",
            file2: "active_agent"
        }
        
        # Cleanup stale locks
        cleaned_count = await shared_memory.cleanup_stale_file_locks(threshold_minutes=30)
        
        assert cleaned_count == 1
        assert file1 not in shared_memory._file_locks
        assert file2 in shared_memory._file_locks
    
    def test_file_access_log(self, shared_memory):
        """Test file access logging"""
        file_path = Path("/test/file.py")
        
        # Simulate some file access
        shared_memory._file_access_log = [
            {
                "timestamp": datetime.utcnow() - timedelta(minutes=5),
                "action": "lock",
                "file_path": str(file_path),
                "agent_id": "agent1"
            },
            {
                "timestamp": datetime.utcnow() - timedelta(minutes=3),
                "action": "unlock", 
                "file_path": str(file_path),
                "agent_id": "agent1"
            },
            {
                "timestamp": datetime.utcnow() - timedelta(minutes=1),
                "action": "lock",
                "file_path": "/other/file.py",
                "agent_id": "agent2"
            }
        ]
        
        # Get all access log
        all_log = shared_memory.get_file_access_log()
        assert len(all_log) == 3
        
        # Get log for specific file
        file_log = shared_memory.get_file_access_log(file_path)
        assert len(file_log) == 2
        assert all(entry["file_path"] == str(file_path) for entry in file_log)
        
        # Check ordering (newest first)
        assert file_log[0]["action"] == "unlock"
        assert file_log[1]["action"] == "lock" 