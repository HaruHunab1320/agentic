"""
Shared Memory system for inter-agent communication and coordination
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic.models.project import ProjectStructure
from agentic.models.task import Task, TaskResult
from agentic.utils.logging import LoggerMixin


class SharedMemory(LoggerMixin):
    """Shared memory system for agent coordination and communication"""
    
    def __init__(self):
        super().__init__()
        
        # Project state
        self._project_structure: Optional[ProjectStructure] = None
        self._recent_changes: List[Dict[str, Any]] = []
        
        # Inter-agent messaging
        self._message_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._message_lock = asyncio.Lock()
        
        # Task coordination
        self._active_tasks: Dict[str, Task] = {}
        self._task_progress: Dict[str, Dict[str, Any]] = {}
        self._task_lock = asyncio.Lock()
        
        # File access tracking
        self._file_locks: Dict[Path, str] = {}  # file_path -> agent_id
        self._file_access_log: List[Dict[str, Any]] = []
        self._file_lock = asyncio.Lock()
        
        # Agent status tracking
        self._agent_status: Dict[str, Dict[str, Any]] = {}
        self._agent_heartbeats: Dict[str, datetime] = {}
    
    # Project Structure Management
    def set_project_structure(self, structure: ProjectStructure) -> None:
        """Set the current project structure"""
        self._project_structure = structure
        self.logger.info("Project structure updated in shared memory")
    
    def get_project_structure(self) -> Optional[ProjectStructure]:
        """Get the current project structure"""
        return self._project_structure
    
    def add_recent_change(self, description: str, files: List[Path], agent_id: str) -> None:
        """Record a recent change"""
        change = {
            "timestamp": datetime.utcnow(),
            "description": description,
            "files": [str(f) for f in files],
            "agent_id": agent_id
        }
        
        self._recent_changes.append(change)
        
        # Keep only last 100 changes
        if len(self._recent_changes) > 100:
            self._recent_changes = self._recent_changes[-100:]
        
        self.logger.debug(f"Recorded change: {description}")
    
    def get_recent_changes(self, limit: int = 10, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get recent changes"""
        changes = self._recent_changes
        
        if since:
            changes = [c for c in changes if c["timestamp"] > since]
        
        return sorted(changes, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    # Inter-Agent Messaging
    async def send_inter_agent_message(self, sender_id: str, recipient_id: str, message: Dict[str, Any]) -> None:
        """Send message between agents"""
        async with self._message_lock:
            message_with_metadata = {
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "timestamp": datetime.utcnow(),
                "message": message
            }
            
            self._message_queues[recipient_id].append(message_with_metadata)
            self.logger.debug(f"Message sent from {sender_id} to {recipient_id}")
    
    async def get_messages_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get pending messages for an agent"""
        async with self._message_lock:
            messages = self._message_queues[agent_id].copy()
            self._message_queues[agent_id].clear()
            return messages
    
    # Task Coordination
    async def register_task(self, task: Task) -> None:
        """Register a task in shared memory"""
        async with self._task_lock:
            self._active_tasks[task.id] = task
            self._task_progress[task.id] = {
                "status": "registered",
                "progress": 0.0,
                "message": "Task registered",
                "last_update": datetime.utcnow()
            }
            self.logger.debug(f"Task registered: {task.id}")
    
    async def update_task_progress(self, task_id: str, progress: float, message: str) -> None:
        """Update task progress"""
        async with self._task_lock:
            if task_id in self._task_progress:
                self._task_progress[task_id].update({
                    "progress": progress,
                    "message": message,
                    "last_update": datetime.utcnow()
                })
                self.logger.debug(f"Task {task_id} progress: {progress:.1%} - {message}")
    
    async def complete_task(self, task_id: str, result: TaskResult) -> None:
        """Mark task as completed"""
        async with self._task_lock:
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
            
            if task_id in self._task_progress:
                self._task_progress[task_id].update({
                    "status": "completed",
                    "progress": 1.0,
                    "message": "Task completed",
                    "result": result,
                    "completed_at": datetime.utcnow()
                })
    
    def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task progress"""
        return self._task_progress.get(task_id)
    
    def get_all_task_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get progress for all tasks"""
        return self._task_progress.copy()
    
    # File Access Coordination
    async def request_file_lock(self, file_path: Path, agent_id: str, timeout: float = 30.0) -> bool:
        """Request exclusive access to a file"""
        async with self._file_lock:
            if file_path in self._file_locks:
                existing_agent = self._file_locks[file_path]
                if existing_agent != agent_id:
                    self.logger.warning(f"File {file_path} already locked by {existing_agent}")
                    return False
            
            self._file_locks[file_path] = agent_id
            self._file_access_log.append({
                "timestamp": datetime.utcnow(),
                "action": "lock",
                "file_path": str(file_path),
                "agent_id": agent_id
            })
            
            self.logger.debug(f"File lock acquired: {file_path} by {agent_id}")
            return True
    
    async def release_file_lock(self, file_path: Path, agent_id: str) -> bool:
        """Release file lock"""
        async with self._file_lock:
            if file_path not in self._file_locks:
                return False
            
            if self._file_locks[file_path] != agent_id:
                self.logger.warning(f"Agent {agent_id} attempted to release lock held by {self._file_locks[file_path]}")
                return False
            
            del self._file_locks[file_path]
            self._file_access_log.append({
                "timestamp": datetime.utcnow(),
                "action": "unlock",
                "file_path": str(file_path),
                "agent_id": agent_id
            })
            
            self.logger.debug(f"File lock released: {file_path} by {agent_id}")
            return True
    
    def get_file_locks(self) -> Dict[Path, str]:
        """Get current file locks"""
        return self._file_locks.copy()
    
    def is_file_locked(self, file_path: Path) -> bool:
        """Check if file is locked"""
        return file_path in self._file_locks
    
    def get_file_access_log(self, file_path: Optional[Path] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get file access log"""
        log = self._file_access_log
        
        if file_path:
            log = [entry for entry in log if entry["file_path"] == str(file_path)]
        
        return sorted(log, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    # Agent Status Management
    def update_agent_status(self, agent_id: str, status: Dict[str, Any]) -> None:
        """Update agent status"""
        self._agent_status[agent_id] = {
            **status,
            "last_update": datetime.utcnow()
        }
        self._agent_heartbeats[agent_id] = datetime.utcnow()
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status"""
        return self._agent_status.get(agent_id)
    
    def get_all_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all agents"""
        return self._agent_status.copy()
    
    def record_agent_heartbeat(self, agent_id: str) -> None:
        """Record agent heartbeat"""
        self._agent_heartbeats[agent_id] = datetime.utcnow()
    
    def get_stale_agents(self, threshold_minutes: int = 5) -> List[str]:
        """Get agents that haven't sent heartbeats recently"""
        threshold = datetime.utcnow() - timedelta(minutes=threshold_minutes)
        stale_agents = []
        
        for agent_id, last_heartbeat in self._agent_heartbeats.items():
            if last_heartbeat < threshold:
                stale_agents.append(agent_id)
        
        return stale_agents
    
    # Cleanup Methods
    def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up old completed tasks"""
        threshold = datetime.utcnow() - timedelta(hours=older_than_hours)
        cleaned_count = 0
        
        task_ids_to_remove = []
        for task_id, progress in self._task_progress.items():
            if (progress.get("status") == "completed" and 
                "completed_at" in progress and 
                progress["completed_at"] < threshold):
                task_ids_to_remove.append(task_id)
        
        for task_id in task_ids_to_remove:
            del self._task_progress[task_id]
            cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} old completed tasks")
        return cleaned_count
    
    def cleanup_old_messages(self, older_than_hours: int = 1) -> int:
        """Clean up old inter-agent messages"""
        threshold = datetime.utcnow() - timedelta(hours=older_than_hours)
        cleaned_count = 0
        
        for agent_id in self._message_queues:
            original_count = len(self._message_queues[agent_id])
            self._message_queues[agent_id] = [
                msg for msg in self._message_queues[agent_id]
                if msg["timestamp"] > threshold
            ]
            cleaned_count += original_count - len(self._message_queues[agent_id])
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old messages")
        
        return cleaned_count
    
    async def cleanup_stale_file_locks(self, threshold_minutes: int = 30) -> int:
        """Clean up file locks from stale agents"""
        async with self._file_lock:
            stale_agents = set(self.get_stale_agents(threshold_minutes))
            cleaned_count = 0
            
            locks_to_remove = []
            for file_path, agent_id in self._file_locks.items():
                if agent_id in stale_agents:
                    locks_to_remove.append(file_path)
            
            for file_path in locks_to_remove:
                del self._file_locks[file_path]
                cleaned_count += 1
                self.logger.warning(f"Released stale file lock: {file_path}")
            
            return cleaned_count 