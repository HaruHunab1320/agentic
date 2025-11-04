"""
State Persistence and Recovery System for Multi-Agent Operations

This module provides crash-resistant state persistence, enabling recovery
of interrupted operations and long-running agent tasks.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import zlib
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic.utils.logging import LoggerMixin


class StateType(str, Enum):
    """Types of state that can be persisted"""
    EXECUTION_CONTEXT = "execution_context"
    AGENT_STATE = "agent_state"
    TASK_PROGRESS = "task_progress"
    SHARED_MEMORY = "shared_memory"
    CHECKPOINT = "checkpoint"
    TRANSACTION = "transaction"


class PersistedState(BaseModel):
    """Represents a persisted state entry"""
    id: str
    state_type: StateType
    entity_id: str  # ID of the entity (execution, agent, task, etc.)
    state_data: Dict[str, Any]
    compressed: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecoveryPoint(BaseModel):
    """A point from which execution can be recovered"""
    id: str
    name: str
    description: str
    execution_id: str
    state_snapshot: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_automatic: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StatePersistenceManager(LoggerMixin):
    """
    Manages persistent state storage and recovery for the multi-agent system.
    
    Features:
    - Automatic state snapshots during execution
    - Crash recovery with minimal data loss
    - Efficient storage with compression
    - Expiration and cleanup of old states
    - Support for different state types
    """
    
    def __init__(self, storage_path: Path, auto_checkpoint_interval: int = 60):
        super().__init__()
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "state_persistence.db"
        self.auto_checkpoint_interval = auto_checkpoint_interval  # seconds
        
        # Initialize database
        self._init_database()
        
        # Background tasks
        self._checkpoint_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # In-memory cache for frequently accessed states
        self._state_cache: Dict[str, PersistedState] = {}
        self._cache_size_limit = 100
        
        # Don't start cleanup task immediately - let it be started after first use
        # self._start_cleanup_task()
    
    def _init_database(self):
        """Initialize the SQLite database for state storage"""
        try:
            # Ensure the directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create connection with proper settings
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS persisted_states (
                        id TEXT PRIMARY KEY,
                        state_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        state_data BLOB NOT NULL,
                        compressed INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        expires_at TEXT,
                        metadata TEXT,
                        UNIQUE(state_type, entity_id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_entity_id ON persisted_states(entity_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_state_type ON persisted_states(state_type)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at ON persisted_states(expires_at)
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS recovery_points (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        execution_id TEXT NOT NULL,
                        state_snapshot BLOB NOT NULL,
                        created_at TEXT NOT NULL,
                        is_automatic INTEGER DEFAULT 1,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_execution_id ON recovery_points(execution_id)
                """)
                
                conn.commit()
            
            self.logger.info(f"Initialized state persistence database at {self.db_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize state persistence database: {e}")
            # Re-raise to prevent silent failures
            raise
    
    def _start_cleanup_task(self):
        """Start background task for cleaning up expired states"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    cleaned = self._cleanup_expired_states()
                    if cleaned > 0:
                        self.logger.info(f"Cleaned up {cleaned} expired states")
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def _serialize_state(self, state_data: Dict[str, Any], compress: bool = True) -> bytes:
        """Serialize state data for storage"""
        # Convert to JSON-serializable format first
        json_data = json.dumps(state_data, default=str)
        data_bytes = json_data.encode('utf-8')
        
        # Compress if requested and data is large enough
        if compress and len(data_bytes) > 1024:  # 1KB threshold
            data_bytes = zlib.compress(data_bytes)
            
        return data_bytes
    
    def _deserialize_state(self, data_bytes: bytes, compressed: bool) -> Dict[str, Any]:
        """Deserialize state data from storage"""
        if compressed:
            data_bytes = zlib.decompress(data_bytes)
        
        json_data = data_bytes.decode('utf-8')
        return json.loads(json_data)
    
    async def save_state(self,
                        state_type: StateType,
                        entity_id: str,
                        state_data: Dict[str, Any],
                        ttl_seconds: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save state to persistent storage"""
        # Start cleanup task on first use if not already started
        if self._cleanup_task is None:
            self._start_cleanup_task()
        
        import uuid
        state_id = str(uuid.uuid4())
        
        # Determine if we should compress
        compress = len(json.dumps(state_data, default=str)) > 1024
        
        # Calculate expiration
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        # Create state object
        state = PersistedState(
            id=state_id,
            state_type=state_type,
            entity_id=entity_id,
            state_data=state_data,
            compressed=compress,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        # Serialize data
        serialized_data = self._serialize_state(state_data, compress)
        
        # Save to database
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO persisted_states 
                (id, state_type, entity_id, state_data, compressed, 
                 created_at, updated_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.id,
                state.state_type,
                state.entity_id,
                serialized_data,
                int(state.compressed),
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
                state.expires_at.isoformat() if state.expires_at else None,
                json.dumps(state.metadata)
            ))
            conn.commit()
        
        # Add to cache
        self._add_to_cache(state)
        
        self.logger.debug(f"Saved {state_type} state for {entity_id}")
        
        return state_id
    
    async def load_state(self,
                        state_type: StateType,
                        entity_id: str) -> Optional[PersistedState]:
        """Load state from persistent storage"""
        # Start cleanup task on first use if not already started
        if self._cleanup_task is None:
            self._start_cleanup_task()
        
        # Check cache first
        cache_key = f"{state_type}:{entity_id}"
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]
        
        # Load from database
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM persisted_states 
                WHERE state_type = ? AND entity_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (state_type, entity_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Deserialize state data
            state_data = self._deserialize_state(
                row['state_data'],
                bool(row['compressed'])
            )
            
            # Create state object
            state = PersistedState(
                id=row['id'],
                state_type=row['state_type'],
                entity_id=row['entity_id'],
                state_data=state_data,
                compressed=bool(row['compressed']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            
            # Add to cache
            self._add_to_cache(state)
            
            return state
    
    async def update_state(self,
                         state_type: StateType,
                         entity_id: str,
                         state_data: Dict[str, Any],
                         merge: bool = False) -> str:
        """Update existing state or create new one"""
        # Load existing state if merging
        if merge:
            existing = await self.load_state(state_type, entity_id)
            if existing:
                # Merge with existing data
                merged_data = existing.state_data.copy()
                merged_data.update(state_data)
                state_data = merged_data
        
        return await self.save_state(state_type, entity_id, state_data)
    
    async def delete_state(self,
                         state_type: StateType,
                         entity_id: str) -> bool:
        """Delete state from storage"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                DELETE FROM persisted_states 
                WHERE state_type = ? AND entity_id = ?
            """, (state_type, entity_id))
            conn.commit()
            
            deleted = cursor.rowcount > 0
        
        # Remove from cache
        cache_key = f"{state_type}:{entity_id}"
        if cache_key in self._state_cache:
            del self._state_cache[cache_key]
        
        if deleted:
            self.logger.debug(f"Deleted {state_type} state for {entity_id}")
        
        return deleted
    
    async def create_recovery_point(self,
                                  execution_id: str,
                                  name: str,
                                  description: str = "",
                                  is_automatic: bool = True) -> str:
        """Create a recovery point for an execution"""
        import uuid
        recovery_id = str(uuid.uuid4())
        
        # Collect all relevant states for this execution
        state_snapshot = {}
        
        # Get execution context
        exec_state = await self.load_state(StateType.EXECUTION_CONTEXT, execution_id)
        if exec_state:
            state_snapshot['execution_context'] = exec_state.state_data
        
        # Get all related agent states
        agent_states = await self._load_related_states(
            StateType.AGENT_STATE,
            execution_id
        )
        state_snapshot['agent_states'] = {
            state.entity_id: state.state_data
            for state in agent_states
        }
        
        # Get task progress
        task_states = await self._load_related_states(
            StateType.TASK_PROGRESS,
            execution_id
        )
        state_snapshot['task_progress'] = {
            state.entity_id: state.state_data
            for state in task_states
        }
        
        # Create recovery point
        recovery_point = RecoveryPoint(
            id=recovery_id,
            name=name,
            description=description,
            execution_id=execution_id,
            state_snapshot=state_snapshot,
            is_automatic=is_automatic
        )
        
        # Serialize snapshot
        serialized_snapshot = self._serialize_state(state_snapshot, compress=True)
        
        # Save to database
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO recovery_points 
                (id, name, description, execution_id, state_snapshot, 
                 created_at, is_automatic, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recovery_point.id,
                recovery_point.name,
                recovery_point.description,
                recovery_point.execution_id,
                serialized_snapshot,
                recovery_point.created_at.isoformat(),
                int(recovery_point.is_automatic),
                json.dumps(recovery_point.metadata)
            ))
            conn.commit()
        
        self.logger.info(f"Created recovery point '{name}' for execution {execution_id}")
        
        return recovery_id
    
    async def restore_from_recovery_point(self, recovery_id: str) -> Dict[str, Any]:
        """Restore state from a recovery point"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM recovery_points WHERE id = ?
            """, (recovery_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No recovery point with id {recovery_id}")
        
        # Deserialize snapshot
        state_snapshot = self._deserialize_state(
            row['state_snapshot'],
            compressed=True
        )
        
        # Restore execution context
        if 'execution_context' in state_snapshot:
            await self.save_state(
                StateType.EXECUTION_CONTEXT,
                row['execution_id'],
                state_snapshot['execution_context']
            )
        
        # Restore agent states
        for agent_id, agent_state in state_snapshot.get('agent_states', {}).items():
            await self.save_state(
                StateType.AGENT_STATE,
                agent_id,
                agent_state
            )
        
        # Restore task progress
        for task_id, task_state in state_snapshot.get('task_progress', {}).items():
            await self.save_state(
                StateType.TASK_PROGRESS,
                task_id,
                task_state
            )
        
        self.logger.info(f"Restored from recovery point {recovery_id}")
        
        return {
            'execution_id': row['execution_id'],
            'recovery_point_name': row['name'],
            'created_at': row['created_at'],
            'states_restored': {
                'execution_context': 1 if 'execution_context' in state_snapshot else 0,
                'agent_states': len(state_snapshot.get('agent_states', {})),
                'task_progress': len(state_snapshot.get('task_progress', {}))
            }
        }
    
    async def _load_related_states(self,
                                 state_type: StateType,
                                 execution_id: str) -> List[PersistedState]:
        """Load all states related to an execution"""
        states = []
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            # For agent and task states, entity_id contains execution_id
            cursor = conn.execute("""
                SELECT * FROM persisted_states 
                WHERE state_type = ? AND entity_id LIKE ?
                ORDER BY updated_at DESC
            """, (state_type, f"{execution_id}:%"))
            
            for row in cursor:
                state_data = self._deserialize_state(
                    row['state_data'],
                    bool(row['compressed'])
                )
                
                state = PersistedState(
                    id=row['id'],
                    state_type=row['state_type'],
                    entity_id=row['entity_id'],
                    state_data=state_data,
                    compressed=bool(row['compressed']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                states.append(state)
        
        return states
    
    def _add_to_cache(self, state: PersistedState):
        """Add state to in-memory cache"""
        cache_key = f"{state.state_type}:{state.entity_id}"
        
        # Implement simple LRU by removing oldest if at limit
        if len(self._state_cache) >= self._cache_size_limit:
            # Remove oldest entry
            oldest_key = min(
                self._state_cache.keys(),
                key=lambda k: self._state_cache[k].updated_at
            )
            del self._state_cache[oldest_key]
        
        self._state_cache[cache_key] = state
    
    def _cleanup_expired_states(self) -> int:
        """Clean up expired states from storage"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                DELETE FROM persisted_states 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (datetime.utcnow().isoformat(),))
            conn.commit()
            
            return cursor.rowcount
    
    async def start_auto_checkpoint(self,
                                  execution_id: str,
                                  interval_seconds: Optional[int] = None):
        """Start automatic checkpointing for an execution"""
        interval = interval_seconds or self.auto_checkpoint_interval
        
        async def checkpoint_loop():
            checkpoint_num = 0
            while execution_id in self._checkpoint_tasks:
                try:
                    await asyncio.sleep(interval)
                    
                    # Create automatic checkpoint
                    checkpoint_num += 1
                    await self.create_recovery_point(
                        execution_id=execution_id,
                        name=f"Auto checkpoint {checkpoint_num}",
                        description=f"Automatic checkpoint at {datetime.utcnow()}",
                        is_automatic=True
                    )
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Auto checkpoint error: {e}")
        
        # Start checkpoint task
        task = asyncio.create_task(checkpoint_loop())
        self._checkpoint_tasks[execution_id] = task
        
        self.logger.info(f"Started auto checkpointing for {execution_id} every {interval}s")
    
    def stop_auto_checkpoint(self, execution_id: str):
        """Stop automatic checkpointing for an execution"""
        if execution_id in self._checkpoint_tasks:
            task = self._checkpoint_tasks.pop(execution_id)
            task.cancel()
            self.logger.info(f"Stopped auto checkpointing for {execution_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Count states by type
            cursor = conn.execute("""
                SELECT state_type, COUNT(*) as count, 
                       SUM(LENGTH(state_data)) as total_size
                FROM persisted_states 
                GROUP BY state_type
            """)
            
            state_stats = {}
            for row in cursor:
                state_stats[row[0]] = {
                    'count': row[1],
                    'total_size': row[2]
                }
            
            # Count recovery points
            cursor = conn.execute("""
                SELECT COUNT(*) as count,
                       SUM(LENGTH(state_snapshot)) as total_size
                FROM recovery_points
            """)
            
            recovery_stats = cursor.fetchone()
            
            # Database file size
            db_size = self.db_path.stat().st_size
            
            return {
                'database_size': db_size,
                'state_statistics': state_stats,
                'recovery_points': {
                    'count': recovery_stats[0],
                    'total_size': recovery_stats[1]
                },
                'cache_size': len(self._state_cache),
                'active_checkpoints': len(self._checkpoint_tasks)
            }
    
    async def close(self):
        """Clean up resources"""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Cancel all checkpoint tasks
        for task in self._checkpoint_tasks.values():
            task.cancel()
        
        self._checkpoint_tasks.clear()
        self._state_cache.clear()
        
        self.logger.info("State persistence manager closed")