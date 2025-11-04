"""
File Change Tracking and Rollback System for Multi-Agent Operations

This module provides atomic file operations with rollback capabilities,
essential for safe multi-agent code generation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from agentic.utils.logging import LoggerMixin


class ChangeType(str, Enum):
    """Types of file changes"""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class FileChange(BaseModel):
    """Represents a single file change"""
    file_path: str
    change_type: ChangeType
    original_content: Optional[str] = None
    new_content: Optional[str] = None
    original_hash: Optional[str] = None
    new_hash: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    task_id: str
    
    def get_size_delta(self) -> int:
        """Get the size difference of this change"""
        original_size = len(self.original_content) if self.original_content else 0
        new_size = len(self.new_content) if self.new_content else 0
        return new_size - original_size


class ChangeSet(BaseModel):
    """A set of related changes that should be applied atomically"""
    id: str
    description: str
    changes: List[FileChange] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    committed: bool = False
    rolled_back: bool = False
    parent_changeset_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_change(self, change: FileChange):
        """Add a change to this changeset"""
        self.changes.append(change)
    
    def get_affected_files(self) -> Set[str]:
        """Get all files affected by this changeset"""
        return {change.file_path for change in self.changes}
    
    def get_total_size_delta(self) -> int:
        """Get the total size change of this changeset"""
        return sum(change.get_size_delta() for change in self.changes)


class Checkpoint(BaseModel):
    """A checkpoint representing a consistent state"""
    id: str
    name: str
    description: str
    changeset_ids: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChangeTracker(LoggerMixin):
    """
    Tracks file changes and provides rollback capabilities for multi-agent operations.
    
    This is designed to handle the complexity of multiple agents modifying files
    simultaneously while maintaining consistency and providing recovery options.
    """
    
    def __init__(self, workspace_path: Path, storage_path: Optional[Path] = None):
        super().__init__()
        self.workspace_path = workspace_path
        self.storage_path = storage_path or workspace_path / ".agentic" / "changes"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory tracking
        self.active_changesets: Dict[str, ChangeSet] = {}
        self.committed_changesets: Dict[str, ChangeSet] = {}
        self.checkpoints: Dict[str, Checkpoint] = {}
        
        # File state tracking
        self.file_states: Dict[str, str] = {}  # file_path -> current_hash
        self.file_locks: Dict[str, str] = {}   # file_path -> changeset_id
        
        # Backup directory for rollback
        self.backup_dir = self.storage_path / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load existing state
        self._load_state()
    
    def _load_state(self):
        """Load persisted state from disk"""
        state_file = self.storage_path / "tracker_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                # Restore changesets
                for cs_data in state.get('committed_changesets', []):
                    cs = ChangeSet(**cs_data)
                    self.committed_changesets[cs.id] = cs
                    
                # Restore checkpoints
                for cp_data in state.get('checkpoints', []):
                    cp = Checkpoint(**cp_data)
                    self.checkpoints[cp.id] = cp
                    
                self.logger.info(f"Loaded {len(self.committed_changesets)} changesets, "
                               f"{len(self.checkpoints)} checkpoints")
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Persist state to disk"""
        state = {
            'committed_changesets': [cs.dict() for cs in self.committed_changesets.values()],
            'checkpoints': [cp.dict() for cp in self.checkpoints.values()],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        state_file = self.storage_path / "tracker_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _calculate_file_hash(self, content: str) -> str:
        """Calculate hash of file content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _backup_file(self, file_path: Path, changeset_id: str) -> Path:
        """Create a backup of a file"""
        backup_path = self.backup_dir / changeset_id / file_path.relative_to(self.workspace_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.exists():
            shutil.copy2(file_path, backup_path)
        else:
            # For deleted files, create a marker
            backup_path.with_suffix('.deleted').touch()
            
        return backup_path
    
    def begin_changeset(self, description: str, agent_id: str, parent_id: Optional[str] = None) -> str:
        """Begin tracking a new set of changes"""
        import uuid
        changeset_id = str(uuid.uuid4())
        
        changeset = ChangeSet(
            id=changeset_id,
            description=description,
            parent_changeset_id=parent_id,
            metadata={'agent_id': agent_id}
        )
        
        self.active_changesets[changeset_id] = changeset
        self.logger.info(f"Began changeset {changeset_id}: {description}")
        
        return changeset_id
    
    def track_file_change(self, 
                         changeset_id: str,
                         file_path: Path,
                         new_content: Optional[str],
                         agent_id: str,
                         task_id: str) -> FileChange:
        """Track a file change within a changeset"""
        
        if changeset_id not in self.active_changesets:
            raise ValueError(f"No active changeset with id {changeset_id}")
        
        # Check if file is locked by another changeset
        file_str = str(file_path)
        if file_str in self.file_locks and self.file_locks[file_str] != changeset_id:
            raise RuntimeError(f"File {file_path} is locked by another changeset")
        
        # Lock the file for this changeset
        self.file_locks[file_str] = changeset_id
        
        # Determine change type and capture original state
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            original_hash = self._calculate_file_hash(original_content)
            
            if new_content is None:
                change_type = ChangeType.DELETE
            else:
                change_type = ChangeType.MODIFY
        else:
            original_content = None
            original_hash = None
            change_type = ChangeType.CREATE
        
        # Create backup
        self._backup_file(file_path, changeset_id)
        
        # Calculate new hash
        new_hash = self._calculate_file_hash(new_content) if new_content else None
        
        # Create change record
        change = FileChange(
            file_path=file_str,
            change_type=change_type,
            original_content=original_content,
            new_content=new_content,
            original_hash=original_hash,
            new_hash=new_hash,
            agent_id=agent_id,
            task_id=task_id
        )
        
        # Add to changeset
        self.active_changesets[changeset_id].add_change(change)
        
        # Apply the change
        if new_content is not None:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        elif change_type == ChangeType.DELETE:
            file_path.unlink()
        
        self.logger.debug(f"Tracked {change_type.value} of {file_path} in changeset {changeset_id}")
        
        return change
    
    def commit_changeset(self, changeset_id: str) -> ChangeSet:
        """Commit a changeset, making it permanent"""
        if changeset_id not in self.active_changesets:
            raise ValueError(f"No active changeset with id {changeset_id}")
        
        changeset = self.active_changesets.pop(changeset_id)
        changeset.committed = True
        
        # Release file locks
        for file_path in changeset.get_affected_files():
            if self.file_locks.get(file_path) == changeset_id:
                del self.file_locks[file_path]
        
        # Move to committed
        self.committed_changesets[changeset_id] = changeset
        
        # Update file states
        for change in changeset.changes:
            if change.new_hash:
                self.file_states[change.file_path] = change.new_hash
            elif change.change_type == ChangeType.DELETE:
                self.file_states.pop(change.file_path, None)
        
        # Save state
        self._save_state()
        
        self.logger.info(f"Committed changeset {changeset_id} with {len(changeset.changes)} changes")
        
        return changeset
    
    def rollback_changeset(self, changeset_id: str) -> List[str]:
        """Rollback a changeset, restoring original files"""
        # Check if it's active or committed
        if changeset_id in self.active_changesets:
            changeset = self.active_changesets.pop(changeset_id)
        elif changeset_id in self.committed_changesets:
            changeset = self.committed_changesets[changeset_id]
        else:
            raise ValueError(f"No changeset with id {changeset_id}")
        
        if changeset.rolled_back:
            raise ValueError(f"Changeset {changeset_id} already rolled back")
        
        rolled_back_files = []
        errors = []
        
        # Rollback changes in reverse order
        for change in reversed(changeset.changes):
            try:
                file_path = Path(change.file_path)
                backup_path = self.backup_dir / changeset_id / file_path.relative_to(self.workspace_path)
                
                if change.change_type == ChangeType.CREATE:
                    # Delete the created file
                    if file_path.exists():
                        file_path.unlink()
                        
                elif change.change_type == ChangeType.DELETE:
                    # Restore the deleted file
                    if backup_path.exists():
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_path, file_path)
                        
                elif change.change_type == ChangeType.MODIFY:
                    # Restore original content
                    if backup_path.exists():
                        shutil.copy2(backup_path, file_path)
                    elif change.original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(change.original_content)
                
                rolled_back_files.append(change.file_path)
                
                # Update file state
                if change.original_hash:
                    self.file_states[change.file_path] = change.original_hash
                else:
                    self.file_states.pop(change.file_path, None)
                    
            except Exception as e:
                errors.append(f"Failed to rollback {change.file_path}: {e}")
                self.logger.error(f"Rollback error: {e}")
        
        # Mark as rolled back
        changeset.rolled_back = True
        
        # Release any remaining locks
        for file_path in changeset.get_affected_files():
            if self.file_locks.get(file_path) == changeset_id:
                del self.file_locks[file_path]
        
        # Save state
        self._save_state()
        
        if errors:
            self.logger.warning(f"Rollback completed with {len(errors)} errors")
        else:
            self.logger.info(f"Successfully rolled back {len(rolled_back_files)} files")
        
        return rolled_back_files
    
    def create_checkpoint(self, name: str, description: str, 
                         changeset_ids: Optional[List[str]] = None) -> str:
        """Create a checkpoint of the current state"""
        import uuid
        checkpoint_id = str(uuid.uuid4())
        
        # If no changesets specified, use all committed ones
        if changeset_ids is None:
            changeset_ids = list(self.committed_changesets.keys())
        
        checkpoint = Checkpoint(
            id=checkpoint_id,
            name=name,
            description=description,
            changeset_ids=changeset_ids
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        self._save_state()
        
        self.logger.info(f"Created checkpoint {name} with {len(changeset_ids)} changesets")
        
        return checkpoint_id
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> Tuple[List[str], List[str]]:
        """Rollback to a specific checkpoint"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"No checkpoint with id {checkpoint_id}")
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        # Find changesets to rollback (committed after checkpoint)
        changesets_to_rollback = []
        for cs_id, cs in self.committed_changesets.items():
            if cs_id not in checkpoint.changeset_ids and not cs.rolled_back:
                changesets_to_rollback.append(cs_id)
        
        # Sort by timestamp (newest first)
        changesets_to_rollback.sort(
            key=lambda cs_id: self.committed_changesets[cs_id].created_at,
            reverse=True
        )
        
        rolled_back_files = []
        errors = []
        
        # Rollback each changeset
        for cs_id in changesets_to_rollback:
            try:
                files = self.rollback_changeset(cs_id)
                rolled_back_files.extend(files)
            except Exception as e:
                errors.append(f"Failed to rollback changeset {cs_id}: {e}")
        
        self.logger.info(f"Rolled back to checkpoint {checkpoint.name}: "
                        f"{len(rolled_back_files)} files, {len(errors)} errors")
        
        return rolled_back_files, errors
    
    def get_file_history(self, file_path: Path) -> List[FileChange]:
        """Get the history of changes for a specific file"""
        file_str = str(file_path)
        history = []
        
        # Search through all changesets
        for cs in self.committed_changesets.values():
            for change in cs.changes:
                if change.file_path == file_str:
                    history.append(change)
        
        # Sort by timestamp
        history.sort(key=lambda c: c.timestamp)
        
        return history
    
    def get_changeset_summary(self, changeset_id: str) -> Dict[str, any]:
        """Get a summary of a changeset"""
        if changeset_id in self.active_changesets:
            changeset = self.active_changesets[changeset_id]
        elif changeset_id in self.committed_changesets:
            changeset = self.committed_changesets[changeset_id]
        else:
            raise ValueError(f"No changeset with id {changeset_id}")
        
        return {
            'id': changeset.id,
            'description': changeset.description,
            'created_at': changeset.created_at,
            'committed': changeset.committed,
            'rolled_back': changeset.rolled_back,
            'total_changes': len(changeset.changes),
            'affected_files': list(changeset.get_affected_files()),
            'size_delta': changeset.get_total_size_delta(),
            'changes_by_type': {
                change_type.value: len([c for c in changeset.changes if c.change_type == change_type])
                for change_type in ChangeType
            }
        }
    
    def cleanup_old_backups(self, days_to_keep: int = 7):
        """Clean up old backup files"""
        import time
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        cleaned = 0
        for changeset_dir in self.backup_dir.iterdir():
            if changeset_dir.is_dir():
                # Check if all files in this backup are old
                all_old = True
                for backup_file in changeset_dir.rglob('*'):
                    if backup_file.is_file() and backup_file.stat().st_mtime > cutoff_time:
                        all_old = False
                        break
                
                if all_old:
                    shutil.rmtree(changeset_dir)
                    cleaned += 1
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} old backup directories")
        
        return cleaned