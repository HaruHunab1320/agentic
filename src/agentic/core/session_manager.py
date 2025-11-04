"""
Session Manager for Agentic

Provides session persistence and context management between Agentic runs.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from agentic.utils.logging import LoggerMixin


@dataclass
class SessionEntry:
    """Represents a single command/response in a session"""
    timestamp: float
    command: str
    response: str
    agent_used: Optional[str] = None
    files_modified: List[str] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.files_modified is None:
            self.files_modified = []


@dataclass
class Session:
    """Represents a complete Agentic session"""
    session_id: str
    workspace: str
    start_time: float
    end_time: Optional[float] = None
    entries: List[SessionEntry] = None
    
    def __post_init__(self):
        if self.entries is None:
            self.entries = []


class SessionManager(LoggerMixin):
    """Manages session persistence and history for Agentic"""
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path
        self.session_dir = Path.home() / ".agentic" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self.current_session: Optional[Session] = None
        
        # Session index file
        self.index_file = self.session_dir / "index.json"
        self.session_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load session index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except:
                return {"sessions": [], "last_session": None}
        return {"sessions": [], "last_session": None}
    
    def _save_index(self):
        """Save session index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.session_index, f, indent=2)
    
    def start_session(self) -> Session:
        """Start a new session"""
        session_id = f"session_{int(time.time())}_{self.workspace_path.name}"
        self.current_session = Session(
            session_id=session_id,
            workspace=str(self.workspace_path),
            start_time=time.time()
        )
        
        # Add to index
        self.session_index["sessions"].append({
            "session_id": session_id,
            "workspace": str(self.workspace_path),
            "start_time": self.current_session.start_time,
            "start_date": datetime.now().isoformat()
        })
        self.session_index["last_session"] = session_id
        self._save_index()
        
        self.logger.info(f"Started session: {session_id}")
        return self.current_session
    
    def add_entry(self, command: str, response: str, **kwargs):
        """Add an entry to the current session"""
        if not self.current_session:
            self.start_session()
        
        # For analysis tasks, store more of the response for context
        max_response_length = 2000 if 'analysis' in command.lower() else 500
        
        entry = SessionEntry(
            timestamp=time.time(),
            command=command,
            response=response[:max_response_length],  # Store more for analysis
            **kwargs
        )
        
        self.current_session.entries.append(entry)
        self._save_session()
    
    def end_session(self):
        """End the current session"""
        if self.current_session:
            self.current_session.end_time = time.time()
            self._save_session()
            
            # Update index
            for session in self.session_index["sessions"]:
                if session["session_id"] == self.current_session.session_id:
                    session["end_time"] = self.current_session.end_time
                    session["end_date"] = datetime.now().isoformat()
                    session["num_entries"] = len(self.current_session.entries)
                    break
            
            self._save_index()
            self.logger.info(f"Ended session: {self.current_session.session_id}")
            self.current_session = None
    
    def _save_session(self):
        """Save current session to disk"""
        if not self.current_session:
            return
        
        session_file = self.session_dir / f"{self.current_session.session_id}.json"
        
        # Convert to dict for JSON serialization
        session_data = {
            "session_id": self.current_session.session_id,
            "workspace": self.current_session.workspace,
            "start_time": self.current_session.start_time,
            "end_time": self.current_session.end_time,
            "entries": [asdict(entry) for entry in self.current_session.entries]
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from disk"""
        session_file = self.session_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            # Convert back to Session object
            session = Session(
                session_id=data["session_id"],
                workspace=data["workspace"],
                start_time=data["start_time"],
                end_time=data.get("end_time")
            )
            
            # Convert entries
            for entry_data in data.get("entries", []):
                session.entries.append(SessionEntry(**entry_data))
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def get_last_session(self) -> Optional[Session]:
        """Get the last session for the current workspace"""
        for session_info in reversed(self.session_index["sessions"]):
            if session_info["workspace"] == str(self.workspace_path):
                return self.load_session(session_info["session_id"])
        return None
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a session without loading all data"""
        for session_info in self.session_index["sessions"]:
            if session_info["session_id"] == session_id:
                return session_info
        return None
    
    def list_sessions(self, workspace_only: bool = True) -> List[Dict[str, Any]]:
        """List all sessions, optionally filtered by current workspace"""
        sessions = self.session_index["sessions"]
        
        if workspace_only:
            sessions = [s for s in sessions if s["workspace"] == str(self.workspace_path)]
        
        return sorted(sessions, key=lambda x: x["start_time"], reverse=True)
    
    def get_context_summary(self) -> str:
        """Get a summary of recent context for the AI"""
        last_session = self.get_last_session()
        if not last_session or not last_session.entries:
            return ""
        
        # Get last few entries
        recent_entries = last_session.entries[-5:]
        
        context_parts = ["Previous session context:"]
        for entry in recent_entries:
            context_parts.append(f"- Command: {entry.command[:100]}...")
            if entry.files_modified:
                context_parts.append(f"  Files: {', '.join(entry.files_modified)}")
        
        return "\n".join(context_parts)
    
    def get_recent_analysis(self) -> Optional[str]:
        """Get the most recent analysis response from current or last session"""
        # Check current session first
        if self.current_session and self.current_session.entries:
            for entry in reversed(self.current_session.entries[-10:]):
                if any(word in entry.response.lower() for word in ['analysis', 'implementation', 'missing', 'need to', 'would need']):
                    return entry.response
        
        # Check last session
        last_session = self.get_last_session()
        if last_session and last_session.entries:
            for entry in reversed(last_session.entries[-10:]):
                if any(word in entry.response.lower() for word in ['analysis', 'implementation', 'missing', 'need to', 'would need']):
                    return entry.response
        
        return None