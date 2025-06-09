"""
Agent monitoring interface for status reporting
"""

from typing import Optional, Callable
from agentic.core.swarm_monitor import AgentStatus


class MonitoringMixin:
    """Mixin for agents to report status to monitor"""
    
    def __init__(self):
        self._status_callback: Optional[Callable] = None
        self._task_start_callback: Optional[Callable] = None
        self._task_complete_callback: Optional[Callable] = None
        self._file_created_callback: Optional[Callable] = None
        self._error_callback: Optional[Callable] = None
    
    def set_monitoring_callbacks(self, 
                                status_update: Optional[Callable] = None,
                                task_start: Optional[Callable] = None,
                                task_complete: Optional[Callable] = None,
                                file_created: Optional[Callable] = None,
                                error_report: Optional[Callable] = None):
        """Set callbacks for monitoring events"""
        self._status_callback = status_update
        self._task_start_callback = task_start
        self._task_complete_callback = task_complete
        self._file_created_callback = file_created
        self._error_callback = error_report
    
    def report_status(self, status: AgentStatus, message: Optional[str] = None):
        """Report current status"""
        if self._status_callback:
            try:
                self._status_callback(status, message)
            except Exception:
                pass  # Don't let monitoring errors affect execution
    
    def report_task_start(self, task_id: str, task_name: str):
        """Report task starting"""
        if self._task_start_callback:
            try:
                self._task_start_callback(task_id, task_name)
            except Exception:
                pass
    
    def report_task_complete(self, task_id: str, success: bool = True):
        """Report task completion"""
        if self._task_complete_callback:
            try:
                self._task_complete_callback(task_id, success)
            except Exception:
                pass
    
    def report_file_created(self, file_path: str):
        """Report file creation"""
        if self._file_created_callback:
            try:
                self._file_created_callback(file_path)
            except Exception:
                pass
    
    def report_error(self, error: str):
        """Report error"""
        if self._error_callback:
            try:
                self._error_callback(error)
            except Exception:
                pass