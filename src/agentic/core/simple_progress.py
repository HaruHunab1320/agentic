"""
Simple Progress Monitor for Single Agent Queries

Provides lightweight progress updates without screen clearing.
"""

import asyncio
import time
from typing import Optional, Callable
from dataclasses import dataclass

from agentic.utils.logging import LoggerMixin


@dataclass
class SimpleProgress:
    """Simple progress tracking for single-agent execution"""
    agent_id: str
    agent_name: str
    current_status: str = "Initializing..."
    last_activity: str = ""
    start_time: float = 0.0
    
    def __post_init__(self):
        self.start_time = time.time()
    
    @property
    def elapsed(self) -> int:
        return int(time.time() - self.start_time)


class SimpleProgressMonitor(LoggerMixin):
    """Lightweight progress monitor that doesn't clear the screen"""
    
    def __init__(self, status_updater: Optional[Callable] = None):
        super().__init__()
        self.status_updater = status_updater
        self.progress: Optional[SimpleProgress] = None
        self._activity_queue = asyncio.Queue()
        self._update_task: Optional[asyncio.Task] = None
    
    def start_monitoring(self, agent_id: str, agent_name: str):
        """Start monitoring an agent"""
        self.progress = SimpleProgress(agent_id, agent_name)
        if self.status_updater:
            self.status_updater("Starting Claude Code agent...")
        
        # Start background updater
        self._update_task = asyncio.create_task(self._run_updater())
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
    
    def update_status(self, status: str, activity: Optional[str] = None):
        """Update agent status"""
        if self.progress:
            self.progress.current_status = status
            if activity:
                self.progress.last_activity = activity
            
            # Queue update
            asyncio.create_task(self._queue_update(status, activity))
    
    async def _queue_update(self, status: str, activity: Optional[str]):
        """Queue an update"""
        await self._activity_queue.put((status, activity))
    
    async def _run_updater(self):
        """Background task to update display"""
        activity_messages = [
            "Analyzing codebase...",
            "Reading project files...",
            "Understanding requirements...",
            "Formulating response...",
            "Checking dependencies...",
            "Processing information..."
        ]
        message_index = 0
        last_update = time.time()
        
        while True:
            try:
                # Check for queued updates
                try:
                    status, activity = await asyncio.wait_for(
                        self._activity_queue.get(), 
                        timeout=3.0
                    )
                    
                    if self.status_updater:
                        if activity:
                            self.status_updater(activity)
                        else:
                            self.status_updater(status)
                    last_update = time.time()
                    
                except asyncio.TimeoutError:
                    # No updates, show periodic message
                    if time.time() - last_update > 5:
                        if self.status_updater:
                            self.status_updater(activity_messages[message_index % len(activity_messages)])
                        message_index += 1
                        last_update = time.time()
                
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in progress updater: {e}")
                await asyncio.sleep(1)