"""
Utilities for clean swarm monitor display
"""

import sys
import io
import contextlib
from typing import Optional


class MonitorOutputRedirect:
    """Context manager to redirect stdout during swarm monitoring"""
    
    def __init__(self):
        self.original_stdout = None
        self.buffer = io.StringIO()
        
    def __enter__(self):
        # Save original stdout
        self.original_stdout = sys.stdout
        # Redirect stdout to our buffer
        sys.stdout = self.buffer
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout
        sys.stdout = self.original_stdout
        
        # If there was buffered output, print it after monitor stops
        buffered_content = self.buffer.getvalue()
        if buffered_content.strip():
            print("\n--- Buffered Output ---")
            print(buffered_content)
            print("--- End Buffered Output ---\n")


@contextlib.contextmanager
def suppress_output_during_monitor():
    """Suppress all stdout during swarm monitor display"""
    with MonitorOutputRedirect():
        yield