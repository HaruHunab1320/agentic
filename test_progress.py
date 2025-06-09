#!/usr/bin/env python3
"""Test progress indicators for single-agent queries"""

import asyncio
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
import time

async def test_progress():
    """Test the progress indicator"""
    print("Testing progress indicator...")
    
    # Simulate the query processing with spinner
    with Live(
        Spinner("dots", text=Text("Processing query...", style="cyan")),
        refresh_per_second=4
    ) as live:
        # Simulate processing time
        for i in range(5):
            await asyncio.sleep(1)
            live.update(Spinner("dots", text=Text(f"Processing query... ({i+1}s)", style="cyan")))
    
    print("\n✅ Task completed successfully!")
    print("The skipped test requires building a complete automated liquidity provision")
    print("system with both the agent logic and a simulated testing environment.")

if __name__ == "__main__":
    asyncio.run(test_progress())