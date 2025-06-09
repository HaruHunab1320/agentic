#!/usr/bin/env python3
"""
Diagnose swarm monitor truncation issues
"""

import asyncio
import os
import sys
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def test_terminal_width():
    """Test terminal width detection and table rendering"""
    
    print("🔍 Diagnosing Terminal and Display Issues")
    print("=" * 80)
    
    # Get terminal size
    terminal_size = shutil.get_terminal_size()
    print(f"\n📐 Terminal Size: {terminal_size.columns} columns x {terminal_size.lines} lines")
    
    # Test Rich console width detection
    console = Console()
    print(f"📊 Rich Console Width: {console.width}")
    
    # Test stderr console (what swarm monitor uses)
    stderr_console = Console(file=sys.stderr)
    print(f"📊 Stderr Console Width: {stderr_console.width}")
    
    # Create a test table similar to swarm monitor
    print("\n🧪 Testing Table Display (stdout):")
    table = Table(show_header=True, header_style="bold magenta", box=None, expand=True)
    table.add_column("Agent", style="cyan", width=25, overflow="fold")
    table.add_column("Status", justify="center", width=15, overflow="fold")
    table.add_column("Progress", justify="center", width=30, overflow="fold")
    table.add_column("Current Task", style="blue", width=50, overflow="fold")
    table.add_column("Time", justify="right", width=10, overflow="fold")
    
    # Add test data
    table.add_row(
        "Claude Code\n[dim]claude_code[/dim]",
        "⚡ [yellow]executing[/yellow]",
        "[████████████░░░░░░░] 60%",
        "can you make sure all our tests are p...\n[dim italic]Tests passing: Press Ctrl-C again to exit[/dim italic]",
        "26.4s"
    )
    
    console.print(table)
    
    # Test with Panel
    print("\n🧪 Testing Panel Display:")
    panel_content = "Total width test: " + "x" * (console.width - 25)
    panel = Panel(panel_content, title="Width Test", expand=True)
    console.print(panel)
    
    # Test stderr output
    print("\n🧪 Testing Stderr Output (what swarm monitor uses):")
    stderr_console.print("[yellow]This is printed to stderr - check if it appears correctly[/yellow]")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if terminal_size.columns < 120:
        print(f"⚠️  Your terminal width ({terminal_size.columns}) is less than recommended (120+)")
        print("   Consider widening your terminal for better display")
    else:
        print(f"✅ Your terminal width ({terminal_size.columns}) is sufficient")
    
    if console.width != terminal_size.columns:
        print(f"⚠️  Console width mismatch: Console={console.width}, Terminal={terminal_size.columns}")
        print("   This might cause display issues")
    
    # Environment variables that might affect display
    print("\n🔧 Environment Variables:")
    for var in ['TERM', 'COLUMNS', 'LINES', 'CI', 'GITHUB_ACTIONS', 'AGENTIC_NO_ALTERNATE_SCREEN']:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")

if __name__ == "__main__":
    test_terminal_width()