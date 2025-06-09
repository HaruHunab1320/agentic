#!/usr/bin/env python3
"""Check recent log entries to see what errors are occurring"""

import subprocess
import sys
from pathlib import Path

# Find log files
log_locations = [
    Path.home() / ".agentic" / "logs",
    Path("/tmp") / "agentic",
    Path("/var/log") / "agentic",
    Path.cwd() / "logs",
    Path.cwd() / ".agentic" / "logs"
]

print("=== Checking for Agentic Log Files ===\n")

found_logs = []
for loc in log_locations:
    if loc.exists():
        log_files = list(loc.glob("*.log"))
        if log_files:
            print(f"Found logs in: {loc}")
            found_logs.extend(log_files)

if not found_logs:
    print("No log files found. Checking system logs...")
    
    # Try to find in system journal
    try:
        result = subprocess.run(
            ["journalctl", "-u", "agentic", "-n", "50", "--no-pager"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            print("\nSystem journal entries:")
            print(result.stdout)
    except:
        pass
else:
    # Show recent entries from found logs
    print(f"\nFound {len(found_logs)} log files")
    
    # Get most recent log
    most_recent = max(found_logs, key=lambda f: f.stat().st_mtime)
    print(f"\nMost recent log: {most_recent}")
    print(f"Last modified: {most_recent.stat().st_mtime}")
    
    print("\n=== Last 100 lines of most recent log ===")
    with open(most_recent, 'r') as f:
        lines = f.readlines()
        for line in lines[-100:]:
            if any(keyword in line for keyword in ['ERROR', 'CRITICAL', 'Traceback', 'Exception']):
                print(f"❌ {line.rstrip()}")
            elif 'WARNING' in line:
                print(f"⚠️  {line.rstrip()}")
            else:
                print(f"   {line.rstrip()}")

# Also check if there's a debug mode we can enable
print("\n\n=== Debug Mode ===")
print("To enable debug logging, you can:")
print("1. Set environment variable: export AGENTIC_LOG_LEVEL=DEBUG")
print("2. Run with debug flag: agentic --debug")
print("3. Check ~/.agentic/config.yaml for log settings")