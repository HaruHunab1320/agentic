#!/usr/bin/env python3
"""
Script to migrate all swarm monitor imports to use the unified implementation
"""

import os
import re
from pathlib import Path


def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Patterns to replace
        replacements = [
            # Direct imports
            (r'from agentic\.core\.swarm_monitor import (.+)', 
             r'from agentic.core.swarm_monitor_unified import \1'),
            (r'from agentic\.core\.swarm_monitor_enhanced import (.+)', 
             r'from agentic.core.swarm_monitor_unified import \1'),
            (r'from agentic\.core\.swarm_monitor_simple import (.+)', 
             r'from agentic.core.swarm_monitor_unified import \1'),
            (r'from agentic\.core\.swarm_monitor_fixed import (.+)', 
             r'from agentic.core.swarm_monitor_unified import \1'),
            
            # Import with aliases
            (r'from agentic\.core\.swarm_monitor_enhanced import SwarmMonitorEnhanced as SwarmMonitor',
             'from agentic.core.swarm_monitor_unified import SwarmMonitorUnified as SwarmMonitor'),
            (r'from agentic\.core\.swarm_monitor_simple import SwarmMonitorSimple as SwarmMonitor',
             'from agentic.core.swarm_monitor_unified import SwarmMonitorUnified as SwarmMonitor'),
            (r'from agentic\.core\.swarm_monitor_fixed import SwarmMonitorFixed as SwarmMonitor',
             'from agentic.core.swarm_monitor_unified import SwarmMonitorUnified as SwarmMonitor'),
             
            # Class instantiation
            (r'SwarmMonitorEnhanced\(', 'SwarmMonitorUnified('),
            (r'SwarmMonitorSimple\(', 'SwarmMonitorUnified('),
            (r'SwarmMonitorFixed\(', 'SwarmMonitorUnified('),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main migration function"""
    # Find project root
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "agentic"
    
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return
    
    updated_files = []
    
    # Process all Python files
    for py_file in src_dir.rglob("*.py"):
        # Skip the unified monitor itself and old monitors
        if py_file.name in ['swarm_monitor_unified.py', 'swarm_monitor.py', 
                           'swarm_monitor_enhanced.py', 'swarm_monitor_simple.py',
                           'swarm_monitor_fixed.py']:
            continue
        
        if update_imports_in_file(py_file):
            updated_files.append(py_file)
            print(f"Updated: {py_file.relative_to(project_root)}")
    
    print(f"\nMigration complete!")
    print(f"Updated {len(updated_files)} files")
    
    # List old monitor files that can be deleted
    old_monitors = [
        src_dir / "core" / "swarm_monitor.py",
        src_dir / "core" / "swarm_monitor_enhanced.py",
        src_dir / "core" / "swarm_monitor_simple.py",
        src_dir / "core" / "swarm_monitor_fixed.py",
    ]
    
    existing_old_monitors = [m for m in old_monitors if m.exists()]
    if existing_old_monitors:
        print("\nOld monitor files that can be deleted:")
        for monitor in existing_old_monitors:
            print(f"  - {monitor.relative_to(project_root)}")


if __name__ == "__main__":
    main()