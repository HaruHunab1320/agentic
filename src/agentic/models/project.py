"""
Project structure and analysis models
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class TechStack(BaseModel):
    """Detected technology stack"""
    languages: List[str] = Field(default_factory=list, description="Programming languages detected")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks detected")
    databases: List[str] = Field(default_factory=list, description="Database systems detected")
    testing_frameworks: List[str] = Field(default_factory=list, description="Testing frameworks detected")
    build_tools: List[str] = Field(default_factory=list, description="Build tools detected")
    deployment_tools: List[str] = Field(default_factory=list, description="Deployment tools detected")


class ProjectStructure(BaseModel):
    """Project structure analysis"""
    root_path: Path = Field(description="Root path of the project")
    tech_stack: TechStack = Field(default_factory=TechStack, description="Detected technology stack")
    entry_points: List[Path] = Field(default_factory=list, description="Entry point files")
    config_files: List[Path] = Field(default_factory=list, description="Configuration files")
    source_directories: List[Path] = Field(default_factory=list, description="Source code directories")
    test_directories: List[Path] = Field(default_factory=list, description="Test directories")
    documentation_files: List[Path] = Field(default_factory=list, description="Documentation files")
    dependency_files: List[Path] = Field(default_factory=list, description="Dependency management files")
    
    class Config:
        # Allow Path objects to be serialized
        arbitrary_types_allowed = True


class DependencyGraph(BaseModel):
    """File and module dependencies"""
    nodes: Dict[str, Dict] = Field(default_factory=dict, description="Dependency graph nodes")
    edges: List[Dict[str, str]] = Field(default_factory=list, description="Dependency graph edges")
    
    def get_dependents(self, file_path: str) -> List[str]:
        """Get files that depend on the given file"""
        return [edge["target"] for edge in self.edges if edge["source"] == file_path]
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """Get files that the given file depends on"""
        return [edge["source"] for edge in self.edges if edge["target"] == file_path]
    
    def has_circular_dependencies(self) -> bool:
        """Check if there are circular dependencies in the graph"""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def _has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
                
            visited.add(node)
            rec_stack.add(node)
            
            # Check all dependencies of this node
            for dep in self.get_dependencies(node):
                if _has_cycle(dep):
                    return True
                    
            rec_stack.remove(node)
            return False
        
        # Check all nodes
        for node in self.nodes:
            if node not in visited:
                if _has_cycle(node):
                    return True
        
        return False
    
    def get_dependency_depth(self, file_path: str) -> int:
        """Get the dependency depth for a given file"""
        visited = set()
        
        def _get_depth(node: str) -> int:
            if node in visited:
                return 0  # Avoid infinite loops
            
            visited.add(node)
            dependencies = self.get_dependencies(node)
            
            if not dependencies:
                return 0
            
            return 1 + max(_get_depth(dep) for dep in dependencies)
        
        return _get_depth(file_path) 