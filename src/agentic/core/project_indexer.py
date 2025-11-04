"""
Project Indexing System for maintaining awareness of project state
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from agentic.models.task import ProjectContext
from agentic.utils.logging import LoggerMixin


class FileInfo(BaseModel):
    """Information about a single file"""
    path: str
    size: int
    modified: datetime
    language: Optional[str] = None
    hash: str
    imports: List[str] = Field(default_factory=list)
    exports: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class ProjectIndex(BaseModel):
    """Complete project index"""
    root_path: str
    indexed_at: datetime
    total_files: int
    total_size: int
    file_index: Dict[str, FileInfo] = Field(default_factory=dict)
    language_stats: Dict[str, int] = Field(default_factory=dict)
    framework_indicators: Dict[str, List[str]] = Field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    
    
class ProjectIndexer(LoggerMixin):
    """Indexes project files for better context awareness"""
    
    # Language detection patterns
    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript", 
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".swift": "swift",
        ".kt": "kotlin",
        ".dart": "dart",
        ".vue": "vue",
        ".svelte": "svelte"
    }
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        "react": {
            "files": ["package.json"],
            "patterns": [r'"react":', r'from\s+["\']react["\']', r'import\s+.*React']
        },
        "vue": {
            "files": ["package.json", "vue.config.js"],
            "patterns": [r'"vue":', r'<template>', r'new Vue\(']
        },
        "angular": {
            "files": ["angular.json", "package.json"],
            "patterns": [r'"@angular/core":', r'@Component\(', r'import.*from.*@angular']
        },
        "django": {
            "files": ["manage.py", "settings.py"],
            "patterns": [r'from django', r'INSTALLED_APPS', r'django\.']
        },
        "flask": {
            "files": ["app.py", "application.py"],
            "patterns": [r'from flask import', r'Flask\(__name__\)', r'@app\.route']
        },
        "express": {
            "files": ["package.json"],
            "patterns": [r'"express":', r'require\(["\']express["\']', r'app\.listen\(']
        },
        "nextjs": {
            "files": ["next.config.js", "package.json"],
            "patterns": [r'"next":', r'from ["\']next/', r'pages/', r'app/']
        },
        "rails": {
            "files": ["Gemfile", "config/routes.rb"],
            "patterns": [r'gem ["\']rails["\']', r'Rails\.application', r'class.*<.*ApplicationController']
        }
    }
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path
        self.index: Optional[ProjectIndex] = None
        self._cache_file = workspace_path / ".agentic" / "project_index.json"
        
    async def index_project(self, force_reindex: bool = False) -> ProjectIndex:
        """Index the entire project"""
        # Check cache first
        if not force_reindex and self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is recent (within last hour)
                    indexed_at = datetime.fromisoformat(data['indexed_at'])
                    if (datetime.utcnow() - indexed_at).seconds < 3600:
                        self.logger.info("Using cached project index")
                        self.index = ProjectIndex(**data)
                        return self.index
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        self.logger.info("Indexing project...")
        
        self.index = ProjectIndex(
            root_path=str(self.workspace_path),
            indexed_at=datetime.utcnow(),
            total_files=0,
            total_size=0
        )
        
        # Walk through all files
        for file_path in self._walk_project():
            try:
                file_info = await self._index_file(file_path)
                if file_info:
                    relative_path = str(file_path.relative_to(self.workspace_path))
                    self.index.file_index[relative_path] = file_info
                    self.index.total_files += 1
                    self.index.total_size += file_info.size
                    
                    # Update language stats
                    if file_info.language:
                        self.index.language_stats[file_info.language] = \
                            self.index.language_stats.get(file_info.language, 0) + 1
                            
            except Exception as e:
                self.logger.warning(f"Failed to index {file_path}: {e}")
        
        # Detect frameworks
        self._detect_frameworks()
        
        # Build dependency graph
        self._build_dependency_graph()
        
        # Save to cache
        self._save_cache()
        
        self.logger.info(f"Indexed {self.index.total_files} files")
        
        return self.index
    
    def _walk_project(self) -> List[Path]:
        """Walk project directory, respecting .gitignore"""
        ignore_patterns = self._load_gitignore()
        files = []
        
        for root, dirs, filenames in os.walk(self.workspace_path):
            root_path = Path(root)
            
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                'node_modules', '__pycache__', 'venv', 'env', 'dist', 'build',
                'target', 'out', '.git', '.svn', 'coverage', 'htmlcov'
            }]
            
            for filename in filenames:
                file_path = root_path / filename
                
                # Skip hidden files and common ignore patterns
                if filename.startswith('.') or filename.endswith('.pyc'):
                    continue
                    
                # Skip binary and large files
                if file_path.suffix in {'.exe', '.dll', '.so', '.dylib', '.bin', 
                                       '.jpg', '.png', '.gif', '.mp4', '.zip'}:
                    continue
                
                files.append(file_path)
                
        return files
    
    def _load_gitignore(self) -> List[str]:
        """Load .gitignore patterns"""
        gitignore_path = self.workspace_path / ".gitignore"
        patterns = []
        
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception:
                pass
                
        return patterns
    
    async def _index_file(self, file_path: Path) -> Optional[FileInfo]:
        """Index a single file"""
        try:
            stat = file_path.stat()
            
            # Skip large files (> 1MB)
            if stat.st_size > 1024 * 1024:
                return None
                
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Skip binary files
                return None
            
            # Calculate hash
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Detect language
            language = self.LANGUAGE_EXTENSIONS.get(file_path.suffix)
            
            # Extract code structure
            file_info = FileInfo(
                path=str(file_path),
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
                language=language,
                hash=file_hash
            )
            
            # Language-specific parsing
            if language == "python":
                self._parse_python_file(content, file_info)
            elif language in ["javascript", "typescript"]:
                self._parse_javascript_file(content, file_info)
            elif language == "java":
                self._parse_java_file(content, file_info)
                
            return file_info
            
        except Exception as e:
            self.logger.debug(f"Error indexing {file_path}: {e}")
            return None
    
    def _parse_python_file(self, content: str, file_info: FileInfo):
        """Parse Python file for structure"""
        # Extract imports
        import_pattern = r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)'
        for match in re.finditer(import_pattern, content):
            module = match.group(1) or match.group(2).split()[0]
            file_info.imports.append(module)
        
        # Extract classes
        class_pattern = r'class\s+(\w+)\s*(?:\([^)]*\))?:'
        file_info.classes = re.findall(class_pattern, content)
        
        # Extract functions
        func_pattern = r'def\s+(\w+)\s*\('
        file_info.functions = re.findall(func_pattern, content)
        
        # Identify dependencies
        for imp in file_info.imports:
            if not imp.startswith('.') and '.' in imp:
                file_info.dependencies.append(imp.split('.')[0])
    
    def _parse_javascript_file(self, content: str, file_info: FileInfo):
        """Parse JavaScript/TypeScript file for structure"""
        # Extract imports
        import_patterns = [
            r'import\s+(?:{[^}]+}|[\w\s,]+)\s+from\s+["\']([^"\']+)["\']',
            r'require\s*\(["\']([^"\']+)["\']\)',
            r'import\s*\(["\']([^"\']+)["\']\)'
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                file_info.imports.append(match.group(1))
        
        # Extract classes
        class_pattern = r'class\s+(\w+)\s*(?:extends\s+\w+)?\s*{'
        file_info.classes = re.findall(class_pattern, content)
        
        # Extract functions
        func_patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>'
        ]
        
        for pattern in func_patterns:
            file_info.functions.extend(re.findall(pattern, content))
        
        # Extract React components
        component_pattern = r'(?:export\s+)?(?:default\s+)?(?:function|const)\s+(\w+)\s*[=:]\s*(?:\([^)]*\)|[^=])\s*=>\s*(?:<|\()'
        components = re.findall(component_pattern, content)
        file_info.exports.extend([c for c in components if c[0].isupper()])
    
    def _parse_java_file(self, content: str, file_info: FileInfo):
        """Parse Java file for structure"""
        # Extract imports
        import_pattern = r'import\s+([^;]+);'
        file_info.imports = re.findall(import_pattern, content)
        
        # Extract classes
        class_pattern = r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)'
        file_info.classes = re.findall(class_pattern, content)
        
        # Extract methods
        method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)?(\w+)\s*\('
        file_info.functions = re.findall(method_pattern, content)
    
    def _detect_frameworks(self):
        """Detect frameworks used in the project"""
        for framework, indicators in self.FRAMEWORK_PATTERNS.items():
            detected_files = []
            
            # Check for indicator files
            for indicator_file in indicators['files']:
                file_path = self.workspace_path / indicator_file
                if file_path.exists():
                    detected_files.append(indicator_file)
            
            # Check for patterns in code
            for file_path, file_info in self.index.file_index.items():
                if file_info.language in ["python", "javascript", "typescript"]:
                    try:
                        with open(self.workspace_path / file_path, 'r') as f:
                            content = f.read()
                            for pattern in indicators['patterns']:
                                if re.search(pattern, content):
                                    detected_files.append(file_path)
                                    break
                    except Exception:
                        pass
            
            if detected_files:
                self.index.framework_indicators[framework] = detected_files[:5]  # Limit to 5 examples
    
    def _build_dependency_graph(self):
        """Build dependency graph from imports"""
        # This is simplified - real implementation would resolve relative imports
        for file_path, file_info in self.index.file_index.items():
            deps = []
            
            # Try to resolve local imports
            for imp in file_info.imports:
                if imp.startswith('.'):
                    # Relative import - try to resolve
                    continue
                else:
                    # Check if it's a local module
                    for other_path in self.index.file_index:
                        if other_path.endswith(f"{imp}.py") or \
                           other_path.endswith(f"{imp}/__init__.py"):
                            deps.append(other_path)
                            
            if deps:
                self.index.dependency_graph[file_path] = deps
    
    def _save_cache(self):
        """Save index to cache file"""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to JSON-serializable format
            data = self.index.dict()
            # Convert datetime objects to ISO format
            data['indexed_at'] = data['indexed_at'].isoformat()
            for file_info in data['file_index'].values():
                file_info['modified'] = file_info['modified'].isoformat()
            
            with open(self._cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def get_project_context(self) -> ProjectContext:
        """Get project context from index"""
        if not self.index:
            raise ValueError("Project not indexed yet")
        
        # Determine primary language
        primary_language = None
        if self.index.language_stats:
            primary_language = max(self.index.language_stats.items(), 
                                 key=lambda x: x[1])[0]
        
        # Determine primary framework
        primary_framework = None
        frameworks = set(self.index.framework_indicators.keys())
        if frameworks:
            # Prioritize by number of indicator files
            primary_framework = max(self.index.framework_indicators.items(),
                                  key=lambda x: len(x[1]))[0]
        
        # Determine package manager
        package_manager = None
        if (self.workspace_path / "package.json").exists():
            if (self.workspace_path / "yarn.lock").exists():
                package_manager = "yarn"
            elif (self.workspace_path / "pnpm-lock.yaml").exists():
                package_manager = "pnpm"
            else:
                package_manager = "npm"
        elif (self.workspace_path / "requirements.txt").exists() or \
              (self.workspace_path / "pyproject.toml").exists():
            package_manager = "pip"
        elif (self.workspace_path / "go.mod").exists():
            package_manager = "go"
        elif (self.workspace_path / "Cargo.toml").exists():
            package_manager = "cargo"
        
        # Determine project type
        project_type = self._determine_project_type()
        
        return ProjectContext(
            primary_language=primary_language,
            languages=set(self.index.language_stats.keys()),
            framework=primary_framework,
            frameworks=frameworks,
            package_manager=package_manager,
            project_type=project_type,
            existing_patterns=self._extract_patterns()
        )
    
    def _determine_project_type(self) -> str:
        """Determine the type of project"""
        indicators = {
            "web": ["index.html", "app.js", "App.tsx", "pages/", "components/"],
            "api": ["api/", "routes/", "controllers/", "server.js", "main.go"],
            "cli": ["cli.py", "__main__.py", "cmd/", "main.go"],
            "library": ["setup.py", "lib/", "__init__.py", "index.js"],
            "mobile": ["ios/", "android/", "App.tsx", "react-native"]
        }
        
        for project_type, files in indicators.items():
            for indicator in files:
                if any(indicator in path for path in self.index.file_index.keys()):
                    return project_type
                    
        return "unknown"
    
    def _extract_patterns(self) -> Dict[str, List[str]]:
        """Extract common patterns from codebase"""
        patterns = {
            "component_style": [],
            "import_style": [],
            "naming_convention": []
        }
        
        # Analyze component naming patterns
        if self.index.framework_indicators.get("react"):
            components = []
            for file_info in self.index.file_index.values():
                if file_info.language in ["javascript", "typescript"]:
                    components.extend([c for c in file_info.exports if c[0].isupper()])
            
            if components:
                # Check if PascalCase or kebab-case files
                patterns["component_style"] = ["PascalCase" if components else "kebab-case"]
        
        return patterns
    
    def search_files(self, query: str) -> List[Tuple[str, float]]:
        """Search for files matching query with relevance scores"""
        if not self.index:
            return []
        
        results = []
        query_lower = query.lower()
        
        for file_path, file_info in self.index.file_index.items():
            score = 0.0
            
            # Exact filename match
            if query_lower in Path(file_path).name.lower():
                score += 1.0
            
            # Path contains query
            if query_lower in file_path.lower():
                score += 0.5
            
            # Class/function names contain query
            for class_name in file_info.classes:
                if query_lower in class_name.lower():
                    score += 0.8
                    
            for func_name in file_info.functions:
                if query_lower in func_name.lower():
                    score += 0.6
            
            if score > 0:
                results.append((file_path, score))
        
        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:20]  # Top 20 results
    
    def get_related_files(self, file_path: str) -> List[str]:
        """Get files related to the given file"""
        if not self.index or file_path not in self.index.file_index:
            return []
        
        related = set()
        
        # Direct dependencies
        if file_path in self.index.dependency_graph:
            related.update(self.index.dependency_graph[file_path])
        
        # Files that depend on this file
        for other_path, deps in self.index.dependency_graph.items():
            if file_path in deps:
                related.add(other_path)
        
        # Files in same directory
        parent_dir = str(Path(file_path).parent)
        for other_path in self.index.file_index:
            if str(Path(other_path).parent) == parent_dir:
                related.add(other_path)
        
        # Remove self
        related.discard(file_path)
        
        return list(related)