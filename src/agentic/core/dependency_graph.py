"""
Advanced Dependency Graph Builder for Phase 3 Intelligent Orchestration
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Set

from agentic.models.project import DependencyGraph, ProjectStructure, TechStack
from agentic.utils.logging import LoggerMixin


class DependencyGraphBuilder(LoggerMixin):
    """Builds comprehensive dependency graphs for projects"""
    
    def __init__(self):
        super().__init__()
        self.parsers = {
            'python': PythonDependencyParser(),
            'javascript': JavaScriptDependencyParser(),
            'typescript': TypeScriptDependencyParser(),
        }
    
    async def build_dependency_graph(self, project: ProjectStructure) -> DependencyGraph:
        """Build complete dependency graph"""
        self.logger.info(f"Building dependency graph for {project.root_path}")
        
        graph = DependencyGraph()
        
        # Add file nodes
        for source_dir in project.source_directories:
            await self._add_directory_nodes(graph, source_dir, project.tech_stack)
        
        # Add dependency edges
        for language in project.tech_stack.languages:
            if language in self.parsers:
                parser = self.parsers[language]
                await parser.add_dependencies(graph, project.root_path)
        
        # Add cross-language dependencies (e.g., TypeScript importing from Python API)
        await self._add_cross_language_dependencies(graph, project)
        
        # Calculate impact scores
        self._calculate_impact_scores(graph)
        
        self.logger.info(f"Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
    
    async def _add_directory_nodes(self, graph: DependencyGraph, directory: Path, tech_stack: TechStack):
        """Add file nodes from directory"""
        extensions = self._get_relevant_extensions(tech_stack)
        
        for ext in extensions:
            for file_path in directory.rglob(f"*.{ext}"):
                if not self._should_ignore_file(file_path):
                    node_info = await self._analyze_file(file_path)
                    graph.nodes[str(file_path)] = node_info
    
    def _get_relevant_extensions(self, tech_stack: TechStack) -> Set[str]:
        """Get file extensions relevant to the tech stack"""
        extensions = set()
        
        for language in tech_stack.languages:
            if language == 'python':
                extensions.update(['py', 'pyi'])
            elif language == 'javascript':
                extensions.update(['js', 'jsx', 'mjs'])
            elif language == 'typescript':
                extensions.update(['ts', 'tsx', 'mts'])
            elif language == 'go':
                extensions.add('go')
            elif language == 'rust':
                extensions.add('rs')
        
        return extensions
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = {
            '__pycache__', '.git', 'node_modules', '.pytest_cache',
            'dist', 'build', '.coverage', '.venv', 'venv'
        }
        
        # Check if any parent directory matches ignore patterns
        for part in file_path.parts:
            if part in ignore_patterns:
                return True
        
        # Check file patterns
        if file_path.name.startswith('.') and file_path.suffix not in ['.py', '.js', '.ts']:
            return True
        
        return False
    
    async def _analyze_file(self, file_path: Path) -> Dict:
        """Analyze individual file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            return {
                'file_path': str(file_path),
                'size': len(content),
                'lines': content.count('\n') + 1,
                'extension': file_path.suffix,
                'functions': self._count_functions(content, file_path.suffix),
                'classes': self._count_classes(content, file_path.suffix),
                'imports': self._extract_imports(content, file_path.suffix),
                'exports': self._extract_exports(content, file_path.suffix),
                'complexity_score': self._calculate_file_complexity(content, file_path.suffix),
                'last_modified': file_path.stat().st_mtime,
                'impact_score': 0.0,  # Will be calculated later
                'importance': 'unknown'  # Will be classified later
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e),
                'impact_score': 0.0,
                'importance': 'unknown'
            }
    
    def _count_functions(self, content: str, extension: str) -> int:
        """Count functions in file"""
        if extension == '.py':
            return len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            # Count function declarations and arrow functions
            func_decl = len(re.findall(r'^\s*function\s+\w+', content, re.MULTILINE))
            arrow_func = len(re.findall(r'^\s*\w+\s*=\s*\([^)]*\)\s*=>', content, re.MULTILINE))
            method_func = len(re.findall(r'^\s*\w+\s*\([^)]*\)\s*{', content, re.MULTILINE))
            return func_decl + arrow_func + method_func
        return 0
    
    def _count_classes(self, content: str, extension: str) -> int:
        """Count classes in file"""
        if extension == '.py':
            return len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            return len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
        return 0
    
    def _extract_imports(self, content: str, extension: str) -> List[str]:
        """Extract import statements"""
        imports = []
        
        if extension == '.py':
            # Python imports
            imports.extend(re.findall(r'^\s*import\s+([^\s#]+)', content, re.MULTILINE))
            imports.extend(re.findall(r'^\s*from\s+([^\s]+)\s+import', content, re.MULTILINE))
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            # JavaScript/TypeScript imports
            imports.extend(re.findall(r'^\s*import.*from\s+[\'"]([^\'"]+)[\'"]', content, re.MULTILINE))
            imports.extend(re.findall(r'^\s*const.*=\s*require\([\'"]([^\'"]+)[\'"]\)', content, re.MULTILINE))
        
        return [imp.strip() for imp in imports if imp.strip()]
    
    def _extract_exports(self, content: str, extension: str) -> List[str]:
        """Extract export statements"""
        exports = []
        
        if extension == '.py':
            # Python __all__ exports
            all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if all_match:
                exports = re.findall(r'[\'"]([^\'"]+)[\'"]', all_match.group(1))
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            # JavaScript/TypeScript exports
            exports.extend(re.findall(r'^\s*export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)', content, re.MULTILINE))
            exports.extend(re.findall(r'^\s*export\s*{\s*([^}]+)\s*}', content, re.MULTILINE))
        
        return [exp.strip() for exp in exports if exp.strip()]
    
    def _calculate_file_complexity(self, content: str, extension: str) -> float:
        """Calculate file complexity score"""
        # Basic complexity metrics
        lines = content.count('\n') + 1
        
        # Cyclomatic complexity indicators
        if extension == '.py':
            complexity_keywords = ['if', 'elif', 'for', 'while', 'try', 'except', 'with']
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            complexity_keywords = ['if', 'for', 'while', 'switch', 'try', 'catch']
        else:
            complexity_keywords = []
        
        complexity_count = sum(content.count(keyword) for keyword in complexity_keywords)
        
        # Normalize complexity (0.0 to 1.0)
        base_complexity = min(lines / 500.0, 1.0)  # 500 lines = max base complexity
        keyword_complexity = min(complexity_count / 50.0, 1.0)  # 50 keywords = max keyword complexity
        
        return (base_complexity + keyword_complexity) / 2.0
    
    async def _add_cross_language_dependencies(self, graph: DependencyGraph, project: ProjectStructure):
        """Add cross-language dependencies"""
        # This is a simplified implementation
        # In a real system, this would analyze API calls, config references, etc.
        
        # For now, just add dependencies between config files and source files
        for config_file in project.config_files:
            config_path = str(config_file)
            if config_path in graph.nodes:
                # Config files typically affect all source files
                for source_dir in project.source_directories:
                    for node_path in graph.nodes:
                        if str(source_dir) in node_path and node_path != config_path:
                            graph.edges.append({
                                "source": config_path,
                                "target": node_path,
                                "type": "configuration"
                            })
    
    def _calculate_impact_scores(self, graph: DependencyGraph):
        """Calculate impact scores for each node"""
        for node_id in graph.nodes:
            # Impact score based on number of dependents
            dependents = graph.get_dependents(node_id)
            dependencies = graph.get_dependencies(node_id)
            
            # Higher score = more critical file
            impact_score = len(dependents) * 2 + len(dependencies) * 0.5
            
            # Consider file complexity
            if 'complexity_score' in graph.nodes[node_id]:
                impact_score += graph.nodes[node_id]['complexity_score'] * 10
            
            graph.nodes[node_id]['impact_score'] = impact_score
            
            # Classify file importance
            if impact_score > 20:
                graph.nodes[node_id]['importance'] = 'critical'
            elif impact_score > 10:
                graph.nodes[node_id]['importance'] = 'high'
            elif impact_score > 5:
                graph.nodes[node_id]['importance'] = 'medium'
            else:
                graph.nodes[node_id]['importance'] = 'low'


class PythonDependencyParser:
    """Parser for Python dependencies"""
    
    async def add_dependencies(self, graph: DependencyGraph, project_root: Path):
        """Add Python-specific dependencies"""
        for node_path in graph.nodes:
            if node_path.endswith('.py'):
                try:
                    file_path = Path(node_path)
                    if file_path.exists():
                        await self._parse_python_file(graph, file_path, project_root)
                except Exception as e:
                    pass  # Skip files that can't be parsed
    
    async def _parse_python_file(self, graph: DependencyGraph, file_path: Path, project_root: Path):
        """Parse Python file for dependencies"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        await self._add_python_dependency(graph, str(file_path), alias.name, project_root)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        await self._add_python_dependency(graph, str(file_path), node.module, project_root)
                        
        except Exception:
            # If AST parsing fails, fall back to regex
            await self._parse_python_with_regex(graph, file_path, project_root)
    
    async def _add_python_dependency(self, graph: DependencyGraph, source_file: str, 
                                   module_name: str, project_root: Path):
        """Add Python dependency if it's a local module"""
        # Check if it's a relative import within the project
        potential_paths = [
            project_root / f"{module_name.replace('.', '/')}.py",
            project_root / f"{module_name.replace('.', '/')}/__init__.py",
            project_root / "src" / f"{module_name.replace('.', '/')}.py",
            project_root / "src" / f"{module_name.replace('.', '/')}/__init__.py"
        ]
        
        for potential_path in potential_paths:
            if potential_path.exists() and str(potential_path) in graph.nodes:
                graph.edges.append({
                    "source": str(potential_path),
                    "target": source_file,
                    "type": "import"
                })
                break
    
    async def _parse_python_with_regex(self, graph: DependencyGraph, file_path: Path, project_root: Path):
        """Fallback regex parsing for Python"""
        content = file_path.read_text(encoding='utf-8')
        
        # Find import statements
        import_patterns = [
            r'^\s*import\s+([^\s#]+)',
            r'^\s*from\s+([^\s]+)\s+import'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                await self._add_python_dependency(graph, str(file_path), match.strip(), project_root)


class JavaScriptDependencyParser:
    """Parser for JavaScript dependencies"""
    
    async def add_dependencies(self, graph: DependencyGraph, project_root: Path):
        """Add JavaScript-specific dependencies"""
        for node_path in graph.nodes:
            if any(node_path.endswith(ext) for ext in ['.js', '.jsx', '.mjs']):
                try:
                    file_path = Path(node_path)
                    if file_path.exists():
                        await self._parse_javascript_file(graph, file_path, project_root)
                except Exception:
                    pass  # Skip files that can't be parsed
    
    async def _parse_javascript_file(self, graph: DependencyGraph, file_path: Path, project_root: Path):
        """Parse JavaScript file for dependencies"""
        content = file_path.read_text(encoding='utf-8')
        
        # Import patterns
        import_patterns = [
            r'^\s*import.*from\s+[\'"]([^\'"]+)[\'"]',
            r'^\s*const.*=\s*require\([\'"]([^\'"]+)[\'"]\)',
            r'^\s*import\([\'"]([^\'"]+)[\'"]\)'  # Dynamic imports
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                await self._add_javascript_dependency(graph, str(file_path), match.strip(), project_root)
    
    async def _add_javascript_dependency(self, graph: DependencyGraph, source_file: str,
                                       import_path: str, project_root: Path):
        """Add JavaScript dependency if it's a local file"""
        # Skip node_modules and external packages
        if import_path.startswith('.'):
            # Relative import
            source_path = Path(source_file)
            resolved_path = (source_path.parent / import_path).resolve()
            
            # Try different extensions
            potential_paths = [
                resolved_path.with_suffix('.js'),
                resolved_path.with_suffix('.jsx'),
                resolved_path.with_suffix('.ts'),
                resolved_path.with_suffix('.tsx'),
                resolved_path / 'index.js',
                resolved_path / 'index.jsx',
                resolved_path / 'index.ts',
                resolved_path / 'index.tsx'
            ]
            
            for potential_path in potential_paths:
                if potential_path.exists() and str(potential_path) in graph.nodes:
                    graph.edges.append({
                        "source": str(potential_path),
                        "target": source_file,
                        "type": "import"
                    })
                    break


class TypeScriptDependencyParser(JavaScriptDependencyParser):
    """Parser for TypeScript dependencies (extends JavaScript parser)"""
    
    async def add_dependencies(self, graph: DependencyGraph, project_root: Path):
        """Add TypeScript-specific dependencies"""
        for node_path in graph.nodes:
            if any(node_path.endswith(ext) for ext in ['.ts', '.tsx', '.mts']):
                try:
                    file_path = Path(node_path)
                    if file_path.exists():
                        await self._parse_javascript_file(graph, file_path, project_root)  # Reuse JS parser
                        await self._parse_typescript_specific(graph, file_path, project_root)
                except Exception:
                    pass
    
    async def _parse_typescript_specific(self, graph: DependencyGraph, file_path: Path, project_root: Path):
        """Parse TypeScript-specific syntax"""
        content = file_path.read_text(encoding='utf-8')
        
        # Type imports
        type_import_pattern = r'^\s*import\s+type.*from\s+[\'"]([^\'"]+)[\'"]'
        matches = re.findall(type_import_pattern, content, re.MULTILINE)
        
        for match in matches:
            await self._add_javascript_dependency(graph, str(file_path), match.strip(), project_root) 