"""
Enhanced Project Analyzer for Phase 3 with dependency graphs and pattern recognition
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from agentic.core.dependency_graph import DependencyGraphBuilder
from agentic.core.project_analyzer import ProjectAnalyzer
from agentic.models.project import DependencyGraph, ProjectStructure
from agentic.utils.logging import LoggerMixin


class CodePattern:
    """Represents a detected code pattern"""
    
    def __init__(self, name: str, pattern_type: str, location: str, 
                 confidence: float, description: str, files: List[str]):
        self.name = name
        self.type = pattern_type  # 'design_pattern', 'anti_pattern', 'architecture_pattern'
        self.location = location
        self.confidence = confidence
        self.description = description
        self.files = files


class Suggestion:
    """Represents an improvement suggestion"""
    
    def __init__(self, suggestion_type: str, description: str, priority: str, 
                 estimated_effort: str, affected_files: List[str]):
        self.type = suggestion_type  # 'refactor', 'implement', 'optimize', 'security'
        self.description = description
        self.priority = priority  # 'low', 'medium', 'high', 'critical'
        self.estimated_effort = estimated_effort
        self.affected_files = affected_files


class EnhancedProjectAnalyzer(LoggerMixin):
    """Enhanced project analyzer with dependency graphs and pattern recognition"""
    
    def __init__(self):
        super().__init__()
        self.base_analyzer = None  # Will be initialized when needed
        self.dependency_builder = DependencyGraphBuilder()
        self.pattern_analyzer = CodePatternAnalyzer()
        self.architecture_analyzer = ArchitectureAnalyzer()
    
    async def analyze_project_comprehensive(self, project_path: Path) -> EnhancedProjectAnalysis:
        """Perform comprehensive project analysis with all Phase 3 features"""
        self.logger.info(f"Starting comprehensive analysis of {project_path}")
        
        # Initialize base analyzer with project path
        if self.base_analyzer is None:
            self.base_analyzer = ProjectAnalyzer(project_path)
        
        # Basic project structure analysis
        basic_structure = await self.base_analyzer.analyze()
        
        # Build dependency graph
        dependency_graph = await self.dependency_builder.build_dependency_graph(basic_structure)
        
        # Analyze code patterns
        code_patterns = await self.pattern_analyzer.analyze_patterns(basic_structure)
        
        # Analyze architecture
        architecture_info = await self.architecture_analyzer.analyze_architecture(
            basic_structure, dependency_graph
        )
        
        # Generate suggestions
        suggestions = await self._generate_suggestions(
            basic_structure, dependency_graph, code_patterns, architecture_info
        )
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(
            basic_structure, dependency_graph, code_patterns
        )
        
        # Security analysis
        security_issues = await self._analyze_security(basic_structure)
        
        # Performance analysis
        performance_insights = await self._analyze_performance(
            basic_structure, dependency_graph
        )
        
        return EnhancedProjectAnalysis(
            basic_structure=basic_structure,
            dependency_graph=dependency_graph,
            code_patterns=code_patterns,
            architecture_info=architecture_info,
            suggestions=suggestions,
            quality_metrics=quality_metrics,
            security_issues=security_issues,
            performance_insights=performance_insights
        )
    
    async def _generate_suggestions(self, structure: ProjectStructure, 
                                  graph: DependencyGraph, patterns: List[CodePattern],
                                  architecture: Dict) -> List[Suggestion]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []
        
        # Pattern-based suggestions
        suggestions.extend(self.pattern_analyzer.suggest_improvements(patterns))
        
        # Dependency-based suggestions
        if graph.has_circular_dependencies():
            suggestions.append(Suggestion(
                suggestion_type='refactor',
                description='Break circular dependencies to improve maintainability',
                priority='high',
                estimated_effort='4-8 hours',
                affected_files=list(graph.nodes.keys())[:10]  # Show first 10 affected files
            ))
        
        # Architecture-based suggestions
        if architecture.get('complexity_score', 0) > 0.8:
            suggestions.append(Suggestion(
                suggestion_type='refactor',
                description='High architectural complexity detected - consider modularization',
                priority='medium',
                estimated_effort='1-2 weeks',
                affected_files=[]
            ))
        
        # Find files with high impact but low test coverage
        high_impact_files = [
            node_id for node_id, data in graph.nodes.items()
            if data.get('importance') == 'critical'
        ]
        
        test_coverage_files = []
        for test_dir in structure.test_directories:
            test_coverage_files.extend([str(f) for f in test_dir.rglob("*.py")])
            test_coverage_files.extend([str(f) for f in test_dir.rglob("*.js")])
            test_coverage_files.extend([str(f) for f in test_dir.rglob("*.ts")])
        
        if len(test_coverage_files) < len(high_impact_files) * 0.5:
            suggestions.append(Suggestion(
                suggestion_type='implement',
                description='Add comprehensive test coverage for critical files',
                priority='high',
                estimated_effort='2-3 days',
                affected_files=high_impact_files[:5]
            ))
        
        return suggestions
    
    async def _calculate_quality_metrics(self, structure: ProjectStructure,
                                       graph: DependencyGraph, 
                                       patterns: List[CodePattern]) -> Dict:
        """Calculate code quality metrics"""
        metrics = {}
        
        # Dependency metrics
        if graph.nodes:
            avg_dependencies = sum(len(graph.get_dependencies(node)) for node in graph.nodes) / len(graph.nodes)
            avg_dependents = sum(len(graph.get_dependents(node)) for node in graph.nodes) / len(graph.nodes)
            
            metrics['dependency_metrics'] = {
                'total_files': len(graph.nodes),
                'total_dependencies': len(graph.edges),
                'average_dependencies_per_file': round(avg_dependencies, 2),
                'average_dependents_per_file': round(avg_dependents, 2),
                'has_circular_dependencies': graph.has_circular_dependencies()
            }
        
        # Complexity metrics
        total_complexity = sum(
            data.get('complexity_score', 0) 
            for data in graph.nodes.values()
        )
        avg_complexity = total_complexity / len(graph.nodes) if graph.nodes else 0
        
        metrics['complexity_metrics'] = {
            'average_file_complexity': round(avg_complexity, 3),
            'high_complexity_files': len([
                node for node, data in graph.nodes.items()
                if data.get('complexity_score', 0) > 0.8
            ])
        }
        
        # Pattern metrics
        design_patterns = [p for p in patterns if p.type == 'design_pattern']
        anti_patterns = [p for p in patterns if p.type == 'anti_pattern']
        
        metrics['pattern_metrics'] = {
            'design_patterns_found': len(design_patterns),
            'anti_patterns_found': len(anti_patterns),
            'pattern_coverage': len(design_patterns) / len(graph.nodes) if graph.nodes else 0
        }
        
        # File size metrics
        file_sizes = [data.get('size', 0) for data in graph.nodes.values()]
        if file_sizes:
            metrics['size_metrics'] = {
                'average_file_size': round(sum(file_sizes) / len(file_sizes)),
                'largest_file_size': max(file_sizes),
                'files_over_1000_lines': len([
                    data for data in graph.nodes.values()
                    if data.get('lines', 0) > 1000
                ])
            }
        
        return metrics
    
    async def _analyze_security(self, structure: ProjectStructure) -> List[Dict]:
        """Analyze potential security issues"""
        security_issues = []
        
        # Scan for common security anti-patterns
        security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*[\'"][^\'"]+[\'"]',
                r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
                r'secret\s*=\s*[\'"][^\'"]+[\'"]',
            ],
            'sql_injection': [
                r'cursor\.execute\([\'"].*%.*[\'"]',
                r'query\s*=\s*[\'"].*\+.*[\'"]',
            ],
            'unsafe_imports': [
                r'import\s+pickle',
                r'from\s+pickle\s+import',
                r'eval\s*\(',
                r'exec\s*\(',
            ]
        }
        
        for source_dir in structure.source_directories:
            for file_path in source_dir.rglob("*.py"):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    for issue_type, patterns in security_patterns.items():
                        for pattern in patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                security_issues.append({
                                    'type': issue_type,
                                    'file': str(file_path),
                                    'severity': 'medium' if issue_type == 'hardcoded_secrets' else 'high',
                                    'description': f'Potential {issue_type.replace("_", " ")} found',
                                    'matches': len(matches)
                                })
                                
                except Exception as e:
                    self.logger.warning(f"Failed to scan {file_path} for security issues: {e}")
        
        return security_issues
    
    async def _analyze_performance(self, structure: ProjectStructure,
                                 graph: DependencyGraph) -> Dict:
        """Analyze potential performance issues"""
        insights = {
            'hot_spots': [],
            'optimization_opportunities': [],
            'dependency_bottlenecks': []
        }
        
        # Find potential hot spots (files with high complexity and many dependents)
        for node_id, data in graph.nodes.items():
            complexity = data.get('complexity_score', 0)
            
            # Count dependents by looking at edges where this node is the target
            dependents = [edge for edge in graph.edges if edge.get('target') == node_id]
            dependent_count = len(dependents)
            
            if complexity > 0.7 and dependent_count > 5:
                insights['hot_spots'].append({
                    'file': node_id,
                    'complexity': complexity,
                    'dependents': dependent_count,
                    'reason': 'High complexity with many dependents'
                })
        
        # Find files with many dependencies (potential performance bottlenecks)
        for node_id, data in graph.nodes.items():
            # Count dependencies by looking at edges where this node is the source
            dependencies = [edge for edge in graph.edges if edge.get('source') == node_id]
            dependency_count = len(dependencies)
            
            if dependency_count > 10:
                insights['dependency_bottlenecks'].append({
                    'file': node_id,
                    'dependencies': dependency_count,
                    'suggestion': 'Consider dependency injection or modularization'
                })
        
        return insights


class EnhancedProjectAnalysis:
    """Container for comprehensive project analysis results"""
    
    def __init__(self, basic_structure: ProjectStructure, dependency_graph: DependencyGraph,
                 code_patterns: List[CodePattern], architecture_info: Dict,
                 suggestions: List[Suggestion], quality_metrics: Dict,
                 security_issues: List[Dict], performance_insights: Dict):
        self.basic_structure = basic_structure
        self.dependency_graph = dependency_graph
        self.code_patterns = code_patterns
        self.architecture_info = architecture_info
        self.suggestions = suggestions
        self.quality_metrics = quality_metrics
        self.security_issues = security_issues
        self.performance_insights = performance_insights


class CodePatternAnalyzer:
    """Analyzes and catalogs code patterns"""
    
    def __init__(self):
        self.pattern_matchers = [
            SingletonPatternMatcher(),
            FactoryPatternMatcher(),
            MVCPatternMatcher(),
            APIPatternMatcher(),
        ]
    
    async def analyze_patterns(self, project: ProjectStructure) -> List[CodePattern]:
        """Identify code patterns in project"""
        patterns = []
        
        for source_dir in project.source_directories:
            for matcher in self.pattern_matchers:
                found_patterns = await matcher.find_patterns(source_dir)
                patterns.extend(found_patterns)
        
        return patterns
    
    def suggest_improvements(self, patterns: List[CodePattern]) -> List[Suggestion]:
        """Suggest improvements based on patterns"""
        suggestions = []
        
        # Detect anti-patterns
        for pattern in patterns:
            if pattern.type == 'anti_pattern':
                suggestions.append(Suggestion(
                    suggestion_type='refactor',
                    description=f"Refactor {pattern.name} anti-pattern in {pattern.location}",
                    priority='medium',
                    estimated_effort='2-4 hours',
                    affected_files=pattern.files
                ))
        
        # Suggest missing patterns
        has_error_handling = any(p.name == 'error_handling' for p in patterns)
        if not has_error_handling:
            suggestions.append(Suggestion(
                suggestion_type='implement',
                description="Implement consistent error handling pattern",
                priority='high',
                estimated_effort='4-8 hours',
                affected_files=[]
            ))
        
        return suggestions


class ArchitectureAnalyzer:
    """Analyzes project architecture patterns"""
    
    async def analyze_architecture(self, project: ProjectStructure, 
                                 graph: DependencyGraph) -> Dict:
        """Analyze architectural patterns and characteristics"""
        
        architecture_info = {
            'style': await self._detect_architecture_style(project, graph),
            'complexity_score': self._calculate_architecture_complexity(graph),
            'modularity_score': self._calculate_modularity(graph),
            'layers': await self._identify_layers(project),
            'entry_points': self._analyze_entry_points(project),
        }
        
        return architecture_info
    
    async def _detect_architecture_style(self, project: ProjectStructure, 
                                       graph: DependencyGraph) -> str:
        """Detect the architectural style"""
        
        # Check for microservices indicators
        if len(project.source_directories) > 5:
            return 'microservices'
        
        # Check for MVC patterns
        mvc_indicators = ['models', 'views', 'controllers', 'templates']
        has_mvc = sum(1 for indicator in mvc_indicators 
                     if any(indicator in str(d).lower() 
                           for d in project.source_directories))
        
        if has_mvc >= 2:
            return 'mvc'
        
        # Check for layered architecture
        layer_indicators = ['api', 'service', 'repository', 'domain']
        has_layers = sum(1 for indicator in layer_indicators 
                        if any(indicator in str(d).lower() 
                              for d in project.source_directories))
        
        if has_layers >= 3:
            return 'layered'
        
        # Default to monolithic
        return 'monolithic'
    
    def _calculate_architecture_complexity(self, graph: DependencyGraph) -> float:
        """Calculate overall architectural complexity"""
        if not graph.nodes:
            return 0.0
        
        # Factors that contribute to complexity
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        avg_dependencies = edge_count / node_count if node_count > 0 else 0
        
        # Normalize complexity score
        complexity = min((node_count / 100 + avg_dependencies / 10) / 2, 1.0)
        
        return complexity
    
    def _calculate_modularity(self, graph: DependencyGraph) -> float:
        """Calculate modularity score (how well-separated modules are)"""
        if not graph.nodes:
            return 1.0
        
        # Simple modularity calculation based on clustering
        # Higher modularity = better separation
        
        total_possible_edges = len(graph.nodes) * (len(graph.nodes) - 1)
        actual_edges = len(graph.edges)
        
        if total_possible_edges == 0:
            return 1.0
        
        # Lower edge ratio indicates better modularity
        edge_ratio = actual_edges / total_possible_edges
        modularity = max(1.0 - edge_ratio * 2, 0.0)
        
        return modularity
    
    async def _identify_layers(self, project: ProjectStructure) -> List[str]:
        """Identify architectural layers"""
        layers = []
        
        layer_keywords = {
            'presentation': ['ui', 'view', 'component', 'template', 'frontend'],
            'api': ['api', 'endpoint', 'route', 'controller'],
            'business': ['service', 'business', 'domain', 'logic'],
            'data': ['model', 'repository', 'dao', 'database', 'storage']
        }
        
        for layer_name, keywords in layer_keywords.items():
            for source_dir in project.source_directories:
                if any(keyword in str(source_dir).lower() for keyword in keywords):
                    if layer_name not in layers:
                        layers.append(layer_name)
                    break
        
        return layers
    
    def _analyze_entry_points(self, project: ProjectStructure) -> List[Dict]:
        """Analyze application entry points"""
        entry_points = []
        
        for entry_file in project.entry_points:
            entry_points.append({
                'file': str(entry_file),
                'type': self._classify_entry_point(entry_file)
            })
        
        return entry_points
    
    def _classify_entry_point(self, file_path: Path) -> str:
        """Classify the type of entry point"""
        filename = file_path.name.lower()
        
        if filename in ['main.py', '__main__.py']:
            return 'cli_application'
        elif filename in ['app.py', 'server.py', 'wsgi.py']:
            return 'web_application'
        elif filename in ['manage.py']:
            return 'django_application'
        elif filename in ['index.js', 'server.js']:
            return 'node_application'
        else:
            return 'unknown'


# Pattern matcher implementations

class SingletonPatternMatcher:
    """Detects singleton patterns"""
    
    async def find_patterns(self, directory: Path) -> List[CodePattern]:
        patterns = []
        
        for file_path in directory.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Look for singleton pattern indicators
                if ('__new__' in content and 
                    'instance' in content and 
                    'cls' in content):
                    
                    patterns.append(CodePattern(
                        name='singleton',
                        pattern_type='design_pattern',
                        location=str(file_path),
                        confidence=0.8,
                        description='Singleton pattern implementation detected',
                        files=[str(file_path)]
                    ))
                    
            except Exception:
                pass
        
        return patterns


class FactoryPatternMatcher:
    """Detects factory patterns"""
    
    async def find_patterns(self, directory: Path) -> List[CodePattern]:
        patterns = []
        
        for file_path in directory.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Look for factory pattern indicators
                if (re.search(r'def\s+create_\w+', content) or
                    re.search(r'class\s+\w+Factory', content)):
                    
                    patterns.append(CodePattern(
                        name='factory',
                        pattern_type='design_pattern',
                        location=str(file_path),
                        confidence=0.7,
                        description='Factory pattern implementation detected',
                        files=[str(file_path)]
                    ))
                    
            except Exception:
                pass
        
        return patterns


class MVCPatternMatcher:
    """Detects MVC patterns"""
    
    async def find_patterns(self, directory: Path) -> List[CodePattern]:
        patterns = []
        
        # Look for MVC structure in directory layout
        has_models = any(directory.rglob("*model*"))
        has_views = any(directory.rglob("*view*"))
        has_controllers = any(directory.rglob("*controller*"))
        
        if sum([has_models, has_views, has_controllers]) >= 2:
            patterns.append(CodePattern(
                name='mvc',
                pattern_type='architecture_pattern',
                location=str(directory),
                confidence=0.8,
                description='MVC architectural pattern detected',
                files=[]
            ))
        
        return patterns


class APIPatternMatcher:
    """Detects API patterns"""
    
    async def find_patterns(self, directory: Path) -> List[CodePattern]:
        patterns = []
        
        for file_path in directory.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Look for REST API patterns
                rest_indicators = [
                    r'@app\.route',
                    r'@api\.route',
                    r'@bp\.route',
                    r'@router\.(get|post|put|delete)',
                    r'app\.(get|post|put|delete)'
                ]
                
                if any(re.search(pattern, content) for pattern in rest_indicators):
                    patterns.append(CodePattern(
                        name='rest_api',
                        pattern_type='architecture_pattern',
                        location=str(file_path),
                        confidence=0.9,
                        description='REST API pattern detected',
                        files=[str(file_path)]
                    ))
                    
            except Exception:
                pass
        
        return patterns 