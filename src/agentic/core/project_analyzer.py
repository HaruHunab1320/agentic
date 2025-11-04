"""
Project structure analysis and technology stack detection
"""

import json
from pathlib import Path
from typing import List, Set

from agentic.models.project import ProjectStructure, TechStack
from agentic.utils.logging import LoggerMixin


class ProjectAnalyzer(LoggerMixin):
    """Analyzes project structure and detects technology stacks"""
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path).resolve()
        self.logger.debug(f"Initialized ProjectAnalyzer for {self.project_path}")
    
    async def analyze(self) -> ProjectStructure:
        """
        Perform comprehensive project analysis
        
        Returns:
            ProjectStructure with detected tech stack and organization
        """
        self.logger.info(f"Starting analysis of {self.project_path}")
        
        # Detect technology stack
        tech_stack = await self._detect_tech_stack()
        
        # Find project structure
        entry_points = self._find_entry_points(tech_stack)
        config_files = self._find_config_files()
        source_directories = self._find_source_directories(tech_stack)
        test_directories = self._find_test_directories(tech_stack)
        documentation_files = self._find_documentation_files()
        dependency_files = self._find_dependency_files()
        
        project_structure = ProjectStructure(
            root_path=self.project_path,
            tech_stack=tech_stack,
            entry_points=entry_points,
            config_files=config_files,
            source_directories=source_directories,
            test_directories=test_directories,
            documentation_files=documentation_files,
            dependency_files=dependency_files
        )
        
        self.logger.info(f"Analysis complete. Detected: {len(tech_stack.languages)} languages, "
                        f"{len(tech_stack.frameworks)} frameworks")
        
        return project_structure
    
    async def _detect_tech_stack(self) -> TechStack:
        """Detect the technology stack of the project"""
        languages = set()
        frameworks = set()
        databases = set()
        testing_frameworks = set()
        build_tools = set()
        deployment_tools = set()
        
        # Check for common configuration files and analyze them
        self._detect_javascript_stack(languages, frameworks, testing_frameworks, build_tools)
        self._detect_python_stack(languages, frameworks, testing_frameworks, build_tools)
        self._detect_go_stack(languages, frameworks, testing_frameworks, build_tools)
        self._detect_rust_stack(languages, frameworks, testing_frameworks, build_tools)
        self._detect_java_stack(languages, frameworks, testing_frameworks, build_tools)
        self._detect_database_stack(databases)
        self._detect_deployment_stack(deployment_tools)
        
        # Detect languages from file extensions if not already detected
        if not languages:
            languages.update(self._detect_languages_from_files())
        
        return TechStack(
            languages=sorted(list(languages)),
            frameworks=sorted(list(frameworks)),
            databases=sorted(list(databases)),
            testing_frameworks=sorted(list(testing_frameworks)),
            build_tools=sorted(list(build_tools)),
            deployment_tools=sorted(list(deployment_tools))
        )
    
    def _detect_javascript_stack(self, languages: Set[str], frameworks: Set[str], 
                                testing_frameworks: Set[str], build_tools: Set[str]) -> None:
        """Detect JavaScript/TypeScript stack"""
        package_json = self.project_path / "package.json"
        if not package_json.exists():
            return
        
        try:
            with open(package_json) as f:
                data = json.load(f)
            
            # Add JavaScript as language
            languages.add("javascript")
            
            # Check for TypeScript
            all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "typescript" in all_deps or any("@types/" in dep for dep in all_deps):
                languages.add("typescript")
            
            # Detect frameworks
            js_frameworks = {
                "react": "react",
                "vue": "vue", 
                "@vue/cli": "vue",
                "angular": "angular",
                "@angular/core": "angular",
                "svelte": "svelte",
                "next": "nextjs",
                "nuxt": "nuxt",
                "express": "express",
                "fastify": "fastify",
                "koa": "koa",
                "nestjs": "nestjs",
                "@nestjs/core": "nestjs"
            }
            
            for dep, framework in js_frameworks.items():
                if dep in all_deps:
                    frameworks.add(framework)
            
            # Detect testing frameworks
            js_testing = {
                "jest": "jest",
                "mocha": "mocha", 
                "jasmine": "jasmine",
                "cypress": "cypress",
                "playwright": "playwright",
                "@testing-library/react": "react-testing-library",
                "vitest": "vitest"
            }
            
            for dep, test_framework in js_testing.items():
                if dep in all_deps:
                    testing_frameworks.add(test_framework)
            
            # Detect build tools
            js_build_tools = {
                "webpack": "webpack",
                "vite": "vite",
                "rollup": "rollup", 
                "parcel": "parcel",
                "esbuild": "esbuild",
                "turbo": "turborepo"
            }
            
            for dep, build_tool in js_build_tools.items():
                if dep in all_deps:
                    build_tools.add(build_tool)
                    
        except (json.JSONDecodeError, FileNotFoundError):
            self.logger.warning(f"Could not parse {package_json}")
    
    def _detect_python_stack(self, languages: Set[str], frameworks: Set[str],
                           testing_frameworks: Set[str], build_tools: Set[str]) -> None:
        """Detect Python stack"""
        python_files = [
            self.project_path / "requirements.txt",
            self.project_path / "pyproject.toml",
            self.project_path / "setup.py",
            self.project_path / "Pipfile",
            self.project_path / "poetry.lock"
        ]
        
        has_python = any(f.exists() for f in python_files) or list(self.project_path.glob("*.py"))
        
        if not has_python:
            return
        
        languages.add("python")
        
        # Read requirements and detect frameworks
        requirements_content = ""
        
        # Try different requirement sources
        if (self.project_path / "requirements.txt").exists():
            with open(self.project_path / "requirements.txt") as f:
                requirements_content += f.read()
        
        if (self.project_path / "pyproject.toml").exists():
            # Basic toml parsing - just look for common patterns
            with open(self.project_path / "pyproject.toml") as f:
                toml_content = f.read()
                requirements_content += toml_content
        
        # Detect Python frameworks
        python_frameworks = {
            "django": "django",
            "flask": "flask", 
            "fastapi": "fastapi",
            "tornado": "tornado",
            "pyramid": "pyramid",
            "bottle": "bottle",
            "streamlit": "streamlit",
            "dash": "dash"
        }
        
        for package, framework in python_frameworks.items():
            if package in requirements_content.lower():
                frameworks.add(framework)
        
        # Detect testing frameworks
        python_testing = {
            "pytest": "pytest",
            "unittest": "unittest",
            "nose": "nose",
            "tox": "tox"
        }
        
        for package, test_framework in python_testing.items():
            if package in requirements_content.lower():
                testing_frameworks.add(test_framework)
        
        # Detect build tools
        if "poetry" in requirements_content or (self.project_path / "poetry.lock").exists():
            build_tools.add("poetry")
        if (self.project_path / "setup.py").exists():
            build_tools.add("setuptools")
    
    def _detect_go_stack(self, languages: Set[str], frameworks: Set[str],
                        testing_frameworks: Set[str], build_tools: Set[str]) -> None:
        """Detect Go stack"""
        go_mod = self.project_path / "go.mod"
        if not go_mod.exists() and not list(self.project_path.glob("*.go")):
            return
        
        languages.add("go")
        
        if go_mod.exists():
            try:
                with open(go_mod) as f:
                    content = f.read()
                
                # Detect Go frameworks
                go_frameworks = {
                    "github.com/gin-gonic/gin": "gin",
                    "github.com/gorilla/mux": "gorilla",
                    "github.com/labstack/echo": "echo",
                    "github.com/gofiber/fiber": "fiber"
                }
                
                for package, framework in go_frameworks.items():
                    if package in content:
                        frameworks.add(framework)
                        
            except FileNotFoundError:
                pass
        
        # Go has built-in testing
        if list(self.project_path.glob("*_test.go")):
            testing_frameworks.add("go-test")
    
    def _detect_rust_stack(self, languages: Set[str], frameworks: Set[str],
                          testing_frameworks: Set[str], build_tools: Set[str]) -> None:
        """Detect Rust stack"""
        cargo_toml = self.project_path / "Cargo.toml"
        if not cargo_toml.exists() and not list(self.project_path.glob("*.rs")):
            return
        
        languages.add("rust")
        build_tools.add("cargo")
        
        if cargo_toml.exists():
            try:
                with open(cargo_toml) as f:
                    content = f.read()
                
                # Detect Rust frameworks
                rust_frameworks = {
                    "actix-web": "actix",
                    "rocket": "rocket",
                    "warp": "warp",
                    "axum": "axum"
                }
                
                for package, framework in rust_frameworks.items():
                    if package in content:
                        frameworks.add(framework)
                        
            except FileNotFoundError:
                pass
    
    def _detect_java_stack(self, languages: Set[str], frameworks: Set[str],
                          testing_frameworks: Set[str], build_tools: Set[str]) -> None:
        """Detect Java stack"""
        java_files = [
            self.project_path / "pom.xml",  # Maven
            self.project_path / "build.gradle",  # Gradle
            self.project_path / "build.gradle.kts"  # Gradle Kotlin DSL
        ]
        
        has_java = any(f.exists() for f in java_files) or list(self.project_path.glob("**/*.java"))
        
        if not has_java:
            return
        
        languages.add("java")
        
        # Detect build tools
        if (self.project_path / "pom.xml").exists():
            build_tools.add("maven")
        if (self.project_path / "build.gradle").exists() or (self.project_path / "build.gradle.kts").exists():
            build_tools.add("gradle")
    
    def _detect_database_stack(self, databases: Set[str]) -> None:
        """Detect database technologies"""
        # Look for common database configuration files
        db_files = {
            "docker-compose.yml": ["postgres", "mysql", "mongodb", "redis"],
            ".env": ["DATABASE_URL", "POSTGRES", "MYSQL", "MONGO", "REDIS"],
            "requirements.txt": ["psycopg2", "pymongo", "redis", "mysql"],
            "package.json": ["pg", "mysql", "mongodb", "redis"]
        }
        
        for filename, indicators in db_files.items():
            file_path = self.project_path / filename
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        content = f.read().lower()
                    
                    for indicator in indicators:
                        if indicator.lower() in content:
                            if "postgres" in indicator.lower() or "psycopg" in indicator.lower() or "pg" == indicator.lower():
                                databases.add("postgresql")
                            elif "mysql" in indicator.lower():
                                databases.add("mysql")
                            elif "mongo" in indicator.lower():
                                databases.add("mongodb")
                            elif "redis" in indicator.lower():
                                databases.add("redis")
                                
                except (FileNotFoundError, UnicodeDecodeError):
                    pass
    
    def _detect_deployment_stack(self, deployment_tools: Set[str]) -> None:
        """Detect deployment and infrastructure tools"""
        deployment_files = {
            "Dockerfile": "docker",
            "docker-compose.yml": "docker-compose",
            ".github/workflows": "github-actions",
            "Jenkinsfile": "jenkins",
            "terraform": "terraform",
            "k8s": "kubernetes",
            "kubernetes": "kubernetes",
            "serverless.yml": "serverless"
        }
        
        for file_pattern, tool in deployment_files.items():
            if "/" in file_pattern:
                # Check for directory
                if (self.project_path / file_pattern).is_dir():
                    deployment_tools.add(tool)
            else:
                # Check for file or glob pattern
                if (self.project_path / file_pattern).exists() or list(self.project_path.glob(f"**/{file_pattern}")):
                    deployment_tools.add(tool)
    
    def _detect_languages_from_files(self) -> Set[str]:
        """Detect programming languages from file extensions"""
        languages = set()
        
        # Map file extensions to languages
        extension_map = {
            ".py": "python",
            ".js": "javascript", 
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".kt": "kotlin",
            ".scala": "scala",
            ".rb": "ruby",
            ".php": "php",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp"
        }
        
        # Scan for files with these extensions
        for ext, lang in extension_map.items():
            if list(self.project_path.glob(f"**/*{ext}")):
                languages.add(lang)
        
        return languages
    
    def _find_entry_points(self, tech_stack: TechStack) -> List[Path]:
        """Find likely entry point files"""
        entry_points = []
        
        # Common entry point patterns
        entry_patterns = [
            "main.py", "app.py", "__main__.py", "manage.py",  # Python
            "index.js", "app.js", "server.js", "main.js",    # JavaScript
            "index.ts", "app.ts", "server.ts", "main.ts",    # TypeScript
            "index.jsx", "app.jsx", "main.jsx",              # JSX
            "index.tsx", "app.tsx", "main.tsx", "App.tsx",   # TSX
            "main.go", "cmd/main.go",                         # Go
            "main.rs", "src/main.rs",                         # Rust
            "Main.java", "Application.java"                   # Java
        ]
        
        for pattern in entry_patterns:
            matches = list(self.project_path.glob(f"**/{pattern}"))
            entry_points.extend(matches)
        
        # Also look for any standalone entry-like files in root
        root_patterns = [
            "index.*", "main.*", "app.*", "App.*", "server.*"
        ]
        
        for pattern in root_patterns:
            matches = list(self.project_path.glob(pattern))
            # Filter to common entry point extensions
            for match in matches:
                if match.suffix in ['.js', '.ts', '.jsx', '.tsx', '.py', '.go', '.rs', '.java']:
                    entry_points.append(match)
        
        # Look for package.json main field
        if "javascript" in tech_stack.languages or "typescript" in tech_stack.languages:
            package_json = self.project_path / "package.json"
            if package_json.exists():
                try:
                    with open(package_json) as f:
                        data = json.load(f)
                    
                    main_file = data.get("main")
                    if main_file:
                        main_path = self.project_path / main_file
                        if main_path.exists():
                            entry_points.append(main_path)
                            
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
        
        return list(set(entry_points))  # Remove duplicates
    
    def _find_config_files(self) -> List[Path]:
        """Find configuration files"""
        config_patterns = [
            "*.json", "*.yml", "*.yaml", "*.toml", "*.ini", "*.conf",
            ".env*", "Dockerfile", "docker-compose*", "Makefile",
            "requirements*.txt", "package.json", "go.mod", "Cargo.toml",
            "pyproject.toml", "setup.py", "pom.xml", "build.gradle*"
        ]
        
        config_files = []
        for pattern in config_patterns:
            matches = list(self.project_path.glob(pattern))
            config_files.extend(matches)
        
        return config_files
    
    def _find_source_directories(self, tech_stack: TechStack) -> List[Path]:
        """Find source code directories"""
        common_source_dirs = ["src", "lib", "app", "source", "code"]
        
        # Language-specific source directories
        if "python" in tech_stack.languages:
            common_source_dirs.extend(["python", "py"])
        if "javascript" in tech_stack.languages or "typescript" in tech_stack.languages:
            common_source_dirs.extend(["js", "ts", "client", "frontend", "components"])
        if "go" in tech_stack.languages:
            common_source_dirs.extend(["cmd", "internal", "pkg"])
        if "rust" in tech_stack.languages:
            common_source_dirs.extend(["src"])
        if "java" in tech_stack.languages:
            common_source_dirs.extend(["src/main/java", "java"])
        
        source_directories = []
        for dir_name in common_source_dirs:
            dir_path = self.project_path / dir_name
            if dir_path.is_dir():
                source_directories.append(dir_path)
        
        return source_directories
    
    def _find_test_directories(self, tech_stack: TechStack) -> List[Path]:
        """Find test directories"""
        test_patterns = [
            "test", "tests", "testing", "__tests__", "spec", "specs",
            "src/test", "src/tests", "test/unit", "test/integration"
        ]
        
        test_directories = []
        for pattern in test_patterns:
            test_path = self.project_path / pattern
            if test_path.is_dir():
                test_directories.append(test_path)
        
        return test_directories
    
    def _find_documentation_files(self) -> List[Path]:
        """Find documentation files"""
        doc_patterns = [
            "README*", "CHANGELOG*", "LICENSE*", "CONTRIBUTING*",
            "docs/**/*.md", "doc/**/*.md", "documentation/**/*.md",
            "*.md", "*.rst", "*.txt"
        ]
        
        doc_files = []
        for pattern in doc_patterns:
            matches = list(self.project_path.glob(pattern))
            doc_files.extend(matches)
        
        return doc_files
    
    def _find_dependency_files(self) -> List[Path]:
        """Find dependency management files"""
        dependency_patterns = [
            "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            "requirements*.txt", "Pipfile", "Pipfile.lock", "poetry.lock", "pyproject.toml",
            "go.mod", "go.sum", "Cargo.toml", "Cargo.lock",
            "pom.xml", "build.gradle", "build.gradle.kts"
        ]
        
        dependency_files = []
        for pattern in dependency_patterns:
            matches = list(self.project_path.glob(pattern))
            dependency_files.extend(matches)
        
        return dependency_files 