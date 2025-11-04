"""
Result Validation System for Code Verification

This module provides comprehensive validation of agent-generated code,
including syntax checking, build validation, test execution, and
quality assurance.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic.utils.logging import LoggerMixin


class ValidationResult(BaseModel):
    """Result of a validation check"""
    validator_name: str
    passed: bool
    severity: str = "error"  # error, warning, info
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationSuite(BaseModel):
    """A suite of validation results"""
    suite_name: str
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings: int = 0
    results: List[ValidationResult] = Field(default_factory=list)
    execution_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks
    
    @property
    def has_errors(self) -> bool:
        return any(r.severity == "error" and not r.passed for r in self.results)
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the suite"""
        self.results.append(result)
        self.total_checks += 1
        
        if result.passed:
            self.passed_checks += 1
        else:
            if result.severity == "error":
                self.failed_checks += 1
            else:
                self.warnings += 1


class LanguageValidator:
    """Base class for language-specific validators"""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    async def validate_syntax(self, file_path: Path) -> List[ValidationResult]:
        """Validate syntax of a file"""
        raise NotImplementedError
    
    async def validate_imports(self, file_path: Path) -> List[ValidationResult]:
        """Validate imports and dependencies"""
        raise NotImplementedError
    
    async def validate_style(self, file_path: Path) -> List[ValidationResult]:
        """Validate code style and conventions"""
        raise NotImplementedError


class PythonValidator(LanguageValidator):
    """Python-specific validation"""
    
    async def validate_syntax(self, file_path: Path) -> List[ValidationResult]:
        """Validate Python syntax using AST"""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the file
            ast.parse(content, filename=str(file_path))
            
            results.append(ValidationResult(
                validator_name="python_syntax",
                passed=True,
                severity="info",
                message="Python syntax is valid",
                file_path=str(file_path)
            ))
            
        except SyntaxError as e:
            results.append(ValidationResult(
                validator_name="python_syntax",
                passed=False,
                severity="error",
                message=f"Syntax error: {e.msg}",
                file_path=str(file_path),
                line_number=e.lineno,
                column_number=e.offset,
                suggestion="Check for missing colons, parentheses, or indentation errors"
            ))
        except Exception as e:
            results.append(ValidationResult(
                validator_name="python_syntax",
                passed=False,
                severity="error",
                message=f"Failed to parse file: {str(e)}",
                file_path=str(file_path)
            ))
        
        return results
    
    async def validate_imports(self, file_path: Path) -> List[ValidationResult]:
        """Validate Python imports"""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append((f"{module}.{alias.name}", node.lineno))
            
            # Check if imports are valid (basic check)
            for import_name, line_no in imports:
                # Check for relative imports in module files
                if import_name.startswith('.'):
                    if not file_path.name == '__init__.py' and not file_path.parent.name == 'tests':
                        results.append(ValidationResult(
                            validator_name="python_imports",
                            passed=False,
                            severity="warning",
                            message=f"Relative import '{import_name}' outside of package",
                            file_path=str(file_path),
                            line_number=line_no,
                            suggestion="Use absolute imports for clarity"
                        ))
            
            if not results:
                results.append(ValidationResult(
                    validator_name="python_imports",
                    passed=True,
                    severity="info",
                    message="All imports appear valid",
                    file_path=str(file_path)
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                validator_name="python_imports",
                passed=False,
                severity="error",
                message=f"Failed to analyze imports: {str(e)}",
                file_path=str(file_path)
            ))
        
        return results
    
    async def validate_style(self, file_path: Path) -> List[ValidationResult]:
        """Basic style validation for Python"""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check line length
            for i, line in enumerate(lines, 1):
                if len(line.rstrip()) > 100:  # Relaxed from PEP8's 79
                    results.append(ValidationResult(
                        validator_name="python_style",
                        passed=False,
                        severity="warning",
                        message=f"Line too long ({len(line.rstrip())} > 100 characters)",
                        file_path=str(file_path),
                        line_number=i,
                        suggestion="Consider breaking long lines for better readability"
                    ))
            
            # Check for trailing whitespace
            for i, line in enumerate(lines, 1):
                if line.rstrip() != line.rstrip('\n').rstrip('\r'):
                    results.append(ValidationResult(
                        validator_name="python_style",
                        passed=False,
                        severity="warning",
                        message="Trailing whitespace",
                        file_path=str(file_path),
                        line_number=i,
                        suggestion="Remove trailing whitespace"
                    ))
            
            if not results:
                results.append(ValidationResult(
                    validator_name="python_style",
                    passed=True,
                    severity="info",
                    message="Basic style checks passed",
                    file_path=str(file_path)
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                validator_name="python_style",
                passed=False,
                severity="error",
                message=f"Failed to check style: {str(e)}",
                file_path=str(file_path)
            ))
        
        return results


class JavaScriptValidator(LanguageValidator):
    """JavaScript/TypeScript validation"""
    
    async def validate_syntax(self, file_path: Path) -> List[ValidationResult]:
        """Validate JavaScript syntax"""
        results = []
        
        # Try using Node.js to check syntax
        try:
            cmd = ["node", "--check", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                results.append(ValidationResult(
                    validator_name="javascript_syntax",
                    passed=True,
                    severity="info",
                    message="JavaScript syntax is valid",
                    file_path=str(file_path)
                ))
            else:
                # Parse error message
                error_match = re.search(r'(.+):(\d+):(\d+): (.+)', result.stderr)
                if error_match:
                    results.append(ValidationResult(
                        validator_name="javascript_syntax",
                        passed=False,
                        severity="error",
                        message=error_match.group(4),
                        file_path=str(file_path),
                        line_number=int(error_match.group(2)),
                        column_number=int(error_match.group(3))
                    ))
                else:
                    results.append(ValidationResult(
                        validator_name="javascript_syntax",
                        passed=False,
                        severity="error",
                        message=result.stderr.strip(),
                        file_path=str(file_path)
                    ))
                    
        except FileNotFoundError:
            results.append(ValidationResult(
                validator_name="javascript_syntax",
                passed=False,
                severity="warning",
                message="Node.js not found, cannot validate JavaScript syntax",
                file_path=str(file_path)
            ))
        except Exception as e:
            results.append(ValidationResult(
                validator_name="javascript_syntax",
                passed=False,
                severity="error",
                message=f"Failed to validate syntax: {str(e)}",
                file_path=str(file_path)
            ))
        
        return results
    
    async def validate_imports(self, file_path: Path) -> List[ValidationResult]:
        """Validate JavaScript imports"""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all import statements
            import_pattern = r'import\s+(?:{[^}]+}|[\w\s,]+)\s+from\s+["\']([^"\']+)["\']'
            require_pattern = r'require\s*\(\s*["\']([^"\']+)["\']\s*\)'
            
            imports = re.findall(import_pattern, content) + re.findall(require_pattern, content)
            
            for import_path in imports:
                # Check for missing file extensions in relative imports
                if import_path.startswith('.') and not any(import_path.endswith(ext) 
                                                          for ext in ['.js', '.jsx', '.ts', '.tsx', '.json']):
                    results.append(ValidationResult(
                        validator_name="javascript_imports",
                        passed=False,
                        severity="warning",
                        message=f"Relative import '{import_path}' missing file extension",
                        file_path=str(file_path),
                        suggestion="Consider adding explicit file extension for clarity"
                    ))
            
            if not results:
                results.append(ValidationResult(
                    validator_name="javascript_imports",
                    passed=True,
                    severity="info",
                    message="Import statements look valid",
                    file_path=str(file_path)
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                validator_name="javascript_imports",
                passed=False,
                severity="error",
                message=f"Failed to analyze imports: {str(e)}",
                file_path=str(file_path)
            ))
        
        return results
    
    async def validate_style(self, file_path: Path) -> List[ValidationResult]:
        """Basic style validation for JavaScript"""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check for console.log statements (common in dev but should be removed)
            for i, line in enumerate(lines, 1):
                if 'console.log' in line and not line.strip().startswith('//'):
                    results.append(ValidationResult(
                        validator_name="javascript_style",
                        passed=False,
                        severity="warning",
                        message="console.log statement found",
                        file_path=str(file_path),
                        line_number=i,
                        suggestion="Remove console.log statements before production"
                    ))
            
            # Check for debugger statements
            for i, line in enumerate(lines, 1):
                if 'debugger' in line and not line.strip().startswith('//'):
                    results.append(ValidationResult(
                        validator_name="javascript_style",
                        passed=False,
                        severity="error",
                        message="debugger statement found",
                        file_path=str(file_path),
                        line_number=i,
                        suggestion="Remove debugger statements"
                    ))
            
            if not results:
                results.append(ValidationResult(
                    validator_name="javascript_style",
                    passed=True,
                    severity="info",
                    message="Basic style checks passed",
                    file_path=str(file_path)
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                validator_name="javascript_style",
                passed=False,
                severity="error",
                message=f"Failed to check style: {str(e)}",
                file_path=str(file_path)
            ))
        
        return results


class ResultValidationManager(LoggerMixin):
    """
    Manages validation of agent-generated code and results.
    
    Features:
    - Multi-language syntax validation
    - Build system validation
    - Test execution and validation
    - Code quality checks
    - Security scanning
    """
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path
        
        # Language validators
        self.validators = {
            '.py': PythonValidator(workspace_path),
            '.js': JavaScriptValidator(workspace_path),
            '.jsx': JavaScriptValidator(workspace_path),
            '.ts': JavaScriptValidator(workspace_path),
            '.tsx': JavaScriptValidator(workspace_path)
        }
        
        # Build commands by project type
        self.build_commands = {
            'npm': ['npm', 'run', 'build'],
            'yarn': ['yarn', 'build'],
            'pip': ['python', '-m', 'py_compile'],
            'maven': ['mvn', 'compile'],
            'gradle': ['gradle', 'build'],
            'cargo': ['cargo', 'build']
        }
        
        # Test commands by project type
        self.test_commands = {
            'npm': ['npm', 'test'],
            'yarn': ['yarn', 'test'],
            'pytest': ['pytest'],
            'unittest': ['python', '-m', 'unittest'],
            'jest': ['jest'],
            'mocha': ['mocha'],
            'cargo': ['cargo', 'test']
        }
    
    async def validate_file(self, file_path: Path) -> ValidationSuite:
        """Validate a single file"""
        suite = ValidationSuite(suite_name=f"File validation: {file_path.name}")
        
        # Get appropriate validator
        validator = self.validators.get(file_path.suffix)
        if not validator:
            suite.add_result(ValidationResult(
                validator_name="file_validation",
                passed=False,
                severity="warning",
                message=f"No validator available for {file_path.suffix} files",
                file_path=str(file_path)
            ))
            return suite
        
        # Run validations
        start_time = asyncio.get_event_loop().time()
        
        # Syntax validation
        syntax_results = await validator.validate_syntax(file_path)
        for result in syntax_results:
            suite.add_result(result)
        
        # Only continue if syntax is valid
        if not any(r.severity == "error" for r in syntax_results):
            # Import validation
            import_results = await validator.validate_imports(file_path)
            for result in import_results:
                suite.add_result(result)
            
            # Style validation
            style_results = await validator.validate_style(file_path)
            for result in style_results:
                suite.add_result(result)
        
        suite.execution_time = asyncio.get_event_loop().time() - start_time
        
        return suite
    
    async def validate_files(self, file_paths: List[Path]) -> ValidationSuite:
        """Validate multiple files"""
        suite = ValidationSuite(suite_name=f"Batch file validation ({len(file_paths)} files)")
        
        # Validate each file
        for file_path in file_paths:
            if file_path.exists() and file_path.is_file():
                file_suite = await self.validate_file(file_path)
                
                # Aggregate results
                suite.total_checks += file_suite.total_checks
                suite.passed_checks += file_suite.passed_checks
                suite.failed_checks += file_suite.failed_checks
                suite.warnings += file_suite.warnings
                suite.results.extend(file_suite.results)
        
        return suite
    
    async def validate_build(self, project_type: Optional[str] = None) -> ValidationSuite:
        """Validate that the project builds successfully"""
        suite = ValidationSuite(suite_name="Build validation")
        
        # Detect project type if not provided
        if not project_type:
            project_type = self._detect_project_type()
        
        if not project_type:
            suite.add_result(ValidationResult(
                validator_name="build_validation",
                passed=False,
                severity="warning",
                message="Could not detect project type for build validation"
            ))
            return suite
        
        # Get build command
        build_cmd = self.build_commands.get(project_type)
        if not build_cmd:
            suite.add_result(ValidationResult(
                validator_name="build_validation",
                passed=False,
                severity="warning",
                message=f"No build command configured for project type: {project_type}"
            ))
            return suite
        
        # Run build
        self.logger.info(f"Running build command: {' '.join(build_cmd)}")
        
        try:
            result = await asyncio.create_subprocess_exec(
                *build_cmd,
                cwd=str(self.workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                suite.add_result(ValidationResult(
                    validator_name="build_validation",
                    passed=True,
                    severity="info",
                    message="Build completed successfully"
                ))
            else:
                suite.add_result(ValidationResult(
                    validator_name="build_validation",
                    passed=False,
                    severity="error",
                    message=f"Build failed with exit code {result.returncode}",
                    metadata={
                        'stdout': stdout.decode('utf-8', errors='replace'),
                        'stderr': stderr.decode('utf-8', errors='replace')
                    }
                ))
                
        except Exception as e:
            suite.add_result(ValidationResult(
                validator_name="build_validation",
                passed=False,
                severity="error",
                message=f"Build execution failed: {str(e)}"
            ))
        
        return suite
    
    async def validate_tests(self, test_command: Optional[List[str]] = None) -> ValidationSuite:
        """Run and validate test suite"""
        suite = ValidationSuite(suite_name="Test validation")
        
        # Use provided command or detect
        if not test_command:
            test_command = self._detect_test_command()
        
        if not test_command:
            suite.add_result(ValidationResult(
                validator_name="test_validation",
                passed=False,
                severity="warning",
                message="No test command found or provided"
            ))
            return suite
        
        # Run tests
        self.logger.info(f"Running tests: {' '.join(test_command)}")
        
        try:
            result = await asyncio.create_subprocess_exec(
                *test_command,
                cwd=str(self.workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                suite.add_result(ValidationResult(
                    validator_name="test_validation",
                    passed=True,
                    severity="info",
                    message="All tests passed"
                ))
                
                # Try to parse test results for more detail
                test_output = stdout.decode('utf-8', errors='replace')
                
                # Look for common test result patterns
                if 'passed' in test_output.lower():
                    # Extract test counts if possible
                    import re
                    count_match = re.search(r'(\d+)\s+passed', test_output)
                    if count_match:
                        suite.metadata['tests_passed'] = int(count_match.group(1))
                        
            else:
                suite.add_result(ValidationResult(
                    validator_name="test_validation",
                    passed=False,
                    severity="error",
                    message=f"Tests failed with exit code {result.returncode}",
                    metadata={
                        'stdout': stdout.decode('utf-8', errors='replace'),
                        'stderr': stderr.decode('utf-8', errors='replace')
                    }
                ))
                
        except Exception as e:
            suite.add_result(ValidationResult(
                validator_name="test_validation",
                passed=False,
                severity="error",
                message=f"Test execution failed: {str(e)}"
            ))
        
        return suite
    
    def _detect_project_type(self) -> Optional[str]:
        """Detect the project type from files"""
        if (self.workspace_path / "package.json").exists():
            # Check for yarn.lock
            if (self.workspace_path / "yarn.lock").exists():
                return "yarn"
            return "npm"
        elif (self.workspace_path / "requirements.txt").exists() or \
             (self.workspace_path / "pyproject.toml").exists():
            return "pip"
        elif (self.workspace_path / "pom.xml").exists():
            return "maven"
        elif (self.workspace_path / "build.gradle").exists():
            return "gradle"
        elif (self.workspace_path / "Cargo.toml").exists():
            return "cargo"
        
        return None
    
    def _detect_test_command(self) -> Optional[List[str]]:
        """Detect the test command to use"""
        # Check package.json for test script
        package_json = self.workspace_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)
                    if 'scripts' in data and 'test' in data['scripts']:
                        if (self.workspace_path / "yarn.lock").exists():
                            return ["yarn", "test"]
                        return ["npm", "test"]
            except:
                pass
        
        # Check for Python test frameworks
        if (self.workspace_path / "pytest.ini").exists() or \
           any((self.workspace_path / "tests").glob("test_*.py")):
            return ["pytest"]
        
        # Check for Jest config
        if (self.workspace_path / "jest.config.js").exists():
            return ["jest"]
        
        return None
    
    async def validate_security(self, file_paths: List[Path]) -> ValidationSuite:
        """Basic security validation"""
        suite = ValidationSuite(suite_name="Security validation")
        
        # Common security patterns to check
        security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'eval\s*\(', "Use of eval() detected - security risk"),
            (r'exec\s*\(', "Use of exec() detected - security risk"),
            (r'__import__\s*\(', "Dynamic import detected - potential security risk")
        ]
        
        for file_path in file_paths:
            if not file_path.exists() or not file_path.is_file():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                
                for pattern, message in security_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Find line number
                        line_no = content[:match.start()].count('\n') + 1
                        
                        suite.add_result(ValidationResult(
                            validator_name="security_validation",
                            passed=False,
                            severity="error",
                            message=message,
                            file_path=str(file_path),
                            line_number=line_no,
                            suggestion="Remove hardcoded secrets and use environment variables"
                        ))
                
            except Exception as e:
                suite.add_result(ValidationResult(
                    validator_name="security_validation",
                    passed=False,
                    severity="warning",
                    message=f"Could not scan file for security issues: {str(e)}",
                    file_path=str(file_path)
                ))
        
        if not suite.results:
            suite.add_result(ValidationResult(
                validator_name="security_validation",
                passed=True,
                severity="info",
                message="No obvious security issues detected"
            ))
        
        return suite
    
    def format_validation_report(self, suites: List[ValidationSuite]) -> str:
        """Format validation results into a readable report"""
        report_lines = ["# Validation Report", ""]
        
        # Summary
        total_checks = sum(s.total_checks for s in suites)
        total_passed = sum(s.passed_checks for s in suites)
        total_failed = sum(s.failed_checks for s in suites)
        total_warnings = sum(s.warnings for s in suites)
        
        report_lines.extend([
            "## Summary",
            f"- Total checks: {total_checks}",
            f"- Passed: {total_passed}",
            f"- Failed: {total_failed}",
            f"- Warnings: {total_warnings}",
            f"- Success rate: {total_passed/total_checks*100:.1f}%" if total_checks > 0 else "- Success rate: N/A",
            ""
        ])
        
        # Details by suite
        for suite in suites:
            report_lines.extend([
                f"## {suite.suite_name}",
                f"Execution time: {suite.execution_time:.2f}s",
                ""
            ])
            
            # Group by severity
            errors = [r for r in suite.results if r.severity == "error" and not r.passed]
            warnings = [r for r in suite.results if r.severity == "warning"]
            info = [r for r in suite.results if r.severity == "info"]
            
            if errors:
                report_lines.append("### Errors")
                for result in errors:
                    location = f"{result.file_path}:{result.line_number}" if result.line_number else result.file_path or "N/A"
                    report_lines.append(f"- [{result.validator_name}] {location}: {result.message}")
                    if result.suggestion:
                        report_lines.append(f"  Suggestion: {result.suggestion}")
                report_lines.append("")
            
            if warnings:
                report_lines.append("### Warnings")
                for result in warnings:
                    location = f"{result.file_path}:{result.line_number}" if result.line_number else result.file_path or "N/A"
                    report_lines.append(f"- [{result.validator_name}] {location}: {result.message}")
                report_lines.append("")
        
        return "\n".join(report_lines)