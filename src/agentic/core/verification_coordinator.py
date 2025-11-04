"""
Verification Coordinator for Multi-Agent System

Ensures that generated code actually works by running tests and coordinating fixes.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

from agentic.models.task import Task
from agentic.utils.logging import LoggerMixin


@dataclass
class TestResult:
    """Result of a test execution"""
    test_type: str  # unit, integration, e2e, etc.
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    errors: List[str] = None
    output: str = ""
    duration: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class VerificationResult:
    """Overall verification result"""
    success: bool
    test_results: Dict[str, TestResult]
    system_health: Dict[str, bool]
    failures: List[Dict[str, Any]]
    suggestions: List[str]
    
    @property
    def all_tests_passed(self) -> bool:
        return all(result.passed for result in self.test_results.values())
    
    @property
    def total_failures(self) -> int:
        return sum(result.failed_tests for result in self.test_results.values())


class VerificationCoordinator(LoggerMixin):
    """Coordinates verification and iterative fixes for multi-agent generated code"""
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path
        self.max_fix_iterations = 3
        self.test_commands = {
            'npm': {
                'test': 'npm test',
                'lint': 'npm run lint',
                'build': 'npm run build',
                'start': 'npm start'
            },
            'python': {
                'test': 'pytest',
                'lint': 'flake8',
                'type': 'mypy',
                'start': 'python main.py'
            },
            'typescript': {
                'test': 'npm test',
                'lint': 'npm run lint',
                'typecheck': 'npm run typecheck',
                'build': 'npm run build'
            }
        }
    
    async def verify_system(self, project_type: str = 'auto') -> VerificationResult:
        """Run comprehensive system verification"""
        self.logger.info("Starting system verification...")
        
        # Detect project type if auto
        if project_type == 'auto':
            project_type = await self._detect_project_type()
            self.logger.info(f"Detected project type: {project_type}")
        
        test_results = {}
        failures = []
        
        # If project type is unknown, that's a failure
        if project_type == 'unknown':
            failures.append({
                'type': 'project_detection',
                'error': 'No recognizable project structure found',
                'output': 'Expected package.json, requirements.txt, or other project files'
            })
            # Return early with failure
            return VerificationResult(
                success=False,
                test_results={},
                system_health={},
                failures=failures,
                suggestions=['Create a proper project structure with package.json or requirements.txt']
            )
        
        # Run different types of tests based on project
        if project_type in ['npm', 'typescript', 'react']:
            # Check if package.json exists
            if (self.workspace_path / 'package.json').exists():
                # Install dependencies first
                install_result = await self._run_command('npm install')
                if not install_result[0]:
                    failures.append({
                        'type': 'dependency_install',
                        'error': 'Failed to install npm dependencies',
                        'output': install_result[1]
                    })
                    # Can't run tests without dependencies
                    return VerificationResult(
                        success=False,
                        test_results={},
                        system_health={},
                        failures=failures,
                        suggestions=['Fix package.json and ensure npm install succeeds']
                    )
                
                # Run tests
                test_results['unit_tests'] = await self._run_npm_tests()
                test_results['lint'] = await self._run_npm_lint()
                test_results['build'] = await self._run_npm_build()
            else:
                failures.append({
                    'type': 'missing_package_json',
                    'error': 'No package.json found for npm project',
                    'output': 'Expected package.json in project root'
                })
            
        elif project_type == 'python':
            # Run Python tests
            test_results['unit_tests'] = await self._run_python_tests()
            test_results['lint'] = await self._run_python_lint()
        
        # Test system startup
        system_health = await self._check_system_health(project_type)
        
        # CRITICAL FIX: Require at least some tests to exist
        if not test_results:
            failures.append({
                'type': 'no_tests',
                'error': 'No tests were found or run',
                'output': 'A production system must have tests'
            })
            success = False
        else:
            # Only consider success if we actually ran tests AND they passed
            success = all(result.passed for result in test_results.values()) and \
                     all(system_health.values()) and \
                     len(test_results) > 0  # Must have at least one test
        
        # Generate suggestions for failures
        suggestions = self._generate_fix_suggestions(test_results, system_health)
        
        return VerificationResult(
            success=success,
            test_results=test_results,
            system_health=system_health,
            failures=failures,
            suggestions=suggestions
        )
    
    async def _detect_project_type(self) -> str:
        """Detect the type of project based on files present"""
        if (self.workspace_path / 'package.json').exists():
            # Read package.json to determine if React, Node, etc.
            try:
                with open(self.workspace_path / 'package.json', 'r') as f:
                    package_data = json.load(f)
                    deps = package_data.get('dependencies', {})
                    dev_deps = package_data.get('devDependencies', {})
                    
                    if 'react' in deps or 'react' in dev_deps:
                        return 'react'
                    elif '@types/node' in dev_deps:
                        return 'typescript'
                    else:
                        return 'npm'
            except:
                return 'npm'
        
        elif (self.workspace_path / 'requirements.txt').exists() or \
             (self.workspace_path / 'setup.py').exists() or \
             (self.workspace_path / 'pyproject.toml').exists():
            return 'python'
        
        return 'unknown'
    
    async def _run_command(self, command: str, timeout: int = 300) -> Tuple[bool, str, str]:
        """Run a shell command and return (success, stdout, stderr)"""
        try:
            self.logger.debug(f"Running command: {command}")
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_path)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')
                
                success = process.returncode == 0
                return success, stdout_str, stderr_str
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False, "", f"Command timed out after {timeout} seconds"
                
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return False, "", str(e)
    
    async def _run_npm_tests(self) -> TestResult:
        """Run npm tests and parse results"""
        start_time = asyncio.get_event_loop().time()
        
        # Check if test script exists in package.json
        if not await self._check_npm_script('test'):
            return TestResult(
                test_type='unit_tests',
                passed=False,  # No tests IS a failure for production code
                output="No test script defined in package.json",
                errors=["Missing test script in package.json"]
            )
        
        # Also check if test files actually exist
        test_patterns = ['**/*.test.js', '**/*.test.ts', '**/*.spec.js', '**/*.spec.ts', 
                        'test/**/*.js', 'tests/**/*.js', '__tests__/**/*.js']
        has_test_files = False
        for pattern in test_patterns:
            if list(self.workspace_path.glob(pattern)):
                has_test_files = True
                break
        
        if not has_test_files:
            return TestResult(
                test_type='unit_tests',
                passed=False,
                output="No test files found",
                errors=["No test files (*.test.js, *.spec.js, etc.) found in project"]
            )
        
        # Don't use --passWithNoTests - we want it to fail if no tests exist
        success, stdout, stderr = await self._run_command('npm test')
        duration = asyncio.get_event_loop().time() - start_time
        
        # Parse test results from output
        test_stats = self._parse_test_output(stdout + stderr)
        
        # Check if no tests were found
        if success and test_stats.get('total', 0) == 0:
            # This means the test runner passed but found no tests
            success = False
            errors = ["No test files found - production code must have tests"]
        else:
            errors = self._extract_test_errors(stdout + stderr) if not success else []
        
        return TestResult(
            test_type='unit_tests',
            passed=success,
            total_tests=test_stats.get('total', 0),
            passed_tests=test_stats.get('passed', 0),
            failed_tests=test_stats.get('failed', 0),
            errors=errors,
            output=stdout + stderr,
            duration=duration
        )
    
    async def _run_npm_lint(self) -> TestResult:
        """Run npm lint"""
        if not await self._check_npm_script('lint'):
            return TestResult(
                test_type='lint',
                passed=False,  # Production code should have linting
                output="No lint script defined",
                errors=["Missing lint script in package.json"]
            )
        
        success, stdout, stderr = await self._run_command('npm run lint')
        
        return TestResult(
            test_type='lint',
            passed=success,
            errors=self._extract_lint_errors(stdout + stderr) if not success else [],
            output=stdout + stderr
        )
    
    async def _run_npm_build(self) -> TestResult:
        """Run npm build"""
        if not await self._check_npm_script('build'):
            # For TypeScript projects, build is essential
            return TestResult(
                test_type='build',
                passed=False,
                output="No build script defined",
                errors=["Missing build script in package.json - required for TypeScript projects"]
            )
        
        success, stdout, stderr = await self._run_command('npm run build')
        
        return TestResult(
            test_type='build',
            passed=success,
            errors=self._extract_build_errors(stdout + stderr) if not success else [],
            output=stdout + stderr
        )
    
    async def _run_python_tests(self) -> TestResult:
        """Run Python tests"""
        # Try pytest first
        success, stdout, stderr = await self._run_command('pytest -v')
        
        if "command not found" in stderr.lower():
            # Try unittest as fallback
            success, stdout, stderr = await self._run_command('python -m unittest discover')
        
        return TestResult(
            test_type='unit_tests',
            passed=success,
            errors=self._extract_test_errors(stdout + stderr) if not success else [],
            output=stdout + stderr
        )
    
    async def _run_python_lint(self) -> TestResult:
        """Run Python linting"""
        # Try flake8
        success, stdout, stderr = await self._run_command('flake8 .')
        
        if "command not found" in stderr.lower():
            # Try pylint as fallback
            success, stdout, stderr = await self._run_command('pylint **/*.py')
        
        return TestResult(
            test_type='lint',
            passed=success,
            errors=self._extract_lint_errors(stdout + stderr) if not success else [],
            output=stdout + stderr
        )
    
    async def _check_system_health(self, project_type: str) -> Dict[str, bool]:
        """Check if the system can start and run"""
        health = {}
        
        if project_type in ['npm', 'typescript', 'react']:
            # Check if server can start
            if (self.workspace_path / 'server.js').exists() or \
               (self.workspace_path / 'index.js').exists():
                health['backend_starts'] = await self._check_service_starts('npm start', 5)
            
            # Check if frontend dev server starts
            if project_type == 'react':
                health['frontend_starts'] = await self._check_service_starts('npm start', 5)
        
        elif project_type == 'python':
            # Check if main.py or app.py exists and runs
            if (self.workspace_path / 'main.py').exists():
                health['app_starts'] = await self._check_service_starts('python main.py', 5)
            elif (self.workspace_path / 'app.py').exists():
                health['app_starts'] = await self._check_service_starts('python app.py', 5)
        
        # Check if API endpoints respond (if applicable)
        if any('api' in str(f).lower() for f in self.workspace_path.glob('**/*.js')):
            health['api_responds'] = await self._check_api_health()
        
        return health
    
    async def _check_service_starts(self, command: str, wait_time: int = 5) -> bool:
        """Check if a service starts successfully"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_path)
            )
            
            # Wait a bit to see if it crashes immediately
            await asyncio.sleep(wait_time)
            
            # Check if process is still running
            if process.returncode is None:
                # Still running, that's good
                process.terminate()
                await process.wait()
                return True
            else:
                # Process exited, that's bad
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to check service: {e}")
            return False
    
    async def _check_api_health(self) -> bool:
        """Check if API endpoints are responding"""
        # Simple check - try to connect to localhost:3000 or :5000
        for port in [3000, 5000, 8000, 8080]:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection('localhost', port),
                    timeout=2.0
                )
                writer.close()
                await writer.wait_closed()
                return True
            except:
                continue
        
        return False
    
    async def _check_npm_script(self, script_name: str) -> bool:
        """Check if an npm script exists in package.json"""
        try:
            with open(self.workspace_path / 'package.json', 'r') as f:
                package_data = json.load(f)
                return script_name in package_data.get('scripts', {})
        except:
            return False
    
    def _parse_test_output(self, output: str) -> Dict[str, int]:
        """Parse test statistics from output"""
        stats = {'total': 0, 'passed': 0, 'failed': 0}
        
        # Look for common test output patterns
        import re
        
        # Jest pattern: "Tests: 1 failed, 2 passed, 3 total"
        jest_match = re.search(r'Tests?:\s*(?:(\d+)\s*failed,\s*)?(\d+)\s*passed,\s*(\d+)\s*total', output)
        if jest_match:
            stats['failed'] = int(jest_match.group(1) or 0)
            stats['passed'] = int(jest_match.group(2))
            stats['total'] = int(jest_match.group(3))
            return stats
        
        # Mocha pattern: "2 passing", "1 failing"
        passing = re.search(r'(\d+)\s*passing', output)
        failing = re.search(r'(\d+)\s*failing', output)
        if passing or failing:
            stats['passed'] = int(passing.group(1)) if passing else 0
            stats['failed'] = int(failing.group(1)) if failing else 0
            stats['total'] = stats['passed'] + stats['failed']
            return stats
        
        # Pytest pattern: "1 failed, 2 passed"
        pytest_match = re.search(r'(\d+)\s*failed.*?(\d+)\s*passed', output)
        if pytest_match:
            stats['failed'] = int(pytest_match.group(1))
            stats['passed'] = int(pytest_match.group(2))
            stats['total'] = stats['passed'] + stats['failed']
        
        return stats
    
    def _extract_test_errors(self, output: str) -> List[str]:
        """Extract test error messages from output"""
        errors = []
        lines = output.split('\n')
        
        # Look for failure indicators
        in_failure = False
        current_error = []
        
        for line in lines:
            if any(indicator in line.lower() for indicator in ['fail', 'error', 'assert']):
                in_failure = True
            elif in_failure and (line.strip() == '' or line.startswith(' ')):
                current_error.append(line)
            elif in_failure:
                if current_error:
                    errors.append('\n'.join(current_error))
                current_error = []
                in_failure = False
        
        # Get last 5 errors max
        return errors[-5:] if errors else ["Test failures detected, check output for details"]
    
    def _extract_lint_errors(self, output: str) -> List[str]:
        """Extract linting errors"""
        errors = []
        lines = output.split('\n')
        
        for line in lines:
            if any(level in line for level in ['error', 'Error', 'ERROR']):
                errors.append(line.strip())
        
        return errors[:10]  # First 10 errors
    
    def _extract_build_errors(self, output: str) -> List[str]:
        """Extract build errors"""
        errors = []
        lines = output.split('\n')
        
        for line in lines:
            if 'error' in line.lower() and not line.strip().startswith('#'):
                errors.append(line.strip())
        
        return errors[:10]
    
    def _generate_fix_suggestions(self, test_results: Dict[str, TestResult], 
                                 system_health: Dict[str, bool]) -> List[str]:
        """Generate suggestions for fixing failures"""
        suggestions = []
        
        # Test failures
        for test_type, result in test_results.items():
            if not result.passed:
                if test_type == 'unit_tests':
                    suggestions.append(f"Fix {result.failed_tests} failing tests")
                elif test_type == 'lint':
                    suggestions.append("Fix linting errors")
                elif test_type == 'build':
                    suggestions.append("Fix build errors")
        
        # System health issues
        if not system_health.get('backend_starts', True):
            suggestions.append("Fix backend startup issues")
        if not system_health.get('frontend_starts', True):
            suggestions.append("Fix frontend startup issues")
        if not system_health.get('api_responds', True):
            suggestions.append("Fix API endpoint issues")
        
        return suggestions
    
    async def analyze_failures(self, verification_result: VerificationResult) -> List[Task]:
        """Analyze failures and create fix tasks"""
        fix_tasks = []
        
        # Analyze test failures
        for test_type, result in verification_result.test_results.items():
            if not result.passed:
                if test_type == 'unit_tests' and result.errors:
                    # Create specific fix tasks for test failures
                    fix_tasks.append(self._create_test_fix_task(result))
                
                elif test_type == 'lint' and result.errors:
                    # Create lint fix task
                    fix_tasks.append(self._create_lint_fix_task(result))
                
                elif test_type == 'build' and result.errors:
                    # Create build fix task
                    fix_tasks.append(self._create_build_fix_task(result))
        
        # Analyze system health issues
        if not verification_result.system_health.get('backend_starts', True):
            fix_tasks.append(self._create_startup_fix_task('backend'))
        
        if not verification_result.system_health.get('api_responds', True):
            fix_tasks.append(self._create_api_fix_task())
        
        return fix_tasks
    
    def _create_test_fix_task(self, test_result: TestResult) -> Task:
        """Create task to fix failing tests"""
        error_summary = '\n'.join(test_result.errors[:3])  # First 3 errors
        
        command = f"""Fix the {test_result.failed_tests} failing tests. 

Errors:
{error_summary}

Analyze the test failures and fix the implementation code or update the tests as appropriate.
Ensure all tests pass after your changes."""
        
        from agentic.models.task import Task
        import uuid
        
        task = Task(
            id=str(uuid.uuid4()),
            command=command,
            affected_areas=['testing', 'debugging'],
            complexity_score=0.7,
            requires_context=True
        )
        task.agent_type_hint = 'aider_backend'  # Good at fixing code
        task.coordination_context = {
            'role': 'test_fixer',
            'priority': 'high'
        }
        
        return task
    
    def _create_lint_fix_task(self, lint_result: TestResult) -> Task:
        """Create task to fix linting errors"""
        error_summary = '\n'.join(lint_result.errors[:5])
        
        command = f"""Fix all linting errors in the codebase.

Linting errors:
{error_summary}

Run the linter and fix all issues. Common issues include:
- Unused variables
- Missing semicolons
- Incorrect indentation
- Import order issues"""
        
        from agentic.models.task import Task
        import uuid
        
        task = Task(
            id=str(uuid.uuid4()),
            command=command,
            affected_areas=['code_quality'],
            complexity_score=0.4,
            requires_context=True
        )
        task.agent_type_hint = 'aider_backend'
        task.coordination_context = {
            'role': 'code_cleaner',
            'priority': 'medium'
        }
        
        return task
    
    def _create_build_fix_task(self, build_result: TestResult) -> Task:
        """Create task to fix build errors"""
        error_summary = '\n'.join(build_result.errors[:5])
        
        command = f"""Fix the build errors preventing compilation.

Build errors:
{error_summary}

Common issues:
- TypeScript type errors
- Missing imports
- Syntax errors
- Configuration issues"""
        
        from agentic.models.task import Task
        import uuid
        
        task = Task(
            id=str(uuid.uuid4()),
            command=command,
            affected_areas=['build', 'configuration'],
            complexity_score=0.6,
            requires_context=True
        )
        task.agent_type_hint = 'aider_backend'
        task.coordination_context = {
            'role': 'build_fixer',
            'priority': 'high'
        }
        
        return task
    
    def _create_startup_fix_task(self, service_type: str) -> Task:
        """Create task to fix startup issues"""
        command = f"""Fix the {service_type} service startup issues.

The {service_type} is failing to start properly. Common issues:
- Missing dependencies in package.json
- Port conflicts
- Missing environment variables
- Syntax errors in main files
- Missing required files

Check logs, fix the issues, and ensure the service starts successfully."""
        
        from agentic.models.task import Task
        import uuid
        
        task = Task(
            id=str(uuid.uuid4()),
            command=command,
            affected_areas=['backend', 'configuration'],
            complexity_score=0.7,
            requires_context=True
        )
        task.agent_type_hint = 'aider_backend'
        task.coordination_context = {
            'role': 'devops_fixer',
            'priority': 'critical'
        }
        
        return task
    
    def _create_api_fix_task(self) -> Task:
        """Create task to fix API issues"""
        command = """Fix API endpoint issues - the API is not responding properly.

Common issues to check:
- Server not binding to correct port
- Routes not properly defined
- Middleware configuration issues
- CORS problems
- Database connection issues

Ensure all API endpoints are accessible and responding correctly."""
        
        from agentic.models.task import Task
        import uuid
        
        task = Task(
            id=str(uuid.uuid4()),
            command=command,
            affected_areas=['api', 'backend'],
            complexity_score=0.8,
            requires_context=True
        )
        task.agent_type_hint = 'aider_backend'
        task.coordination_context = {
            'role': 'api_fixer',
            'priority': 'critical'
        }
        
        return task