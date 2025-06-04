# End-to-end tests for user scenarios
import pytest
import asyncio
import tempfile
from pathlib import Path
import subprocess
import json
import time
from typing import Dict, List

from agentic.cli.main import AgenticCLI
from agentic.core.orchestrator import Orchestrator
from agentic.models.project import Project
from agentic.models.config import AgenticConfig
from agentic.models.task import TaskResult


class TestUserScenarios:
    """End-to-end tests simulating real user workflows"""

    @pytest.fixture
    async def clean_environment(self):
        """Setup clean test environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            test_dir = Path(temp_dir)
            
            # Change to test directory
            import os
            os.chdir(test_dir)
            
            yield test_dir
            
            # Restore original directory
            os.chdir(original_cwd)

    @pytest.mark.e2e
    async def test_new_user_first_project_setup(self, clean_environment):
        """Test: New user sets up their first project"""
        project_dir = clean_environment / "my-first-app"
        project_dir.mkdir()
        
        # User initializes project
        cli = AgenticCLI()
        init_result = await cli.run_command([
            "init", 
            "--project-path", str(project_dir),
            "--project-type", "react-app"
        ])
        
        assert init_result.success
        assert (project_dir / ".agentic").exists()
        assert (project_dir / "agentic.config.yaml").exists()
        
        # User runs their first command
        first_command_result = await cli.run_command([
            "create", "component", "UserProfile",
            "--project-path", str(project_dir)
        ])
        
        assert first_command_result.success
        assert (project_dir / "src" / "components" / "UserProfile.jsx").exists()
        
        # Verify component is properly structured
        component_content = (project_dir / "src" / "components" / "UserProfile.jsx").read_text()
        assert "export default UserProfile" in component_content
        assert "function UserProfile" in component_content or "const UserProfile" in component_content

    @pytest.mark.e2e
    async def test_developer_building_fullstack_app(self, clean_environment):
        """Test: Developer building a complete fullstack application"""
        app_dir = clean_environment / "fullstack-todo-app"
        app_dir.mkdir()
        
        cli = AgenticCLI()
        
        # Initialize fullstack project
        await cli.run_command([
            "init",
            "--project-path", str(app_dir),
            "--project-type", "fullstack",
            "--backend", "node-express",
            "--frontend", "react",
            "--database", "postgresql"
        ])
        
        # Build backend API
        backend_result = await cli.run_command([
            "implement", "todo-api",
            "--description", "RESTful API for todo items with CRUD operations",
            "--project-path", str(app_dir)
        ])
        
        assert backend_result.success
        assert (app_dir / "backend" / "routes" / "todos.js").exists()
        assert (app_dir / "backend" / "models" / "Todo.js").exists()
        
        # Build frontend
        frontend_result = await cli.run_command([
            "implement", "todo-frontend", 
            "--description", "React frontend with todo list, add, edit, delete functionality",
            "--project-path", str(app_dir)
        ])
        
        assert frontend_result.success
        assert (app_dir / "frontend" / "src" / "components" / "TodoList.jsx").exists()
        assert (app_dir / "frontend" / "src" / "components" / "TodoItem.jsx").exists()
        
        # Add database setup
        db_result = await cli.run_command([
            "setup", "database",
            "--description", "PostgreSQL schema and migrations for todos",
            "--project-path", str(app_dir)
        ])
        
        assert db_result.success
        assert (app_dir / "database" / "migrations").exists()
        
        # Add tests
        test_result = await cli.run_command([
            "generate", "tests",
            "--coverage", "all",
            "--project-path", str(app_dir)
        ])
        
        assert test_result.success
        assert (app_dir / "backend" / "tests").exists()
        assert (app_dir / "frontend" / "src" / "__tests__").exists()

    @pytest.mark.e2e
    async def test_team_collaboration_workflow(self, clean_environment):
        """Test: Multiple developers working on same project"""
        project_dir = clean_environment / "team-project"
        project_dir.mkdir()
        
        cli = AgenticCLI()
        
        # Team lead initializes project
        await cli.run_command([
            "init",
            "--project-path", str(project_dir),
            "--team-mode", "true"
        ])
        
        # Developer 1: Works on authentication
        auth_result = await cli.run_command([
            "implement", "user-auth",
            "--assignee", "dev1",
            "--project-path", str(project_dir)
        ])
        
        # Developer 2: Works on dashboard (parallel)
        dashboard_task = asyncio.create_task(cli.run_command([
            "implement", "dashboard",
            "--assignee", "dev2", 
            "--project-path", str(project_dir)
        ]))
        
        # Wait for both to complete
        dashboard_result = await dashboard_task
        
        assert auth_result.success
        assert dashboard_result.success
        
        # Verify no conflicts
        assert len(auth_result.conflicts) == 0
        assert len(dashboard_result.conflicts) == 0
        
        # Team lead reviews and integrates
        integration_result = await cli.run_command([
            "integrate", "features",
            "--features", "user-auth,dashboard",
            "--project-path", str(project_dir)
        ])
        
        assert integration_result.success

    @pytest.mark.e2e
    async def test_legacy_code_modernization(self, clean_environment):
        """Test: Developer modernizing legacy codebase"""
        legacy_dir = clean_environment / "legacy-app"
        legacy_dir.mkdir()
        
        # Create legacy codebase
        await self._create_legacy_codebase(legacy_dir)
        
        cli = AgenticCLI()
        
        # Initialize Agentic on existing project
        await cli.run_command([
            "init",
            "--project-path", str(legacy_dir),
            "--existing-project", "true"
        ])
        
        # Analyze legacy code
        analysis_result = await cli.run_command([
            "analyze", "codebase",
            "--generate-report", "true",
            "--project-path", str(legacy_dir)
        ])
        
        assert analysis_result.success
        assert (legacy_dir / "agentic-analysis-report.md").exists()
        
        # Modernize step by step
        modernization_steps = [
            "update-dependencies",
            "add-type-annotations", 
            "improve-error-handling",
            "add-tests",
            "optimize-performance"
        ]
        
        for step in modernization_steps:
            step_result = await cli.run_command([
                "modernize", step,
                "--project-path", str(legacy_dir)
            ])
            assert step_result.success
        
        # Verify modernization
        final_analysis = await cli.run_command([
            "analyze", "codebase", 
            "--project-path", str(legacy_dir)
        ])
        
        assert final_analysis.code_quality_score > 7.0
        assert final_analysis.maintainability_score > 8.0

    @pytest.mark.e2e
    async def test_debugging_and_issue_resolution(self, clean_environment):
        """Test: Developer debugging issues with Agentic's help"""
        project_dir = clean_environment / "buggy-app"
        project_dir.mkdir()
        
        # Create project with intentional bugs
        await self._create_project_with_bugs(project_dir)
        
        cli = AgenticCLI()
        await cli.run_command(["init", "--project-path", str(project_dir)])
        
        # Developer runs into issues
        test_result = await cli.run_command([
            "test", "run",
            "--project-path", str(project_dir)
        ])
        
        # Tests should fail initially
        assert not test_result.all_passed
        assert len(test_result.failures) > 0
        
        # Use Agentic to debug
        debug_result = await cli.run_command([
            "debug", "test-failures",
            "--auto-fix", "true",
            "--project-path", str(project_dir)
        ])
        
        assert debug_result.success
        assert len(debug_result.fixes_applied) > 0
        
        # Run tests again - should pass now
        final_test_result = await cli.run_command([
            "test", "run",
            "--project-path", str(project_dir)
        ])
        
        assert final_test_result.all_passed

    @pytest.mark.e2e
    async def test_performance_optimization_workflow(self, clean_environment):
        """Test: Developer optimizing application performance"""
        app_dir = clean_environment / "slow-app"
        app_dir.mkdir()
        
        # Create app with performance issues
        await self._create_slow_application(app_dir)
        
        cli = AgenticCLI()
        await cli.run_command(["init", "--project-path", str(app_dir)])
        
        # Analyze performance
        perf_analysis = await cli.run_command([
            "analyze", "performance", 
            "--project-path", str(app_dir)
        ])
        
        assert perf_analysis.success
        assert len(perf_analysis.bottlenecks) > 0
        
        # Apply optimizations
        optimization_result = await cli.run_command([
            "optimize", "performance",
            "--target", "all-bottlenecks",
            "--project-path", str(app_dir)
        ])
        
        assert optimization_result.success
        
        # Verify improvements
        final_analysis = await cli.run_command([
            "analyze", "performance",
            "--project-path", str(app_dir)
        ])
        
        assert final_analysis.performance_score > perf_analysis.performance_score
        assert len(final_analysis.bottlenecks) < len(perf_analysis.bottlenecks)

    @pytest.mark.e2e
    async def test_deployment_preparation(self, clean_environment):
        """Test: Developer preparing app for deployment"""
        app_dir = clean_environment / "deploy-ready-app"
        app_dir.mkdir()
        
        # Create development app
        await self._create_development_app(app_dir)
        
        cli = AgenticCLI()
        await cli.run_command(["init", "--project-path", str(app_dir)])
        
        # Prepare for production
        deployment_prep = await cli.run_command([
            "prepare", "deployment",
            "--target", "production",
            "--platform", "aws",
            "--project-path", str(app_dir)
        ])
        
        assert deployment_prep.success
        
        # Verify deployment artifacts
        assert (app_dir / "Dockerfile").exists()
        assert (app_dir / "docker-compose.yml").exists()
        assert (app_dir / ".github" / "workflows" / "deploy.yml").exists()
        assert (app_dir / "terraform").exists()
        
        # Security audit
        security_result = await cli.run_command([
            "audit", "security",
            "--project-path", str(app_dir)
        ])
        
        assert security_result.success
        assert security_result.critical_issues == 0

    async def _create_legacy_codebase(self, project_path: Path):
        """Create a realistic legacy codebase"""
        # Legacy JavaScript with old patterns
        (project_path / "src").mkdir()
        (project_path / "src" / "app.js").write_text('''
var express = require('express');
var app = express();

app.get('/', function(req, res) {
    var data = getData();
    res.send(data);
});

function getData() {
    // Synchronous file reading (bad practice)
    var fs = require('fs');
    try {
        var data = fs.readFileSync('data.json', 'utf8');
        return JSON.parse(data);
    } catch (e) {
        return null;
    }
}

app.listen(3000);
        ''')
        
        # Legacy package.json
        (project_path / "package.json").write_text(json.dumps({
            "name": "legacy-app",
            "version": "1.0.0",
            "dependencies": {
                "express": "^3.0.0"  # Very old version
            }
        }))

    async def _create_project_with_bugs(self, project_path: Path):
        """Create project with intentional bugs for testing"""
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        
        # Buggy code
        (project_path / "src" / "calculator.js").write_text('''
function add(a, b) {
    return a + b;
}

function divide(a, b) {
    return a / b;  // Bug: no division by zero check
}

function multiply(a, b) {
    return a * b;
}

module.exports = { add, divide, multiply };
        ''')
        
        # Test that will fail
        (project_path / "tests" / "calculator.test.js").write_text('''
const { add, divide, multiply } = require('../src/calculator');

test('add function', () => {
    expect(add(2, 3)).toBe(5);
});

test('divide function', () => {
    expect(divide(10, 2)).toBe(5);
    expect(divide(10, 0)).toBe(Infinity);  // This will fail
});
        ''')
        
        (project_path / "package.json").write_text(json.dumps({
            "name": "buggy-app",
            "scripts": {"test": "jest"},
            "devDependencies": {"jest": "^27.0.0"}
        }))

    async def _create_slow_application(self, project_path: Path):
        """Create application with performance issues"""
        (project_path / "src").mkdir()
        
        # Inefficient code
        (project_path / "src" / "slow.js").write_text('''
function inefficientSort(arr) {
    // Bubble sort (inefficient)
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                let temp = arr[j];
                arr[j] = arr[j + 1]; 
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}

function memoryLeak() {
    // Intentional memory leak
    const largeArray = new Array(1000000).fill('data');
    setTimeout(() => memoryLeak(), 100);  // Recursive call
}

module.exports = { inefficientSort, memoryLeak };
        ''')

    async def _create_development_app(self, project_path: Path):
        """Create a development-ready app needing deployment prep"""
        (project_path / "src").mkdir()
        (project_path / "src" / "server.js").write_text('''
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.json({ message: 'Hello World' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
        ''')
        
        (project_path / "package.json").write_text(json.dumps({
            "name": "deploy-ready-app",
            "version": "1.0.0",
            "main": "src/server.js",
            "scripts": {
                "start": "node src/server.js",
                "dev": "nodemon src/server.js"
            },
            "dependencies": {
                "express": "^4.18.0"
            },
            "devDependencies": {
                "nodemon": "^2.0.0"
            }
        }))


class TestCLIUserExperience:
    """Test CLI user experience and usability"""
    
    @pytest.mark.e2e
    async def test_help_and_documentation_access(self, clean_environment):
        """Test that users can easily access help and documentation"""
        cli = AgenticCLI()
        
        # Test main help
        help_result = await cli.run_command(["--help"])
        assert help_result.success
        assert "Available commands:" in help_result.output
        
        # Test command-specific help
        init_help = await cli.run_command(["init", "--help"])
        assert init_help.success
        assert "Initialize" in init_help.output
        
        # Test examples
        examples_result = await cli.run_command(["examples"])
        assert examples_result.success
        assert len(examples_result.examples) > 0

    @pytest.mark.e2e
    async def test_error_handling_and_user_feedback(self, clean_environment):
        """Test that errors are handled gracefully with helpful feedback"""
        cli = AgenticCLI()
        
        # Test invalid command
        invalid_result = await cli.run_command(["invalid-command"])
        assert not invalid_result.success
        assert "Did you mean" in invalid_result.error_message
        
        # Test missing required arguments
        missing_args_result = await cli.run_command(["init"])
        assert not missing_args_result.success
        assert "required" in missing_args_result.error_message.lower()
        
        # Test invalid project path
        invalid_path_result = await cli.run_command([
            "init", "--project-path", "/nonexistent/path"
        ])
        assert not invalid_path_result.success
        assert "path" in invalid_path_result.error_message.lower()

    @pytest.mark.e2e
    async def test_progress_feedback_and_interruption(self, clean_environment):
        """Test progress feedback and ability to interrupt long operations"""
        project_dir = clean_environment / "progress-test"
        project_dir.mkdir()
        
        cli = AgenticCLI()
        await cli.run_command(["init", "--project-path", str(project_dir)])
        
        # Start long-running operation
        long_task = asyncio.create_task(cli.run_command([
            "implement", "complex-feature",
            "--description", "Very complex feature requiring long processing",
            "--project-path", str(project_dir)
        ]))
        
        # Wait for task to start
        await asyncio.sleep(2)
        
        # Check progress is being reported
        progress = await cli.get_current_progress()
        assert progress is not None
        assert progress.percentage >= 0
        
        # Test interruption
        await cli.interrupt_current_task()
        
        # Task should be cancelled gracefully
        result = await long_task
        assert result.status == "interrupted"
        assert "gracefully stopped" in result.message.lower()


class TestNewUserExperience:
    """Test end-to-end scenarios for new users"""

    @pytest.fixture
    async def clean_project(self):
        """Create a clean project directory for new user simulation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "my_first_project" 
            project_path.mkdir()
            yield project_path

    @pytest.fixture
    async def user_orchestrator(self, clean_project):
        """Create an orchestrator as a new user would"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize(clean_project)
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.e2e
    async def test_first_time_user_setup(self, clean_project):
        """Test the first-time user experience from project setup"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        
        # User initializes Agentic with their project
        await orchestrator.initialize(clean_project)
        
        # Verify system is ready for use
        status = orchestrator.get_system_status()
        assert status["initialized"] is True
        assert status["project_analyzed"] is True
        
        # User can get system status
        agent_status = orchestrator.get_agent_status()
        assert "total_agents" in agent_status
        
        await orchestrator.shutdown()

    @pytest.mark.e2e
    async def test_simple_hello_world_creation(self, user_orchestrator):
        """Test creating a simple hello world application"""
        commands = [
            "Create a simple hello world program",
            "Add basic documentation",
            "Create a test for the hello world function"
        ]
        
        results = []
        for command in commands:
            result = await user_orchestrator.execute_command(command)
            results.append(result)
            # Brief pause between commands
            await asyncio.sleep(0.1)
        
        # Verify all commands were handled
        assert len(results) == len(commands)
        for result in results:
            assert isinstance(result, TaskResult)
            assert result.task_id is not None

    @pytest.mark.e2e
    async def test_progressive_feature_development(self, user_orchestrator):
        """Test building features progressively"""
        # Start simple
        result1 = await user_orchestrator.execute_command("Create a basic function that adds two numbers")
        assert isinstance(result1, TaskResult)
        
        # Add complexity
        result2 = await user_orchestrator.execute_command("Add input validation to the function")
        assert isinstance(result2, TaskResult)
        
        # Add testing
        result3 = await user_orchestrator.execute_command("Create comprehensive tests for the function")
        assert isinstance(result3, TaskResult)
        
        # All should work together
        assert all(isinstance(r, TaskResult) for r in [result1, result2, result3])

    @pytest.mark.e2e
    async def test_user_help_and_guidance(self, user_orchestrator):
        """Test user getting help and guidance"""
        help_commands = [
            "help",
            "what can you do",
            "show me examples",
            "how do I create a React component"
        ]
        
        for command in help_commands:
            result = await user_orchestrator.execute_command(command)
            # Should provide helpful response without error
            assert isinstance(result, TaskResult)


class TestExperiencedUserWorkflows:
    """Test workflows for experienced users"""

    @pytest.fixture
    async def existing_project(self):
        """Create an existing project with some structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "existing_project"
            project_path.mkdir()
            
            # Create existing structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()
            (project_path / "src" / "main.py").write_text("def main():\n    print('Hello')")
            (project_path / "README.md").write_text("# My Project")
            
            yield project_path

    @pytest.fixture
    async def experienced_orchestrator(self, existing_project):
        """Create orchestrator for experienced user"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize(existing_project)
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.e2e
    async def test_complex_refactoring_workflow(self, experienced_orchestrator):
        """Test complex refactoring workflow"""
        commands = [
            "Analyze the current code structure",
            "Suggest improvements for maintainability", 
            "Add type hints to existing functions",
            "Create unit tests for all functions"
        ]
        
        results = []
        for command in commands:
            result = await experienced_orchestrator.execute_command(command)
            results.append(result)
            await asyncio.sleep(0.1)
        
        # All operations should complete
        assert len(results) == len(commands)
        assert all(isinstance(r, TaskResult) for r in results)

    @pytest.mark.e2e
    async def test_multi_file_coordination(self, experienced_orchestrator):
        """Test coordination across multiple files"""
        commands = [
            "Create a new module for database operations",
            "Create corresponding tests for the database module",
            "Update the main module to use the new database module",
            "Update documentation to reflect the changes"
        ]
        
        # Execute all commands
        for command in commands:
            result = await experienced_orchestrator.execute_command(command)
            assert isinstance(result, TaskResult)

    @pytest.mark.e2e  
    async def test_performance_optimization_workflow(self, experienced_orchestrator):
        """Test performance optimization workflow"""
        # User requests performance analysis and optimization
        result = await experienced_orchestrator.execute_command(
            "Analyze performance bottlenecks and optimize the code"
        )
        
        assert isinstance(result, TaskResult)
        assert result.task_id is not None


class TestCollaborativeWorkflows:
    """Test collaborative development scenarios"""

    @pytest.fixture
    async def team_project(self):
        """Create a project simulating team development"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "team_project"
            project_path.mkdir()
            
            # Create team project structure
            (project_path / "frontend").mkdir()
            (project_path / "backend").mkdir()
            (project_path / "shared").mkdir()
            (project_path / "docs").mkdir()
            
            yield project_path

    @pytest.fixture
    async def team_orchestrator(self, team_project):
        """Create orchestrator for team scenarios"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize(team_project)
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.e2e
    async def test_feature_branch_workflow(self, team_orchestrator):
        """Test feature development workflow"""
        # Simulate working on a feature
        commands = [
            "Create a new user authentication feature",
            "Add frontend components for login",
            "Add backend API endpoints",
            "Create integration tests",
            "Update documentation"
        ]
        
        for command in commands:
            result = await team_orchestrator.execute_command(command)
            assert isinstance(result, TaskResult)
            # Small delay to simulate real development
            await asyncio.sleep(0.1)

    @pytest.mark.e2e
    async def test_bug_fix_workflow(self, team_orchestrator):
        """Test bug fixing workflow"""
        # Create a bug scenario
        await team_orchestrator.execute_command("Create a function with a subtle bug")
        
        # Fix the bug
        result = await team_orchestrator.execute_command("Find and fix any bugs in the code")
        assert isinstance(result, TaskResult)
        
        # Add tests to prevent regression
        result = await team_orchestrator.execute_command("Add tests to prevent this bug from recurring")
        assert isinstance(result, TaskResult)

    @pytest.mark.e2e
    async def test_code_review_preparation(self, team_orchestrator):
        """Test preparing code for review"""
        # Make some changes
        await team_orchestrator.execute_command("Add a new utility function")
        
        # Prepare for code review
        commands = [
            "Add documentation to the new function",
            "Add comprehensive tests", 
            "Check code style and formatting",
            "Verify all tests pass"
        ]
        
        for command in commands:
            result = await team_orchestrator.execute_command(command)
            assert isinstance(result, TaskResult)


class TestProductionWorkflows:
    """Test production-ready development workflows"""

    @pytest.fixture
    async def production_project(self):
        """Create a production-like project"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "production_app"
            project_path.mkdir()
            
            # Create production structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()
            (project_path / "config").mkdir()
            (project_path / "docker").mkdir()
            (project_path / "docs").mkdir()
            
            # Add some basic files
            (project_path / "requirements.txt").write_text("flask==2.0.1")
            (project_path / "Dockerfile").write_text("FROM python:3.9")
            
            yield project_path

    @pytest.fixture
    async def production_orchestrator(self, production_project):
        """Create orchestrator for production scenarios"""
        config = AgenticConfig()
        orchestrator = Orchestrator(config)
        await orchestrator.initialize(production_project)
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.e2e
    async def test_deployment_preparation(self, production_orchestrator):
        """Test preparing application for deployment"""
        commands = [
            "Review deployment configuration",
            "Ensure all dependencies are properly specified",
            "Add health check endpoints",
            "Verify production readiness"
        ]
        
        for command in commands:
            result = await production_orchestrator.execute_command(command)
            assert isinstance(result, TaskResult)

    @pytest.mark.e2e
    async def test_security_review_workflow(self, production_orchestrator):
        """Test security review workflow"""
        # Security-focused commands
        commands = [
            "Review code for security vulnerabilities",
            "Check for hardcoded secrets",
            "Verify input validation",
            "Add security headers"
        ]
        
        for command in commands:
            result = await production_orchestrator.execute_command(command)
            assert isinstance(result, TaskResult)

    @pytest.mark.e2e
    async def test_monitoring_setup(self, production_orchestrator):
        """Test setting up monitoring and logging"""
        commands = [
            "Add application logging",
            "Set up error tracking",
            "Add performance monitoring",
            "Create health check endpoints"
        ]
        
        for command in commands:
            result = await production_orchestrator.execute_command(command)
            assert isinstance(result, TaskResult)


class TestUserExperienceFlow:
    """Test the overall user experience flow"""

    @pytest.mark.e2e
    async def test_complete_development_lifecycle(self):
        """Test complete development lifecycle from start to finish"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "lifecycle_project"
            project_path.mkdir()
            
            config = AgenticConfig()
            orchestrator = Orchestrator(config)
            
            try:
                # Initialize project
                await orchestrator.initialize(project_path)
                
                # Development phases
                phases = [
                    "Initialize a new web application project",
                    "Create basic application structure", 
                    "Add user authentication",
                    "Create API endpoints",
                    "Add frontend components",
                    "Create comprehensive tests",
                    "Add documentation",
                    "Prepare for deployment"
                ]
                
                for phase in phases:
                    result = await orchestrator.execute_command(phase)
                    assert isinstance(result, TaskResult)
                    # Brief pause between phases
                    await asyncio.sleep(0.1)
                
                # Verify project is in good state
                status = orchestrator.get_system_status()
                assert status["initialized"] is True
                
            finally:
                await orchestrator.shutdown()

    @pytest.mark.e2e
    async def test_error_recovery_and_user_guidance(self):
        """Test how system recovers from errors and guides users"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "error_test_project"
            project_path.mkdir()
            
            config = AgenticConfig()
            orchestrator = Orchestrator(config)
            
            try:
                await orchestrator.initialize(project_path)
                
                # Try some commands that might cause issues
                problematic_commands = [
                    "",  # Empty command
                    "do something impossible",  # Vague command
                    "create file with invalid characters \\/:*?\"<>|",  # Invalid filename
                ]
                
                for command in problematic_commands:
                    result = await orchestrator.execute_command(command)
                    # System should handle gracefully
                    assert isinstance(result, TaskResult)
                    assert result.task_id is not None
                
                # System should still work after errors
                result = await orchestrator.execute_command("Create a simple hello world function")
                assert isinstance(result, TaskResult)
                
            finally:
                await orchestrator.shutdown() 