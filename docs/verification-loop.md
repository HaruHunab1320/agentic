# Verification Loop System

## Overview

The Verification Loop System is an automated quality assurance feature that ensures code generated by the multi-agent system actually works before the process completes. Similar to how Claude Code iteratively fixes issues until tests pass, this system automatically runs tests, identifies failures, and dispatches fix tasks to available agents.

## Key Features

### 1. Automated Test Execution
- Automatically detects project type (React, TypeScript, Python, etc.)
- Runs appropriate test commands (npm test, pytest, etc.)
- Executes lint checks and build verification
- Checks system health (service startup, API responses)

### 2. Intelligent Failure Analysis
- Parses test output to identify specific failures
- Extracts error messages and stack traces
- Generates targeted fix tasks for each type of failure
- Prioritizes fixes based on severity

### 3. Iterative Fix Loop
- Runs up to 3 iterations of test → fix → retest
- Assigns fix tasks to available agents in the swarm
- Tracks progress and updates verification status
- Stops when all tests pass or max iterations reached

### 4. Multi-Language Support
- **JavaScript/TypeScript**: npm test, lint, build
- **Python**: pytest, flake8, mypy
- **React**: Component tests, E2E tests
- **General**: Custom test commands

## How It Works

### 1. Initial Execution Phase
```python
# Agents complete their initial tasks
result = await coordination_engine.execute_coordinated_tasks(tasks)
```

### 2. Verification Phase (if enabled)
```python
if result.status == "completed" and context.get('verify', True):
    verification_passed = await self._run_verification_loop(result)
```

### 3. Verification Loop Process
```
1. Run all tests and checks
   ├── Unit tests
   ├── Lint checks
   ├── Build verification
   └── System health checks

2. If failures detected:
   ├── Analyze failures
   ├── Create fix tasks
   ├── Execute fixes
   └── Re-run verification

3. Repeat up to 3 times or until all pass
```

## Usage

### Enable Verification in Multi-Agent Commands

```python
# In your CLI or script
result = await coordination_engine.execute_multi_agent_command(
    "Build a complete React application with tests",
    context={
        'verify': True,  # Enable verification loop
        'agent_type_strategy': 'dynamic'
    }
)

# Check verification status
if result.verification_status == "passed":
    print("✅ All tests passed!")
elif result.verification_status == "failed_after_retries":
    print("⚠️ Some tests still failing after fixes")
```

### Disable Verification (if needed)

```python
context = {'verify': False}  # Skip verification phase
```

## Verification Result Status

The `ExecutionResult` now includes a `verification_status` field:

- `None` - Verification was not run
- `"passed"` - All tests passed (possibly after fixes)
- `"failed_after_retries"` - Tests still failing after max iterations

## Example Scenarios

### Scenario 1: React Component with Tests
```
Command: "Create a UserProfile component with unit tests"

1. Agent creates component and test files
2. Verification runs `npm test`
3. Test fails due to missing mock
4. Fix task generated: "Add missing mock for API call"
5. Agent fixes the test
6. Verification runs again - all tests pass ✅
```

### Scenario 2: Python API with Lint Errors
```
Command: "Build a REST API with authentication"

1. Agent creates API endpoints
2. Verification runs pytest and flake8
3. Lint errors found: unused imports, line too long
4. Fix task generated: "Fix linting errors"
5. Agent cleans up code
6. Verification passes ✅
```

### Scenario 3: Build Failures
```
Command: "Create TypeScript library with strict typing"

1. Agent creates library code
2. Verification runs `npm run build`
3. TypeScript errors: type mismatches
4. Fix task generated: "Fix TypeScript type errors"
5. Agent fixes type issues
6. Build succeeds ✅
```

## Fix Task Generation

The system generates specific fix tasks based on failure type:

### Test Failures
```python
command = f"""Fix the {test_result.failed_tests} failing tests. 

Errors:
{error_summary}

Analyze the test failures and fix the implementation code or update the tests as appropriate.
"""
```

### Lint Errors
```python
command = f"""Fix all linting errors in the codebase.

Linting errors:
{error_summary}

Common issues include:
- Unused variables
- Missing semicolons
- Incorrect indentation
"""
```

### Build Errors
```python
command = f"""Fix the build errors preventing compilation.

Build errors:
{error_summary}

Common issues:
- TypeScript type errors
- Missing imports
- Syntax errors
"""
```

## Configuration

### Project Detection
The system automatically detects project type by checking for:
- `package.json` → Node.js/React/TypeScript
- `requirements.txt` → Python
- `pyproject.toml` → Python with modern tooling

### Custom Test Commands
You can extend the verification system with custom commands:

```python
self.test_commands = {
    'custom': {
        'test': 'make test',
        'lint': 'make lint',
        'build': 'make build'
    }
}
```

## Benefits

1. **Quality Assurance**: Ensures generated code actually works
2. **Time Savings**: Fixes issues automatically without manual intervention
3. **Learning**: System learns from failures and improves over time
4. **Confidence**: Know that delivered code passes all tests
5. **Documentation**: Verification logs provide audit trail

## Best Practices

1. **Write Good Tests**: The system is only as good as your test coverage
2. **Clear Error Messages**: Help the system understand failures
3. **Incremental Changes**: Smaller changes are easier to verify
4. **Monitor Progress**: Watch the verification output for insights

## Limitations

1. **Max 3 Iterations**: Prevents infinite loops
2. **Test Quality**: Can't fix poorly written tests
3. **Complex Failures**: Some issues may need human intervention
4. **Resource Usage**: Verification adds time to execution

## Future Enhancements

1. **Smart Retry Logic**: Learn which fixes work for common errors
2. **Parallel Testing**: Run test suites in parallel
3. **Partial Verification**: Test only changed components
4. **Historical Analysis**: Track common failure patterns
5. **Custom Validators**: Add project-specific checks