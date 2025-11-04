# Agentic Codebase Cleanup Plan

## Overview
This document outlines the systematic cleanup of the Agentic codebase based on the comprehensive review findings.

## Phase 1: Quick Wins (Immediate)

### 1.1 Remove Unused Imports
```bash
# Install autoflake if not already installed
pip install autoflake

# Remove unused imports from all Python files
find src/agentic -name "*.py" -type f -exec autoflake --remove-all-unused-imports --in-place {} \;
```

### 1.2 Move Test Files to Proper Location
- Move `test_aider_exploration.py` → `tests/integration/`
- Move `test_language_context.py` → `tests/unit/`
- Remove `test_*.py` files that are just debugging scripts

### 1.3 Archive Documentation Clutter
Create `docs/archive/` and move:
- All `*_FIX.md` files
- All `*_SUMMARY.md` files
- Old exploration/analysis files

### 1.4 Remove Example/Demo Files
- Delete `fibonacci.py` and `test_fibonacci.py`
- Delete other one-off example files

## Phase 2: Consolidation (Week 1)

### 2.1 Consolidate Swarm Monitor
1. Analyze which features from each version are actually needed
2. Create single `swarm_monitor.py` with configurable display modes
3. Update all imports to use the consolidated version
4. Archive old versions in `legacy/` folder

### 2.2 Merge Coordination Engines
1. Integrate safety features from `coordination_engine_safe.py` into main
2. Use feature flags or configuration for safety mode
3. Update orchestrator to use single implementation

### 2.3 Centralize Configuration
1. Create `src/agentic/config/` module
2. Move all DEFAULT_* constants to central location
3. Create environment variable documentation
4. Add configuration validation

## Phase 3: Code Quality (Week 2)

### 3.1 Standardize Error Handling
1. Create `src/agentic/utils/error_handling.py`
2. Define common error patterns and utilities
3. Replace duplicated error handling code

### 3.2 Fix Test Coverage
1. Add real integration tests for agent interactions
2. Create unit tests for agent implementations
3. Test CLI commands properly
4. Remove mock-heavy tests that don't test real functionality

### 3.3 Remove Dead Code
1. Delete unused modules (enterprise features that aren't integrated)
2. Remove stub implementations
3. Clean up incomplete features

## Phase 4: Documentation & Cleanup (Week 3)

### 4.1 Update Documentation
1. Create proper README with setup instructions
2. Document all environment variables
3. Create architecture documentation
4. Add API documentation

### 4.2 Final Cleanup
1. Run linting and formatting
2. Update all imports
3. Run full test suite
4. Create migration guide for any breaking changes

## Tracking Progress

- [ ] Phase 1.1: Remove unused imports
- [ ] Phase 1.2: Move test files
- [ ] Phase 1.3: Archive documentation
- [ ] Phase 1.4: Remove examples
- [ ] Phase 2.1: Consolidate swarm monitor
- [ ] Phase 2.2: Merge coordination engines
- [ ] Phase 2.3: Centralize configuration
- [ ] Phase 3.1: Standardize error handling
- [ ] Phase 3.2: Fix test coverage
- [ ] Phase 3.3: Remove dead code
- [ ] Phase 4.1: Update documentation
- [ ] Phase 4.2: Final cleanup

## Expected Outcomes

1. **Codebase Size**: Reduced by 25-30%
2. **Maintainability**: Significantly improved
3. **Test Coverage**: Increased from ~40% to 70%+
4. **Configuration**: Centralized and documented
5. **Performance**: Faster imports, cleaner execution paths

## Notes

- Always run tests after each major change
- Keep backups of removed code in `legacy/` folder initially
- Document any breaking changes
- Consider creating a `v2.0` tag before major refactoring