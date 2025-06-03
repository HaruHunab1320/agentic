"""
Pytest configuration and shared fixtures for Agentic tests
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from click.testing import CliRunner


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_project_dir(temp_dir: Path) -> Path:
    """Create a sample project directory for testing"""
    # Create basic project structure
    (temp_dir / "src").mkdir()
    (temp_dir / "tests").mkdir()
    (temp_dir / "docs").mkdir()
    
    # Create package.json for JavaScript detection
    package_json = {
        "name": "test-project",
        "dependencies": {
            "react": "^18.0.0",
            "typescript": "^5.0.0"
        },
        "devDependencies": {
            "jest": "^29.0.0",
            "@testing-library/react": "^13.0.0"
        }
    }
    
    import json
    (temp_dir / "package.json").write_text(json.dumps(package_json, indent=2))
    
    # Create some source files
    (temp_dir / "src" / "App.tsx").write_text("""
import React from 'react';

const App: React.FC = () => {
  return <div>Hello World</div>;
};

export default App;
""")
    
    return temp_dir


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI runner for testing"""
    return CliRunner()


@pytest.fixture
def python_project_dir(temp_dir: Path) -> Path:
    """Create a sample Python project directory"""
    # Create Python project structure
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "mypackage").mkdir()
    (temp_dir / "tests").mkdir()
    
    # Create pyproject.toml
    pyproject_content = """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "test-package"
dependencies = [
    "fastapi>=0.68.0",
    "pydantic>=2.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0"
]
"""
    (temp_dir / "pyproject.toml").write_text(pyproject_content.strip())
    
    # Create source files
    (temp_dir / "src" / "mypackage" / "__init__.py").write_text('__version__ = "0.1.0"')
    (temp_dir / "src" / "mypackage" / "main.py").write_text("""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
""")
    
    return temp_dir 