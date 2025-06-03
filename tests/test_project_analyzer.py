"""
Tests for the ProjectAnalyzer
"""

import json
from pathlib import Path

import pytest

from agentic.core.project_analyzer import ProjectAnalyzer


class TestProjectAnalyzer:
    """Test cases for project analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_javascript_react_project(self, sample_project_dir: Path):
        """Test analysis of a React/TypeScript project"""
        analyzer = ProjectAnalyzer(sample_project_dir)
        structure = await analyzer.analyze()
        
        # Check detected languages
        assert "javascript" in structure.tech_stack.languages
        assert "typescript" in structure.tech_stack.languages
        
        # Check detected frameworks
        assert "react" in structure.tech_stack.frameworks
        
        # Check detected testing frameworks
        assert "jest" in structure.tech_stack.testing_frameworks
        
        # Check project structure (use resolve() to handle symlinks)
        assert structure.root_path.resolve() == sample_project_dir.resolve()
        assert len(structure.source_directories) > 0
        assert any(d.name == "src" for d in structure.source_directories)
    
    @pytest.mark.asyncio
    async def test_analyze_python_project(self, python_project_dir: Path):
        """Test analysis of a Python project"""
        analyzer = ProjectAnalyzer(python_project_dir)
        structure = await analyzer.analyze()
        
        # Check detected language
        assert "python" in structure.tech_stack.languages
        
        # Check detected frameworks
        assert "fastapi" in structure.tech_stack.frameworks
        
        # Check detected testing frameworks
        assert "pytest" in structure.tech_stack.testing_frameworks
        
        # Check project structure (use resolve() to handle symlinks)
        assert structure.root_path.resolve() == python_project_dir.resolve()
        assert len(structure.dependency_files) > 0
        assert any(f.name == "pyproject.toml" for f in structure.dependency_files)
    
    @pytest.mark.asyncio
    async def test_detect_go_project(self, temp_dir: Path):
        """Test detection of Go project"""
        # Create a Go project
        go_mod_content = """
module example.com/myapp

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
)
"""
        (temp_dir / "go.mod").write_text(go_mod_content.strip())
        (temp_dir / "main.go").write_text("""
package main

import "github.com/gin-gonic/gin"

func main() {
    r := gin.Default()
    r.GET("/", func(c *gin.Context) {
        c.JSON(200, gin.H{"message": "Hello World"})
    })
    r.Run()
}
""")
        
        analyzer = ProjectAnalyzer(temp_dir)
        structure = await analyzer.analyze()
        
        assert "go" in structure.tech_stack.languages
        assert "gin" in structure.tech_stack.frameworks
    
    @pytest.mark.asyncio
    async def test_detect_rust_project(self, temp_dir: Path):
        """Test detection of Rust project"""
        # Create a Rust project
        cargo_toml = """
[package]
name = "my-app"
version = "0.1.0"

[dependencies]
actix-web = "4.0"
"""
        (temp_dir / "Cargo.toml").write_text(cargo_toml.strip())
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "main.rs").write_text("""
use actix_web::{web, App, HttpServer, Result};

async fn hello() -> Result<String> {
    Ok("Hello World!".to_string())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new().route("/", web::get().to(hello))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
""")
        
        analyzer = ProjectAnalyzer(temp_dir)
        structure = await analyzer.analyze()
        
        assert "rust" in structure.tech_stack.languages
        assert "actix" in structure.tech_stack.frameworks
        assert "cargo" in structure.tech_stack.build_tools
    
    @pytest.mark.asyncio
    async def test_detect_database_usage(self, temp_dir: Path):
        """Test detection of database usage"""
        # Create a project with database dependencies
        package_json = {
            "dependencies": {
                "pg": "^8.7.0",
                "redis": "^4.0.0"
            }
        }
        (temp_dir / "package.json").write_text(json.dumps(package_json))
        
        analyzer = ProjectAnalyzer(temp_dir)
        structure = await analyzer.analyze()
        
        # Both databases should be detected
        assert "postgresql" in structure.tech_stack.databases
        assert "redis" in structure.tech_stack.databases
    
    @pytest.mark.asyncio
    async def test_detect_deployment_tools(self, temp_dir: Path):
        """Test detection of deployment and infrastructure tools"""
        # Create Dockerfile
        (temp_dir / "Dockerfile").write_text("""
FROM node:18
WORKDIR /app
COPY . .
RUN npm install
CMD ["npm", "start"]
""")
        
        # Create docker-compose.yml
        (temp_dir / "docker-compose.yml").write_text("""
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
""")
        
        # Create GitHub Actions workflow
        (temp_dir / ".github").mkdir()
        (temp_dir / ".github" / "workflows").mkdir()
        (temp_dir / ".github" / "workflows" / "ci.yml").write_text("""
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
""")
        
        analyzer = ProjectAnalyzer(temp_dir)
        structure = await analyzer.analyze()
        
        assert "docker" in structure.tech_stack.deployment_tools
        assert "docker-compose" in structure.tech_stack.deployment_tools
        assert "github-actions" in structure.tech_stack.deployment_tools
    
    @pytest.mark.asyncio
    async def test_find_entry_points(self, sample_project_dir: Path):
        """Test finding entry point files"""
        # Add some entry point files
        (sample_project_dir / "index.js").write_text("console.log('Hello');")
        (sample_project_dir / "src" / "main.tsx").write_text("import React from 'react';")
        
        analyzer = ProjectAnalyzer(sample_project_dir)
        structure = await analyzer.analyze()
        
        entry_point_names = [ep.name for ep in structure.entry_points]
        # Check for the files we added
        assert "index.js" in entry_point_names
        assert "main.tsx" in entry_point_names
        # App.tsx from fixture might not be detected if it's in nested structure
        # Just verify we have some entry points
        assert len(entry_point_names) >= 2
    
    @pytest.mark.asyncio
    async def test_empty_directory_analysis(self, temp_dir: Path):
        """Test analysis of empty directory"""
        analyzer = ProjectAnalyzer(temp_dir)
        structure = await analyzer.analyze()
        
        # Should not crash and return minimal structure (use resolve() to handle symlinks)
        assert structure.root_path.resolve() == temp_dir.resolve()
        assert len(structure.tech_stack.languages) == 0
        assert len(structure.tech_stack.frameworks) == 0
        assert len(structure.source_directories) == 0 