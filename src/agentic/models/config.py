"""
Configuration models for Agentic
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict

from agentic.models.agent import AgentConfig, AgentType
from agentic.models.project import ProjectStructure


class AgenticConfig(BaseModel):
    """Main configuration for Agentic workspace"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    workspace_name: str = Field(description="Name of the workspace")
    workspace_path: Path = Field(description="Path to the workspace directory")
    
    # Agent configurations
    agents: Dict[str, AgentConfig] = Field(default_factory=dict, description="Configured agents")
    default_agents: List[AgentType] = Field(default_factory=list, description="Default agents to spawn")
    
    # Project settings
    max_concurrent_agents: int = Field(default=3, description="Maximum concurrent agent instances")
    auto_spawn_agents: bool = Field(default=True, description="Whether to auto-spawn agents on analysis")
    
    # Analysis settings
    analysis_depth: str = Field(default="standard", description="Analysis depth: quick, standard, deep")
    include_tests: bool = Field(default=True, description="Include test files in analysis")
    include_docs: bool = Field(default=True, description="Include documentation in analysis")
    
    # Model configuration (unified)
    primary_model: str = Field(default="claude-3-5-sonnet", description="Primary AI model")
    fallback_model: str = Field(default="claude-3-haiku", description="Fallback AI model")
    temperature: float = Field(default=0.1, description="Model temperature")
    max_tokens: int = Field(default=100000, description="Maximum tokens per request")
    
    # File patterns
    ignore_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.pyc", "__pycache__", ".git", "node_modules", 
            "*.log", ".DS_Store", "*.tmp", "*.temp"
        ],
        description="File patterns to ignore during analysis"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    
    def save_to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file"""
        config_dict = self.model_dump(mode='json')
        
        # Convert Path objects to strings for YAML serialization
        config_dict['workspace_path'] = str(config_dict['workspace_path'])
        if config_dict.get('log_file'):
            config_dict['log_file'] = str(config_dict['log_file'])
        
        # Convert agent configs
        for agent_name, agent_config in config_dict.get('agents', {}).items():
            agent_config['workspace_path'] = str(agent_config['workspace_path'])
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_yaml(cls, path: Path) -> 'AgenticConfig':
        """Load configuration from YAML file"""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert string paths back to Path objects
        config_dict['workspace_path'] = Path(config_dict['workspace_path'])
        if config_dict.get('log_file'):
            config_dict['log_file'] = Path(config_dict['log_file'])
        
        # Convert agent configs
        for agent_name, agent_config in config_dict.get('agents', {}).items():
            agent_config['workspace_path'] = Path(agent_config['workspace_path'])
        
        return cls(**config_dict)
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent"""
        return self.agents.get(agent_name)
    
    def add_agent_config(self, agent_name: str, config: AgentConfig) -> None:
        """Add or update agent configuration"""
        self.agents[agent_name] = config
    
    def remove_agent_config(self, agent_name: str) -> bool:
        """Remove agent configuration"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            return True
        return False
    
    @classmethod
    def create_default(cls, workspace_path: Path, workspace_name: Optional[str] = None) -> 'AgenticConfig':
        """Create default configuration for a workspace"""
        if workspace_name is None:
            workspace_name = workspace_path.name
        
        return cls(
            workspace_name=workspace_name,
            workspace_path=workspace_path,
            default_agents=[AgentType.CLAUDE_CODE],
            max_concurrent_agents=3,
            auto_spawn_agents=True,
            analysis_depth="standard",
            include_tests=True,
            include_docs=True,
            log_level="INFO"
        )
    
    @classmethod
    def create_from_project_structure(cls, project_structure: ProjectStructure) -> AgenticConfig:
        """Create configuration from analyzed project structure"""
        project_name = project_structure.root_path.name
        
        # Create default agent configurations based on detected tech stack
        agents = {}
        
        # Always include a reasoning agent for Claude Code
        agents["reasoning"] = AgentConfig(
            agent_type=AgentType.CLAUDE_CODE,
            name="reasoning",
            workspace_path=project_structure.root_path,
            focus_areas=["debugging", "analysis", "explanation"],
            model_config={"model": "claude-3-5-sonnet"},
            temperature=0.1
        )
        
        # Add frontend agent if web frameworks detected
        web_frameworks = {"react", "vue", "angular", "svelte", "nextjs", "nuxt"}
        if any(fw.lower() in web_frameworks for fw in project_structure.tech_stack.frameworks):
            agents["frontend"] = AgentConfig(
                agent_type=AgentType.AIDER_FRONTEND,
                name="frontend",
                workspace_path=project_structure.root_path,
                focus_areas=["components", "ui", "styling", "frontend"],
                model_config={"model": "claude-3-5-sonnet"},
                temperature=0.1
            )
        
        # Add backend agent if backend frameworks/languages detected  
        backend_languages = {"python", "javascript", "typescript", "go", "rust", "java", "c#"}
        backend_frameworks = {"fastapi", "django", "flask", "express", "nestjs", "gin", "actix", "spring"}
        
        has_backend = (
            any(lang.lower() in backend_languages for lang in project_structure.tech_stack.languages) or
            any(fw.lower() in backend_frameworks for fw in project_structure.tech_stack.frameworks)
        )
        
        if has_backend:
            agents["backend"] = AgentConfig(
                agent_type=AgentType.AIDER_BACKEND,
                name="backend",
                workspace_path=project_structure.root_path,
                focus_areas=["api", "database", "server", "backend"],
                model_config={"model": "claude-3-5-sonnet"},
                temperature=0.1
            )
        
        # Add testing agent if test frameworks detected
        if project_structure.tech_stack.testing_frameworks or project_structure.test_directories:
            agents["testing"] = AgentConfig(
                agent_type=AgentType.AIDER_TESTING,
                name="testing",
                workspace_path=project_structure.root_path,
                focus_areas=["tests", "testing", "qa"],
                model_config={"model": "claude-3-5-sonnet"},
                temperature=0.1
            )
        
        # Default exclude patterns
        exclude_patterns = [
            ".git/**",
            "node_modules/**", 
            "__pycache__/**",
            "*.pyc",
            ".env",
            ".env.*",
            "dist/**",
            "build/**",
            "coverage/**",
            ".pytest_cache/**",
            ".mypy_cache/**",
            "*.log"
        ]
        
        # Default include patterns based on detected languages
        include_patterns = []
        for lang in project_structure.tech_stack.languages:
            if lang.lower() == "python":
                include_patterns.extend(["*.py", "*.pyi"])
            elif lang.lower() in ["javascript", "typescript"]:
                include_patterns.extend(["*.js", "*.ts", "*.jsx", "*.tsx"])
            elif lang.lower() == "go":
                include_patterns.append("*.go")
            elif lang.lower() == "rust":
                include_patterns.append("*.rs")
        
        # If no specific patterns, include common code files
        if not include_patterns:
            include_patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.go", "*.rs", "*.java", "*.c", "*.cpp", "*.h"]
        
        return cls(
            workspace_name=project_name,
            workspace_path=project_structure.root_path,
            agents=agents,
            default_agents=[AgentType.CLAUDE_CODE],
            max_concurrent_agents=3,
            auto_spawn_agents=True,
            analysis_depth="standard",
            include_tests=True,
            include_docs=True,
            log_level="INFO"
        )
    
    @classmethod
    def load_or_create(cls, workspace_path: Path) -> 'AgenticConfig':
        """Load configuration from file or create default if not found"""
        # Try to find existing configuration file
        config_files = [
            workspace_path / '.agentic' / 'config.yml',
            workspace_path / '.agentic' / 'config.yaml',
            workspace_path / 'agentic.yml',
            workspace_path / 'agentic.yaml'
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    return cls.load_from_yaml(config_file)
                except Exception as e:
                    # If loading fails, continue to try other files or create default
                    continue
        
        # No valid configuration found, create default
        return cls.create_default(workspace_path) 