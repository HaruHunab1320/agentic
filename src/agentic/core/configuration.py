"""
Phase 4: Configuration System
Comprehensive project and user configuration with inheritance
"""

import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
import os
from datetime import datetime


class AgentBehaviorConfig(BaseModel):
    """Configuration for agent behavior"""
    model_config = ConfigDict(extra='forbid')
    
    thinking_mode: str = "adaptive"  # adaptive, always, never
    max_thinking_time: int = 300  # seconds
    retry_attempts: int = 3
    timeout: int = 600  # seconds
    parallel_tasks: int = 2
    cost_limit_per_hour: float = 10.0  # USD
    auto_save_interval: int = 30  # seconds


class ModelConfig(BaseModel):
    """AI model configuration"""
    model_config = ConfigDict(extra='forbid')
    
    primary_model: str = "claude-3-5-sonnet"
    fallback_model: str = "claude-3-haiku" 
    temperature: float = 0.1
    max_tokens: int = 100000
    thinking_tokens: int = 32000
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds


class IntegrationConfig(BaseModel):
    """External tool integration configuration"""
    model_config = ConfigDict(extra='forbid')
    
    # Git integration
    git_auto_commit: bool = True
    git_commit_template: str = "feat: {summary}"
    git_branch_strategy: str = "feature_branch"  # main, feature_branch
    git_push_on_complete: bool = False
    
    # IDE integration
    ide_integration: bool = True
    vscode_extension: bool = False
    jetbrains_plugin: bool = False
    
    # CI/CD integration
    ci_cd_integration: bool = False
    github_actions: bool = False
    gitlab_ci: bool = False
    
    # Notifications
    notifications: Dict[str, Any] = Field(default_factory=lambda: {
        "email": {"enabled": False, "recipients": []},
        "slack": {"enabled": False, "webhook_url": "", "channel": ""},
        "discord": {"enabled": False, "webhook_url": ""}
    })


class PerformanceConfig(BaseModel):
    """Performance and resource configuration"""
    model_config = ConfigDict(extra='forbid')
    
    # Resource limits
    max_concurrent_agents: int = 5
    memory_limit_mb: int = 2048
    cache_size_mb: int = 512
    log_retention_days: int = 30
    
    # Timeouts
    analysis_timeout: int = 300  # seconds
    coordination_timeout: int = 180  # seconds
    api_timeout: int = 120  # seconds
    
    # Optimization
    auto_optimization: bool = True
    profiling_enabled: bool = False
    metrics_collection: bool = True
    performance_alerts: bool = True


class SecurityConfig(BaseModel):
    """Security and access configuration"""
    model_config = ConfigDict(extra='forbid')
    
    # API security
    api_key_rotation_days: int = 90
    require_api_key_env: bool = True
    allowed_domains: List[str] = Field(default_factory=list)
    
    # File access
    restrict_file_access: bool = True
    allowed_directories: List[str] = Field(default_factory=list)
    blocked_extensions: List[str] = Field(default_factory=lambda: [
        ".env", ".key", ".pem", ".p12", ".jks"
    ])
    
    # Execution security
    allow_shell_execution: bool = False
    sandbox_mode: bool = True


class AgenticConfig(BaseModel):
    """Complete Agentic configuration with validation"""
    model_config = ConfigDict(extra='forbid')
    
    # Meta information
    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Core settings
    project_name: str
    project_root: Path
    workspace_path: Optional[Path] = None
    
    # Component configurations
    agents: Dict[str, AgentBehaviorConfig] = Field(default_factory=dict)
    models: ModelConfig = Field(default_factory=ModelConfig)
    integrations: IntegrationConfig = Field(default_factory=IntegrationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Feature flags
    features: Dict[str, bool] = Field(default_factory=lambda: {
        "ml_routing": True,
        "dependency_analysis": True,
        "pattern_learning": True,
        "shared_memory": True,
        "conflict_detection": True,
        "performance_monitoring": True,
        "interactive_cli": True,
        "cost_tracking": True
    })
    
    # Environment-specific overrides
    environment: str = "development"  # development, staging, production
    debug: bool = False

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation and setup"""
        super().model_post_init(__context)
        
        # Set workspace path if not provided
        if not self.workspace_path:
            self.workspace_path = self.project_root
        
        # Update timestamp
        self.updated_at = datetime.utcnow()
        
        # Validate environment-specific settings
        if self.environment == "production":
            # Production safety checks
            if self.debug:
                self.debug = False
            if self.security.allow_shell_execution:
                self.security.allow_shell_execution = False
            if not self.security.sandbox_mode:
                self.security.sandbox_mode = True


class ConfigurationManager:
    """Manages configuration with inheritance and validation"""
    
    def __init__(self):
        self.global_config_path = Path.home() / '.agentic' / 'global.yml'
        self.user_config_cache: Dict[str, Any] = {}
        self._ensure_config_directories()
    
    def _ensure_config_directories(self):
        """Ensure configuration directories exist"""
        self.global_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_configuration(self, project_path: Path) -> AgenticConfig:
        """Load configuration with inheritance hierarchy"""
        # Start with defaults
        config_data = {
            "project_name": project_path.name,
            "project_root": project_path
        }
        
        # Apply global user configuration
        global_config = self._load_global_config()
        if global_config:
            config_data = self._merge_configs(config_data, global_config)
        
        # Apply project configuration
        project_config = self._load_project_config(project_path)
        if project_config:
            config_data = self._merge_configs(config_data, project_config)
        
        # Apply environment overrides
        env_config = self._load_environment_config(project_path)
        if env_config:
            config_data = self._merge_configs(config_data, env_config)
        
        # Apply environment variable overrides
        env_overrides = self._load_environment_variables()
        if env_overrides:
            config_data = self._merge_configs(config_data, env_overrides)
        
        return AgenticConfig(**config_data)
    
    def _load_global_config(self) -> Optional[Dict[str, Any]]:
        """Load global user configuration"""
        if not self.global_config_path.exists():
            return None
        
        try:
            with open(self.global_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # TODO: Log error
            return None
    
    def _load_project_config(self, project_path: Path) -> Optional[Dict[str, Any]]:
        """Load project-specific configuration"""
        config_files = [
            project_path / '.agentic' / 'config.yml',
            project_path / '.agentic' / 'config.yaml',
            project_path / 'agentic.yml',
            project_path / 'agentic.yaml'
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    # TODO: Log error
                    continue
        
        return None
    
    def _load_environment_config(self, project_path: Path) -> Optional[Dict[str, Any]]:
        """Load environment-specific configuration"""
        environment = os.getenv('AGENTIC_ENV', 'development')
        
        config_files = [
            project_path / '.agentic' / f'config.{environment}.yml',
            project_path / '.agentic' / f'config.{environment}.yaml',
            project_path / f'agentic.{environment}.yml',
            project_path / f'agentic.{environment}.yaml'
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        if config:
                            config['environment'] = environment
                        return config
                except Exception as e:
                    # TODO: Log error
                    continue
        
        return {"environment": environment}
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        overrides = {}
        
        # Model configuration
        if os.getenv('AGENTIC_PRIMARY_MODEL'):
            overrides.setdefault('models', {})['primary_model'] = os.getenv('AGENTIC_PRIMARY_MODEL')
        
        if os.getenv('AGENTIC_TEMPERATURE'):
            try:
                overrides.setdefault('models', {})['temperature'] = float(os.getenv('AGENTIC_TEMPERATURE'))
            except ValueError:
                pass
        
        # Performance configuration
        if os.getenv('AGENTIC_MAX_CONCURRENT_AGENTS'):
            try:
                overrides.setdefault('performance', {})['max_concurrent_agents'] = int(os.getenv('AGENTIC_MAX_CONCURRENT_AGENTS'))
            except ValueError:
                pass
        
        # Debug mode
        if os.getenv('AGENTIC_DEBUG'):
            overrides['debug'] = os.getenv('AGENTIC_DEBUG').lower() in ('true', '1', 'yes', 'on')
        
        return overrides
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_project_config(self, project_path: Path, config: AgenticConfig, 
                          include_defaults: bool = False):
        """Save project-specific configuration"""
        config_path = project_path / '.agentic' / 'config.yml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get configuration data
        if include_defaults:
            config_dict = config.model_dump()
        else:
            config_dict = self._get_non_default_values(config)
        
        # Remove computed fields
        for field in ['created_at', 'updated_at', 'project_root', 'workspace_path']:
            config_dict.pop(field, None)
        
        # Convert Path objects to strings
        config_dict = self._serialize_paths(config_dict)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def save_global_config(self, config_data: Dict[str, Any]):
        """Save global user configuration"""
        self._ensure_config_directories()
        
        # Load existing global config
        existing_config = self._load_global_config() or {}
        
        # Merge with new data
        merged_config = self._merge_configs(existing_config, config_data)
        
        with open(self.global_config_path, 'w') as f:
            yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    
    def update_agent_config(self, project_path: Path, agent_name: str, 
                          updates: Dict[str, Any]):
        """Update specific agent configuration"""
        config = self.load_configuration(project_path)
        
        if agent_name not in config.agents:
            config.agents[agent_name] = AgentBehaviorConfig()
        
        # Apply updates to agent config
        agent_updates = {}
        for key, value in updates.items():
            if hasattr(AgentBehaviorConfig, key):
                agent_updates[key] = value
        
        if agent_updates:
            updated_agent_config = config.agents[agent_name].model_copy(update=agent_updates)
            config.agents[agent_name] = updated_agent_config
            self.save_project_config(project_path, config)
    
    def get_agent_config(self, project_path: Path, agent_name: str) -> AgentBehaviorConfig:
        """Get configuration for specific agent"""
        config = self.load_configuration(project_path)
        return config.agents.get(agent_name, AgentBehaviorConfig())
    
    def validate_configuration(self, config_path: Path) -> List[str]:
        """Validate configuration file and return errors"""
        errors = []
        
        try:
            if not config_path.exists():
                return ["Configuration file does not exist"]
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return ["Configuration file is empty"]
            
            # Try to create AgenticConfig instance for validation
            project_root = config_path.parent.parent if '.agentic' in str(config_path) else config_path.parent
            config_data['project_root'] = project_root
            config_data['project_name'] = config_data.get('project_name', project_root.name)
            
            AgenticConfig(**config_data)
            
        except yaml.YAMLError as e:
            errors.append(f"YAML parsing error: {e}")
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
        
        return errors
    
    def _get_non_default_values(self, config: AgenticConfig) -> Dict[str, Any]:
        """Get only non-default configuration values"""
        defaults = AgenticConfig(project_name="default", project_root=Path("."))
        config_dict = config.model_dump()
        default_dict = defaults.model_dump()
        
        return self._diff_configs(config_dict, default_dict)
    
    def _diff_configs(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Get differences between two configuration dictionaries"""
        diff = {}
        
        for key, value in config.items():
            if key not in defaults:
                diff[key] = value
            elif isinstance(value, dict) and isinstance(defaults[key], dict):
                nested_diff = self._diff_configs(value, defaults[key])
                if nested_diff:
                    diff[key] = nested_diff
            elif value != defaults[key]:
                diff[key] = value
        
        return diff
    
    def _serialize_paths(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Path objects to strings in configuration dictionary"""
        result = {}
        
        for key, value in config_dict.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = self._serialize_paths(value)
            elif isinstance(value, list):
                result[key] = [str(item) if isinstance(item, Path) else item for item in value]
            else:
                result[key] = value
        
        return result


# Configuration validation helpers
def validate_model_config(model_config: ModelConfig) -> List[str]:
    """Validate model configuration"""
    errors = []
    
    if model_config.temperature < 0 or model_config.temperature > 2:
        errors.append("Temperature must be between 0 and 2")
    
    if model_config.max_tokens <= 0:
        errors.append("Max tokens must be positive")
    
    if model_config.thinking_tokens < 0:
        errors.append("Thinking tokens cannot be negative")
    
    return errors


def validate_performance_config(perf_config: PerformanceConfig) -> List[str]:
    """Validate performance configuration"""
    errors = []
    
    if perf_config.max_concurrent_agents <= 0:
        errors.append("Max concurrent agents must be positive")
    
    if perf_config.memory_limit_mb <= 0:
        errors.append("Memory limit must be positive")
    
    if perf_config.analysis_timeout <= 0:
        errors.append("Analysis timeout must be positive")
    
    return errors


# Global configuration manager instance
config_manager = ConfigurationManager() 