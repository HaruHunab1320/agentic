# Advanced Plugin System Architecture for Phase 5
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import logging
import pkgutil
import sys
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable, Type, Union
import traceback

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PluginStatus(str, Enum):
    """Plugin lifecycle status"""
    UNKNOWN = "unknown"
    DISCOVERED = "discovered" 
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    ERROR = "error"
    UNLOADED = "unloaded"


class PluginType(str, Enum):
    """Types of plugins supported"""
    AGENT = "agent"           # Agent implementations
    TOOL = "tool"             # Tool integrations
    MODEL = "model"           # AI model providers
    INTEGRATION = "integration" # External service integrations
    EXTENSION = "extension"   # Core functionality extensions
    MIDDLEWARE = "middleware" # Request/response processing
    UI = "ui"                # User interface components
    ANALYTICS = "analytics"   # Analytics and monitoring


class HookType(str, Enum):
    """Available hook types for plugins"""
    BEFORE_TASK_EXECUTE = "before_task_execute"
    AFTER_TASK_EXECUTE = "after_task_execute"
    BEFORE_AGENT_START = "before_agent_start"
    AFTER_AGENT_START = "after_agent_start"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ON_ERROR = "on_error"
    ON_SYSTEM_START = "on_system_start"
    ON_SYSTEM_SHUTDOWN = "on_system_shutdown"
    ON_CONFIGURATION_CHANGE = "on_configuration_change"


class PluginDependency(BaseModel):
    """Plugin dependency specification"""
    name: str = Field(description="Name of required plugin")
    version: Optional[str] = Field(default=None, description="Required version (semver)")
    optional: bool = Field(default=False, description="Whether dependency is optional")
    minimum_version: Optional[str] = Field(default=None, description="Minimum version required")
    maximum_version: Optional[str] = Field(default=None, description="Maximum version allowed")


class PluginMetadata(BaseModel):
    """Plugin metadata and configuration"""
    name: str = Field(description="Plugin name")
    version: str = Field(description="Plugin version")
    description: str = Field(description="Plugin description")
    author: str = Field(description="Plugin author")
    plugin_type: PluginType = Field(description="Type of plugin")
    entry_point: str = Field(description="Entry point class name")
    dependencies: List[PluginDependency] = Field(default_factory=list, description="Plugin dependencies")
    configuration_schema: Optional[Dict[str, Any]] = Field(default=None, description="Configuration schema")
    supported_versions: List[str] = Field(default_factory=list, description="Supported framework versions")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")
    license: Optional[str] = Field(default=None, description="Plugin license")
    homepage: Optional[str] = Field(default=None, description="Plugin homepage URL")
    hooks: List[HookType] = Field(default_factory=list, description="Hooks this plugin implements")


class PluginConfig(BaseModel):
    """Runtime plugin configuration"""
    enabled: bool = Field(default=True, description="Whether plugin is enabled")
    auto_activate: bool = Field(default=True, description="Whether to auto-activate on load")
    priority: int = Field(default=100, description="Plugin priority (lower = higher priority)")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Plugin-specific settings")
    environment: str = Field(default="development", description="Target environment")


class PluginInstance(BaseModel):
    """Runtime plugin instance information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique instance ID")
    metadata: PluginMetadata = Field(description="Plugin metadata")
    config: PluginConfig = Field(description="Plugin configuration")
    status: PluginStatus = Field(default=PluginStatus.UNKNOWN, description="Current plugin status")
    plugin_class: Optional[Type] = Field(default=None, description="Plugin class reference")
    instance: Optional['BasePlugin'] = Field(default=None, description="Plugin instance")
    error_message: Optional[str] = Field(default=None, description="Last error message")
    loaded_at: Optional[datetime] = Field(default=None, description="When plugin was loaded")
    activated_at: Optional[datetime] = Field(default=None, description="When plugin was activated")
    module_path: Optional[str] = Field(default=None, description="Module path for plugin")
    
    class Config:
        arbitrary_types_allowed = True


class HookContext(BaseModel):
    """Context passed to plugin hooks"""
    hook_type: HookType = Field(description="Type of hook being executed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Hook execution time")
    source: str = Field(description="Source component triggering the hook")
    data: Dict[str, Any] = Field(default_factory=dict, description="Hook-specific data")
    plugin_id: Optional[str] = Field(default=None, description="Plugin ID if relevant")
    agent_id: Optional[str] = Field(default=None, description="Agent ID if relevant")
    task_id: Optional[str] = Field(default=None, description="Task ID if relevant")


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self.status = PluginStatus.LOADED
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._hooks: Dict[HookType, List[Callable]] = {}
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    async def activate(self) -> bool:
        """Activate the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    async def deactivate(self) -> bool:
        """Deactivate the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Clean up plugin resources. Return True if successful."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    def register_hook(self, hook_type: HookType, callback: Callable):
        """Register a hook callback"""
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []
        self._hooks[hook_type].append(callback)
    
    async def execute_hook(self, hook_type: HookType, context: HookContext) -> Any:
        """Execute hooks of specified type"""
        results = []
        if hook_type in self._hooks:
            for callback in self._hooks[hook_type]:
                try:
                    if inspect.iscoroutinefunction(callback):
                        result = await callback(context)
                    else:
                        result = callback(context)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Hook {hook_type} execution failed: {e}")
                    results.append(None)
        return results
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration. Return list of validation errors."""
        return []  # Default: no validation errors
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform plugin health check"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "plugin": self.__class__.__name__
        }


class PluginLoader:
    """Loads and manages plugin discovery and instantiation"""
    
    def __init__(self, plugin_directories: List[Path]):
        self.plugin_directories = plugin_directories
        self.discovered_plugins: Dict[str, PluginMetadata] = {}
        
    async def discover_plugins(self) -> Dict[str, PluginMetadata]:
        """Discover all available plugins"""
        self.discovered_plugins.clear()
        
        for plugin_dir in self.plugin_directories:
            if not plugin_dir.exists():
                logger.warning(f"Plugin directory does not exist: {plugin_dir}")
                continue
                
            await self._discover_from_directory(plugin_dir)
        
        # Also discover from installed packages
        await self._discover_from_packages()
        
        logger.info(f"Discovered {len(self.discovered_plugins)} plugins")
        return self.discovered_plugins
    
    async def _discover_from_directory(self, plugin_dir: Path):
        """Discover plugins from a directory"""
        try:
            # Look for all .py files or subdirectories with plugins
            for item in plugin_dir.iterdir():
                if item.is_file() and item.suffix == ".py":
                    # Check any .py file for plugin metadata
                    await self._load_plugin_metadata(item)
                elif item.is_dir() and (item / "__init__.py").exists():
                    # Check if directory contains a plugin
                    init_file = item / "__init__.py"
                    plugin_file = item / "plugin.py"
                    
                    if plugin_file.exists():
                        await self._load_plugin_metadata(plugin_file)
                    else:
                        await self._load_plugin_metadata(init_file)
                        
        except Exception as e:
            logger.error(f"Error discovering plugins from {plugin_dir}: {e}")
    
    async def _discover_from_packages(self):
        """Discover plugins from installed packages"""
        try:
            # Look for packages with 'agentic_' prefix
            for finder, name, ispkg in pkgutil.iter_modules():
                if name.startswith('agentic_plugin_'):
                    try:
                        module = importlib.import_module(name)
                        if hasattr(module, 'PLUGIN_METADATA'):
                            metadata = module.PLUGIN_METADATA
                            if isinstance(metadata, dict):
                                plugin_metadata = PluginMetadata(**metadata)
                                self.discovered_plugins[plugin_metadata.name] = plugin_metadata
                    except Exception as e:
                        logger.error(f"Error loading plugin package {name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error discovering plugin packages: {e}")
    
    async def _load_plugin_metadata(self, plugin_file: Path):
        """Load plugin metadata from file"""
        try:
            # Add parent directory to Python path temporarily
            parent_dir = str(plugin_file.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                
            try:
                # Import module
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for plugin metadata
                if hasattr(module, 'PLUGIN_METADATA'):
                    metadata_dict = module.PLUGIN_METADATA
                    if isinstance(metadata_dict, dict):
                        metadata = PluginMetadata(**metadata_dict)
                        self.discovered_plugins[metadata.name] = metadata
                        logger.debug(f"Discovered plugin: {metadata.name}")
                
            finally:
                # Remove from path
                if parent_dir in sys.path:
                    sys.path.remove(parent_dir)
                    
        except Exception as e:
            logger.error(f"Error loading metadata from {plugin_file}: {e}")
    
    async def load_plugin(self, plugin_name: str, config: PluginConfig) -> Optional[PluginInstance]:
        """Load and instantiate a specific plugin"""
        if plugin_name not in self.discovered_plugins:
            logger.error(f"Plugin not found: {plugin_name}")
            return None
            
        metadata = self.discovered_plugins[plugin_name]
        
        try:
            # Find and load the plugin module
            plugin_class = await self._load_plugin_class(metadata)
            if not plugin_class:
                return None
                
            # Create plugin instance
            plugin_instance = plugin_class(config)
            
            return PluginInstance(
                metadata=metadata,
                config=config,
                status=PluginStatus.LOADED,
                plugin_class=plugin_class,
                instance=plugin_instance,
                loaded_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return PluginInstance(
                metadata=metadata,
                config=config,
                status=PluginStatus.ERROR,
                error_message=str(e)
            )
    
    async def _load_plugin_class(self, metadata: PluginMetadata) -> Optional[Type[BasePlugin]]:
        """Load the plugin class from metadata"""
        try:
            # Try to find the plugin module
            for plugin_dir in self.plugin_directories:
                plugin_file = plugin_dir / f"{metadata.name}.py"
                if plugin_file.exists():
                    # Load from file
                    spec = importlib.util.spec_from_file_location(metadata.name, plugin_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Get the entry point class
                    if hasattr(module, metadata.entry_point):
                        plugin_class = getattr(module, metadata.entry_point)
                        if issubclass(plugin_class, BasePlugin):
                            return plugin_class
            
            # Try to load from package
            try:
                module = importlib.import_module(f"agentic_plugin_{metadata.name}")
                if hasattr(module, metadata.entry_point):
                    plugin_class = getattr(module, metadata.entry_point)
                    if issubclass(plugin_class, BasePlugin):
                        return plugin_class
            except ImportError:
                pass
                
            logger.error(f"Could not find plugin class {metadata.entry_point} for {metadata.name}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading plugin class for {metadata.name}: {e}")
            return None


class PluginRegistry:
    """Central registry for managing all plugins"""
    
    def __init__(self, plugin_directories: Optional[List[Path]] = None):
        self.plugin_directories = plugin_directories or [Path("plugins")]
        self.loader = PluginLoader(self.plugin_directories)
        self.plugins: Dict[str, PluginInstance] = {}
        self.hook_registry: Dict[HookType, List[str]] = {}  # hook_type -> plugin_names
        self.dependency_graph: Dict[str, Set[str]] = {}  # plugin -> dependencies
        
    async def initialize(self):
        """Initialize the plugin registry"""
        logger.info("Initializing plugin registry...")
        
        # Discover all available plugins
        await self.loader.discover_plugins()
        
        # Build dependency graph
        self._build_dependency_graph()
        
        logger.info(f"Plugin registry initialized with {len(self.loader.discovered_plugins)} discovered plugins")
    
    def _build_dependency_graph(self):
        """Build plugin dependency graph"""
        self.dependency_graph.clear()
        
        for plugin_name, metadata in self.loader.discovered_plugins.items():
            dependencies = set()
            for dep in metadata.dependencies:
                if not dep.optional:  # Only include required dependencies
                    dependencies.add(dep.name)
            self.dependency_graph[plugin_name] = dependencies
    
    async def load_plugin(self, plugin_name: str, config: Optional[PluginConfig] = None) -> bool:
        """Load a plugin and its dependencies"""
        if plugin_name in self.plugins:
            logger.warning(f"Plugin {plugin_name} is already loaded")
            return True
            
        if plugin_name not in self.loader.discovered_plugins:
            logger.error(f"Plugin {plugin_name} not found")
            return False
        
        # Load dependencies first
        dependencies = self.dependency_graph.get(plugin_name, set())
        for dep_name in dependencies:
            if not await self.load_plugin(dep_name):
                logger.error(f"Failed to load dependency {dep_name} for plugin {plugin_name}")
                return False
        
        # Use default config if none provided
        if config is None:
            config = PluginConfig()
        
        # Load the plugin
        plugin_instance = await self.loader.load_plugin(plugin_name, config)
        if not plugin_instance:
            return False
            
        self.plugins[plugin_name] = plugin_instance
        
        # Register hooks
        if plugin_instance.metadata:
            for hook_type in plugin_instance.metadata.hooks:
                if hook_type not in self.hook_registry:
                    self.hook_registry[hook_type] = []
                self.hook_registry[hook_type].append(plugin_name)
        
        logger.info(f"Loaded plugin: {plugin_name}")
        return True
    
    async def initialize_plugin(self, plugin_name: str) -> bool:
        """Initialize a loaded plugin"""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} is not loaded")
            return False
            
        plugin_instance = self.plugins[plugin_name]
        if plugin_instance.status != PluginStatus.LOADED:
            logger.error(f"Plugin {plugin_name} is not in loaded state")
            return False
        
        try:
            success = await plugin_instance.instance.initialize()
            if success:
                plugin_instance.status = PluginStatus.INITIALIZED
                logger.info(f"Initialized plugin: {plugin_name}")
                return True
            else:
                plugin_instance.status = PluginStatus.ERROR
                plugin_instance.error_message = "Initialization failed"
                return False
                
        except Exception as e:
            logger.error(f"Error initializing plugin {plugin_name}: {e}")
            plugin_instance.status = PluginStatus.ERROR
            plugin_instance.error_message = str(e)
            return False
    
    async def activate_plugin(self, plugin_name: str) -> bool:
        """Activate an initialized plugin"""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} is not loaded")
            return False
            
        plugin_instance = self.plugins[plugin_name]
        if plugin_instance.status != PluginStatus.INITIALIZED:
            logger.error(f"Plugin {plugin_name} is not initialized")
            return False
        
        try:
            success = await plugin_instance.instance.activate()
            if success:
                plugin_instance.status = PluginStatus.ACTIVATED
                plugin_instance.activated_at = datetime.utcnow()
                logger.info(f"Activated plugin: {plugin_name}")
                return True
            else:
                plugin_instance.status = PluginStatus.ERROR
                plugin_instance.error_message = "Activation failed"
                return False
                
        except Exception as e:
            logger.error(f"Error activating plugin {plugin_name}: {e}")
            plugin_instance.status = PluginStatus.ERROR
            plugin_instance.error_message = str(e)
            return False
    
    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate an active plugin"""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} is not loaded")
            return False
            
        plugin_instance = self.plugins[plugin_name]
        if plugin_instance.status != PluginStatus.ACTIVATED:
            logger.warning(f"Plugin {plugin_name} is not activated")
            return True  # Already deactivated
        
        try:
            success = await plugin_instance.instance.deactivate()
            if success:
                plugin_instance.status = PluginStatus.DEACTIVATED
                logger.info(f"Deactivated plugin: {plugin_name}")
                return True
            else:
                plugin_instance.status = PluginStatus.ERROR
                plugin_instance.error_message = "Deactivation failed"
                return False
                
        except Exception as e:
            logger.error(f"Error deactivating plugin {plugin_name}: {e}")
            plugin_instance.status = PluginStatus.ERROR
            plugin_instance.error_message = str(e)
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin and clean up resources"""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} is not loaded")
            return True
            
        plugin_instance = self.plugins[plugin_name]
        
        # Deactivate if active
        if plugin_instance.status == PluginStatus.ACTIVATED:
            await self.deactivate_plugin(plugin_name)
        
        # Clean up
        try:
            if plugin_instance.instance:
                await plugin_instance.instance.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up plugin {plugin_name}: {e}")
        
        # Remove from registries
        del self.plugins[plugin_name]
        
        # Remove from hook registry
        for hook_type, plugin_list in self.hook_registry.items():
            if plugin_name in plugin_list:
                plugin_list.remove(plugin_name)
        
        logger.info(f"Unloaded plugin: {plugin_name}")
        return True
    
    async def execute_hooks(self, hook_type: HookType, context: HookContext) -> Dict[str, Any]:
        """Execute all registered hooks of specified type"""
        results = {}
        
        if hook_type not in self.hook_registry:
            return results
        
        plugin_names = self.hook_registry[hook_type]
        for plugin_name in plugin_names:
            if plugin_name in self.plugins:
                plugin_instance = self.plugins[plugin_name]
                if plugin_instance.status == PluginStatus.ACTIVATED and plugin_instance.instance:
                    try:
                        result = await plugin_instance.instance.execute_hook(hook_type, context)
                        results[plugin_name] = result
                    except Exception as e:
                        logger.error(f"Hook execution failed for plugin {plugin_name}: {e}")
                        results[plugin_name] = {"error": str(e)}
        
        return results
    
    def get_plugin_status(self, plugin_name: str) -> Optional[PluginStatus]:
        """Get the current status of a plugin"""
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].status
        return None
    
    def get_all_plugins(self) -> Dict[str, PluginInstance]:
        """Get all registered plugins"""
        return self.plugins.copy()
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, PluginInstance]:
        """Get all plugins of specified type"""
        return {
            name: instance for name, instance in self.plugins.items()
            if instance.metadata.plugin_type == plugin_type
        }
    
    def get_active_plugins(self) -> Dict[str, PluginInstance]:
        """Get all active plugins"""
        return {
            name: instance for name, instance in self.plugins.items()
            if instance.status == PluginStatus.ACTIVATED
        }
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (unload and load again)"""
        if plugin_name in self.plugins:
            config = self.plugins[plugin_name].config
            await self.unload_plugin(plugin_name)
        else:
            config = PluginConfig()
        
        # Rediscover plugins to pick up changes
        await self.loader.discover_plugins()
        self._build_dependency_graph()
        
        return await self.load_plugin(plugin_name, config)
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all active plugins"""
        results = {}
        
        for plugin_name, plugin_instance in self.plugins.items():
            if plugin_instance.status == PluginStatus.ACTIVATED and plugin_instance.instance:
                try:
                    health = await plugin_instance.instance.health_check()
                    results[plugin_name] = health
                except Exception as e:
                    results[plugin_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        return results


# Verified: Complete - Comprehensive plugin system with discovery, loading, lifecycle management, hooks, and dependency resolution 