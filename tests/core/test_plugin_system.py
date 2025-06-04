# Comprehensive tests for Phase 5 Plugin System Architecture
import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import importlib.util

from src.agentic.core.plugin_system import (
    BasePlugin, PluginLoader, PluginRegistry, PluginMetadata,
    PluginConfig, PluginInstance, PluginStatus, PluginType,
    HookType, HookContext, PluginDependency
)


class MockPlugin(BasePlugin):
    """Mock plugin for testing"""
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.initialized = False
        self.activated = False
        self.cleanup_called = False
        
    async def initialize(self) -> bool:
        self.initialized = True
        self.status = PluginStatus.INITIALIZED
        return True
        
    async def activate(self) -> bool:
        self.activated = True
        self.status = PluginStatus.ACTIVATED
        return True
        
    async def deactivate(self) -> bool:
        self.activated = False
        self.status = PluginStatus.DEACTIVATED
        return True
        
    async def cleanup(self) -> bool:
        self.cleanup_called = True
        return True
        
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="mock_plugin",
            version="1.0.0",
            description="Mock plugin for testing",
            author="Test Author",
            plugin_type=PluginType.TOOL,
            entry_point="MockPlugin",
            hooks=[HookType.BEFORE_TASK_EXECUTE, HookType.AFTER_TASK_EXECUTE]
        )


class FailingPlugin(BasePlugin):
    """Plugin that fails during lifecycle operations"""
    
    async def initialize(self) -> bool:
        raise Exception("Initialization failed")
        
    async def activate(self) -> bool:
        raise Exception("Activation failed")
        
    async def deactivate(self) -> bool:
        raise Exception("Deactivation failed")
        
    async def cleanup(self) -> bool:
        raise Exception("Cleanup failed")
        
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="failing_plugin",
            version="1.0.0", 
            description="Plugin that fails",
            author="Test Author",
            plugin_type=PluginType.TOOL,
            entry_point="FailingPlugin"
        )


class DependentPlugin(BasePlugin):
    """Plugin with dependencies"""
    
    async def initialize(self) -> bool:
        self.status = PluginStatus.INITIALIZED
        return True
        
    async def activate(self) -> bool:
        self.status = PluginStatus.ACTIVATED
        return True
        
    async def deactivate(self) -> bool:
        self.status = PluginStatus.DEACTIVATED
        return True
        
    async def cleanup(self) -> bool:
        return True
        
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="dependent_plugin",
            version="1.0.0",
            description="Plugin with dependencies",
            author="Test Author",
            plugin_type=PluginType.EXTENSION,
            entry_point="DependentPlugin",
            dependencies=[
                PluginDependency(name="mock_plugin", version="1.0.0"),
                PluginDependency(name="optional_plugin", optional=True)
            ]
        )


@pytest.fixture
def temp_plugin_dir():
    """Create a temporary directory for plugin testing"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_plugin_metadata():
    """Create mock plugin metadata"""
    return PluginMetadata(
        name="test_plugin",
        version="1.0.0",
        description="Test plugin",
        author="Test Author", 
        plugin_type=PluginType.TOOL,
        entry_point="TestPlugin",
        hooks=[HookType.BEFORE_TASK_EXECUTE]
    )


@pytest.fixture
def plugin_config():
    """Create plugin configuration"""
    return PluginConfig(
        enabled=True,
        auto_activate=True,
        priority=100,
        settings={"test_setting": "test_value"}
    )


class TestPluginMetadata:
    """Test plugin metadata handling"""
    
    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata"""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            plugin_type=PluginType.AGENT,
            entry_point="TestPlugin"
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.AGENT
        assert metadata.entry_point == "TestPlugin"
        assert metadata.dependencies == []
        assert metadata.hooks == []
    
    def test_plugin_dependencies(self):
        """Test plugin dependency specification"""
        dependency = PluginDependency(
            name="required_plugin",
            version="2.0.0",
            optional=False,
            minimum_version="1.0.0"
        )
        
        assert dependency.name == "required_plugin"
        assert dependency.version == "2.0.0"
        assert not dependency.optional
        assert dependency.minimum_version == "1.0.0"


class TestBasePlugin:
    """Test base plugin functionality"""
    
    def test_plugin_creation(self, plugin_config):
        """Test creating a plugin instance"""
        plugin = MockPlugin(plugin_config)
        
        assert plugin.config == plugin_config
        assert plugin.status == PluginStatus.LOADED
        assert not plugin.initialized
        assert not plugin.activated
    
    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self, plugin_config):
        """Test plugin lifecycle methods"""
        plugin = MockPlugin(plugin_config)
        
        # Initialize
        success = await plugin.initialize()
        assert success
        assert plugin.initialized
        assert plugin.status == PluginStatus.INITIALIZED
        
        # Activate
        success = await plugin.activate()
        assert success
        assert plugin.activated
        assert plugin.status == PluginStatus.ACTIVATED
        
        # Deactivate
        success = await plugin.deactivate()
        assert success
        assert not plugin.activated
        assert plugin.status == PluginStatus.DEACTIVATED
        
        # Cleanup
        success = await plugin.cleanup()
        assert success
        assert plugin.cleanup_called
    
    def test_hook_registration(self, plugin_config):
        """Test hook registration and execution"""
        plugin = MockPlugin(plugin_config)
        
        # Register a hook
        hook_called = False
        
        def test_hook(context: HookContext):
            nonlocal hook_called
            hook_called = True
            return "hook_result"
        
        plugin.register_hook(HookType.BEFORE_TASK_EXECUTE, test_hook)
        
        # Execute hook
        context = HookContext(
            hook_type=HookType.BEFORE_TASK_EXECUTE,
            source="test",
            data={"test": "data"}
        )
        
        results = asyncio.run(plugin.execute_hook(HookType.BEFORE_TASK_EXECUTE, context))
        
        assert hook_called
        assert results == ["hook_result"]
    
    @pytest.mark.asyncio
    async def test_async_hook_execution(self, plugin_config):
        """Test async hook execution"""
        plugin = MockPlugin(plugin_config)
        
        async def async_hook(context: HookContext):
            await asyncio.sleep(0.01)  # Simulate async work
            return "async_result"
        
        plugin.register_hook(HookType.AFTER_TASK_EXECUTE, async_hook)
        
        context = HookContext(
            hook_type=HookType.AFTER_TASK_EXECUTE,
            source="test"
        )
        
        results = await plugin.execute_hook(HookType.AFTER_TASK_EXECUTE, context)
        assert results == ["async_result"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, plugin_config):
        """Test plugin health check"""
        plugin = MockPlugin(plugin_config)
        
        health = await plugin.health_check()
        
        assert health["status"] == "healthy"
        assert health["plugin"] == "MockPlugin"
        assert "timestamp" in health


class TestPluginLoader:
    """Test plugin loading functionality"""
    
    @pytest.fixture
    def plugin_loader(self, temp_plugin_dir):
        """Create plugin loader with temp directory"""
        return PluginLoader([temp_plugin_dir])
    
    def test_loader_creation(self, plugin_loader):
        """Test plugin loader creation"""
        assert len(plugin_loader.plugin_directories) == 1
        assert plugin_loader.discovered_plugins == {}
    
    @pytest.mark.asyncio
    async def test_discover_plugins_empty_directory(self, plugin_loader):
        """Test discovery with empty directory"""
        plugins = await plugin_loader.discover_plugins()
        assert plugins == {}
    
    @pytest.mark.asyncio
    async def test_discover_plugins_with_metadata(self, plugin_loader, temp_plugin_dir):
        """Test plugin discovery with metadata file"""
        # Create a plugin file with metadata
        plugin_file = temp_plugin_dir / "test_plugin.py"
        plugin_content = '''
PLUGIN_METADATA = {
    "name": "test_plugin",
    "version": "1.0.0",
    "description": "Test plugin",
    "author": "Test Author",
    "plugin_type": "tool",
    "entry_point": "TestPlugin"
}

class TestPlugin:
    pass
'''
        plugin_file.write_text(plugin_content)
        
        plugins = await plugin_loader.discover_plugins()
        
        assert "test_plugin" in plugins
        metadata = plugins["test_plugin"]
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.TOOL
    
    @pytest.mark.asyncio
    async def test_load_plugin_success(self, plugin_loader, temp_plugin_dir, plugin_config):
        """Test successful plugin loading"""
        # Create a plugin file
        plugin_file = temp_plugin_dir / "mock_plugin.py"
        plugin_content = '''
from src.agentic.core.plugin_system import BasePlugin, PluginMetadata, PluginType, PluginStatus

PLUGIN_METADATA = {
    "name": "mock_plugin",
    "version": "1.0.0",
    "description": "Mock plugin",
    "author": "Test Author",
    "plugin_type": "tool",
    "entry_point": "MockPlugin"
}

class MockPlugin(BasePlugin):
    async def initialize(self):
        return True
    async def activate(self):
        return True
    async def deactivate(self):
        return True
    async def cleanup(self):
        return True
    def get_metadata(self):
        return PluginMetadata(**PLUGIN_METADATA)
'''
        plugin_file.write_text(plugin_content)
        
        # Discover and load
        await plugin_loader.discover_plugins()
        plugin_instance = await plugin_loader.load_plugin("mock_plugin", plugin_config)
        
        assert plugin_instance is not None
        assert plugin_instance.metadata.name == "mock_plugin"
        assert plugin_instance.status == PluginStatus.LOADED
        assert plugin_instance.instance is not None
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_plugin(self, plugin_loader, plugin_config):
        """Test loading non-existent plugin"""
        plugin_instance = await plugin_loader.load_plugin("nonexistent_plugin", plugin_config)
        assert plugin_instance is None


class TestPluginRegistry:
    """Test plugin registry functionality"""
    
    @pytest.fixture
    def plugin_registry(self, temp_plugin_dir):
        """Create plugin registry with temp directory"""
        return PluginRegistry([temp_plugin_dir])
    
    @pytest.mark.asyncio
    async def test_registry_initialization(self, plugin_registry):
        """Test registry initialization"""
        await plugin_registry.initialize()
        assert plugin_registry.plugins == {}
        assert plugin_registry.hook_registry == {}
    
    @pytest.mark.asyncio
    async def test_load_plugin_to_registry(self, plugin_registry, temp_plugin_dir):
        """Test loading plugin into registry"""
        # Create mock plugin file
        plugin_file = temp_plugin_dir / "mock_plugin.py"
        plugin_content = '''
from tests.core.test_plugin_system import MockPlugin
from src.agentic.core.plugin_system import PluginMetadata, PluginType

PLUGIN_METADATA = {
    "name": "mock_plugin",
    "version": "1.0.0",
    "description": "Mock plugin",
    "author": "Test Author",
    "plugin_type": "tool",
    "entry_point": "MockPlugin",
    "hooks": ["before_task_execute", "after_task_execute"]
}
'''
        plugin_file.write_text(plugin_content)
        
        await plugin_registry.initialize()
        
        # Mock the loader to return our MockPlugin
        async def mock_load_plugin(name, config):
            if name == "mock_plugin":
                metadata = PluginMetadata(
                    name="mock_plugin",
                    version="1.0.0",
                    description="Mock plugin",
                    author="Test Author",
                    plugin_type=PluginType.TOOL,
                    entry_point="MockPlugin",
                    hooks=[HookType.BEFORE_TASK_EXECUTE, HookType.AFTER_TASK_EXECUTE]
                )
                return PluginInstance(
                    metadata=metadata,
                    config=config,
                    status=PluginStatus.LOADED,
                    instance=MockPlugin(config)
                )
            return None
        
        plugin_registry.loader.load_plugin = mock_load_plugin
        plugin_registry.loader.discovered_plugins = {
            "mock_plugin": PluginMetadata(
                name="mock_plugin",
                version="1.0.0",
                description="Mock plugin",
                author="Test Author",
                plugin_type=PluginType.TOOL,
                entry_point="MockPlugin",
                hooks=[HookType.BEFORE_TASK_EXECUTE, HookType.AFTER_TASK_EXECUTE]
            )
        }
        
        success = await plugin_registry.load_plugin("mock_plugin")
        
        assert success
        assert "mock_plugin" in plugin_registry.plugins
        assert plugin_registry.plugins["mock_plugin"].status == PluginStatus.LOADED
        
        # Check hooks were registered
        assert HookType.BEFORE_TASK_EXECUTE in plugin_registry.hook_registry
        assert "mock_plugin" in plugin_registry.hook_registry[HookType.BEFORE_TASK_EXECUTE]
    
    @pytest.mark.asyncio
    async def test_plugin_lifecycle_through_registry(self, plugin_registry):
        """Test complete plugin lifecycle through registry"""
        # Mock plugin loading
        mock_plugin = MockPlugin(PluginConfig())
        metadata = mock_plugin.get_metadata()
        
        plugin_instance = PluginInstance(
            metadata=metadata,
            config=PluginConfig(),
            status=PluginStatus.LOADED,
            instance=mock_plugin
        )
        
        plugin_registry.plugins["mock_plugin"] = plugin_instance
        
        # Initialize
        success = await plugin_registry.initialize_plugin("mock_plugin")
        assert success
        assert plugin_instance.status == PluginStatus.INITIALIZED
        assert mock_plugin.initialized
        
        # Activate
        success = await plugin_registry.activate_plugin("mock_plugin")
        assert success
        assert plugin_instance.status == PluginStatus.ACTIVATED
        assert mock_plugin.activated
        
        # Deactivate
        success = await plugin_registry.deactivate_plugin("mock_plugin")
        assert success
        assert plugin_instance.status == PluginStatus.DEACTIVATED
        assert not mock_plugin.activated
        
        # Unload
        success = await plugin_registry.unload_plugin("mock_plugin")
        assert success
        assert "mock_plugin" not in plugin_registry.plugins
        assert mock_plugin.cleanup_called
    
    @pytest.mark.asyncio
    async def test_plugin_with_dependencies(self, plugin_registry):
        """Test loading plugin with dependencies"""
        # Setup mock plugins
        mock_plugin = MockPlugin(PluginConfig())
        dependent_plugin = DependentPlugin(PluginConfig())
        
        # Mock the loader
        async def mock_load_plugin(name, config):
            if name == "mock_plugin":
                return PluginInstance(
                    metadata=mock_plugin.get_metadata(),
                    config=config,
                    status=PluginStatus.LOADED,
                    instance=mock_plugin
                )
            elif name == "dependent_plugin":
                return PluginInstance(
                    metadata=dependent_plugin.get_metadata(),
                    config=config,
                    status=PluginStatus.LOADED,
                    instance=dependent_plugin
                )
            return None
        
        plugin_registry.loader.load_plugin = mock_load_plugin
        plugin_registry.loader.discovered_plugins = {
            "mock_plugin": mock_plugin.get_metadata(),
            "dependent_plugin": dependent_plugin.get_metadata()
        }
        plugin_registry._build_dependency_graph()
        
        # Load dependent plugin (should load mock_plugin first)
        success = await plugin_registry.load_plugin("dependent_plugin")
        
        assert success
        assert "mock_plugin" in plugin_registry.plugins
        assert "dependent_plugin" in plugin_registry.plugins
    
    @pytest.mark.asyncio
    async def test_hook_execution(self, plugin_registry):
        """Test hook execution through registry"""
        # Setup plugin with hooks
        mock_plugin = MockPlugin(PluginConfig())
        
        hook_executed = False
        
        def test_hook(context):
            nonlocal hook_executed
            hook_executed = True
            return "hook_result"
        
        mock_plugin.register_hook(HookType.BEFORE_TASK_EXECUTE, test_hook)
        
        plugin_instance = PluginInstance(
            metadata=mock_plugin.get_metadata(),
            config=PluginConfig(),
            status=PluginStatus.ACTIVATED,
            instance=mock_plugin
        )
        
        plugin_registry.plugins["mock_plugin"] = plugin_instance
        plugin_registry.hook_registry[HookType.BEFORE_TASK_EXECUTE] = ["mock_plugin"]
        
        # Execute hooks
        context = HookContext(
            hook_type=HookType.BEFORE_TASK_EXECUTE,
            source="test"
        )
        
        results = await plugin_registry.execute_hooks(HookType.BEFORE_TASK_EXECUTE, context)
        
        assert hook_executed
        assert "mock_plugin" in results
        assert results["mock_plugin"] == ["hook_result"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, plugin_registry):
        """Test error handling during plugin operations"""
        failing_plugin = FailingPlugin(PluginConfig())
        
        plugin_instance = PluginInstance(
            metadata=failing_plugin.get_metadata(),
            config=PluginConfig(),
            status=PluginStatus.LOADED,
            instance=failing_plugin
        )
        
        plugin_registry.plugins["failing_plugin"] = plugin_instance
        
        # Test initialization failure
        success = await plugin_registry.initialize_plugin("failing_plugin")
        assert not success
        assert plugin_instance.status == PluginStatus.ERROR
        assert "Initialization failed" in plugin_instance.error_message
    
    def test_plugin_filtering(self, plugin_registry):
        """Test plugin filtering methods"""
        # Setup different types of plugins
        tool_plugin = PluginInstance(
            metadata=PluginMetadata(
                name="tool_plugin",
                version="1.0.0",
                description="Tool plugin",
                author="Test",
                plugin_type=PluginType.TOOL,
                entry_point="ToolPlugin"
            ),
            config=PluginConfig(),
            status=PluginStatus.ACTIVATED
        )
        
        agent_plugin = PluginInstance(
            metadata=PluginMetadata(
                name="agent_plugin",
                version="1.0.0",
                description="Agent plugin",
                author="Test",
                plugin_type=PluginType.AGENT,
                entry_point="AgentPlugin"
            ),
            config=PluginConfig(),
            status=PluginStatus.LOADED
        )
        
        plugin_registry.plugins["tool_plugin"] = tool_plugin
        plugin_registry.plugins["agent_plugin"] = agent_plugin
        
        # Test filtering by type
        tool_plugins = plugin_registry.get_plugins_by_type(PluginType.TOOL)
        assert len(tool_plugins) == 1
        assert "tool_plugin" in tool_plugins
        
        agent_plugins = plugin_registry.get_plugins_by_type(PluginType.AGENT)
        assert len(agent_plugins) == 1
        assert "agent_plugin" in agent_plugins
        
        # Test filtering by status
        active_plugins = plugin_registry.get_active_plugins()
        assert len(active_plugins) == 1
        assert "tool_plugin" in active_plugins
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, plugin_registry):
        """Test health check for all plugins"""
        mock_plugin = MockPlugin(PluginConfig())
        
        plugin_instance = PluginInstance(
            metadata=mock_plugin.get_metadata(),
            config=PluginConfig(),
            status=PluginStatus.ACTIVATED,
            instance=mock_plugin
        )
        
        plugin_registry.plugins["mock_plugin"] = plugin_instance
        
        health_results = await plugin_registry.health_check_all()
        
        assert "mock_plugin" in health_results
        assert health_results["mock_plugin"]["status"] == "healthy"
        assert health_results["mock_plugin"]["plugin"] == "MockPlugin"


class TestHookContext:
    """Test hook context functionality"""
    
    def test_hook_context_creation(self):
        """Test creating hook context"""
        context = HookContext(
            hook_type=HookType.BEFORE_TASK_EXECUTE,
            source="test_component",
            data={"task_id": "123", "command": "test command"},
            agent_id="agent_123"
        )
        
        assert context.hook_type == HookType.BEFORE_TASK_EXECUTE
        assert context.source == "test_component"
        assert context.data["task_id"] == "123"
        assert context.agent_id == "agent_123"
        assert isinstance(context.timestamp, datetime)


# Verified: Complete - Comprehensive test suite for plugin system architecture 