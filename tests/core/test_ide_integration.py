"""
Tests for IDE Integration
"""

import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic.core.ide_integration import (
    IDEIntegrationManager,
    VSCodeExtension,
    JetBrainsPlugin,
    GitHubIntegration,
    FileEditor,
    FileSelection,
    IDECommand,
    IDEResponse,
    IDEType,
    IntegrationType,
    initialize_ide_integration,
    get_ide_integration_manager,
    create_ide_command_from_selection,
    execute_file_edits
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        yield workspace


@pytest.fixture
def sample_file(temp_workspace):
    """Create a sample file for testing"""
    sample_path = temp_workspace / "test_file.py"
    content = """def hello_world():
    print("Hello, World!")
    return "success"

def another_function():
    pass
"""
    sample_path.write_text(content)
    return sample_path


class TestFileSelection:
    """Test FileSelection dataclass"""
    
    def test_file_selection_creation(self, temp_workspace):
        """Test creating FileSelection"""
        file_path = temp_workspace / "test.py"
        selection = FileSelection(
            file_path=file_path,
            start_line=1,
            end_line=3,
            start_column=0,
            end_column=10,
            selected_text="test code"
        )
        
        assert selection.file_path == file_path
        assert selection.start_line == 1
        assert selection.end_line == 3
        assert selection.selected_text == "test code"
    
    def test_file_selection_to_dict(self, temp_workspace):
        """Test converting FileSelection to dict"""
        file_path = temp_workspace / "test.py"
        selection = FileSelection(
            file_path=file_path,
            start_line=1,
            end_line=3,
            selected_text="test code"
        )
        
        result = selection.to_dict()
        expected = {
            'file_path': str(file_path),
            'start_line': 1,
            'end_line': 3,
            'start_column': 0,
            'end_column': 0,
            'selected_text': 'test code'
        }
        
        assert result == expected


class TestIDECommand:
    """Test IDECommand dataclass"""
    
    def test_ide_command_creation(self, temp_workspace):
        """Test creating IDECommand"""
        file_selection = FileSelection(
            file_path=temp_workspace / "test.py",
            start_line=1,
            end_line=2
        )
        
        command = IDECommand(
            command_id="test-123",
            command_text="explain this code",
            file_selection=file_selection,
            workspace_path=temp_workspace
        )
        
        assert command.command_id == "test-123"
        assert command.command_text == "explain this code"
        assert command.file_selection == file_selection
        assert command.workspace_path == temp_workspace


class TestIDEResponse:
    """Test IDEResponse dataclass"""
    
    def test_ide_response_creation(self):
        """Test creating IDEResponse"""
        response = IDEResponse(
            command_id="test-123",
            success=True,
            output="Code explanation here",
            suggestions=["Add error handling", "Add tests"]
        )
        
        assert response.command_id == "test-123"
        assert response.success is True
        assert response.output == "Code explanation here"
        assert len(response.suggestions) == 2


class TestVSCodeExtension:
    """Test VS Code extension integration"""
    
    @pytest.fixture
    def vscode_extension(self, temp_workspace):
        """Create VSCodeExtension instance"""
        return VSCodeExtension(temp_workspace)
    
    @pytest.mark.asyncio
    async def test_vscode_initialization(self, vscode_extension, temp_workspace):
        """Test VS Code extension initialization"""
        result = await vscode_extension.initialize()
        
        assert result is True
        assert vscode_extension.extension_path.exists()
        assert (vscode_extension.extension_path / 'package.json').exists()
        assert (vscode_extension.extension_path / 'commands.json').exists()
        assert (vscode_extension.extension_path / 'responses.json').exists()
        assert (vscode_extension.extension_path / 'status.json').exists()
    
    @pytest.mark.asyncio
    async def test_vscode_package_json_content(self, vscode_extension):
        """Test VS Code package.json content"""
        await vscode_extension.initialize()
        
        package_json_path = vscode_extension.extension_path / 'package.json'
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
        
        assert package_data['name'] == 'agentic-vscode'
        assert package_data['displayName'] == 'Agentic AI Assistant'
        assert 'contributes' in package_data
        assert 'commands' in package_data['contributes']
        assert len(package_data['contributes']['commands']) == 6
    
    @pytest.mark.asyncio
    async def test_vscode_handle_command(self, vscode_extension, temp_workspace):
        """Test handling VS Code command"""
        await vscode_extension.initialize()
        
        file_selection = FileSelection(
            file_path=temp_workspace / "test.py",
            start_line=1,
            end_line=2
        )
        
        command = IDECommand(
            command_id="test-123",
            command_text="explain this code",
            file_selection=file_selection,
            workspace_path=temp_workspace
        )
        
        response = await vscode_extension.handle_command(command)
        
        assert response.success is True
        assert response.command_id == "test-123"
        assert "Executed: explain this code" in response.output
        assert len(response.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_vscode_get_agent_status(self, vscode_extension):
        """Test getting agent status"""
        await vscode_extension.initialize()
        
        status = await vscode_extension.get_agent_status()
        
        assert 'initialized' in status
        assert 'agents' in status
        assert 'last_update' in status
        assert status['initialized'] is True


class TestJetBrainsPlugin:
    """Test JetBrains plugin integration"""
    
    @pytest.fixture
    def jetbrains_plugin(self, temp_workspace):
        """Create JetBrainsPlugin instance"""
        return JetBrainsPlugin(temp_workspace, IDEType.INTELLIJ)
    
    @pytest.mark.asyncio
    async def test_jetbrains_initialization(self, jetbrains_plugin):
        """Test JetBrains plugin initialization"""
        result = await jetbrains_plugin.initialize()
        
        assert result is True
        assert jetbrains_plugin.plugin_path.exists()
        assert (jetbrains_plugin.plugin_path / 'plugin.xml').exists()
        assert (jetbrains_plugin.plugin_path / 'communication.json').exists()
    
    @pytest.mark.asyncio
    async def test_jetbrains_plugin_xml_content(self, jetbrains_plugin):
        """Test JetBrains plugin.xml content"""
        await jetbrains_plugin.initialize()
        
        plugin_xml_path = jetbrains_plugin.plugin_path / 'plugin.xml'
        content = plugin_xml_path.read_text()
        
        assert 'com.agentic.plugin' in content
        assert 'Agentic AI Assistant' in content
        assert 'ExplainCodeAction' in content
        assert 'RefactorCodeAction' in content
        assert 'GenerateTestsAction' in content
    
    @pytest.mark.asyncio
    async def test_jetbrains_communication_setup(self, jetbrains_plugin):
        """Test JetBrains communication setup"""
        await jetbrains_plugin.initialize()
        
        comm_file = jetbrains_plugin.plugin_path / 'communication.json'
        with open(comm_file, 'r') as f:
            comm_data = json.load(f)
        
        assert comm_data['plugin_active'] is True
        assert comm_data['ide_type'] == 'intellij'
        assert 'commands' in comm_data
        assert 'responses' in comm_data


class TestGitHubIntegration:
    """Test GitHub integration"""
    
    @pytest.fixture
    def github_integration(self):
        """Create GitHubIntegration instance"""
        return GitHubIntegration()
    
    @pytest.mark.asyncio
    async def test_github_initialization_no_token(self, github_integration):
        """Test GitHub initialization without token"""
        result = await github_integration.initialize()
        
        # Should return False when no token is provided
        assert result is False
    
    @pytest.mark.asyncio
    async def test_github_initialization_with_token(self):
        """Test GitHub initialization with token"""
        integration = GitHubIntegration(github_token="fake-token")
        result = await integration.initialize()
        
        # Should return True with token (mocked connection test)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_github_create_pull_request(self, github_integration):
        """Test creating pull request"""
        pr_data = await github_integration.create_pull_request(
            title="Test PR",
            body="Test PR body",
            base_branch="main",
            head_branch="feature/test"
        )
        
        assert pr_data['title'] == "Test PR"
        assert pr_data['body'] == "Test PR body"
        assert pr_data['base'] == "main"
        assert pr_data['head'] == "feature/test"
        assert 'number' in pr_data
        assert 'url' in pr_data
    
    @pytest.mark.asyncio
    async def test_github_review_pull_request(self, github_integration):
        """Test reviewing pull request"""
        review_data = await github_integration.review_pull_request(
            pr_number=123,
            review_type="approve"
        )
        
        assert review_data['pr_number'] == 123
        assert review_data['review_type'] == "approve"
        assert review_data['approved'] is True
        assert len(review_data['comments']) > 0
    
    @pytest.mark.asyncio
    async def test_github_analyze_repository(self, github_integration):
        """Test repository analysis"""
        analysis = await github_integration.analyze_repository(
            "https://github.com/owner/repo"
        )
        
        assert analysis['repository'] == "https://github.com/owner/repo"
        assert 'tech_stack' in analysis
        assert 'code_quality_score' in analysis
        assert 'suggestions' in analysis
        assert len(analysis['suggestions']) > 0


class TestFileEditor:
    """Test file editing capabilities"""
    
    @pytest.fixture
    def file_editor(self, temp_workspace):
        """Create FileEditor instance"""
        return FileEditor(temp_workspace)
    
    @pytest.mark.asyncio
    async def test_file_editor_replace_line(self, file_editor, sample_file):
        """Test replacing a line in file"""
        relative_path = sample_file.relative_to(file_editor.workspace_path)
        
        edits = [{
            'type': 'replace',
            'line': 2,
            'content': '    print("Hello, Agentic!")'
        }]
        
        result = await file_editor.edit_file(relative_path, edits)
        
        assert result is True
        
        # Verify the edit was applied
        content = sample_file.read_text()
        lines = content.split('\n')
        assert 'Hello, Agentic!' in lines[1]
    
    @pytest.mark.asyncio
    async def test_file_editor_insert_line(self, file_editor, sample_file):
        """Test inserting a line in file"""
        relative_path = sample_file.relative_to(file_editor.workspace_path)
        
        edits = [{
            'type': 'insert',
            'line': 3,
            'content': '    # This is a new comment'
        }]
        
        result = await file_editor.edit_file(relative_path, edits)
        
        assert result is True
        
        # Verify the line was inserted
        content = sample_file.read_text()
        lines = content.split('\n')
        assert '# This is a new comment' in lines[2]
    
    @pytest.mark.asyncio
    async def test_file_editor_delete_line(self, file_editor, sample_file):
        """Test deleting a line from file"""
        relative_path = sample_file.relative_to(file_editor.workspace_path)
        
        # Count original lines
        original_content = sample_file.read_text()
        original_lines = len(original_content.split('\n'))
        
        edits = [{
            'type': 'delete',
            'line': 5
        }]
        
        result = await file_editor.edit_file(relative_path, edits)
        
        assert result is True
        
        # Verify line was deleted
        new_content = sample_file.read_text()
        new_lines = len(new_content.split('\n'))
        assert new_lines < original_lines
    
    @pytest.mark.asyncio
    async def test_file_editor_multiple_edits(self, file_editor, sample_file):
        """Test applying multiple edits"""
        relative_path = sample_file.relative_to(file_editor.workspace_path)
        
        edits = [
            {
                'type': 'replace',
                'line': 1,
                'content': 'def hello_agentic():'
            },
            {
                'type': 'insert',
                'line': 6,
                'content': '# End of file'
            }
        ]
        
        result = await file_editor.edit_file(relative_path, edits)
        
        assert result is True
        
        # Verify both edits were applied
        content = sample_file.read_text()
        assert 'hello_agentic' in content
        assert '# End of file' in content
    
    @pytest.mark.asyncio
    async def test_file_editor_backup_creation(self, file_editor, sample_file):
        """Test that backup files are created"""
        relative_path = sample_file.relative_to(file_editor.workspace_path)
        
        edits = [{
            'type': 'replace',
            'line': 1,
            'content': 'def modified_function():'
        }]
        
        # Count backup files before
        backup_files_before = list(sample_file.parent.glob(f"{sample_file.name}.backup_*"))
        
        result = await file_editor.edit_file(relative_path, edits)
        
        assert result is True
        
        # Count backup files after
        backup_files_after = list(sample_file.parent.glob(f"{sample_file.name}.backup_*"))
        assert len(backup_files_after) > len(backup_files_before)
    
    @pytest.mark.asyncio
    async def test_file_editor_create_file(self, file_editor, temp_workspace):
        """Test creating a new file"""
        new_file_path = Path("new_module.py")
        content = """def new_function():
    return "Hello from new file"
"""
        
        result = await file_editor.create_file(new_file_path, content)
        
        assert result is True
        
        # Verify file was created
        actual_file = temp_workspace / new_file_path
        assert actual_file.exists()
        assert "new_function" in actual_file.read_text()
    
    @pytest.mark.asyncio
    async def test_file_editor_get_edit_history(self, file_editor, sample_file):
        """Test getting edit history"""
        relative_path = sample_file.relative_to(file_editor.workspace_path)
        
        edits = [{
            'type': 'replace',
            'line': 1,
            'content': 'def edited_function():'
        }]
        
        await file_editor.edit_file(relative_path, edits)
        
        history = await file_editor.get_edit_history()
        
        assert len(history) > 0
        assert history[0]['file_path'] == str(relative_path)
        assert len(history[0]['edits']) == 1
        assert 'timestamp' in history[0]


class TestIDEIntegrationManager:
    """Test IDE Integration Manager"""
    
    @pytest.fixture
    def integration_manager(self, temp_workspace):
        """Create IDEIntegrationManager instance"""
        return IDEIntegrationManager(temp_workspace)
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, integration_manager):
        """Test initializing all integrations"""
        results = await integration_manager.initialize_all()
        
        assert isinstance(results, dict)
        assert 'vscode' in results
        assert 'jetbrains' in results
        assert 'github' in results
        
        # VS Code and JetBrains should initialize successfully
        assert results['vscode'] is True
        assert results['jetbrains'] is True
        # GitHub may fail without token
        assert isinstance(results['github'], bool)
    
    @pytest.mark.asyncio
    async def test_manager_handle_ide_command(self, integration_manager, temp_workspace):
        """Test handling IDE command"""
        await integration_manager.initialize_all()
        
        command = IDECommand(
            command_id="test-456",
            command_text="refactor this function",
            workspace_path=temp_workspace,
            context={'vscode': True}
        )
        
        response = await integration_manager.handle_ide_command(command)
        
        assert response.command_id == "test-456"
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_manager_get_integration_status(self, integration_manager):
        """Test getting integration status"""
        await integration_manager.initialize_all()
        
        status = await integration_manager.get_integration_status()
        
        assert 'workspace_path' in status
        assert 'integrations' in status
        assert 'last_updated' in status
        
        integrations = status['integrations']
        assert 'vscode' in integrations
        assert 'jetbrains' in integrations
        assert 'github' in integrations


class TestUtilityFunctions:
    """Test utility functions"""
    
    @pytest.mark.asyncio
    async def test_create_ide_command_from_selection(self, sample_file, temp_workspace):
        """Test creating IDE command from file selection"""
        command = await create_ide_command_from_selection(
            file_path=sample_file,
            start_line=1,
            end_line=3,
            command_text="explain this function",
            workspace_path=temp_workspace
        )
        
        assert command.command_text == "explain this function"
        assert command.file_selection is not None
        assert command.file_selection.start_line == 1
        assert command.file_selection.end_line == 3
        assert command.file_selection.selected_text is not None
        assert "def hello_world" in command.file_selection.selected_text
    
    @pytest.mark.asyncio
    async def test_execute_file_edits_no_manager(self, temp_workspace):
        """Test executing file edits without manager"""
        result = await execute_file_edits(
            file_path=Path("test.py"),
            edits=[{'type': 'replace', 'line': 1, 'content': 'new content'}],
            workspace_path=temp_workspace
        )
        
        # Should return False when no manager is initialized
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_file_edits_with_manager(self, temp_workspace, sample_file):
        """Test executing file edits with manager"""
        # Initialize manager
        manager = initialize_ide_integration(temp_workspace)
        
        relative_path = sample_file.relative_to(temp_workspace)
        
        result = await execute_file_edits(
            file_path=relative_path,
            edits=[{'type': 'replace', 'line': 1, 'content': 'def modified_hello():'}],
            workspace_path=temp_workspace
        )
        
        assert result is True
        
        # Verify edit was applied
        content = sample_file.read_text()
        assert "modified_hello" in content
    
    def test_initialize_ide_integration(self, temp_workspace):
        """Test initializing IDE integration"""
        manager = initialize_ide_integration(temp_workspace)
        
        assert isinstance(manager, IDEIntegrationManager)
        assert manager.workspace_path == temp_workspace
        
        # Test getting the manager instance
        retrieved_manager = get_ide_integration_manager()
        assert retrieved_manager is manager


@pytest.mark.asyncio
async def test_integration_full_workflow(temp_workspace):
    """Test full IDE integration workflow"""
    # Create test file
    test_file = temp_workspace / "workflow_test.py"
    test_file.write_text("""def old_function():
    return "old"
""")
    
    # Initialize IDE integration
    manager = initialize_ide_integration(temp_workspace)
    await manager.initialize_all()
    
    # Create IDE command
    command = await create_ide_command_from_selection(
        file_path=test_file,
        start_line=1,
        end_line=2,
        command_text="refactor this function",
        workspace_path=temp_workspace
    )
    
    # Handle command
    response = await manager.handle_ide_command(command)
    assert response.success is True
    
    # Edit file through integration
    relative_path = test_file.relative_to(temp_workspace)
    edit_result = await execute_file_edits(
        file_path=relative_path,
        edits=[{'type': 'replace', 'line': 1, 'content': 'def new_function():'}],
        workspace_path=temp_workspace
    )
    
    assert edit_result is True
    
    # Verify edit was applied
    content = test_file.read_text()
    assert "new_function" in content
    assert "old_function" not in content
    
    # Get integration status
    status = await manager.get_integration_status()
    
    # Check VS Code status structure (it returns different fields than 'enabled')
    vscode_status = status['integrations']['vscode']
    assert 'initialized' in vscode_status or 'enabled' in vscode_status
    
    # Check JetBrains status (it explicitly has 'enabled' field)
    assert status['integrations']['jetbrains']['enabled'] is not False


if __name__ == "__main__":
    pytest.main([__file__]) 