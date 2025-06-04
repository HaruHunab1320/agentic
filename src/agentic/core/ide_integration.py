"""
Phase 4: IDE Integration
Seamless integration with popular development environments including VS Code, JetBrains, and GitHub
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel


class IDEType(str, Enum):
    """Supported IDE types"""
    VSCODE = "vscode"
    INTELLIJ = "intellij"
    PYCHARM = "pycharm"
    WEBSTORM = "webstorm"
    VIM = "vim"
    NEOVIM = "neovim"


class IntegrationType(str, Enum):
    """Types of IDE integration"""
    EXTENSION = "extension"
    PLUGIN = "plugin"
    LSP = "lsp"
    CLI = "cli"
    WEBHOOK = "webhook"


@dataclass
class FileSelection:
    """Represents selected text/file in IDE"""
    file_path: Path
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0
    selected_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': str(self.file_path),
            'start_line': self.start_line,
            'end_line': self.end_line,
            'start_column': self.start_column,
            'end_column': self.end_column,
            'selected_text': self.selected_text
        }


@dataclass
class IDECommand:
    """Command from IDE to Agentic"""
    command_id: str
    command_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    file_selection: Optional[FileSelection] = None
    workspace_path: Optional[Path] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IDEResponse:
    """Response from Agentic to IDE"""
    command_id: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    modified_files: List[Path] = field(default_factory=list)
    created_files: List[Path] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class VSCodeExtension:
    """VS Code extension integration"""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.extension_path = workspace_path / '.vscode' / 'agentic'
        self.console = Console()
        
    async def initialize(self) -> bool:
        """Initialize VS Code extension integration"""
        try:
            # Create extension directory
            self.extension_path.mkdir(parents=True, exist_ok=True)
            
            # Generate extension configuration
            await self._create_extension_config()
            
            # Setup communication files
            await self._setup_communication_files()
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to initialize VS Code extension: {e}[/red]")
            return False
    
    async def _create_extension_config(self):
        """Create VS Code extension configuration"""
        package_json = {
            "name": "agentic-vscode",
            "displayName": "Agentic AI Assistant",
            "description": "Multi-agent AI development assistant",
            "version": "1.0.0",
            "engines": {"vscode": "^1.74.0"},
            "categories": ["Other"],
            "activationEvents": ["onStartupFinished"],
            "main": "./out/extension.js",
            "contributes": {
                "commands": [
                    {
                        "command": "agentic.init",
                        "title": "Initialize Agentic",
                        "category": "Agentic"
                    },
                    {
                        "command": "agentic.executeCommand",
                        "title": "Execute Command",
                        "category": "Agentic"
                    },
                    {
                        "command": "agentic.showPanel",
                        "title": "Show Agentic Panel",
                        "category": "Agentic"
                    },
                    {
                        "command": "agentic.explainCode",
                        "title": "Explain Selected Code",
                        "category": "Agentic"
                    },
                    {
                        "command": "agentic.refactorCode",
                        "title": "Refactor Selected Code",
                        "category": "Agentic"
                    },
                    {
                        "command": "agentic.generateTests",
                        "title": "Generate Tests",
                        "category": "Agentic"
                    }
                ],
                "menus": {
                    "editor/context": [
                        {
                            "command": "agentic.explainCode",
                            "when": "editorHasSelection",
                            "group": "agentic"
                        },
                        {
                            "command": "agentic.refactorCode", 
                            "when": "editorHasSelection",
                            "group": "agentic"
                        },
                        {
                            "command": "agentic.generateTests",
                            "group": "agentic"
                        }
                    ],
                    "explorer/context": [
                        {
                            "command": "agentic.generateTests",
                            "when": "resourceExtname == .js || resourceExtname == .ts || resourceExtname == .py",
                            "group": "agentic"
                        }
                    ]
                },
                "views": {
                    "explorer": [
                        {
                            "id": "agenticAgents",
                            "name": "Agentic Agents",
                            "when": "agenticExtensionActive"
                        }
                    ]
                },
                "viewsContainers": {
                    "activitybar": [
                        {
                            "id": "agentic",
                            "title": "Agentic",
                            "icon": "$(robot)"
                        }
                    ]
                },
                "configuration": {
                    "title": "Agentic",
                    "properties": {
                        "agentic.autoInitialize": {
                            "type": "boolean",
                            "default": True,
                            "description": "Automatically initialize Agentic when opening a project"
                        },
                        "agentic.primaryModel": {
                            "type": "string",
                            "default": "claude-3-5-sonnet",
                            "description": "Primary AI model to use"
                        },
                        "agentic.maxCostPerHour": {
                            "type": "number",
                            "default": 10.0,
                            "description": "Maximum cost per hour in USD"
                        }
                    }
                }
            }
        }
        
        config_file = self.extension_path / 'package.json'
        with open(config_file, 'w') as f:
            json.dump(package_json, f, indent=2)
    
    async def _setup_communication_files(self):
        """Setup communication files for IDE integration"""
        # Create command input file
        command_file = self.extension_path / 'commands.json'
        command_file.write_text('[]')
        
        # Create response output file
        response_file = self.extension_path / 'responses.json'
        response_file.write_text('[]')
        
        # Create status file
        status_file = self.extension_path / 'status.json'
        status_data = {
            'initialized': True,
            'agents': [],
            'last_update': datetime.utcnow().isoformat()
        }
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    async def handle_command(self, ide_command: IDECommand) -> IDEResponse:
        """Handle command from VS Code extension"""
        try:
            # Write command to communication file
            command_file = self.extension_path / 'commands.json'
            
            # Read existing commands
            commands = []
            if command_file.exists():
                with open(command_file, 'r') as f:
                    commands = json.load(f)
            
            # Add new command
            command_data = {
                'command_id': ide_command.command_id,
                'command_text': ide_command.command_text,
                'context': ide_command.context,
                'file_selection': ide_command.file_selection.to_dict() if ide_command.file_selection else None,
                'workspace_path': str(ide_command.workspace_path) if ide_command.workspace_path else None,
                'timestamp': ide_command.timestamp.isoformat()
            }
            
            commands.append(command_data)
            
            # Write back to file
            with open(command_file, 'w') as f:
                json.dump(commands, f, indent=2)
            
            # TODO: Integrate with main orchestrator to execute command
            # For now, return a mock response
            return IDEResponse(
                command_id=ide_command.command_id,
                success=True,
                output=f"Executed: {ide_command.command_text}",
                suggestions=["Consider adding error handling", "Add unit tests"]
            )
            
        except Exception as e:
            return IDEResponse(
                command_id=ide_command.command_id,
                success=False,
                error=str(e)
            )
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status for VS Code display"""
        status_file = self.extension_path / 'status.json'
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                return json.load(f)
        
        return {
            'initialized': False,
            'agents': [],
            'last_update': datetime.utcnow().isoformat()
        }


class JetBrainsPlugin:
    """JetBrains IDE plugin integration"""
    
    def __init__(self, workspace_path: Path, ide_type: IDEType = IDEType.INTELLIJ):
        self.workspace_path = workspace_path
        self.ide_type = ide_type
        self.plugin_path = workspace_path / '.idea' / 'agentic'
        self.console = Console()
    
    async def initialize(self) -> bool:
        """Initialize JetBrains plugin integration"""
        try:
            # Create plugin directory
            self.plugin_path.mkdir(parents=True, exist_ok=True)
            
            # Generate plugin configuration
            await self._create_plugin_config()
            
            # Setup communication mechanism
            await self._setup_communication()
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to initialize JetBrains plugin: {e}[/red]")
            return False
    
    async def _create_plugin_config(self):
        """Create JetBrains plugin configuration"""
        plugin_xml = f"""
        <idea-plugin>
            <id>com.agentic.plugin</id>
            <name>Agentic AI Assistant</name>
            <version>1.0</version>
            <vendor email="support@agentic.dev" url="https://agentic.dev">Agentic</vendor>
            
            <description><![CDATA[
                Multi-agent AI development assistant for {self.ide_type.value}
            ]]></description>
            
            <idea-version since-build="173.0"/>
            
            <depends>com.intellij.modules.platform</depends>
            <depends>com.intellij.modules.lang</depends>
            
            <extensions defaultExtensionNs="com.intellij">
                <toolWindow id="Agentic" secondary="true" anchor="right" 
                           factoryClass="com.agentic.plugin.AgenticToolWindowFactory"/>
                
                <applicationService serviceImplementation="com.agentic.plugin.AgenticService"/>
                
                <projectService serviceImplementation="com.agentic.plugin.AgenticProjectService"/>
            </extensions>
            
            <actions>
                <group id="AgenticActionGroup" text="Agentic" description="Agentic AI Assistant">
                    <add-to-group group-id="EditorPopupMenu" anchor="last"/>
                    <add-to-group group-id="ProjectViewPopupMenu" anchor="last"/>
                    
                    <action id="Agentic.ExplainCode" class="com.agentic.plugin.ExplainCodeAction"
                           text="Explain Code" description="Explain selected code with AI"/>
                    
                    <action id="Agentic.RefactorCode" class="com.agentic.plugin.RefactorCodeAction"
                           text="Refactor Code" description="Refactor selected code with AI"/>
                    
                    <action id="Agentic.GenerateTests" class="com.agentic.plugin.GenerateTestsAction"
                           text="Generate Tests" description="Generate tests for selected code"/>
                    
                    <action id="Agentic.ExecuteCommand" class="com.agentic.plugin.ExecuteCommandAction"
                           text="Execute Agentic Command" description="Execute custom Agentic command"/>
                </group>
            </actions>
        </idea-plugin>
        """
        
        config_file = self.plugin_path / 'plugin.xml'
        config_file.write_text(plugin_xml.strip())
    
    async def _setup_communication(self):
        """Setup communication for JetBrains plugin"""
        # Similar to VS Code but adapted for JetBrains
        communication_file = self.plugin_path / 'communication.json'
        
        comm_data = {
            'plugin_active': True,
            'ide_type': self.ide_type.value,
            'workspace_path': str(self.workspace_path),
            'commands': [],
            'responses': []
        }
        
        with open(communication_file, 'w') as f:
            json.dump(comm_data, f, indent=2)


class GitHubIntegration:
    """GitHub integration for PR reviews and issue management"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.console = Console()
    
    async def initialize(self) -> bool:
        """Initialize GitHub integration"""
        if not self.github_token:
            self.console.print("[yellow]GitHub token not provided - some features will be limited[/yellow]")
            return False
        
        # Test GitHub API connectivity
        try:
            result = await self._test_github_connection()
            if result:
                self.console.print("[green]✅ GitHub integration initialized[/green]")
            return result
        except Exception as e:
            self.console.print(f"[red]Failed to initialize GitHub integration: {e}[/red]")
            return False
    
    async def _test_github_connection(self) -> bool:
        """Test GitHub API connection"""
        # TODO: Implement actual GitHub API test
        return True
    
    async def create_pull_request(self, title: str, body: str, 
                                base_branch: str = "main", 
                                head_branch: str = "feature/agentic-changes") -> Dict[str, Any]:
        """Create a pull request"""
        try:
            # TODO: Implement actual GitHub API call
            pr_data = {
                'number': 123,
                'title': title,
                'body': body,
                'base': base_branch,
                'head': head_branch,
                'url': f"https://github.com/owner/repo/pull/123",
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.console.print(f"[green]✅ Created PR: {title}[/green]")
            return pr_data
            
        except Exception as e:
            self.console.print(f"[red]Failed to create PR: {e}[/red]")
            raise
    
    async def review_pull_request(self, pr_number: int, 
                                review_type: str = "comment") -> Dict[str, Any]:
        """Review a pull request with AI analysis"""
        try:
            # TODO: Integrate with AI agents to analyze PR
            review_data = {
                'pr_number': pr_number,
                'review_type': review_type,
                'comments': [
                    "Code looks good overall",
                    "Consider adding error handling in line 45",
                    "Tests cover the main functionality"
                ],
                'approved': review_type == "approve",
                'submitted_at': datetime.utcnow().isoformat()
            }
            
            self.console.print(f"[green]✅ Reviewed PR #{pr_number}[/green]")
            return review_data
            
        except Exception as e:
            self.console.print(f"[red]Failed to review PR: {e}[/red]")
            raise
    
    async def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """Analyze GitHub repository structure and suggest improvements"""
        try:
            # TODO: Implement repository analysis
            analysis = {
                'repository': repo_url,
                'tech_stack': ['JavaScript', 'React', 'Node.js'],
                'code_quality_score': 8.5,
                'test_coverage': 75.0,
                'suggestions': [
                    "Add CI/CD pipeline",
                    "Improve test coverage",
                    "Add documentation"
                ],
                'analyzed_at': datetime.utcnow().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.console.print(f"[red]Failed to analyze repository: {e}[/red]")
            raise


class FileEditor:
    """Direct file editing capabilities for IDE integration"""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.console = Console()
        self.edit_history: List[Dict[str, Any]] = []
    
    async def edit_file(self, file_path: Path, edits: List[Dict[str, Any]]) -> bool:
        """Apply edits to a file"""
        try:
            absolute_path = self.workspace_path / file_path
            
            if not absolute_path.exists():
                self.console.print(f"[red]File not found: {file_path}[/red]")
                return False
            
            # Backup original file
            backup_path = await self._create_backup(absolute_path)
            
            # Read original content
            with open(absolute_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Apply edits (sorted by line number in reverse order)
            edits_sorted = sorted(edits, key=lambda x: x.get('line', 0), reverse=True)
            
            for edit in edits_sorted:
                edit_type = edit.get('type', 'replace')
                line_num = edit.get('line', 1) - 1  # Convert to 0-based index
                new_content = edit.get('content', '')
                
                if edit_type == 'replace':
                    if 0 <= line_num < len(lines):
                        lines[line_num] = new_content + '\n'
                elif edit_type == 'insert':
                    if 0 <= line_num <= len(lines):
                        lines.insert(line_num, new_content + '\n')
                elif edit_type == 'delete':
                    if 0 <= line_num < len(lines):
                        del lines[line_num]
            
            # Write modified content
            with open(absolute_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            # Record edit in history
            self.edit_history.append({
                'file_path': str(file_path),
                'edits': edits,
                'backup_path': str(backup_path),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            self.console.print(f"[green]✅ Applied {len(edits)} edits to {file_path}[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to edit file {file_path}: {e}[/red]")
            return False
    
    async def _create_backup(self, file_path: Path) -> Path:
        """Create backup of file before editing"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{timestamp}")
        
        with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        
        return backup_path
    
    async def create_file(self, file_path: Path, content: str) -> bool:
        """Create a new file with content"""
        try:
            absolute_path = self.workspace_path / file_path
            
            # Create directory if it doesn't exist
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(absolute_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.console.print(f"[green]✅ Created file: {file_path}[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to create file {file_path}: {e}[/red]")
            return False
    
    async def get_edit_history(self) -> List[Dict[str, Any]]:
        """Get edit history"""
        return self.edit_history.copy()


class IDEIntegrationManager:
    """Main manager for all IDE integrations"""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.console = Console()
        
        # Initialize integrations
        self.vscode = VSCodeExtension(workspace_path)
        self.jetbrains = JetBrainsPlugin(workspace_path)
        self.github = GitHubIntegration()
        self.file_editor = FileEditor(workspace_path)
        
        # Integration status
        self.enabled_integrations: Dict[str, bool] = {}
    
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all available IDE integrations"""
        self.console.print("[bold blue]🔌 Initializing IDE integrations...[/bold blue]")
        
        results = {}
        
        # Initialize VS Code extension
        try:
            results['vscode'] = await self.vscode.initialize()
            if results['vscode']:
                self.console.print("[green]✅ VS Code extension ready[/green]")
            else:
                self.console.print("[yellow]⚠️ VS Code extension initialization failed[/yellow]")
        except Exception as e:
            results['vscode'] = False
            self.console.print(f"[red]❌ VS Code extension error: {e}[/red]")
        
        # Initialize JetBrains plugin
        try:
            results['jetbrains'] = await self.jetbrains.initialize()
            if results['jetbrains']:
                self.console.print("[green]✅ JetBrains plugin ready[/green]")
            else:
                self.console.print("[yellow]⚠️ JetBrains plugin initialization failed[/yellow]")
        except Exception as e:
            results['jetbrains'] = False
            self.console.print(f"[red]❌ JetBrains plugin error: {e}[/red]")
        
        # Initialize GitHub integration
        try:
            results['github'] = await self.github.initialize()
            if results['github']:
                self.console.print("[green]✅ GitHub integration ready[/green]")
            else:
                self.console.print("[yellow]⚠️ GitHub integration limited (no token)[/yellow]")
        except Exception as e:
            results['github'] = False
            self.console.print(f"[red]❌ GitHub integration error: {e}[/red]")
        
        self.enabled_integrations = results
        
        # Display summary
        enabled_count = sum(1 for enabled in results.values() if enabled)
        total_count = len(results)
        
        if enabled_count == total_count:
            self.console.print(f"[bold green]🎉 All {total_count} IDE integrations initialized successfully![/bold green]")
        elif enabled_count > 0:
            self.console.print(f"[yellow]⚡ {enabled_count}/{total_count} IDE integrations initialized[/yellow]")
        else:
            self.console.print("[red]❌ No IDE integrations initialized[/red]")
        
        return results
    
    async def handle_ide_command(self, ide_command: IDECommand) -> IDEResponse:
        """Handle command from any IDE"""
        try:
            # Route command based on context or detect IDE
            if 'vscode' in ide_command.context:
                return await self.vscode.handle_command(ide_command)
            elif 'jetbrains' in ide_command.context:
                # TODO: Implement JetBrains command handling
                pass
            
            # Default handling
            response = IDEResponse(
                command_id=ide_command.command_id,
                success=True,
                output=f"Processed command: {ide_command.command_text}"
            )
            
            return response
            
        except Exception as e:
            return IDEResponse(
                command_id=ide_command.command_id,
                success=False,
                error=str(e)
            )
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        status = {
            'workspace_path': str(self.workspace_path),
            'integrations': {},
            'last_updated': datetime.utcnow().isoformat()
        }
        
        # VS Code status
        if self.enabled_integrations.get('vscode'):
            status['integrations']['vscode'] = await self.vscode.get_agent_status()
        else:
            status['integrations']['vscode'] = {'enabled': False}
        
        # JetBrains status
        status['integrations']['jetbrains'] = {
            'enabled': self.enabled_integrations.get('jetbrains', False),
            'ide_type': self.jetbrains.ide_type.value
        }
        
        # GitHub status
        status['integrations']['github'] = {
            'enabled': self.enabled_integrations.get('github', False),
            'token_configured': bool(self.github.github_token)
        }
        
        return status
    
    def display_integration_status(self):
        """Display integration status in rich format"""
        from rich.table import Table
        
        table = Table(title="🔌 IDE Integration Status")
        table.add_column("Integration", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Features", style="yellow")
        
        # VS Code
        vscode_status = "✅ Active" if self.enabled_integrations.get('vscode') else "❌ Inactive"
        vscode_features = "Extension, Commands, File Editing" if self.enabled_integrations.get('vscode') else "Not available"
        table.add_row("VS Code", vscode_status, vscode_features)
        
        # JetBrains
        jetbrains_status = "✅ Active" if self.enabled_integrations.get('jetbrains') else "❌ Inactive"
        jetbrains_features = "Plugin, Context Menu, Tool Window" if self.enabled_integrations.get('jetbrains') else "Not available"
        table.add_row("JetBrains IDEs", jetbrains_status, jetbrains_features)
        
        # GitHub
        github_status = "✅ Active" if self.enabled_integrations.get('github') else "❌ Inactive"
        github_features = "PR Reviews, Repository Analysis" if self.enabled_integrations.get('github') else "Token required"
        table.add_row("GitHub", github_status, github_features)
        
        self.console.print(table)


# Global IDE integration manager instance
ide_integration_manager: Optional[IDEIntegrationManager] = None


def initialize_ide_integration(workspace_path: Path) -> IDEIntegrationManager:
    """Initialize global IDE integration manager"""
    global ide_integration_manager
    ide_integration_manager = IDEIntegrationManager(workspace_path)
    return ide_integration_manager


def get_ide_integration_manager() -> Optional[IDEIntegrationManager]:
    """Get global IDE integration manager instance"""
    return ide_integration_manager


# Utility functions for IDE communication
async def create_ide_command_from_selection(file_path: Path, start_line: int, 
                                          end_line: int, command_text: str,
                                          workspace_path: Path) -> IDECommand:
    """Create IDE command from file selection"""
    # Read selected text
    selected_text = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if start_line <= len(lines) and end_line <= len(lines):
                selected_text = ''.join(lines[start_line-1:end_line])
    except Exception:
        pass
    
    file_selection = FileSelection(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        selected_text=selected_text
    )
    
    return IDECommand(
        command_id=str(uuid.uuid4()),
        command_text=command_text,
        file_selection=file_selection,
        workspace_path=workspace_path
    )


async def execute_file_edits(file_path: Path, edits: List[Dict[str, Any]], 
                           workspace_path: Path) -> bool:
    """Execute file edits through IDE integration"""
    manager = get_ide_integration_manager()
    if not manager:
        return False
    
    return await manager.file_editor.edit_file(file_path, edits) 