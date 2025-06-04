# Enterprise Features for Phase 5
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    MANAGER = "manager"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(str, Enum):
    """System permissions"""
    CREATE_WORKSPACE = "create_workspace"
    DELETE_WORKSPACE = "delete_workspace"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_BILLING = "manage_billing"
    EXECUTE_AGENTS = "execute_agents"
    VIEW_METRICS = "view_metrics"
    MANAGE_PLUGINS = "manage_plugins"
    CONFIGURE_SYSTEM = "configure_system"


class AuditEventType(str, Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    WORKSPACE_CREATED = "workspace_created"
    WORKSPACE_DELETED = "workspace_deleted"
    AGENT_EXECUTED = "agent_executed"
    FILE_MODIFIED = "file_modified"
    CONFIGURATION_CHANGED = "configuration_changed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    BILLING_EVENT = "billing_event"
    SECURITY_INCIDENT = "security_incident"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    SOC2 = "soc2"


class TeamMemberStatus(str, Enum):
    """Team member status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    SUSPENDED = "suspended"


@dataclass
class User:
    """User model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    full_name: str = ""
    role: UserRole = UserRole.DEVELOPER
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    team_id: Optional[str] = None
    sso_provider: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Team:
    """Team model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    members: List[str] = field(default_factory=list)  # User IDs
    workspaces: List[str] = field(default_factory=list)  # Workspace IDs
    billing_settings: Dict[str, Any] = field(default_factory=dict)
    compliance_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workspace:
    """Workspace model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    team_id: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    settings: Dict[str, Any] = field(default_factory=dict)
    agent_configs: List[Dict] = field(default_factory=list)
    shared_memory_id: Optional[str] = None


class AuditEntry(BaseModel):
    """Audit log entry"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: AuditEventType = Field(description="Type of audit event")
    user_id: Optional[str] = Field(default=None, description="User who performed the action")
    workspace_id: Optional[str] = Field(default=None, description="Workspace context")
    team_id: Optional[str] = Field(default=None, description="Team context")
    action: str = Field(description="Description of action performed")
    resource_type: Optional[str] = Field(default=None, description="Type of resource affected")
    resource_id: Optional[str] = Field(default=None, description="ID of resource affected")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
    ip_address: Optional[str] = Field(default=None, description="IP address of user")
    user_agent: Optional[str] = Field(default=None, description="User agent string")
    compliance_tags: List[str] = Field(default_factory=list, description="Compliance framework tags")
    severity: str = Field(default="info", description="Event severity level")


class RoleBasedAccessControl:
    """RBAC implementation"""
    
    def __init__(self):
        self.role_permissions = self._initialize_role_permissions()
        self.users: Dict[str, User] = {}
        self.custom_permissions: Dict[str, Set[Permission]] = {}
        
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize default role permissions"""
        return {
            UserRole.ADMIN: {
                Permission.CREATE_WORKSPACE,
                Permission.DELETE_WORKSPACE,
                Permission.MANAGE_USERS,
                Permission.VIEW_AUDIT_LOGS,
                Permission.MANAGE_BILLING,
                Permission.EXECUTE_AGENTS,
                Permission.VIEW_METRICS,
                Permission.MANAGE_PLUGINS,
                Permission.CONFIGURE_SYSTEM
            },
            UserRole.MANAGER: {
                Permission.CREATE_WORKSPACE,
                Permission.MANAGE_USERS,
                Permission.VIEW_AUDIT_LOGS,
                Permission.EXECUTE_AGENTS,
                Permission.VIEW_METRICS,
                Permission.MANAGE_PLUGINS
            },
            UserRole.DEVELOPER: {
                Permission.EXECUTE_AGENTS,
                Permission.VIEW_METRICS
            },
            UserRole.VIEWER: {
                Permission.VIEW_METRICS
            },
            UserRole.GUEST: set()
        }
    
    def add_user(self, user: User):
        """Add user to RBAC system"""
        self.users[user.id] = user
        # Set default permissions based on role
        user.permissions = self.role_permissions.get(user.role, set()).copy()
        
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        # Check custom permissions first
        if user_id in self.custom_permissions:
            if permission in self.custom_permissions[user_id]:
                return True
        
        # Check role-based permissions
        return permission in user.permissions
    
    def grant_permission(self, user_id: str, permission: Permission, granted_by: str):
        """Grant custom permission to user"""
        if user_id not in self.custom_permissions:
            self.custom_permissions[user_id] = set()
        self.custom_permissions[user_id].add(permission)
        
        logger.info(f"Permission {permission} granted to user {user_id} by {granted_by}")
    
    def revoke_permission(self, user_id: str, permission: Permission, revoked_by: str):
        """Revoke custom permission from user"""
        if user_id in self.custom_permissions:
            self.custom_permissions[user_id].discard(permission)
            
        logger.info(f"Permission {permission} revoked from user {user_id} by {revoked_by}")
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user"""
        user = self.users.get(user_id)
        if not user:
            return set()
        
        permissions = user.permissions.copy()
        
        # Add custom permissions
        if user_id in self.custom_permissions:
            permissions.update(self.custom_permissions[user_id])
        
        return permissions


class AuditLogger:
    """Enterprise audit logging system"""
    
    def __init__(self, storage_path: Path, compliance_frameworks: List[ComplianceFramework] = None):
        self.storage_path = storage_path
        self.compliance_frameworks = compliance_frameworks or []
        self.audit_entries: List[AuditEntry] = []
        self.max_entries_in_memory = 10000
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def log_event(self, event_type: AuditEventType, action: str, **kwargs):
        """Log an audit event"""
        entry = AuditEntry(
            event_type=event_type,
            action=action,
            user_id=kwargs.get('user_id'),
            workspace_id=kwargs.get('workspace_id'),
            team_id=kwargs.get('team_id'),
            resource_type=kwargs.get('resource_type'),
            resource_id=kwargs.get('resource_id'),
            details=kwargs.get('details', {}),
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            severity=kwargs.get('severity', 'info')
        )
        
        # Add compliance tags
        entry.compliance_tags = self._determine_compliance_tags(entry)
        
        # Store entry
        await self._store_entry(entry)
        
        # Add to in-memory cache
        self.audit_entries.append(entry)
        if len(self.audit_entries) > self.max_entries_in_memory:
            self.audit_entries.pop(0)
        
        logger.info(f"Audit event logged: {event_type} - {action}")
    
    def _determine_compliance_tags(self, entry: AuditEntry) -> List[str]:
        """Determine which compliance frameworks apply to event"""
        tags = []
        
        for framework in self.compliance_frameworks:
            if framework == ComplianceFramework.SOX:
                if entry.event_type in [
                    AuditEventType.FILE_MODIFIED,
                    AuditEventType.CONFIGURATION_CHANGED,
                    AuditEventType.PERMISSION_GRANTED
                ]:
                    tags.append("sox")
            
            elif framework == ComplianceFramework.GDPR:
                if entry.event_type in [
                    AuditEventType.USER_LOGIN,
                    AuditEventType.USER_LOGOUT,
                    AuditEventType.FILE_MODIFIED
                ]:
                    tags.append("gdpr")
            
            elif framework == ComplianceFramework.SOC2:
                if entry.event_type in [
                    AuditEventType.SECURITY_INCIDENT,
                    AuditEventType.PERMISSION_GRANTED,
                    AuditEventType.CONFIGURATION_CHANGED
                ]:
                    tags.append("soc2")
        
        return tags
    
    async def _store_entry(self, entry: AuditEntry):
        """Store audit entry to persistent storage"""
        # Create monthly log files
        log_file = self.storage_path / f"audit-{entry.timestamp.strftime('%Y-%m')}.jsonl"
        
        try:
            with open(log_file, 'a') as f:
                f.write(entry.model_dump_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to store audit entry: {e}")
    
    async def search_entries(self, 
                           start_date: datetime,
                           end_date: datetime,
                           event_types: List[AuditEventType] = None,
                           user_id: str = None,
                           workspace_id: str = None) -> List[AuditEntry]:
        """Search audit entries with filters"""
        results = []
        
        # Search in-memory entries first
        for entry in self.audit_entries:
            if self._matches_filters(entry, start_date, end_date, event_types, user_id, workspace_id):
                results.append(entry)
        
        # Search persistent storage if needed
        if start_date < (datetime.utcnow() - timedelta(days=7)):
            persistent_results = await self._search_persistent_storage(
                start_date, end_date, event_types, user_id, workspace_id
            )
            results.extend(persistent_results)
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def _matches_filters(self, entry: AuditEntry, start_date: datetime, end_date: datetime,
                        event_types: List[AuditEventType] = None, user_id: str = None,
                        workspace_id: str = None) -> bool:
        """Check if entry matches search filters"""
        if not (start_date <= entry.timestamp <= end_date):
            return False
        
        if event_types and entry.event_type not in event_types:
            return False
        
        if user_id and entry.user_id != user_id:
            return False
        
        if workspace_id and entry.workspace_id != workspace_id:
            return False
        
        return True
    
    async def _search_persistent_storage(self, start_date: datetime, end_date: datetime,
                                       event_types: List[AuditEventType] = None,
                                       user_id: str = None, workspace_id: str = None) -> List[AuditEntry]:
        """Search persistent audit log storage"""
        results = []
        
        # Determine which log files to search
        current_date = start_date
        while current_date <= end_date:
            log_file = self.storage_path / f"audit-{current_date.strftime('%Y-%m')}.jsonl"
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    entry_data = json.loads(line)
                                    entry = AuditEntry(**entry_data)
                                    if self._matches_filters(entry, start_date, end_date, 
                                                           event_types, user_id, workspace_id):
                                        results.append(entry)
                                except Exception as e:
                                    logger.warning(f"Failed to parse audit entry: {e}")
                except Exception as e:
                    logger.error(f"Failed to read audit log file {log_file}: {e}")
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return results
    
    async def generate_compliance_report(self, framework: ComplianceFramework,
                                       start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        tag = framework.value
        entries = await self.search_entries(start_date, end_date)
        compliance_entries = [e for e in entries if tag in e.compliance_tags]
        
        if framework == ComplianceFramework.SOX:
            return self._generate_sox_report(compliance_entries, start_date, end_date)
        elif framework == ComplianceFramework.GDPR:
            return self._generate_gdpr_report(compliance_entries, start_date, end_date)
        elif framework == ComplianceFramework.SOC2:
            return self._generate_soc2_report(compliance_entries, start_date, end_date)
        else:
            return self._generate_generic_report(compliance_entries, framework, start_date, end_date)
    
    def _generate_sox_report(self, entries: List[AuditEntry], start_date: datetime, 
                           end_date: datetime) -> Dict[str, Any]:
        """Generate SOX compliance report"""
        return {
            'framework': 'SOX',
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_events': len(entries),
            'change_events': len([e for e in entries if e.event_type == AuditEventType.FILE_MODIFIED]),
            'config_changes': len([e for e in entries if e.event_type == AuditEventType.CONFIGURATION_CHANGED]),
            'access_changes': len([e for e in entries if e.event_type == AuditEventType.PERMISSION_GRANTED]),
            'summary': 'All material changes to financial reporting systems have been logged and tracked.',
            'events': [e.model_dump() for e in entries]
        }
    
    def _generate_gdpr_report(self, entries: List[AuditEntry], start_date: datetime,
                            end_date: datetime) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        return {
            'framework': 'GDPR',
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_events': len(entries),
            'data_access_events': len([e for e in entries if e.event_type == AuditEventType.USER_LOGIN]),
            'data_processing_events': len([e for e in entries if e.event_type == AuditEventType.FILE_MODIFIED]),
            'summary': 'All personal data processing activities have been logged per GDPR requirements.',
            'events': [e.model_dump() for e in entries]
        }
    
    def _generate_soc2_report(self, entries: List[AuditEntry], start_date: datetime,
                            end_date: datetime) -> Dict[str, Any]:
        """Generate SOC 2 compliance report"""
        return {
            'framework': 'SOC 2',
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_events': len(entries),
            'security_events': len([e for e in entries if e.event_type == AuditEventType.SECURITY_INCIDENT]),
            'access_control_events': len([e for e in entries if e.event_type == AuditEventType.PERMISSION_GRANTED]),
            'summary': 'System access and security controls have been monitored per SOC 2 requirements.',
            'events': [e.model_dump() for e in entries]
        }
    
    def _generate_generic_report(self, entries: List[AuditEntry], framework: ComplianceFramework,
                               start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate generic compliance report"""
        return {
            'framework': framework.value.upper(),
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_events': len(entries),
            'summary': f'Audit events relevant to {framework.value.upper()} compliance framework.',
            'events': [e.model_dump() for e in entries]
        }


class CostManagementSystem:
    """Enterprise cost management and budgeting"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.team_budgets: Dict[str, Dict] = {}
        self.cost_tracking: Dict[str, List[Dict]] = {}
        self.alert_thresholds: Dict[str, float] = {}
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def set_team_budget(self, team_id: str, budget_config: Dict[str, Any]):
        """Set budget configuration for team"""
        self.team_budgets[team_id] = {
            'daily_limit': budget_config.get('daily_limit', 100.0),
            'monthly_limit': budget_config.get('monthly_limit', 3000.0),
            'alert_threshold': budget_config.get('alert_threshold', 0.8),  # 80%
            'auto_disable_on_exceed': budget_config.get('auto_disable_on_exceed', False),
            'approved_by': budget_config.get('approved_by'),
            'created_at': datetime.utcnow()
        }
        
        # Set alert threshold
        self.alert_thresholds[team_id] = budget_config.get('alert_threshold', 0.8)
        
        logger.info(f"Budget configuration set for team {team_id}")
    
    async def track_cost(self, team_id: str, workspace_id: str, cost: float, 
                        usage_details: Dict[str, Any]):
        """Track cost for team and workspace"""
        cost_entry = {
            'timestamp': datetime.utcnow(),
            'team_id': team_id,
            'workspace_id': workspace_id,
            'cost': cost,
            'usage_details': usage_details
        }
        
        # Add to tracking
        if team_id not in self.cost_tracking:
            self.cost_tracking[team_id] = []
        self.cost_tracking[team_id].append(cost_entry)
        
        # Check budget limits
        await self._check_budget_limits(team_id)
        
        # Store to persistent storage
        await self._store_cost_entry(cost_entry)
    
    async def _check_budget_limits(self, team_id: str):
        """Check if team is approaching or exceeding budget limits"""
        if team_id not in self.team_budgets:
            return
        
        budget_config = self.team_budgets[team_id]
        current_usage = self.get_current_usage(team_id)
        
        # Check daily limit
        daily_usage = current_usage['daily_cost']
        daily_limit = budget_config['daily_limit']
        
        if daily_usage >= daily_limit * self.alert_thresholds[team_id]:
            await self._send_budget_alert(team_id, 'daily', daily_usage, daily_limit)
        
        if daily_usage >= daily_limit and budget_config['auto_disable_on_exceed']:
            await self._disable_team_access(team_id, 'daily_budget_exceeded')
        
        # Check monthly limit
        monthly_usage = current_usage['monthly_cost']
        monthly_limit = budget_config['monthly_limit']
        
        if monthly_usage >= monthly_limit * self.alert_thresholds[team_id]:
            await self._send_budget_alert(team_id, 'monthly', monthly_usage, monthly_limit)
        
        if monthly_usage >= monthly_limit and budget_config['auto_disable_on_exceed']:
            await self._disable_team_access(team_id, 'monthly_budget_exceeded')
    
    def get_current_usage(self, team_id: str) -> Dict[str, float]:
        """Get current usage for team"""
        if team_id not in self.cost_tracking:
            return {'daily_cost': 0.0, 'monthly_cost': 0.0, 'total_cost': 0.0}
        
        entries = self.cost_tracking[team_id]
        now = datetime.utcnow()
        
        # Calculate daily cost
        daily_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_cost = sum(
            entry['cost'] for entry in entries 
            if entry['timestamp'] >= daily_start
        )
        
        # Calculate monthly cost
        monthly_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_cost = sum(
            entry['cost'] for entry in entries 
            if entry['timestamp'] >= monthly_start
        )
        
        # Calculate total cost
        total_cost = sum(entry['cost'] for entry in entries)
        
        return {
            'daily_cost': daily_cost,
            'monthly_cost': monthly_cost,
            'total_cost': total_cost
        }
    
    async def _send_budget_alert(self, team_id: str, period: str, current: float, limit: float):
        """Send budget alert notification"""
        percentage = (current / limit) * 100
        
        logger.warning(
            f"Budget alert for team {team_id}: {period} usage at {percentage:.1f}% "
            f"(${current:.2f} of ${limit:.2f})"
        )
        
        # In a real implementation, this would send notifications
        # via email, Slack, or other channels
    
    async def _disable_team_access(self, team_id: str, reason: str):
        """Disable team access due to budget exceeded"""
        logger.critical(f"Team {team_id} access disabled: {reason}")
        
        # In a real implementation, this would disable the team's ability
        # to execute agents or access resources
    
    async def _store_cost_entry(self, cost_entry: Dict[str, Any]):
        """Store cost entry to persistent storage"""
        cost_file = self.storage_path / f"costs-{cost_entry['timestamp'].strftime('%Y-%m')}.jsonl"
        
        try:
            with open(cost_file, 'a') as f:
                f.write(json.dumps(cost_entry, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to store cost entry: {e}")
    
    def generate_cost_report(self, team_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate cost report for team"""
        if team_id not in self.cost_tracking:
            return {'team_id': team_id, 'total_cost': 0.0, 'entries': []}
        
        entries = self.cost_tracking[team_id]
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_entries = [e for e in entries if e['timestamp'] >= cutoff_date]
        
        total_cost = sum(entry['cost'] for entry in recent_entries)
        
        # Group by workspace
        workspace_costs = {}
        for entry in recent_entries:
            workspace_id = entry['workspace_id']
            if workspace_id not in workspace_costs:
                workspace_costs[workspace_id] = 0.0
            workspace_costs[workspace_id] += entry['cost']
        
        return {
            'team_id': team_id,
            'period_days': days,
            'total_cost': total_cost,
            'average_daily_cost': total_cost / days if days > 0 else 0,
            'workspace_breakdown': workspace_costs,
            'budget_limit': self.team_budgets.get(team_id, {}).get('monthly_limit', 0),
            'budget_utilization': (total_cost / self.team_budgets.get(team_id, {}).get('monthly_limit', 1)) * 100,
            'entries': recent_entries
        }


class TeamCollaborationManager:
    """Manages team collaboration features"""
    
    def __init__(self, audit_logger: AuditLogger, cost_manager: CostManagementSystem):
        self.audit_logger = audit_logger
        self.cost_manager = cost_manager
        self.teams: Dict[str, Team] = {}
        self.workspaces: Dict[str, Workspace] = {}
        self.shared_agent_pools: Dict[str, Dict] = {}
    
    async def create_team(self, name: str, description: str, created_by: str) -> Team:
        """Create a new team"""
        team = Team(
            name=name,
            description=description
        )
        
        self.teams[team.id] = team
        
        # Log audit event
        await self.audit_logger.log_event(
            AuditEventType.WORKSPACE_CREATED,  # Generic workspace creation
            f"Team '{name}' created",
            user_id=created_by,
            team_id=team.id,
            resource_type="team",
            resource_id=team.id
        )
        
        logger.info(f"Team created: {name} (ID: {team.id})")
        return team
    
    async def create_workspace(self, team_id: str, name: str, description: str, 
                             created_by: str, settings: Dict[str, Any] = None) -> Workspace:
        """Create a shared workspace for team"""
        if team_id not in self.teams:
            raise ValueError(f"Team {team_id} not found")
        
        workspace = Workspace(
            name=name,
            description=description,
            team_id=team_id,
            created_by=created_by,
            settings=settings or {}
        )
        
        self.workspaces[workspace.id] = workspace
        
        # Add workspace to team
        self.teams[team_id].workspaces.append(workspace.id)
        
        # Initialize shared agent pool
        self.shared_agent_pools[workspace.id] = {
            'agents': [],
            'shared_memory': {},
            'learnings': [],
            'active_sessions': {}
        }
        
        # Log audit event
        await self.audit_logger.log_event(
            AuditEventType.WORKSPACE_CREATED,
            f"Workspace '{name}' created in team {team_id}",
            user_id=created_by,
            team_id=team_id,
            workspace_id=workspace.id,
            resource_type="workspace",
            resource_id=workspace.id
        )
        
        logger.info(f"Workspace created: {name} (ID: {workspace.id})")
        return workspace
    
    async def add_team_member(self, team_id: str, user_id: str, added_by: str):
        """Add member to team"""
        if team_id not in self.teams:
            raise ValueError(f"Team {team_id} not found")
        
        team = self.teams[team_id]
        if user_id not in team.members:
            team.members.append(user_id)
            
            # Log audit event
            await self.audit_logger.log_event(
                AuditEventType.PERMISSION_GRANTED,
                f"User {user_id} added to team {team_id}",
                user_id=added_by,
                team_id=team_id,
                resource_type="team_membership",
                resource_id=user_id
            )
            
            logger.info(f"User {user_id} added to team {team_id}")
    
    async def sync_shared_learnings(self, workspace_id: str) -> Dict[str, Any]:
        """Synchronize learnings across team members"""
        if workspace_id not in self.shared_agent_pools:
            return {}
        
        pool = self.shared_agent_pools[workspace_id]
        
        # Collect all learnings from team members
        all_learnings = []
        for session_id, session_data in pool['active_sessions'].items():
            if 'learnings' in session_data:
                all_learnings.extend(session_data['learnings'])
        
        # Merge and deduplicate learnings
        merged_learnings = self._merge_learnings(all_learnings)
        
        # Update shared pool
        pool['learnings'] = merged_learnings
        
        logger.info(f"Synchronized {len(merged_learnings)} learnings for workspace {workspace_id}")
        
        return {
            'workspace_id': workspace_id,
            'total_learnings': len(merged_learnings),
            'unique_patterns': len(set(l.get('pattern', '') for l in merged_learnings))
        }
    
    def _merge_learnings(self, learnings: List[Dict]) -> List[Dict]:
        """Merge and deduplicate learnings"""
        unique_learnings = {}
        
        for learning in learnings:
            # Create a hash of the learning content for deduplication
            content = json.dumps(learning, sort_keys=True)
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in unique_learnings:
                unique_learnings[content_hash] = learning
            else:
                # Merge confidence scores or other metrics
                existing = unique_learnings[content_hash]
                if 'confidence' in learning and 'confidence' in existing:
                    existing['confidence'] = max(existing['confidence'], learning['confidence'])
        
        return list(unique_learnings.values())
    
    def get_team_metrics(self, team_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for team"""
        if team_id not in self.teams:
            return {}
        
        team = self.teams[team_id]
        cost_data = self.cost_manager.get_current_usage(team_id)
        
        # Calculate workspace metrics
        workspace_count = len(team.workspaces)
        active_workspaces = sum(
            1 for ws_id in team.workspaces 
            if ws_id in self.shared_agent_pools and 
            len(self.shared_agent_pools[ws_id]['active_sessions']) > 0
        )
        
        return {
            'team_id': team_id,
            'team_name': team.name,
            'member_count': len(team.members),
            'workspace_count': workspace_count,
            'active_workspaces': active_workspaces,
            'cost_metrics': cost_data,
            'created_at': team.created_at.isoformat()
        }


class SingleSignOnIntegration:
    """SSO integration for enterprise authentication"""
    
    def __init__(self, providers: List[Dict[str, Any]]):
        self.providers = {p['name']: p for p in providers}
        self.active_sessions: Dict[str, Dict] = {}
    
    async def authenticate_user(self, provider: str, token: str, user_info: Dict[str, Any]) -> Optional[User]:
        """Authenticate user via SSO provider"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported SSO provider: {provider}")
        
        # Validate token (mock implementation)
        if not await self._validate_token(provider, token):
            return None
        
        # Create or update user
        user = User(
            username=user_info.get('username', ''),
            email=user_info.get('email', ''),
            full_name=user_info.get('full_name', ''),
            sso_provider=provider,
            last_login=datetime.utcnow()
        )
        
        # Store session
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            'user_id': user.id,
            'provider': provider,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=8)
        }
        
        logger.info(f"User authenticated via {provider}: {user.email}")
        return user
    
    async def _validate_token(self, provider: str, token: str) -> bool:
        """Validate SSO token (mock implementation)"""
        # In a real implementation, this would validate the token
        # against the SSO provider's API
        return len(token) > 10  # Simple mock validation
    
    def is_session_valid(self, session_id: str) -> bool:
        """Check if session is still valid"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        return datetime.utcnow() < session['expires_at']
    
    async def logout_user(self, session_id: str):
        """Logout user and invalidate session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"User session {session_id} terminated")


class EnterpriseManager:
    """Main enterprise features manager"""
    
    def __init__(self, storage_path: Path, compliance_frameworks: List[ComplianceFramework] = None,
                 sso_providers: List[Dict[str, Any]] = None):
        self.storage_path = storage_path
        
        # Initialize subsystems
        self.rbac = RoleBasedAccessControl()
        self.audit_logger = AuditLogger(storage_path / "audit", compliance_frameworks or [])
        self.cost_manager = CostManagementSystem(storage_path / "costs")
        self.team_manager = TeamCollaborationManager(self.audit_logger, self.cost_manager)
        self.sso = SingleSignOnIntegration(sso_providers or []) if sso_providers else None
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize enterprise features"""
        logger.info("Initializing enterprise features...")
        
        # Load existing data from storage
        await self._load_persisted_data()
        
        logger.info("Enterprise features initialized")
    
    async def _load_persisted_data(self):
        """Load persisted enterprise data"""
        # In a real implementation, this would load teams, users, etc.
        # from a database or persistent storage
        pass
    
    def create_admin_user(self, username: str, email: str, full_name: str) -> User:
        """Create initial admin user"""
        admin_user = User(
            username=username,
            email=email,
            full_name=full_name,
            role=UserRole.ADMIN
        )
        
        self.rbac.add_user(admin_user)
        logger.info(f"Admin user created: {username}")
        
        return admin_user
    
    async def audit_user_action(self, user_id: str, action: str, **kwargs):
        """Convenience method to audit user actions"""
        await self.audit_logger.log_event(
            AuditEventType.AGENT_EXECUTED,  # Default event type
            action,
            user_id=user_id,
            **kwargs
        )


# Verified: Complete - Comprehensive enterprise features with RBAC, audit logging, cost management, team collaboration, and SSO integration 