"""
Phase 4: Performance Monitoring and Debugging
Comprehensive monitoring with cost tracking, health checks, and debugging tools
"""

import asyncio
import json
import uuid
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .configuration import PerformanceConfig


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics to track"""
    PERFORMANCE = "performance"
    COST = "cost" 
    HEALTH = "health"
    USAGE = "usage"
    ERROR = "error"


@dataclass
class Metrics:
    """System performance metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    avg_response_time: float = 0.0
    success_rate: float = 100.0
    active_agents: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_tasks: int = 0


@dataclass
class CostSummary:
    """Cost tracking summary"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_cost: float = 0.0
    hourly_rate: float = 0.0
    daily_estimate: float = 0.0
    monthly_estimate: float = 0.0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_agent: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_health: str = "healthy"  # healthy, degraded, unhealthy
    agent_health: Dict[str, bool] = field(default_factory=dict)
    system_health: Dict[str, bool] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Recommendation:
    """Performance/cost optimization recommendation"""
    type: str  # performance, cost, security, configuration
    title: str
    description: str
    priority: str  # low, medium, high, critical
    action: str
    estimated_savings: Optional[float] = None
    implementation_effort: str = "low"  # low, medium, high


class PerformanceReport(BaseModel):
    """Comprehensive performance report"""
    timestamp: datetime
    metrics: Metrics
    costs: CostSummary
    health: HealthStatus
    recommendations: List[Recommendation]
    session_duration: timedelta
    
    class Config:
        arbitrary_types_allowed = True


class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self):
        self.metrics_history: List[Metrics] = []
        self.current_metrics = Metrics()
        self.start_time = datetime.utcnow()
        self.request_times: List[float] = []
        self.is_collecting = False
        
    async def start(self):
        """Start metrics collection"""
        self.is_collecting = True
        while self.is_collecting:
            await self._collect_system_metrics()
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    def stop(self):
        """Stop metrics collection"""
        self.is_collecting = False
    
    async def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # System metrics
            self.current_metrics.memory_usage_mb = psutil.virtual_memory().used / 1024 / 1024
            self.current_metrics.cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            # Calculate derived metrics
            if self.request_times:
                self.current_metrics.avg_response_time = sum(self.request_times) / len(self.request_times)
                self.request_times = self.request_times[-100:]  # Keep last 100 requests
            
            if self.current_metrics.total_requests > 0:
                self.current_metrics.success_rate = (
                    (self.current_metrics.total_requests - self.current_metrics.failed_requests) 
                    / self.current_metrics.total_requests * 100
                )
            
            # Store in history
            self.metrics_history.append(Metrics(
                timestamp=datetime.utcnow(),
                avg_response_time=self.current_metrics.avg_response_time,
                success_rate=self.current_metrics.success_rate,
                active_agents=self.current_metrics.active_agents,
                total_requests=self.current_metrics.total_requests,
                failed_requests=self.current_metrics.failed_requests,
                memory_usage_mb=self.current_metrics.memory_usage_mb,
                cpu_usage_percent=self.current_metrics.cpu_usage_percent,
                cache_hit_rate=self.current_metrics.cache_hit_rate,
                concurrent_tasks=self.current_metrics.concurrent_tasks
            ))
            
            # Keep only last hour of metrics
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
        except Exception as e:
            # Log error but don't crash metrics collection
            print(f"Error collecting metrics: {e}")
    
    async def get_current_metrics(self) -> Metrics:
        """Get current metrics snapshot"""
        return self.current_metrics
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request for metrics"""
        self.request_times.append(response_time)
        self.current_metrics.total_requests += 1
        if not success:
            self.current_metrics.failed_requests += 1
    
    def update_agent_count(self, count: int):
        """Update active agent count"""
        self.current_metrics.active_agents = count
    
    def update_concurrent_tasks(self, count: int):
        """Update concurrent task count"""
        self.current_metrics.concurrent_tasks = count


class CostTracker:
    """Track AI model costs and usage"""
    
    def __init__(self):
        self.usage_log: List[Dict[str, Any]] = []
        self.current_session_cost = 0.0
        self.session_start = datetime.utcnow()
        
        # Model pricing (per million tokens)
        self.pricing = {
            'claude-4': {'input': 15.0, 'output': 75.0, 'thinking': 15.0},
            'claude-3.5-sonnet': {'input': 3.0, 'output': 15.0, 'thinking': 3.0},
            'claude-3-haiku': {'input': 0.25, 'output': 1.25, 'thinking': 0.25},
            'gpt-4': {'input': 30.0, 'output': 60.0, 'thinking': 30.0},
            'gpt-3.5-turbo': {'input': 0.5, 'output': 1.5, 'thinking': 0.5}
        }
    
    async def start(self):
        """Start cost tracking"""
        # Cost tracking is event-driven, no background task needed
    
    async def track_api_call(self, model: str, input_tokens: int, 
                           output_tokens: int, thinking_tokens: int = 0,
                           agent_id: Optional[str] = None):
        """Track individual API call costs"""
        cost = self._calculate_cost(model, input_tokens, output_tokens, thinking_tokens)
        
        usage_entry = {
            'timestamp': datetime.utcnow(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'thinking_tokens': thinking_tokens,
            'cost': cost,
            'agent_id': agent_id
        }
        
        self.usage_log.append(usage_entry)
        self.current_session_cost += cost
        
        # Alert if approaching limits
        hourly_cost = self._calculate_hourly_rate()
        if hourly_cost > 8.0:  # Warning threshold
            await self._send_cost_alert(hourly_cost)
    
    def _calculate_cost(self, model: str, input_tokens: int, 
                       output_tokens: int, thinking_tokens: int) -> float:
        """Calculate cost based on model pricing"""
        if model not in self.pricing:
            return 0.0  # Unknown model, no cost
        
        rates = self.pricing[model]
        cost = (
            (input_tokens / 1_000_000) * rates['input'] +
            (output_tokens / 1_000_000) * rates['output'] +
            (thinking_tokens / 1_000_000) * rates['thinking']
        )
        
        return cost
    
    def _calculate_hourly_rate(self) -> float:
        """Calculate current hourly spending rate"""
        session_duration = datetime.utcnow() - self.session_start
        hours = session_duration.total_seconds() / 3600
        
        if hours < 0.1:  # Less than 6 minutes, use session cost
            return self.current_session_cost * 10  # Rough extrapolation
        
        return self.current_session_cost / hours
    
    async def get_cost_summary(self) -> CostSummary:
        """Generate cost summary"""
        hourly_rate = self._calculate_hourly_rate()
        daily_estimate = hourly_rate * 24
        monthly_estimate = daily_estimate * 30
        
        # Calculate costs by model
        cost_by_model = {}
        cost_by_agent = {}
        total_tokens = 0
        
        for entry in self.usage_log:
            model = entry['model']
            agent_id = entry.get('agent_id', 'unknown')
            cost = entry['cost']
            tokens = entry['input_tokens'] + entry['output_tokens'] + entry['thinking_tokens']
            
            cost_by_model[model] = cost_by_model.get(model, 0) + cost
            cost_by_agent[agent_id] = cost_by_agent.get(agent_id, 0) + cost
            total_tokens += tokens
        
        return CostSummary(
            session_cost=self.current_session_cost,
            hourly_rate=hourly_rate,
            daily_estimate=daily_estimate,
            monthly_estimate=monthly_estimate,
            total_tokens_used=total_tokens,
            total_api_calls=len(self.usage_log),
            cost_by_model=cost_by_model,
            cost_by_agent=cost_by_agent
        )
    
    async def _send_cost_alert(self, hourly_rate: float):
        """Send cost alert when threshold exceeded"""
        # TODO: Implement actual alerting (email, Slack, etc.)
        print(f"⚠️  COST ALERT: Hourly rate ${hourly_rate:.2f} exceeds threshold")


class HealthChecker:
    """Monitor system and agent health"""
    
    def __init__(self):
        self.health_checks: List[Dict[str, Any]] = []
        self.agent_health: Dict[str, bool] = {}
        self.system_health: Dict[str, bool] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.is_monitoring = False
    
    async def start(self):
        """Start health monitoring"""
        self.is_monitoring = True
        while self.is_monitoring:
            await self._perform_health_checks()
            await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop health monitoring"""
        self.is_monitoring = False
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks"""
        try:
            # System health checks
            await self._check_system_resources()
            await self._check_api_connectivity()
            await self._check_database_health()
            
            # Clear old alerts
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            self.alerts = [
                alert for alert in self.alerts 
                if alert['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            self._add_alert(AlertLevel.ERROR, "Health Check Failed", str(e))
    
    async def _check_system_resources(self):
        """Check system resource usage"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Memory check
            memory_usage = memory.percent
            if memory_usage > 90:
                self.system_health['memory'] = False
                self._add_alert(AlertLevel.CRITICAL, "High Memory Usage", 
                              f"Memory usage at {memory_usage:.1f}%")
            elif memory_usage > 75:
                self.system_health['memory'] = False
                self._add_alert(AlertLevel.WARNING, "Elevated Memory Usage", 
                              f"Memory usage at {memory_usage:.1f}%")
            else:
                self.system_health['memory'] = True
            
            # Disk check
            disk_usage = disk.percent
            if disk_usage > 95:
                self.system_health['disk'] = False
                self._add_alert(AlertLevel.CRITICAL, "Disk Space Critical", 
                              f"Disk usage at {disk_usage:.1f}%")
            elif disk_usage > 85:
                self.system_health['disk'] = False
                self._add_alert(AlertLevel.WARNING, "Low Disk Space", 
                              f"Disk usage at {disk_usage:.1f}%")
            else:
                self.system_health['disk'] = True
                
        except Exception as e:
            self.system_health['resources'] = False
            self._add_alert(AlertLevel.ERROR, "Resource Check Failed", str(e))
    
    async def _check_api_connectivity(self):
        """Check API connectivity and response times"""
        # TODO: Implement actual API health checks
        self.system_health['api'] = True
    
    async def _check_database_health(self):
        """Check database connectivity and performance"""
        # TODO: Implement database health checks
        self.system_health['database'] = True
    
    def _add_alert(self, level: AlertLevel, title: str, message: str):
        """Add a new alert"""
        alert = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow(),
            'level': level,
            'title': title,
            'message': message
        }
        self.alerts.append(alert)
    
    async def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        # Determine overall health
        all_systems_healthy = all(self.system_health.values()) if self.system_health else True
        all_agents_healthy = all(self.agent_health.values()) if self.agent_health else True
        
        critical_alerts = [a for a in self.alerts if a['level'] == AlertLevel.CRITICAL]
        error_alerts = [a for a in self.alerts if a['level'] == AlertLevel.ERROR]
        
        if critical_alerts or not all_systems_healthy:
            overall_health = "unhealthy"
        elif error_alerts or not all_agents_healthy:
            overall_health = "degraded"
        else:
            overall_health = "healthy"
        
        return HealthStatus(
            overall_health=overall_health,
            agent_health=self.agent_health.copy(),
            system_health=self.system_health.copy(),
            alerts=self.alerts.copy()
        )
    
    def update_agent_health(self, agent_id: str, is_healthy: bool):
        """Update health status for an agent"""
        self.agent_health[agent_id] = is_healthy
        if not is_healthy:
            self._add_alert(AlertLevel.WARNING, f"Agent Unhealthy", 
                          f"Agent {agent_id} reported unhealthy status")


class DebugConsole:
    """Advanced debugging console with agent introspection"""
    
    def __init__(self):
        self.debug_sessions: Dict[str, Dict[str, Any]] = {}
        self.breakpoints: Dict[str, Dict[str, Any]] = {}
        self.console = Console()
        
    async def start_debug_session(self, agent_id: str) -> str:
        """Start debugging session for agent"""
        session_id = str(uuid.uuid4())
        
        self.debug_sessions[session_id] = {
            'agent_id': agent_id,
            'start_time': datetime.utcnow(),
            'events': [],
            'state_snapshots': []
        }
        
        return session_id
    
    async def log_debug_event(self, session_id: str, event_type: str, data: Dict[str, Any]):
        """Log debug event"""
        if session_id not in self.debug_sessions:
            return
        
        event = {
            'timestamp': datetime.utcnow(),
            'type': event_type,
            'data': data
        }
        
        self.debug_sessions[session_id]['events'].append(event)
    
    async def set_breakpoint(self, agent_id: str, condition: str) -> str:
        """Set conditional breakpoint for agent"""
        breakpoint_id = str(uuid.uuid4())
        
        self.breakpoints[breakpoint_id] = {
            'agent_id': agent_id,
            'condition': condition,
            'hit_count': 0,
            'created_at': datetime.utcnow()
        }
        
        return breakpoint_id
    
    async def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current agent state for debugging"""
        # TODO: Integrate with actual agent registry
        return {
            'status': 'active',
            'current_tasks': [],
            'memory_usage': 0,
            'last_activity': datetime.utcnow(),
            'configuration': {},
            'performance_metrics': {}
        }
    
    def display_debug_info(self, session_id: str):
        """Display debug information in rich format"""
        if session_id not in self.debug_sessions:
            self.console.print(f"[red]Debug session {session_id} not found[/red]")
            return
        
        session = self.debug_sessions[session_id]
        
        # Create debug info table
        table = Table(title=f"Debug Session: {session_id}")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Event Type", style="magenta")
        table.add_column("Data", style="green")
        
        for event in session['events'][-10:]:  # Show last 10 events
            timestamp = event['timestamp'].strftime("%H:%M:%S")
            event_type = event['type']
            data = json.dumps(event['data'], indent=2)
            
            table.add_row(timestamp, event_type, data)
        
        self.console.print(table)


class PerformanceMonitor:
    """Comprehensive performance monitoring"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.cost_tracker = CostTracker()
        self.health_checker = HealthChecker()
        self.debug_console = DebugConsole()
        self.console = Console()
        
    async def start_monitoring(self):
        """Start monitoring all systems"""
        await asyncio.gather(
            self.metrics_collector.start(),
            self.cost_tracker.start(),
            self.health_checker.start()
        )
    
    def stop_monitoring(self):
        """Stop all monitoring"""
        self.metrics_collector.stop()
        self.health_checker.stop()
    
    async def get_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
        metrics = await self.metrics_collector.get_current_metrics()
        costs = await self.cost_tracker.get_cost_summary()
        health = await self.health_checker.get_health_status()
        
        session_duration = datetime.utcnow() - self.metrics_collector.start_time
        recommendations = self._generate_recommendations(metrics, costs, health)
        
        return PerformanceReport(
            timestamp=datetime.utcnow(),
            metrics=metrics,
            costs=costs,
            health=health,
            recommendations=recommendations,
            session_duration=session_duration
        )
    
    def _generate_recommendations(self, metrics: Metrics, costs: CostSummary, 
                                health: HealthStatus) -> List[Recommendation]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Performance recommendations
        if metrics.avg_response_time > 30:
            recommendations.append(Recommendation(
                type="performance",
                title="Slow agent response times detected",
                description=f"Average response time is {metrics.avg_response_time:.1f}s. Consider reducing model complexity or enabling caching.",
                priority="medium",
                action="optimize_model_settings",
                implementation_effort="low"
            ))
        
        if metrics.success_rate < 95:
            recommendations.append(Recommendation(
                type="performance",
                title="Low success rate detected",
                description=f"Success rate is {metrics.success_rate:.1f}%. Review error logs and improve error handling.",
                priority="high",
                action="improve_error_handling",
                implementation_effort="medium"
            ))
        
        # Cost recommendations
        if costs.hourly_rate > self.config.max_concurrent_agents * 2:  # Rough threshold
            recommendations.append(Recommendation(
                type="cost",
                title="High hourly cost detected",
                description=f"Current hourly rate ${costs.hourly_rate:.2f} is high. Consider using more efficient models.",
                priority="medium",
                action="optimize_model_usage",
                estimated_savings=costs.hourly_rate * 0.3,
                implementation_effort="low"
            ))
        
        # Find most expensive model
        if costs.cost_by_model:
            most_expensive = max(costs.cost_by_model.items(), key=lambda x: x[1])
            if most_expensive[1] > costs.session_cost * 0.5:
                recommendations.append(Recommendation(
                    type="cost",
                    title=f"Model {most_expensive[0]} consuming high costs",
                    description=f"Model {most_expensive[0]} accounts for ${most_expensive[1]:.2f} of session costs. Consider using a more efficient model for simple tasks.",
                    priority="low",
                    action="review_model_selection",
                    estimated_savings=most_expensive[1] * 0.4,
                    implementation_effort="low"
                ))
        
        # Health recommendations
        if health.overall_health != "healthy":
            recommendations.append(Recommendation(
                type="health",
                title="System health issues detected",
                description="System is reporting health issues. Review alerts and system resources.",
                priority="high",
                action="investigate_health_issues",
                implementation_effort="medium"
            ))
        
        # Resource recommendations
        if metrics.memory_usage_mb > self.config.memory_limit_mb * 0.8:
            recommendations.append(Recommendation(
                type="performance",
                title="High memory usage",
                description=f"Memory usage is {metrics.memory_usage_mb:.0f}MB, approaching limit of {self.config.memory_limit_mb}MB.",
                priority="medium",
                action="optimize_memory_usage",
                implementation_effort="medium"
            ))
        
        return recommendations
    
    def display_performance_dashboard(self, report: PerformanceReport):
        """Display performance dashboard"""
        # Performance metrics panel
        metrics_table = Table(title="📊 Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Average Response Time", f"{report.metrics.avg_response_time:.1f}s")
        metrics_table.add_row("Success Rate", f"{report.metrics.success_rate:.1f}%")
        metrics_table.add_row("Active Agents", str(report.metrics.active_agents))
        metrics_table.add_row("Total Requests", str(report.metrics.total_requests))
        metrics_table.add_row("Memory Usage", f"{report.metrics.memory_usage_mb:.0f}MB")
        metrics_table.add_row("CPU Usage", f"{report.metrics.cpu_usage_percent:.1f}%")
        
        # Cost tracking panel
        cost_table = Table(title="💰 Cost Tracking")
        cost_table.add_column("Metric", style="cyan")
        cost_table.add_column("Value", style="yellow")
        
        cost_table.add_row("Session Cost", f"${report.costs.session_cost:.2f}")
        cost_table.add_row("Hourly Rate", f"${report.costs.hourly_rate:.2f}")
        cost_table.add_row("Daily Estimate", f"${report.costs.daily_estimate:.2f}")
        cost_table.add_row("Total API Calls", str(report.costs.total_api_calls))
        cost_table.add_row("Total Tokens", f"{report.costs.total_tokens_used:,}")
        
        # Health status panel
        health_status = "🟢 Healthy" if report.health.overall_health == "healthy" else "🟡 Degraded" if report.health.overall_health == "degraded" else "🔴 Unhealthy"
        health_panel = Panel(f"Overall Status: {health_status}", title="🏥 System Health")
        
        # Recommendations panel
        if report.recommendations:
            rec_text = "\n".join([
                f"• {rec.title} ({rec.priority} priority)"
                for rec in report.recommendations[:5]
            ])
        else:
            rec_text = "No recommendations at this time"
        
        rec_panel = Panel(rec_text, title="🚨 Recommendations")
        
        # Display all panels
        self.console.print(metrics_table)
        self.console.print()
        self.console.print(cost_table)
        self.console.print()
        self.console.print(health_panel)
        self.console.print()
        self.console.print(rec_panel)


# Global performance monitor instance
performance_monitor: Optional[PerformanceMonitor] = None


def initialize_monitoring(config: PerformanceConfig) -> PerformanceMonitor:
    """Initialize global performance monitoring"""
    global performance_monitor
    performance_monitor = PerformanceMonitor(config)
    return performance_monitor


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get global performance monitor instance"""
    return performance_monitor 