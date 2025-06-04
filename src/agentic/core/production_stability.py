"""Production stability and performance monitoring system."""

import asyncio
import gc
import logging
import psutil
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    RECOVERING = "recovering"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categorization for better handling."""
    TRANSIENT = "transient"  # Network timeouts, temporary failures
    PERMANENT = "permanent"  # Invalid config, authentication failures
    RESOURCE = "resource"    # Memory, disk, CPU exhaustion
    EXTERNAL = "external"    # Third-party service failures
    INTERNAL = "internal"    # Code bugs, unexpected states


@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    open_connections: int
    active_requests: int
    error_rate: float
    response_time_p95: float
    uptime_seconds: int
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def is_healthy(self) -> bool:
        """Check if metrics indicate healthy system."""
        return (
            self.cpu_usage < 80.0 and
            self.memory_usage < 85.0 and
            self.disk_usage < 90.0 and
            self.error_rate < 5.0 and
            self.response_time_p95 < 2000  # 2 seconds
        )
    
    def get_status(self) -> ServiceStatus:
        """Get service status based on metrics."""
        if not self.is_healthy():
            if self.error_rate > 20.0 or self.memory_usage > 95.0:
                return ServiceStatus.UNHEALTHY
            return ServiceStatus.DEGRADED
        return ServiceStatus.HEALTHY


@dataclass
class ErrorRecord:
    """Record of system error with context."""
    error_id: str
    category: ErrorCategory
    message: str
    stack_trace: str
    context: Dict[str, Any]
    timestamp: datetime
    count: int = 1
    last_occurrence: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls."""
    name: str
    failure_threshold: int = 5
    timeout_seconds: int = 60
    reset_timeout_seconds: int = 300
    
    # State tracking
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self._reset()
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return False
        return (datetime.utcnow() - self.last_failure_time).total_seconds() > self.reset_timeout_seconds
    
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


class ResourceManager:
    """Manages system resources and prevents exhaustion."""
    
    def __init__(self, max_memory_mb: int = 2048, max_open_files: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.max_open_files = max_open_files
        self.cleanup_handlers: List[Callable[[], Awaitable[None]]] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring = False
    
    async def start_monitoring(self):
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    def register_cleanup_handler(self, handler: Callable[[], Awaitable[None]]):
        """Register cleanup handler."""
        self.cleanup_handlers.append(handler)
    
    async def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        open_files = len(process.open_files())
        
        return {
            "memory_mb": memory_mb,
            "memory_percent": (memory_mb / self.max_memory_mb) * 100,
            "open_files": open_files,
            "file_percent": (open_files / self.max_open_files) * 100,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads()
        }
    
    async def force_cleanup(self):
        """Force immediate cleanup of resources."""
        logger.info("Forcing resource cleanup")
        
        # Run registered cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Cleanup handler failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Log cleanup results
        resources = await self.check_resources()
        logger.info(f"Post-cleanup resources: {resources}")
    
    async def _periodic_cleanup(self):
        """Periodic resource cleanup."""
        while self._monitoring:
            try:
                resources = await self.check_resources()
                
                # Check if cleanup is needed
                if (resources["memory_percent"] > 80 or 
                    resources["file_percent"] > 80):
                    logger.warning(f"High resource usage detected: {resources}")
                    await self.force_cleanup()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], Awaitable[bool]]] = {}
        self.health_status: Dict[str, bool] = {}
        self.last_check_times: Dict[str, datetime] = {}
        self.check_intervals: Dict[str, int] = {}  # seconds
    
    def register_health_check(self, name: str, check_func: Callable[[], Awaitable[bool]], 
                            interval_seconds: int = 60):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.check_intervals[name] = interval_seconds
        self.health_status[name] = True  # Assume healthy initially
        logger.info(f"Registered health check: {name}")
    
    async def run_health_check(self, name: str) -> bool:
        """Run specific health check."""
        if name not in self.health_checks:
            return False
        
        try:
            result = await self.health_checks[name]()
            self.health_status[name] = result
            self.last_check_times[name] = datetime.utcnow()
            return result
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            self.health_status[name] = False
            self.last_check_times[name] = datetime.utcnow()
            return False
    
    async def run_all_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks."""
        results = {}
        for name in self.health_checks:
            results[name] = await self.run_health_check(name)
        return results
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        # Run checks that are due
        now = datetime.utcnow()
        for name, interval in self.check_intervals.items():
            last_check = self.last_check_times.get(name)
            if not last_check or (now - last_check).total_seconds() > interval:
                await self.run_health_check(name)
        
        all_healthy = all(self.health_status.values())
        failed_checks = [name for name, status in self.health_status.items() if not status]
        
        return {
            "overall_status": "healthy" if all_healthy else "unhealthy",
            "individual_checks": self.health_status.copy(),
            "failed_checks": failed_checks,
            "last_updated": now.isoformat()
        }


class GracefulDegradationManager:
    """Manages graceful degradation of services under stress."""
    
    def __init__(self):
        self.degradation_rules: Dict[str, Dict[str, Any]] = {}
        self.current_degradations: Dict[str, bool] = {}
        self.feature_toggles: Dict[str, bool] = {}
    
    def register_degradation_rule(self, feature: str, condition: str, 
                                 fallback_behavior: Dict[str, Any]):
        """Register a degradation rule for a feature."""
        self.degradation_rules[feature] = {
            "condition": condition,
            "fallback": fallback_behavior,
            "enabled": True
        }
        self.current_degradations[feature] = False
        logger.info(f"Registered degradation rule for {feature}")
    
    def set_feature_toggle(self, feature: str, enabled: bool):
        """Set feature toggle state."""
        self.feature_toggles[feature] = enabled
        logger.info(f"Feature {feature} {'enabled' if enabled else 'disabled'}")
    
    def should_degrade_feature(self, feature: str, metrics: HealthMetrics) -> bool:
        """Check if feature should be degraded based on current metrics."""
        if not self.degradation_rules.get(feature, {}).get("enabled"):
            return False
        
        # Simple rule evaluation (in production, use more sophisticated logic)
        if metrics.cpu_usage > 90 or metrics.memory_usage > 95:
            return True
        
        if metrics.error_rate > 15:
            return True
        
        return False
    
    async def evaluate_degradations(self, metrics: HealthMetrics):
        """Evaluate all degradation rules."""
        for feature in self.degradation_rules:
            should_degrade = self.should_degrade_feature(feature, metrics)
            
            if should_degrade != self.current_degradations[feature]:
                self.current_degradations[feature] = should_degrade
                
                if should_degrade:
                    logger.warning(f"Degrading feature: {feature}")
                    await self._apply_degradation(feature)
                else:
                    logger.info(f"Restoring feature: {feature}")
                    await self._restore_feature(feature)
    
    async def _apply_degradation(self, feature: str):
        """Apply degradation for a feature."""
        # Implementation would depend on specific features
        # For now, just set feature toggle
        self.set_feature_toggle(feature, False)
    
    async def _restore_feature(self, feature: str):
        """Restore feature from degradation."""
        self.set_feature_toggle(feature, True)


class ProductionStabilityManager:
    """Main production stability management system."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.cwd() / "stability"
        self.storage_path.mkdir(exist_ok=True)
        
        # Components
        self.resource_manager = ResourceManager()
        self.health_checker = HealthChecker()
        self.degradation_manager = GracefulDegradationManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error tracking
        self.error_records: Dict[str, ErrorRecord] = {}
        self.error_count_window = 300  # 5 minutes
        
        # Metrics
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        self.response_times: List[float] = []
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize production stability system."""
        logger.info("Initializing production stability system")
        
        # Start resource monitoring
        await self.resource_manager.start_monitoring()
        
        # Register default health checks
        await self._register_default_health_checks()
        
        # Register default degradation rules
        self._register_default_degradation_rules()
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Production stability system initialized")
    
    async def shutdown(self):
        """Shutdown stability system gracefully."""
        logger.info("Shutting down production stability system")
        
        self._shutdown_event.set()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self.resource_manager.stop_monitoring()
        
        logger.info("Production stability system shutdown complete")
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name)
        return self.circuit_breakers[name]
    
    @asynccontextmanager
    async def circuit_breaker_call(self, service_name: str):
        """Context manager for circuit breaker protected calls."""
        breaker = self.get_circuit_breaker(service_name)
        
        if not breaker.should_allow_request():
            raise Exception(f"Circuit breaker {service_name} is open")
        
        start_time = time.time()
        try:
            yield
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            breaker.record_success()
            self.request_count += 1
        except Exception as e:
            breaker.record_failure()
            self.error_count += 1
            await self._record_error(e, {"service": service_name})
            raise
    
    async def _record_error(self, error: Exception, context: Dict[str, Any]):
        """Record error for analysis."""
        error_id = f"{type(error).__name__}_{hash(str(error)) % 10000}"
        stack_trace = traceback.format_exc()
        
        if error_id in self.error_records:
            record = self.error_records[error_id]
            record.count += 1
            record.last_occurrence = datetime.utcnow()
        else:
            # Categorize error
            category = self._categorize_error(error)
            
            self.error_records[error_id] = ErrorRecord(
                error_id=error_id,
                category=category,
                message=str(error),
                stack_trace=stack_trace,
                context=context,
                timestamp=datetime.utcnow()
            )
        
        logger.error(f"Error recorded: {error_id} - {error}")
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for appropriate handling."""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        if "timeout" in error_msg or "connection" in error_msg:
            return ErrorCategory.TRANSIENT
        elif "memory" in error_msg or "disk" in error_msg:
            return ErrorCategory.RESOURCE
        elif "auth" in error_msg or "permission" in error_msg:
            return ErrorCategory.PERMANENT
        elif "external" in error_msg or "api" in error_msg:
            return ErrorCategory.EXTERNAL
        else:
            return ErrorCategory.INTERNAL
    
    async def get_health_metrics(self) -> HealthMetrics:
        """Get current health metrics."""
        process = psutil.Process()
        
        # CPU and memory
        cpu_usage = process.cpu_percent()
        memory_info = process.memory_info()
        memory_usage = (memory_info.rss / 1024 / 1024 / 2048) * 100  # Assume 2GB limit
        
        # Disk usage
        disk_usage = psutil.disk_usage('/').percent
        
        # Network connections
        open_connections = len(process.connections())
        
        # Error rate (last 5 minutes)
        now = datetime.utcnow()
        recent_errors = sum(1 for record in self.error_records.values()
                          if (now - record.last_occurrence).total_seconds() < self.error_count_window)
        recent_requests = max(1, self.request_count)  # Avoid division by zero
        error_rate = (recent_errors / recent_requests) * 100
        
        # Response time P95
        if self.response_times:
            sorted_times = sorted(self.response_times[-1000:])  # Last 1000 requests
            p95_index = int(len(sorted_times) * 0.95)
            response_time_p95 = sorted_times[p95_index] if p95_index < len(sorted_times) else 0
        else:
            response_time_p95 = 0
        
        # Uptime
        uptime_seconds = int((now - self.start_time).total_seconds())
        
        return HealthMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            open_connections=open_connections,
            active_requests=0,  # Would track in real implementation
            error_rate=error_rate,
            response_time_p95=response_time_p95,
            uptime_seconds=uptime_seconds
        )
    
    async def _register_default_health_checks(self):
        """Register default health checks."""
        
        async def database_health() -> bool:
            """Check database connectivity."""
            # Mock implementation
            return True
        
        async def external_api_health() -> bool:
            """Check external API availability."""
            # Mock implementation
            return True
        
        async def disk_space_health() -> bool:
            """Check available disk space."""
            usage = psutil.disk_usage('/')
            return usage.percent < 90
        
        async def memory_health() -> bool:
            """Check memory usage."""
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < 1800  # 1.8GB threshold
        
        self.health_checker.register_health_check("database", database_health, 30)
        self.health_checker.register_health_check("external_api", external_api_health, 60)
        self.health_checker.register_health_check("disk_space", disk_space_health, 120)
        self.health_checker.register_health_check("memory", memory_health, 30)
    
    def _register_default_degradation_rules(self):
        """Register default graceful degradation rules."""
        
        # Disable non-essential features under high load
        self.degradation_manager.register_degradation_rule(
            "advanced_analytics",
            "cpu_usage > 90 OR memory_usage > 95",
            {"disable": True, "show_message": "Analytics temporarily disabled"}
        )
        
        self.degradation_manager.register_degradation_rule(
            "real_time_updates",
            "error_rate > 15 OR response_time > 5000",
            {"polling_interval": 60, "batch_size": 10}
        )
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get current metrics
                metrics = await self.get_health_metrics()
                
                # Evaluate degradations
                await self.degradation_manager.evaluate_degradations(metrics)
                
                # Force cleanup if needed
                if metrics.memory_usage > 90:
                    await self.resource_manager.force_cleanup()
                
                # Sleep until next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error 