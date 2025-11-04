# Production-Ready Resource Manager for Agent System
import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import psutil
import platform

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK_IO = "disk_io"


class ResourceAllocation:
    """Tracks resource allocation for an agent"""
    def __init__(self, agent_id: str, resources: Dict[str, Any]):
        self.agent_id = agent_id
        self.resources = resources
        self.allocated_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        
    def update(self, new_resources: Dict[str, Any]):
        """Update resource allocation"""
        self.resources.update(new_resources)
        self.last_updated = datetime.utcnow()


class NetworkBandwidthEstimator:
    """Estimates network bandwidth usage"""
    def __init__(self):
        self._last_check = time.time()
        self._last_bytes_sent = 0
        self._last_bytes_recv = 0
        self._bandwidth_history: List[Tuple[float, float]] = []  # (sent_rate, recv_rate)
        self._max_history_size = 60  # Keep 1 minute of history
        
    def update_bandwidth(self) -> Tuple[float, float]:
        """Update and return current bandwidth usage in MB/s"""
        current_time = time.time()
        time_delta = current_time - self._last_check
        
        if time_delta < 0.1:  # Avoid division by very small numbers
            return self._get_average_bandwidth()
        
        # Get current network stats
        net_stats = psutil.net_io_counters()
        bytes_sent = net_stats.bytes_sent
        bytes_recv = net_stats.bytes_recv
        
        # Calculate rates (bytes/second)
        if self._last_bytes_sent > 0 and self._last_bytes_recv > 0:
            sent_rate = (bytes_sent - self._last_bytes_sent) / time_delta
            recv_rate = (bytes_recv - self._last_bytes_recv) / time_delta
            
            # Convert to MB/s
            sent_rate_mb = sent_rate / (1024 * 1024)
            recv_rate_mb = recv_rate / (1024 * 1024)
            
            # Add to history
            self._bandwidth_history.append((sent_rate_mb, recv_rate_mb))
            
            # Maintain history size
            if len(self._bandwidth_history) > self._max_history_size:
                self._bandwidth_history.pop(0)
        
        # Update last values
        self._last_check = current_time
        self._last_bytes_sent = bytes_sent
        self._last_bytes_recv = bytes_recv
        
        return self._get_average_bandwidth()
    
    def _get_average_bandwidth(self) -> Tuple[float, float]:
        """Get average bandwidth from history"""
        if not self._bandwidth_history:
            return 0.0, 0.0
        
        avg_sent = sum(s for s, _ in self._bandwidth_history) / len(self._bandwidth_history)
        avg_recv = sum(r for _, r in self._bandwidth_history) / len(self._bandwidth_history)
        
        return avg_sent, avg_recv
    
    def get_estimated_capacity(self) -> float:
        """Estimate available network capacity (0.0 to 1.0)"""
        # Get current bandwidth
        sent_rate, recv_rate = self.update_bandwidth()
        total_rate = sent_rate + recv_rate
        
        # Estimate max bandwidth based on network interface
        try:
            # Get network interface stats
            net_if_stats = psutil.net_if_stats()
            
            # Find the fastest active interface
            max_speed = 0
            for interface, stats in net_if_stats.items():
                if stats.isup and stats.speed > 0:
                    # Speed is in Mbps, convert to MB/s
                    interface_speed_mb = stats.speed / 8
                    max_speed = max(max_speed, interface_speed_mb)
            
            if max_speed > 0:
                # Calculate usage percentage
                usage_percentage = min(total_rate / max_speed, 1.0)
                return max(0.0, 1.0 - usage_percentage)
            
        except Exception as e:
            logger.warning(f"Could not determine network interface speed: {e}")
        
        # Fallback: assume 100 MB/s max bandwidth
        assumed_max_bandwidth = 100.0  # MB/s
        usage_percentage = min(total_rate / assumed_max_bandwidth, 1.0)
        return max(0.0, 1.0 - usage_percentage)


class ResourceManager:
    """Manages system resources and agent allocations"""
    
    def __init__(self, 
                 safety_margin: float = 0.2,  # 20% safety margin
                 max_cpu_usage: float = 0.8,  # Max 80% CPU
                 max_memory_usage: float = 0.8,  # Max 80% memory
                 max_network_usage: float = 0.7):  # Max 70% network
        
        self.safety_margin = safety_margin
        self.max_cpu_usage = max_cpu_usage
        self.max_memory_usage = max_memory_usage
        self.max_network_usage = max_network_usage
        
        # Track agent allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # Resource monitoring
        self.network_estimator = NetworkBandwidthEstimator()
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Resource usage history for better estimation
        self.usage_history: Dict[ResourceType, List[float]] = defaultdict(list)
        self.history_size = 60  # Keep 1 minute of history
        
        # Start monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background resource monitoring"""
        async def monitor_resources():
            while True:
                try:
                    # Update resource usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    # Update history
                    self._update_history(ResourceType.CPU, cpu_percent / 100.0)
                    self._update_history(ResourceType.MEMORY, memory_percent / 100.0)
                    
                    # Update network bandwidth
                    self.network_estimator.update_bandwidth()
                    
                    # Clean up stale allocations (older than 1 hour)
                    await self._cleanup_stale_allocations()
                    
                    await asyncio.sleep(1)  # Update every second
                    
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    await asyncio.sleep(5)  # Wait longer on error
        
        # Create monitoring task
        try:
            loop = asyncio.get_running_loop()
            self._monitoring_task = loop.create_task(monitor_resources())
        except RuntimeError:
            # No event loop running yet, will be started later
            logger.info("Resource monitoring will start when event loop is available")
    
    def _update_history(self, resource_type: ResourceType, value: float):
        """Update resource usage history"""
        history = self.usage_history[resource_type]
        history.append(value)
        
        # Maintain history size
        if len(history) > self.history_size:
            history.pop(0)
    
    def _get_average_usage(self, resource_type: ResourceType) -> float:
        """Get average resource usage from history"""
        history = self.usage_history[resource_type]
        if not history:
            return 0.0
        return sum(history) / len(history)
    
    async def get_available_capacity(self) -> Dict[str, Any]:
        """Get current available resource capacity"""
        # Get current system metrics
        cpu_info = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        # Calculate available percentages
        cpu_available = max(0.0, (self.max_cpu_usage * 100) - cpu_info) / 100.0
        memory_available = max(0.0, self.max_memory_usage - (memory_info.percent / 100.0))
        
        # Get network capacity
        network_capacity = self.network_estimator.get_estimated_capacity()
        network_available = min(network_capacity, self.max_network_usage)
        
        # Calculate allocated resources
        allocated_cpu = sum(
            alloc.resources.get('cpu_cores', 0) 
            for alloc in self.allocations.values()
        )
        allocated_memory = sum(
            alloc.resources.get('memory_mb', 0) 
            for alloc in self.allocations.values()
        ) / 1024  # Convert to GB
        
        # Get system info
        total_cores = psutil.cpu_count()
        total_memory_gb = memory_info.total / (1024 ** 3)
        
        return {
            # Percentages (0.0 to 1.0)
            'cpu_percentage': cpu_available,
            'memory_percentage': memory_available,
            'network_percentage': network_available,
            
            # Absolute values
            'cpu_cores_available': max(0, total_cores * cpu_available - allocated_cpu),
            'memory_gb_available': max(0, total_memory_gb * memory_available - allocated_memory),
            
            # System totals
            'total_cpu_cores': total_cores,
            'total_memory_gb': total_memory_gb,
            
            # Current usage
            'current_cpu_usage': cpu_info / 100.0,
            'current_memory_usage': memory_info.percent / 100.0,
            'current_network_usage': 1.0 - network_capacity,
            
            # Allocated resources
            'allocated_agents': len(self.allocations),
            'allocated_cpu_cores': allocated_cpu,
            'allocated_memory_gb': allocated_memory
        }
    
    async def can_allocate_resources(self, 
                                   requested_resources: Dict[str, Any]) -> bool:
        """Check if requested resources can be allocated"""
        capacity = await self.get_available_capacity()
        
        # Check CPU
        requested_cpu = requested_resources.get('cpu_cores', 0)
        if requested_cpu > capacity['cpu_cores_available']:
            logger.debug(f"Insufficient CPU: requested={requested_cpu}, available={capacity['cpu_cores_available']}")
            return False
        
        # Check memory
        requested_memory_mb = requested_resources.get('memory_mb', 0)
        requested_memory_gb = requested_memory_mb / 1024
        if requested_memory_gb > capacity['memory_gb_available']:
            logger.debug(f"Insufficient memory: requested={requested_memory_gb}GB, available={capacity['memory_gb_available']}GB")
            return False
        
        # Check network bandwidth requirements
        network_requirement = requested_resources.get('network_bandwidth', 'low')
        if network_requirement == 'high' and capacity['network_percentage'] < 0.3:
            logger.debug(f"Insufficient network bandwidth: available={capacity['network_percentage']}")
            return False
        elif network_requirement == 'medium' and capacity['network_percentage'] < 0.1:
            logger.debug(f"Insufficient network bandwidth: available={capacity['network_percentage']}")
            return False
        
        # Apply safety margin check
        # Ensure we don't allocate if any resource is too close to limit
        safety_threshold = 1.0 - self.safety_margin
        
        if capacity['cpu_percentage'] < self.safety_margin:
            logger.debug(f"CPU usage too high for safety margin: {capacity['current_cpu_usage']}")
            return False
            
        if capacity['memory_percentage'] < self.safety_margin:
            logger.debug(f"Memory usage too high for safety margin: {capacity['current_memory_usage']}")
            return False
        
        return True
    
    async def reserve_resources(self, 
                              agent_id: str, 
                              requested_resources: Dict[str, Any]) -> bool:
        """Reserve resources for an agent"""
        # First check if we can allocate
        if not await self.can_allocate_resources(requested_resources):
            logger.warning(f"Cannot allocate resources for agent {agent_id}")
            return False
        
        # Create or update allocation
        if agent_id in self.allocations:
            self.allocations[agent_id].update(requested_resources)
            logger.info(f"Updated resource allocation for agent {agent_id}")
        else:
            self.allocations[agent_id] = ResourceAllocation(agent_id, requested_resources)
            logger.info(f"Created new resource allocation for agent {agent_id}")
        
        # Log current allocations
        capacity = await self.get_available_capacity()
        logger.info(
            f"Resource allocation successful. "
            f"Total agents: {capacity['allocated_agents']}, "
            f"CPU allocated: {capacity['allocated_cpu_cores']:.1f} cores, "
            f"Memory allocated: {capacity['allocated_memory_gb']:.1f} GB"
        )
        
        return True
    
    async def release_resources(self, agent_id: str) -> bool:
        """Release resources allocated to an agent"""
        if agent_id not in self.allocations:
            logger.warning(f"No allocation found for agent {agent_id}")
            return False
        
        allocation = self.allocations.pop(agent_id)
        logger.info(
            f"Released resources for agent {agent_id}: "
            f"CPU: {allocation.resources.get('cpu_cores', 0)} cores, "
            f"Memory: {allocation.resources.get('memory_mb', 0)} MB"
        )
        
        return True
    
    async def _cleanup_stale_allocations(self):
        """Clean up allocations older than 1 hour"""
        current_time = datetime.utcnow()
        stale_threshold = timedelta(hours=1)
        
        stale_agents = []
        for agent_id, allocation in self.allocations.items():
            if current_time - allocation.last_updated > stale_threshold:
                stale_agents.append(agent_id)
        
        for agent_id in stale_agents:
            logger.warning(f"Cleaning up stale allocation for agent {agent_id}")
            await self.release_resources(agent_id)
    
    def get_resource_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of current resource usage"""
        # Get current metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        # Get network stats
        sent_rate, recv_rate = self.network_estimator.update_bandwidth()
        
        # Calculate agent resource totals
        total_allocated_cpu = sum(
            alloc.resources.get('cpu_cores', 0) 
            for alloc in self.allocations.values()
        )
        total_allocated_memory = sum(
            alloc.resources.get('memory_mb', 0) 
            for alloc in self.allocations.values()
        )
        
        return {
            'system': {
                'platform': platform.system(),
                'cpu_count': psutil.cpu_count(),
                'cpu_usage_percent': cpu_percent,
                'memory_total_gb': memory_info.total / (1024 ** 3),
                'memory_used_gb': memory_info.used / (1024 ** 3),
                'memory_usage_percent': memory_info.percent,
                'disk_total_gb': disk_info.total / (1024 ** 3),
                'disk_used_gb': disk_info.used / (1024 ** 3),
                'disk_usage_percent': disk_info.percent,
                'network_sent_mb_per_sec': sent_rate,
                'network_recv_mb_per_sec': recv_rate
            },
            'agents': {
                'total_count': len(self.allocations),
                'total_cpu_cores': total_allocated_cpu,
                'total_memory_mb': total_allocated_memory,
                'allocations': {
                    agent_id: {
                        'cpu_cores': alloc.resources.get('cpu_cores', 0),
                        'memory_mb': alloc.resources.get('memory_mb', 0),
                        'allocated_at': alloc.allocated_at.isoformat(),
                        'last_updated': alloc.last_updated.isoformat()
                    }
                    for agent_id, alloc in self.allocations.items()
                }
            },
            'limits': {
                'max_cpu_usage': self.max_cpu_usage,
                'max_memory_usage': self.max_memory_usage,
                'max_network_usage': self.max_network_usage,
                'safety_margin': self.safety_margin
            }
        }
    
    async def wait_for_resources(self, 
                                requested_resources: Dict[str, Any],
                                timeout: int = 300) -> bool:
        """Wait for resources to become available (with timeout)"""
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        
        while time.time() - start_time < timeout:
            if await self.can_allocate_resources(requested_resources):
                return True
            
            logger.debug(
                f"Waiting for resources: {requested_resources}. "
                f"Time elapsed: {int(time.time() - start_time)}s"
            )
            
            await asyncio.sleep(check_interval)
        
        logger.warning(f"Timeout waiting for resources after {timeout} seconds")
        return False
    
    async def get_agent_resource_usage(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get resource usage for a specific agent"""
        if agent_id not in self.allocations:
            return None
        
        allocation = self.allocations[agent_id]
        return {
            'agent_id': agent_id,
            'resources': allocation.resources,
            'allocated_at': allocation.allocated_at.isoformat(),
            'last_updated': allocation.last_updated.isoformat(),
            'duration_minutes': (datetime.utcnow() - allocation.allocated_at).total_seconds() / 60
        }
    
    def estimate_agent_capacity(self) -> int:
        """Estimate how many more agents can be spawned"""
        # Get current capacity synchronously for estimation
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        # Calculate available resources
        cpu_available = max(0.0, self.max_cpu_usage - (cpu_percent / 100.0))
        memory_available = max(0.0, self.max_memory_usage - (memory_info.percent / 100.0))
        
        # Assume typical agent requirements
        typical_cpu_per_agent = 0.5  # cores
        typical_memory_per_agent = 256  # MB
        
        # Calculate based on CPU
        total_cores = psutil.cpu_count()
        cpu_capacity = int((total_cores * cpu_available) / typical_cpu_per_agent)
        
        # Calculate based on memory
        total_memory_gb = memory_info.total / (1024 ** 3)
        memory_capacity = int((total_memory_gb * memory_available * 1024) / typical_memory_per_agent)
        
        # Return the minimum (bottleneck)
        return max(0, min(cpu_capacity, memory_capacity))
    
    def __del__(self):
        """Cleanup monitoring task"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()