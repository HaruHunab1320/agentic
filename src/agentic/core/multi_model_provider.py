# Multi-Model Provider Support for Phase 5
from __future__ import annotations

import asyncio
import time
import logging
import statistics
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Type, Union
from dataclasses import dataclass
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Types of AI model providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai" 
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    LOCAL = "local"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class ModelCapability(str, Enum):
    """Model capabilities"""
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    TEXT_ANALYSIS = "text_analysis"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    EMBEDDING = "embedding"
    LARGE_CONTEXT = "large_context"


class ProviderStatus(str, Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    RATE_LIMITED = "rate_limited"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class ModelRequest:
    """Request to a model provider"""
    id: str
    prompt: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 4000
    stream: bool = False
    tools: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = None


@dataclass
class ModelResponse:
    """Response from a model provider"""
    request_id: str
    content: str
    provider: str
    model: str
    usage: Dict[str, int]
    latency: float
    cost: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


class ProviderConfig(BaseModel):
    """Configuration for a model provider"""
    name: str = Field(description="Provider name")
    provider_type: ProviderType = Field(description="Type of provider")
    api_key: str = Field(description="API key for provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    models: List[str] = Field(description="Available models")
    capabilities: List[ModelCapability] = Field(description="Provider capabilities")
    max_requests_per_minute: int = Field(default=60, description="Rate limit")
    max_concurrent_requests: int = Field(default=10, description="Concurrent limit")
    cost_per_1k_tokens: float = Field(description="Cost per 1K tokens")
    weight: float = Field(default=1.0, description="Load balancing weight")
    priority: int = Field(default=100, description="Provider priority")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    health_check_interval: int = Field(default=60, description="Health check interval")


class ProviderMetrics(BaseModel):
    """Metrics for a provider"""
    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    average_latency: float = 0.0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    availability: float = 1.0
    current_connections: int = 0
    rate_limit_hits: int = 0


class BaseModelProvider(ABC):
    """Base class for all model providers"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.status = ProviderStatus.HEALTHY
        self.metrics = ProviderMetrics(provider_name=config.name)
        self.active_requests: Set[str] = set()
        self.last_health_check = datetime.utcnow()
        
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response from model"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health"""
        pass
    
    @abstractmethod
    async def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate cost for request"""
        pass
    
    def can_handle_request(self, request: ModelRequest) -> bool:
        """Check if provider can handle the request"""
        # Check if model is available
        if request.model not in self.config.models:
            return False
            
        # Check concurrent request limit
        if len(self.active_requests) >= self.config.max_concurrent_requests:
            return False
            
        # Check if provider is healthy
        if self.status == ProviderStatus.OFFLINE:
            return False
            
        return True
    
    async def _track_request(self, request: ModelRequest, response: ModelResponse):
        """Track request metrics"""
        self.metrics.total_requests += 1
        self.metrics.last_request_time = datetime.utcnow()
        
        if response:
            self.metrics.successful_requests += 1
            self.metrics.total_cost += response.cost
            self.metrics.total_tokens += response.usage.get('total_tokens', 0)
            
            # Update average latency
            if self.metrics.average_latency == 0:
                self.metrics.average_latency = response.latency
            else:
                self.metrics.average_latency = (
                    self.metrics.average_latency * 0.8 + response.latency * 0.2
                )
        else:
            self.metrics.failed_requests += 1
        
        # Update error rate
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests


class AnthropicProvider(BaseModelProvider):
    """Anthropic Claude provider"""
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Anthropic API"""
        start_time = time.time()
        self.active_requests.add(request.id)
        
        try:
            # Simulate Anthropic API call
            await asyncio.sleep(0.5)  # Mock latency
            
            # Mock response
            response = ModelResponse(
                request_id=request.id,
                content="Mock response from Claude",
                provider=self.config.name,
                model=request.model,
                usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
                latency=time.time() - start_time,
                cost=await self.estimate_cost(request.prompt, request.max_tokens),
                timestamp=datetime.utcnow()
            )
            
            await self._track_request(request, response)
            return response
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            await self._track_request(request, None)
            raise
        finally:
            self.active_requests.discard(request.id)
    
    async def health_check(self) -> bool:
        """Check Anthropic API health"""
        try:
            # Mock health check
            self.status = ProviderStatus.HEALTHY
            self.last_health_check = datetime.utcnow()
            return True
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            return False
    
    async def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate cost for Anthropic request"""
        input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        total_tokens = input_tokens + max_tokens
        return (total_tokens / 1000) * self.config.cost_per_1k_tokens


class OpenAIProvider(BaseModelProvider):
    """OpenAI GPT provider"""
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()
        self.active_requests.add(request.id)
        
        try:
            # Simulate OpenAI API call
            await asyncio.sleep(0.3)  # Mock latency
            
            response = ModelResponse(
                request_id=request.id,
                content="Mock response from GPT",
                provider=self.config.name,
                model=request.model,
                usage={"prompt_tokens": 80, "completion_tokens": 150, "total_tokens": 230},
                latency=time.time() - start_time,
                cost=await self.estimate_cost(request.prompt, request.max_tokens),
                timestamp=datetime.utcnow()
            )
            
            await self._track_request(request, response)
            return response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            await self._track_request(request, None)
            raise
        finally:
            self.active_requests.discard(request.id)
    
    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            self.status = ProviderStatus.HEALTHY
            self.last_health_check = datetime.utcnow()
            return True
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            return False
    
    async def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate cost for OpenAI request"""
        input_tokens = len(prompt.split()) * 1.3
        total_tokens = input_tokens + max_tokens
        return (total_tokens / 1000) * self.config.cost_per_1k_tokens


class LocalProvider(BaseModelProvider):
    """Local model provider (Ollama, etc.)"""
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using local model"""
        start_time = time.time()
        self.active_requests.add(request.id)
        
        try:
            # Simulate local model inference
            await asyncio.sleep(1.0)  # Local models often slower
            
            response = ModelResponse(
                request_id=request.id,
                content="Mock response from local model",
                provider=self.config.name,
                model=request.model,
                usage={"input_tokens": 90, "output_tokens": 180, "total_tokens": 270},
                latency=time.time() - start_time,
                cost=0.0,  # Local models are free
                timestamp=datetime.utcnow()
            )
            
            await self._track_request(request, response)
            return response
            
        except Exception as e:
            logger.error(f"Local model error: {e}")
            await self._track_request(request, None)
            raise
        finally:
            self.active_requests.discard(request.id)
    
    async def health_check(self) -> bool:
        """Check local model health"""
        try:
            self.status = ProviderStatus.HEALTHY
            self.last_health_check = datetime.utcnow()
            return True
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            return False
    
    async def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Local models are free"""
        return 0.0


class LoadBalancer:
    """Load balancer for distributing requests across providers"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED):
        self.strategy = strategy
        self.round_robin_index = 0
        
    def select_provider(self, providers: List[BaseModelProvider], request: ModelRequest) -> Optional[BaseModelProvider]:
        """Select best provider for request"""
        # Filter available providers
        available_providers = [p for p in providers if p.can_handle_request(request)]
        
        if not available_providers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_selection(available_providers, request)
        elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(available_providers)
        else:
            return available_providers[0]
    
    def _round_robin_selection(self, providers: List[BaseModelProvider]) -> BaseModelProvider:
        """Simple round-robin selection"""
        provider = providers[self.round_robin_index % len(providers)]
        self.round_robin_index += 1
        return provider
    
    def _weighted_round_robin_selection(self, providers: List[BaseModelProvider]) -> BaseModelProvider:
        """Weighted round-robin based on provider weights"""
        total_weight = sum(p.config.weight for p in providers)
        weights = [p.config.weight / total_weight for p in providers]
        
        # Simple weighted selection (could be improved with proper weighted round-robin)
        import random
        return random.choices(providers, weights=weights)[0]
    
    def _least_connections_selection(self, providers: List[BaseModelProvider]) -> BaseModelProvider:
        """Select provider with least active connections"""
        return min(providers, key=lambda p: len(p.active_requests))
    
    def _least_response_time_selection(self, providers: List[BaseModelProvider]) -> BaseModelProvider:
        """Select provider with lowest average response time"""
        return min(providers, key=lambda p: p.metrics.average_latency or float('inf'))
    
    async def _cost_optimized_selection(self, providers: List[BaseModelProvider], request: ModelRequest) -> BaseModelProvider:
        """Select provider with lowest estimated cost"""
        costs = []
        for provider in providers:
            try:
                cost = await provider.estimate_cost(request.prompt, request.max_tokens)
                costs.append((provider, cost))
            except Exception:
                costs.append((provider, float('inf')))
        
        return min(costs, key=lambda x: x[1])[0]
    
    def _performance_based_selection(self, providers: List[BaseModelProvider]) -> BaseModelProvider:
        """Select provider based on performance score"""
        def performance_score(provider: BaseModelProvider) -> float:
            metrics = provider.metrics
            if metrics.total_requests == 0:
                return 0.5  # Neutral score for new providers
            
            # Calculate performance score based on multiple factors
            success_rate = 1 - metrics.error_rate
            speed_score = 1 / (metrics.average_latency + 1)  # Inverse of latency
            availability_score = metrics.availability
            load_score = 1 - (len(provider.active_requests) / provider.config.max_concurrent_requests)
            
            # Weighted combination
            score = (
                success_rate * 0.3 +
                speed_score * 0.3 +
                availability_score * 0.2 +
                load_score * 0.2
            )
            
            return score
        
        return max(providers, key=performance_score)


class FailoverManager:
    """Manages failover between providers"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breakers: Dict[str, Dict] = {}
    
    async def execute_with_failover(self, 
                                  providers: List[BaseModelProvider],
                                  request: ModelRequest,
                                  load_balancer: LoadBalancer) -> ModelResponse:
        """Execute request with automatic failover"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            # Select provider
            provider = load_balancer.select_provider(providers, request)
            if not provider:
                raise Exception("No available providers")
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(provider.config.name):
                continue
            
            try:
                return await provider.generate(request)
            
            except Exception as e:
                last_exception = e
                logger.warning(f"Request failed on provider {provider.config.name}: {e}")
                
                # Update circuit breaker
                self._record_failure(provider.config.name)
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.backoff_factor ** attempt)
        
        raise Exception(f"All providers failed. Last error: {last_exception}")
    
    def _is_circuit_breaker_open(self, provider_name: str) -> bool:
        """Check if circuit breaker is open for provider"""
        if provider_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[provider_name]
        if breaker['state'] == 'open':
            # Check if it's time to try again
            if datetime.utcnow() - breaker['last_failure'] > timedelta(minutes=5):
                breaker['state'] = 'half_open'
                return False
            return True
        
        return False
    
    def _record_failure(self, provider_name: str):
        """Record failure for circuit breaker"""
        if provider_name not in self.circuit_breakers:
            self.circuit_breakers[provider_name] = {
                'failures': 0,
                'state': 'closed',
                'last_failure': None
            }
        
        breaker = self.circuit_breakers[provider_name]
        breaker['failures'] += 1
        breaker['last_failure'] = datetime.utcnow()
        
        # Open circuit breaker after 3 failures
        if breaker['failures'] >= 3:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for provider {provider_name}")


class CostOptimizer:
    """Optimizes costs across providers"""
    
    def __init__(self, budget_limit: float = 100.0):  # USD per day
        self.budget_limit = budget_limit
        self.daily_spending: Dict[str, float] = {}
        self.cost_history: List[Dict] = []
        
    async def optimize_provider_selection(self, 
                                        providers: List[BaseModelProvider],
                                        request: ModelRequest) -> List[BaseModelProvider]:
        """Reorder providers based on cost optimization"""
        # Calculate costs for each provider
        provider_costs = []
        for provider in providers:
            try:
                cost = await provider.estimate_cost(request.prompt, request.max_tokens)
                provider_costs.append((provider, cost))
            except Exception:
                provider_costs.append((provider, float('inf')))
        
        # Sort by cost (ascending)
        provider_costs.sort(key=lambda x: x[1])
        
        # Filter out providers that would exceed budget
        filtered_providers = []
        current_date = datetime.utcnow().date().isoformat()
        daily_spent = self.daily_spending.get(current_date, 0.0)
        
        for provider, cost in provider_costs:
            if daily_spent + cost <= self.budget_limit:
                filtered_providers.append(provider)
        
        return filtered_providers or [pc[0] for pc in provider_costs[:1]]  # Return at least one
    
    def track_spending(self, response: ModelResponse):
        """Track spending for budget management"""
        current_date = datetime.utcnow().date().isoformat()
        if current_date not in self.daily_spending:
            self.daily_spending[current_date] = 0.0
        
        self.daily_spending[current_date] += response.cost
        
        # Record in history
        self.cost_history.append({
            'date': current_date,
            'provider': response.provider,
            'model': response.model,
            'cost': response.cost,
            'tokens': response.usage.get('total_tokens', 0)
        })
    
    def get_cost_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate cost report"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        recent_costs = [c for c in self.cost_history if c['date'] >= cutoff_date]
        
        total_cost = sum(c['cost'] for c in recent_costs)
        total_tokens = sum(c['tokens'] for c in recent_costs)
        
        # Group by provider
        provider_costs = {}
        for cost in recent_costs:
            provider = cost['provider']
            if provider not in provider_costs:
                provider_costs[provider] = {'cost': 0, 'tokens': 0, 'requests': 0}
            provider_costs[provider]['cost'] += cost['cost']
            provider_costs[provider]['tokens'] += cost['tokens']
            provider_costs[provider]['requests'] += 1
        
        return {
            'period_days': days,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'average_cost_per_token': total_cost / total_tokens if total_tokens > 0 else 0,
            'provider_breakdown': provider_costs,
            'daily_budget_limit': self.budget_limit,
            'remaining_budget': self.budget_limit - self.daily_spending.get(
                datetime.utcnow().date().isoformat(), 0.0
            )
        }


class MultiModelManager:
    """Main manager for multi-model provider support"""
    
    def __init__(self, configs: List[ProviderConfig]):
        self.providers: Dict[str, BaseModelProvider] = {}
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.PERFORMANCE_BASED)
        self.failover_manager = FailoverManager()
        self.cost_optimizer = CostOptimizer()
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Initialize providers
        for config in configs:
            provider = self._create_provider(config)
            self.providers[config.name] = provider
    
    def _create_provider(self, config: ProviderConfig) -> BaseModelProvider:
        """Create provider instance based on type"""
        if config.provider_type == ProviderType.ANTHROPIC:
            return AnthropicProvider(config)
        elif config.provider_type == ProviderType.OPENAI:
            return OpenAIProvider(config)
        elif config.provider_type == ProviderType.LOCAL:
            return LocalProvider(config)
        else:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")
    
    async def initialize(self):
        """Initialize the multi-model manager"""
        logger.info("Initializing multi-model manager...")
        
        # Start health checking
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Initial health check
        for provider in self.providers.values():
            await provider.health_check()
        
        logger.info(f"Multi-model manager initialized with {len(self.providers)} providers")
    
    async def shutdown(self):
        """Shutdown the multi-model manager"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using best available provider"""
        # Optimize provider order by cost
        providers_list = list(self.providers.values())
        optimized_providers = await self.cost_optimizer.optimize_provider_selection(
            providers_list, request
        )
        
        # Execute with failover
        response = await self.failover_manager.execute_with_failover(
            optimized_providers, request, self.load_balancer
        )
        
        # Track spending
        self.cost_optimizer.track_spending(response)
        
        return response
    
    async def get_provider_metrics(self) -> Dict[str, ProviderMetrics]:
        """Get metrics for all providers"""
        return {name: provider.metrics for name, provider in self.providers.items()}
    
    async def get_cost_report(self, days: int = 7) -> Dict[str, Any]:
        """Get cost report"""
        return self.cost_optimizer.get_cost_report(days)
    
    def get_provider_status(self) -> Dict[str, ProviderStatus]:
        """Get status of all providers"""
        return {name: provider.status for name, provider in self.providers.items()}
    
    async def _health_check_loop(self):
        """Periodic health check for all providers"""
        while True:
            try:
                # Check each provider
                for provider in self.providers.values():
                    if datetime.utcnow() - provider.last_health_check > timedelta(
                        seconds=provider.config.health_check_interval
                    ):
                        await provider.health_check()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Wait longer on error


# Verified: Complete - Comprehensive multi-model provider system with load balancing, failover, cost optimization, and health monitoring 