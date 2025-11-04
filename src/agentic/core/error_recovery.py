"""
Smart Error Recovery System with Retry Strategies

This module provides intelligent error handling and recovery mechanisms
for multi-agent operations, including retry strategies, circuit breakers,
and error categorization.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type

from pydantic import BaseModel, Field

from agentic.utils.logging import LoggerMixin


class ErrorCategory(str, Enum):
    """Categories of errors for different handling strategies"""
    TRANSIENT = "transient"          # Temporary errors that may succeed on retry
    RATE_LIMIT = "rate_limit"        # API rate limiting errors
    AUTHENTICATION = "authentication" # Auth/permission errors
    RESOURCE = "resource"            # Resource exhaustion errors
    NETWORK = "network"              # Network connectivity errors
    VALIDATION = "validation"        # Input validation errors
    CONFLICT = "conflict"            # Concurrent modification conflicts
    PERMANENT = "permanent"          # Permanent failures that won't succeed on retry
    UNKNOWN = "unknown"              # Uncategorized errors


class RetryStrategy(str, Enum):
    """Retry strategies for different error types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


class ErrorContext(BaseModel):
    """Context information about an error"""
    error_type: Type[Exception]
    error_message: str
    error_category: ErrorCategory
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    operation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetryConfig(BaseModel):
    """Configuration for retry behavior"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0     # seconds
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on: Set[ErrorCategory] = Field(default_factory=lambda: {
        ErrorCategory.TRANSIENT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.NETWORK
    })


class CircuitBreakerState(str, Enum):
    """States of a circuit breaker"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker(BaseModel):
    """Circuit breaker for preventing cascading failures"""
    name: str
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # seconds
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    def record_success(self):
        """Record a successful operation"""
        self.last_success_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset on success
    
    def record_failure(self):
        """Record a failed operation"""
        self.last_failure_time = datetime.utcnow()
        self.failure_count += 1
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
    
    def can_execute(self) -> bool:
        """Check if requests can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time:
                time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.failure_count = 0
                    return True
            return False
        
        return True  # HALF_OPEN allows execution


class ErrorRecoveryManager(LoggerMixin):
    """
    Manages error recovery strategies for multi-agent operations.
    
    Features:
    - Intelligent error categorization
    - Multiple retry strategies
    - Circuit breakers for cascading failure prevention
    - Error pattern detection
    - Recovery action suggestions
    """
    
    def __init__(self):
        super().__init__()
        
        # Error categorization rules
        self._error_patterns: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.TRANSIENT: [
                lambda e: "temporarily unavailable" in str(e).lower(),
                lambda e: "timeout" in str(e).lower(),
                lambda e: isinstance(e, (asyncio.TimeoutError, TimeoutError))
            ],
            ErrorCategory.RATE_LIMIT: [
                lambda e: "rate limit" in str(e).lower(),
                lambda e: "too many requests" in str(e).lower(),
                lambda e: "429" in str(e)
            ],
            ErrorCategory.AUTHENTICATION: [
                lambda e: "authentication" in str(e).lower(),
                lambda e: "authorization" in str(e).lower(),
                lambda e: "permission denied" in str(e).lower(),
                lambda e: "401" in str(e) or "403" in str(e)
            ],
            ErrorCategory.NETWORK: [
                lambda e: "connection" in str(e).lower(),
                lambda e: "network" in str(e).lower(),
                lambda e: isinstance(e, (ConnectionError, OSError))
            ],
            ErrorCategory.VALIDATION: [
                lambda e: "validation" in str(e).lower(),
                lambda e: "invalid" in str(e).lower(),
                lambda e: isinstance(e, (ValueError, TypeError))
            ],
            ErrorCategory.CONFLICT: [
                lambda e: "conflict" in str(e).lower(),
                lambda e: "already exists" in str(e).lower(),
                lambda e: "concurrent modification" in str(e).lower()
            ]
        }
        
        # Circuit breakers for different services/operations
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error history for pattern detection
        self._error_history: List[ErrorContext] = []
        self._max_history_size = 1000
        
        # Retry configurations by category
        self._retry_configs: Dict[ErrorCategory, RetryConfig] = {
            ErrorCategory.TRANSIENT: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=5,
                initial_delay=1.0
            ),
            ErrorCategory.RATE_LIMIT: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=10,
                initial_delay=5.0,
                max_delay=300.0
            ),
            ErrorCategory.NETWORK: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                initial_delay=2.0
            ),
            ErrorCategory.CONFLICT: RetryConfig(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=3,
                initial_delay=0.5
            )
        }
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on patterns"""
        for category, patterns in self._error_patterns.items():
            for pattern in patterns:
                try:
                    if pattern(error):
                        return category
                except:
                    continue
        
        return ErrorCategory.UNKNOWN
    
    def create_error_context(self,
                           error: Exception,
                           agent_id: Optional[str] = None,
                           task_id: Optional[str] = None,
                           operation: Optional[str] = None) -> ErrorContext:
        """Create error context with categorization"""
        import traceback
        
        category = self.categorize_error(error)
        
        context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            error_category=category,
            agent_id=agent_id,
            task_id=task_id,
            operation=operation,
            stack_trace=traceback.format_exc()
        )
        
        # Add to history
        self._add_to_history(context)
        
        return context
    
    def _add_to_history(self, context: ErrorContext):
        """Add error to history, maintaining size limit"""
        self._error_history.append(context)
        
        # Trim if needed
        if len(self._error_history) > self._max_history_size:
            self._error_history = self._error_history[-self._max_history_size:]
    
    def get_retry_config(self, error_category: ErrorCategory) -> RetryConfig:
        """Get retry configuration for an error category"""
        return self._retry_configs.get(
            error_category,
            RetryConfig(strategy=RetryStrategy.NO_RETRY)
        )
    
    def calculate_retry_delay(self,
                            retry_config: RetryConfig,
                            attempt: int) -> float:
        """Calculate delay before next retry attempt"""
        if retry_config.strategy == RetryStrategy.NO_RETRY:
            return 0
        
        if retry_config.strategy == RetryStrategy.FIXED_DELAY:
            delay = retry_config.initial_delay
        elif retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = retry_config.initial_delay * attempt
        elif retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = retry_config.initial_delay * (retry_config.backoff_factor ** (attempt - 1))
        else:  # IMMEDIATE
            delay = 0
        
        # Apply max delay cap
        delay = min(delay, retry_config.max_delay)
        
        # Add jitter if enabled
        if retry_config.jitter and delay > 0:
            jitter = random.uniform(0, delay * 0.1)  # Up to 10% jitter
            delay += jitter
        
        return delay
    
    async def execute_with_retry(self,
                               operation: Callable,
                               operation_name: str,
                               agent_id: Optional[str] = None,
                               custom_retry_config: Optional[RetryConfig] = None) -> Any:
        """Execute an operation with automatic retry on failure"""
        last_error = None
        
        for attempt in range(1, 100):  # Safety limit
            try:
                # Check circuit breaker
                circuit_breaker = self._get_or_create_circuit_breaker(operation_name)
                if not circuit_breaker.can_execute():
                    raise RuntimeError(f"Circuit breaker OPEN for {operation_name}")
                
                # Execute operation
                result = await operation()
                
                # Record success
                circuit_breaker.record_success()
                
                if attempt > 1:
                    self.logger.info(f"Operation {operation_name} succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Create error context
                error_context = self.create_error_context(
                    e,
                    agent_id=agent_id,
                    operation=operation_name
                )
                
                # Record failure in circuit breaker
                circuit_breaker.record_failure()
                
                # Get retry config
                retry_config = custom_retry_config or self.get_retry_config(
                    error_context.error_category
                )
                
                # Check if we should retry
                if (attempt >= retry_config.max_attempts or
                    error_context.error_category not in retry_config.retry_on):
                    self.logger.error(f"Operation {operation_name} failed after {attempt} attempts: {e}")
                    raise
                
                # Calculate retry delay
                delay = self.calculate_retry_delay(retry_config, attempt)
                
                self.logger.warning(
                    f"Operation {operation_name} failed (attempt {attempt}/{retry_config.max_attempts}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                
                # Wait before retry
                if delay > 0:
                    await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        raise last_error or RuntimeError(f"Max retry attempts exceeded for {operation_name}")
    
    def _get_or_create_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an operation"""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name=name)
        return self._circuit_breakers[name]
    
    def get_error_patterns(self,
                         time_window: Optional[timedelta] = None,
                         agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze error patterns in recent history"""
        # Filter history by time window
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            relevant_errors = [
                e for e in self._error_history
                if e.timestamp >= cutoff_time
            ]
        else:
            relevant_errors = self._error_history
        
        # Filter by agent if specified
        if agent_id:
            relevant_errors = [
                e for e in relevant_errors
                if e.agent_id == agent_id
            ]
        
        if not relevant_errors:
            return {}
        
        # Analyze patterns
        patterns = {
            'total_errors': len(relevant_errors),
            'errors_by_category': {},
            'errors_by_operation': {},
            'error_rate': len(relevant_errors) / max(time_window.total_seconds() / 60 if time_window else 1, 1),
            'most_common_errors': [],
            'recurring_patterns': []
        }
        
        # Count by category
        for error in relevant_errors:
            cat = error.error_category
            patterns['errors_by_category'][cat] = patterns['errors_by_category'].get(cat, 0) + 1
            
            if error.operation:
                op = error.operation
                patterns['errors_by_operation'][op] = patterns['errors_by_operation'].get(op, 0) + 1
        
        # Find most common error messages
        error_messages = {}
        for error in relevant_errors:
            msg = error.error_message
            error_messages[msg] = error_messages.get(msg, 0) + 1
        
        patterns['most_common_errors'] = sorted(
            error_messages.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Detect recurring patterns (same error multiple times in short period)
        for i in range(len(relevant_errors) - 2):
            if (relevant_errors[i].error_message == relevant_errors[i+1].error_message ==
                relevant_errors[i+2].error_message):
                patterns['recurring_patterns'].append({
                    'error': relevant_errors[i].error_message,
                    'category': relevant_errors[i].error_category,
                    'operation': relevant_errors[i].operation
                })
        
        return patterns
    
    def suggest_recovery_actions(self, error_context: ErrorContext) -> List[str]:
        """Suggest recovery actions based on error type"""
        suggestions = []
        
        if error_context.error_category == ErrorCategory.RATE_LIMIT:
            suggestions.extend([
                "Implement request throttling",
                "Use exponential backoff with longer delays",
                "Consider batching requests",
                "Check API quota limits"
            ])
        
        elif error_context.error_category == ErrorCategory.AUTHENTICATION:
            suggestions.extend([
                "Verify API credentials",
                "Check token expiration",
                "Ensure proper permissions are granted",
                "Re-authenticate if needed"
            ])
        
        elif error_context.error_category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check network connectivity",
                "Verify service endpoints",
                "Implement connection pooling",
                "Add timeout and retry logic"
            ])
        
        elif error_context.error_category == ErrorCategory.RESOURCE:
            suggestions.extend([
                "Check system resources (memory, disk, CPU)",
                "Implement resource limits",
                "Add cleanup routines",
                "Scale horizontally if needed"
            ])
        
        elif error_context.error_category == ErrorCategory.CONFLICT:
            suggestions.extend([
                "Implement optimistic locking",
                "Add conflict resolution logic",
                "Use distributed locks",
                "Serialize conflicting operations"
            ])
        
        elif error_context.error_category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Validate inputs before processing",
                "Add input sanitization",
                "Provide clearer error messages",
                "Implement schema validation"
            ])
        
        return suggestions
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        status = {}
        
        for name, breaker in self._circuit_breakers.items():
            status[name] = {
                'state': breaker.state,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count,
                'can_execute': breaker.can_execute(),
                'last_failure': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                'last_success': breaker.last_success_time.isoformat() if breaker.last_success_time else None
            }
        
        return status
    
    def reset_circuit_breaker(self, name: str):
        """Manually reset a circuit breaker"""
        if name in self._circuit_breakers:
            breaker = self._circuit_breakers[name]
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            self.logger.info(f"Reset circuit breaker: {name}")
    
    def clear_error_history(self):
        """Clear error history"""
        self._error_history.clear()
        self.logger.info("Cleared error history")