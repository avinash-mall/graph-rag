"""
Resilience Module for External Service Calls

This module provides centralized retry logic, circuit-breaking, and timeout handling
for external service calls (LLM endpoints, Neo4j, etc.) to ensure consistent error handling
and prevent transient failures from bubbling up as user-facing errors.
"""

import asyncio
import logging
import time
from typing import Optional, TypeVar, Callable, Any, List, Dict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
from functools import wraps

from config import get_config

logger = logging.getLogger("Resilience")

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    
    Prevents cascading failures by stopping requests to failing services
    and allowing them to recover. Implements three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered, allows limited requests
    
    Args:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes needed to close circuit
        timeout: Seconds before transitioning from OPEN to HALF_OPEN
        name: Name identifier for this circuit breaker
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.name = name
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args, **kwargs: Arguments to pass to func
            
        Returns:
            Result from func
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by func
        """
        async with self._lock:
            await self._update_state()
            
            if self.stats.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {self.stats.last_failure_time}"
                )
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _update_state(self):
        """
        Update circuit breaker state based on current conditions.
        
        State transitions:
        - CLOSED → OPEN: When failures >= failure_threshold
        - OPEN → HALF_OPEN: After timeout period
        - HALF_OPEN → CLOSED: When successes >= success_threshold
        - HALF_OPEN → OPEN: When a failure occurs during testing
        """
        now = datetime.now()
        
        if self.stats.state == CircuitState.OPEN:
            # Check if timeout has passed
            if (self.stats.last_failure_time and 
                (now - self.stats.last_failure_time).total_seconds() >= self.timeout):
                self.stats.state = CircuitState.HALF_OPEN
                self.stats.successes = 0
                logger.info(
                    f"Circuit breaker '{self.name}' transitioned to HALF_OPEN",
                    extra={"circuit_breaker": self.name, "state": "half_open"}
                )
        elif self.stats.state == CircuitState.HALF_OPEN:
            # Already set, no action needed
            pass
        else:  # CLOSED
            # Check if failure threshold exceeded
            if self.stats.failures >= self.failure_threshold:
                self.stats.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self.stats.failures} failures",
                    extra={
                        "circuit_breaker": self.name,
                        "state": "open",
                        "failures": self.stats.failures
                    }
                )
    
    async def _record_success(self):
        """Record a successful call"""
        async with self._lock:
            self.stats.total_successes += 1
            self.stats.last_success_time = datetime.now()
            
            if self.stats.state == CircuitState.HALF_OPEN:
                self.stats.successes += 1
                if self.stats.successes >= self.success_threshold:
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failures = 0
                    self.stats.successes = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' closed after {self.stats.successes} successes",
                        extra={
                            "circuit_breaker": self.name,
                            "state": "closed",
                            "successes": self.stats.successes
                        }
                    )
            elif self.stats.state == CircuitState.CLOSED:
                self.stats.failures = 0  # Reset failure count on success
    
    async def _record_failure(self):
        """Record a failed call"""
        async with self._lock:
            self.stats.total_failures += 1
            self.stats.last_failure_time = datetime.now()
            
            if self.stats.state == CircuitState.CLOSED:
                self.stats.failures += 1
            elif self.stats.state == CircuitState.HALF_OPEN:
                # Failure in half-open, go back to open
                self.stats.state = CircuitState.OPEN
                self.stats.successes = 0
                logger.warning(
                    f"Circuit breaker '{self.name}' re-opened after failure in HALF_OPEN",
                    extra={"circuit_breaker": self.name, "state": "open"}
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failures": self.stats.failures,
            "successes": self.stats.successes,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class RetryableError(Exception):
    """Base class for retryable errors"""
    pass


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        retryable_exceptions: tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions


async def retry_with_backoff(
    func: Callable,
    *args,
    retry_config: Optional[RetryConfig] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> T:
    """
    Execute a function with retry logic and exponential backoff.
    
    Args:
        func: Async function to call
        *args: Positional arguments
        retry_config: Retry configuration (uses config from environment if None)
        context: Additional context for logging
        **kwargs: Keyword arguments
        
    Returns:
        Result from func
        
    Raises:
        Exception: Last exception after all retries exhausted
    """
    if retry_config is None:
        cfg = get_config().resilience
        retry_config = RetryConfig(
            max_retries=cfg.max_retries,
            backoff_factor=cfg.retry_backoff_factor,
            initial_delay=cfg.retry_initial_delay
        )
    
    context = context or {}
    last_exception = None
    
    for attempt in range(retry_config.max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(
                    f"Retry succeeded on attempt {attempt + 1}",
                    extra={
                        "attempt": attempt + 1,
                        "function": func.__name__,
                        **context
                    }
                )
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check if error is retryable
            if not isinstance(e, retry_config.retryable_exceptions):
                logger.debug(
                    f"Non-retryable exception: {type(e).__name__}",
                    extra={"exception_type": type(e).__name__, "function": func.__name__, **context}
                )
                raise
            
            if attempt < retry_config.max_retries:
                delay = min(
                    retry_config.initial_delay * (retry_config.backoff_factor ** attempt),
                    retry_config.max_delay
                )
                
                logger.warning(
                    f"Retry attempt {attempt + 1}/{retry_config.max_retries} failed: {e}. "
                    f"Retrying in {delay:.2f}s",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": retry_config.max_retries,
                        "delay": delay,
                        "exception_type": type(e).__name__,
                        "function": func.__name__,
                        **context
                    }
                )
                
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {retry_config.max_retries + 1} retry attempts exhausted",
                    extra={
                        "max_retries": retry_config.max_retries,
                        "exception_type": type(last_exception).__name__,
                        "function": func.__name__,
                        **context
                    },
                    exc_info=True
                )
    
    raise last_exception


async def call_with_resilience(
    func: Callable,
    *args,
    circuit_breaker: Optional[CircuitBreaker] = None,
    retry_config: Optional[RetryConfig] = None,
    timeout: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> T:
    """
    Execute a function with full resilience: circuit breaker, retries, and timeout.
    
    Args:
        func: Async function to call
        *args: Positional arguments
        circuit_breaker: Circuit breaker instance (uses default if None)
        retry_config: Retry configuration
        timeout: Timeout in seconds (uses config default if None)
        context: Additional context for logging
        **kwargs: Keyword arguments
        
    Returns:
        Result from func
        
    Raises:
        CircuitBreakerOpenError: If circuit breaker is open
        asyncio.TimeoutError: If operation times out
        Exception: Last exception after retries
    """
    context = context or {}
    cfg = get_config().resilience
    
    if timeout is None:
        timeout = cfg.request_timeout
    
    async def _call_with_timeout():
        if circuit_breaker:
            return await circuit_breaker.call(func, *args, **kwargs)
        else:
            return await func(*args, **kwargs)
    
    async def _call_with_retries():
        return await retry_with_backoff(
            _call_with_timeout,
            retry_config=retry_config,
            context=context
        )
    
    try:
        return await asyncio.wait_for(_call_with_retries(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(
            f"Operation timed out after {timeout}s",
            extra={
                "timeout": timeout,
                "function": func.__name__,
                **context
            }
        )
        raise


# Global circuit breakers for different services
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a service"""
    if service_name not in _circuit_breakers:
        cfg = get_config().resilience
        _circuit_breakers[service_name] = CircuitBreaker(
            failure_threshold=cfg.circuit_breaker_failure_threshold,
            success_threshold=cfg.circuit_breaker_success_threshold,
            timeout=cfg.circuit_breaker_timeout,
            name=service_name
        )
    return _circuit_breakers[service_name]


def reset_circuit_breaker(service_name: str):
    """Reset a circuit breaker (for testing/admin)"""
    if service_name in _circuit_breakers:
        _circuit_breakers[service_name].stats = CircuitBreakerStats()


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers"""
    return {
        name: cb.get_stats()
        for name, cb in _circuit_breakers.items()
    }


# Decorator for automatic resilience
def resilient(
    service_name: str = "default",
    circuit_breaker: Optional[CircuitBreaker] = None,
    retry_config: Optional[RetryConfig] = None,
    timeout: Optional[float] = None
):
    """
    Decorator to make an async function resilient with retries and circuit breaker.
    
    Usage:
        @resilient(service_name="llm")
        async def call_llm(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cb = circuit_breaker or get_circuit_breaker(service_name)
            return await call_with_resilience(
                func,
                *args,
                circuit_breaker=cb,
                retry_config=retry_config,
                timeout=timeout,
                context={"function": func.__name__},
                **kwargs
            )
        return wrapper
    return decorator

