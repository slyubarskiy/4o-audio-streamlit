import time
import functools
import streamlit as st
from typing import Callable, Any, Optional, Tuple, Type
import requests
import openai
import logging
import random
import signal
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Base exception for API-related errors"""
    pass

class RetryableError(APIError):
    """Error that should trigger a retry"""
    pass

class NonRetryableError(APIError):
    """Error that should not trigger a retry"""
    pass

def classify_error(exception: Exception) -> Tuple[bool, str, str]:
    """
    Classify error as retryable or non-retryable
    Returns: (is_retryable, error_type, user_message)
    """
    error_classifications = {
        # OpenAI/Azure OpenAI errors
        openai.APITimeoutError: (True, "timeout", "Request timed out. Retrying..."),
        openai.RateLimitError: (True, "rate_limit", "Rate limit reached. Waiting to retry..."),
        openai.APIConnectionError: (True, "connection", "Connection error. Checking network..."),
        openai.InternalServerError: (True, "server_error", "Server error. Retrying..."),
        openai.APIStatusError: (False, "api_error", "API error occurred"),
        openai.AuthenticationError: (False, "auth", "Authentication failed. Please check credentials"),
        
        # HTTP errors
        requests.exceptions.Timeout: (True, "timeout", "Request timed out. Retrying..."),
        requests.exceptions.ConnectionError: (True, "connection", "Connection failed. Retrying..."),
        requests.exceptions.HTTPError: (True, "http_error", "HTTP error. Checking..."),
        requests.exceptions.RequestException: (True, "request_error", "Request failed. Retrying..."),
        
        # Generic
        ConnectionError: (True, "connection", "Connection lost. Retrying..."),
        TimeoutError: (True, "timeout", "Operation timed out. Retrying..."),
    }
    
    for error_class, (retryable, error_type, message) in error_classifications.items():
        if isinstance(exception, error_class):
            return retryable, error_type, message
    
    # Check HTTP status codes if available
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        status_code = exception.response.status_code
        if status_code in [429, 500, 502, 503, 504]:
            return True, f"http_{status_code}", f"Server issue (HTTP {status_code}). Retrying..."
        elif status_code in [400, 401, 403, 404]:
            return False, f"http_{status_code}", f"Request error (HTTP {status_code}). Please check configuration"
    
    # Default to non-retryable
    return False, "unknown", "An unexpected error occurred"

def api_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable] = None,
    use_streamlit_ui: bool = True
):
    """
    Decorator for API calls with exponential backoff retry logic
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exceptions to retry on
        on_retry: Callback function on each retry
        use_streamlit_ui: Show progress in Streamlit UI
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            delay = initial_delay
            last_exception = None
            
            # Create a container for retry status if using Streamlit UI
            if use_streamlit_ui:
                status_container = st.empty()
            
            while attempt < max_attempts:
                try:
                    # Clear any previous error messages
                    if use_streamlit_ui and attempt > 0:
                        status_container.success(f"✅ Retry successful on attempt {attempt + 1}")
                    
                    result = func(*args, **kwargs)
                    
                    # Log success after retry
                    if attempt > 0:
                        logger.info(f"Successfully completed {func.__name__} after {attempt + 1} attempts")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    is_retryable, error_type, user_message = classify_error(e)
                    
                    # Log the error
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}")
                    
                    # Check if we should retry
                    if not is_retryable:
                        logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                        if use_streamlit_ui:
                            st.error(f"❌ {user_message}")
                        raise NonRetryableError(str(e)) from e
                    
                    attempt += 1
                    
                    if attempt >= max_attempts:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                        if use_streamlit_ui:
                            st.error(f"❌ Failed after {max_attempts} attempts: {user_message}")
                        raise RetryableError(f"Max retries exceeded: {str(e)}") from e
                    
                    # Calculate next delay with exponential backoff
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())
                    else:
                        actual_delay = delay
                    
                    actual_delay = min(actual_delay, max_delay)
                    
                    # Show retry status
                    if use_streamlit_ui:
                        status_container.warning(
                            f"⏳ {user_message}\n"
                            f"Attempt {attempt}/{max_attempts} failed. "
                            f"Retrying in {actual_delay:.1f} seconds..."
                        )
                    
                    # Execute retry callback if provided
                    if on_retry:
                        on_retry(attempt, actual_delay, e)
                    
                    # Wait before retry
                    time.sleep(actual_delay)
                    
                    # Update delay for next iteration
                    delay *= exponential_base
            
            # Should never reach here, but just in case
            raise RetryableError(f"Retry loop exceeded without resolution") from last_exception
        
        return wrapper
    return decorator

def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """
    Circuit breaker pattern to prevent cascading failures
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time in seconds before attempting to close circuit
        expected_exception: Exception type to catch
    """
    def decorator(func: Callable) -> Callable:
        func._failures = 0
        func._last_failure_time = 0
        func._circuit_open = False
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if circuit is open
            if func._circuit_open:
                if time.time() - func._last_failure_time < recovery_timeout:
                    raise APIError(f"Circuit breaker is open. Service unavailable for {recovery_timeout - (time.time() - func._last_failure_time):.0f} more seconds")
                else:
                    func._circuit_open = False
                    func._failures = 0
                    logger.info(f"Circuit breaker reset for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                func._failures = 0  # Reset on success
                return result
                
            except expected_exception as e:
                func._failures += 1
                func._last_failure_time = time.time()
                
                if func._failures >= failure_threshold:
                    func._circuit_open = True
                    logger.error(f"Circuit breaker opened for {func.__name__} after {func._failures} failures")
                    if st._is_running_with_streamlit:
                        st.error(f"⚠️ Service temporarily unavailable. Please try again in {recovery_timeout} seconds.")
                
                raise
        
        return wrapper
    return decorator

def with_timeout(timeout_seconds: float = 30.0):
    """
    Add timeout to function calls
    
    Args:
        timeout_seconds: Maximum time allowed for function execution
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator

def cache_on_error(cache_key: str, ttl_seconds: float = 3600):
    """
    Cache successful responses to use as fallback during errors
    
    Args:
        cache_key: Key for storing in session state
        ttl_seconds: Time to live for cache
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_full_key = f"cache_{cache_key}"
            cache_time_key = f"cache_time_{cache_key}"
            
            try:
                result = func(*args, **kwargs)
                
                # Cache successful result
                if 'session_state' in dir(st):
                    st.session_state[cache_full_key] = result
                    st.session_state[cache_time_key] = time.time()
                
                return result
                
            except Exception as e:
                # Check if we have cached result
                if 'session_state' in dir(st) and cache_full_key in st.session_state:
                    cache_age = time.time() - st.session_state.get(cache_time_key, 0)
                    
                    if cache_age < ttl_seconds:
                        logger.info(f"Using cached result for {func.__name__} (age: {cache_age:.0f}s)")
                        st.warning(f"⚠️ Using cached data from {cache_age:.0f} seconds ago due to temporary issue")
                        return st.session_state[cache_full_key]
                
                # No valid cache, re-raise exception
                raise
        
        return wrapper
    return decorator

def validate_response(validator: Callable[[Any], bool], error_message: str = "Invalid response"):
    """
    Validate API responses
    
    Args:
        validator: Function to validate response
        error_message: Error message if validation fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            
            if not validator(result):
                logger.error(f"Response validation failed for {func.__name__}: {error_message}")
                raise ValueError(f"Response validation failed: {error_message}")
            
            return result
        
        return wrapper
    return decorator

# Helper function for batch operations with partial failure handling
def batch_api_call(
    items: list,
    api_function: Callable,
    batch_size: int = 10,
    allow_partial: bool = True,
    progress_bar: bool = True
) -> Tuple[list, list]:
    """
    Process items in batches with error handling
    
    Args:
        items: List of items to process
        api_function: Function to call for each item
        batch_size: Number of items per batch
        allow_partial: Continue on individual failures
        progress_bar: Show progress bar
    
    Returns:
        Tuple of (successful_results, failed_items)
    """
    successful_results = []
    failed_items = []
    
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    if progress_bar and st._is_running_with_streamlit:
        progress = st.progress(0)
        status = st.empty()
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if progress_bar and st._is_running_with_streamlit:
            progress.progress(batch_num / total_batches)
            status.text(f"Processing batch {batch_num}/{total_batches}")
        
        for item in batch:
            try:
                result = api_function(item)
                successful_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process item: {str(e)}")
                failed_items.append((item, str(e)))
                
                if not allow_partial:
                    raise
    
    if progress_bar and st._is_running_with_streamlit:
        progress.empty()
        status.empty()
        
        if failed_items:
            st.warning(f"⚠️ Completed with {len(failed_items)} failures out of {len(items)} items")
        else:
            st.success(f"✅ Successfully processed all {len(items)} items")
    
    return successful_results, failed_items