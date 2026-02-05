"""LLM client for OpenRouter integration."""
import time
from datetime import datetime, timedelta
from typing import Any, List

from openai import OpenAI

from lib.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME,
)


# Configure OpenRouter client with required headers
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    },
)

# Rate limiting
_last_request_time = None
_request_times = []
_RATE_LIMIT_PER_MINUTE = 50  # OpenRouter has higher rate limits
_MIN_DELAY_BETWEEN_REQUESTS = 1  # Minimum seconds between requests


def _wait_for_rate_limit():
    """Wait if necessary to respect rate limits."""
    global _last_request_time, _request_times
    
    now = datetime.now()
    
    # Remove requests older than 1 minute
    _request_times = [t for t in _request_times if now - t < timedelta(minutes=1)]
    
    # If we're at the limit, wait
    if len(_request_times) >= _RATE_LIMIT_PER_MINUTE:
        oldest_request = min(_request_times)
        wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
        if wait_time > 0:
            print(f"⏳ Rate limit: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time + 0.5)  # Add small buffer
            _request_times = []  # Reset after waiting
    
    # Ensure minimum delay between requests
    if _last_request_time:
        elapsed = (now - _last_request_time).total_seconds()
        if elapsed < _MIN_DELAY_BETWEEN_REQUESTS:
            wait_time = _MIN_DELAY_BETWEEN_REQUESTS - elapsed
            time.sleep(wait_time)
    
    _last_request_time = datetime.now()
    _request_times.append(_last_request_time)


def _extract_text(message_content: Any) -> str:
    """Normalize OpenAI/OpenRouter responses that may be lists or strings."""
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts: List[str] = []
        for part in message_content:
            if isinstance(part, dict):
                # Support both {"type":"text","text":"..."} and {"content":"..."}
                if part.get("type") == "text" and part.get("text"):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p.strip() for p in parts if p).strip()
    return str(message_content or "").strip()


def call_llm(prompt: str, model: str = None, max_retries: int = 3) -> str:
    """
    Call OpenRouter LLM with a prompt, with rate limiting and retry logic.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Optional model override. Defaults to OPENROUTER_MODEL from config.
        max_retries: Maximum number of retries for rate limit errors
        
    Returns:
        The LLM response as a string
    """
    model_name = model or OPENROUTER_MODEL
    
    for attempt in range(max_retries):
        try:
            # Wait for rate limit
            _wait_for_rate_limit()
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent routing decisions
            )
            
            if not response.choices:
                raise ValueError("LLM returned no choices.")
            
            message = response.choices[0].message
            return _extract_text(getattr(message, "content", ""))
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                if attempt < max_retries - 1:
                    # Extract retry delay if available
                    wait_time = 30  # Default wait time
                    if "retry_delay" in error_str or "retry in" in error_str.lower():
                        import re
                        retry_match = re.search(r'retry in (\d+)', error_str, re.IGNORECASE)
                        if retry_match:
                            wait_time = int(retry_match.group(1)) + 5  # Add buffer
                    
                    print(f"⏳ Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {max_retries} retries. Please wait and try again later.")
            else:
                # Not a rate limit error, re-raise
                raise
    
    raise Exception("Failed to get response after retries")
