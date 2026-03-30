"""
api_client.py - DeepSeek API client for trace generation.

Supports two API modes:
  1. DeepSeek Official API (api.deepseek.com)
     - Model: "deepseek-reasoner" (full DeepSeek-R1)
     - Returns reasoning in `reasoning_content` field
     - Returns final answer in `content` field

  2. OpenAI-compatible API (Together AI, Fireworks, local vLLM server, HPC)
     - Models: distilled variants (R1-Distill-Qwen-7B, R1-Distill-Llama-8B)
     - Returns <think>...</think> tags inside `content` field

Configuration via environment variables:
  DEEPSEEK_API_KEY    - API key (required)
  DEEPSEEK_BASE_URL   - API base URL (default: https://api.deepseek.com)

Usage:
    from src.generation.api_client import DeepSeekClient

    client = DeepSeekClient()
    result = client.generate("Solve: what is 2+2?", model="deepseek-reasoner")
    print(result["reasoning_trace"])
    print(result["answer_text"])
"""

import os
import re
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """
    Client for the DeepSeek API (OpenAI-compatible).

    Handles:
      - Authentication via API key
      - Automatic retries with exponential backoff
      - Rate limit handling (429 responses)
      - Proper parsing of reasoning vs. answer content
      - Token usage tracking for cost estimation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        initial_retry_delay: float = 2.0,
        timeout: float = 300.0,
    ):
        """
        Initialize the DeepSeek API client.

        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.
            base_url: API base URL. If None, reads from DEEPSEEK_BASE_URL or defaults
                      to https://api.deepseek.com.
            max_retries: Maximum number of retry attempts on failure.
            initial_retry_delay: Initial delay (seconds) before first retry.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL", "")
            or "https://api.deepseek.com"
        )
        # Ensure base_url doesn't end with /
        self.base_url = self.base_url.rstrip("/")

        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.timeout = timeout

        # Cumulative usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_requests = 0

        logger.info(f"DeepSeek API client initialized: {self.base_url}")

    def generate(
        self,
        prompt: str,
        model: str = "deepseek-reasoner",
        temperature: float = 0.6,
        max_tokens: int = 32768,
        top_p: float = 0.95,
    ) -> dict:
        """
        Generate a single response from the DeepSeek API.

        Args:
            prompt: The problem/question to send.
            model: Model identifier. Use "deepseek-reasoner" for full R1,
                   or a distilled model name for compatible APIs.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.

        Returns:
            dict with keys:
              - full_response: Complete raw response text
              - reasoning_trace: The reasoning/thinking content
              - answer_text: The final answer content
              - trace_token_count: Tokens in reasoning trace
              - total_generated_tokens: Total completion tokens
              - prompt_tokens: Tokens in the prompt
              - reasoning_tokens: Tokens in reasoning (if reported by API)
              - generation_time_seconds: Wall-clock time
              - mean_log_prob: None (not available via API)
              - model: Model name used
        """
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/v1/chat/completions"

        messages = [{"role": "user", "content": prompt}]

        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # DeepSeek-reasoner doesn't support temperature/top_p
        # (it uses its own internal reasoning parameters)
        is_reasoner = "reasoner" in model.lower()
        if not is_reasoner:
            payload["temperature"] = temperature
            payload["top_p"] = top_p

        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    response_data = json.loads(resp.read().decode("utf-8"))

                generation_time = time.time() - start_time
                self.total_requests += 1

                # Parse the response
                return self._parse_response(response_data, model, generation_time)

            except urllib.error.HTTPError as e:
                last_error = e
                status = e.code

                if status == 429:
                    # Rate limited — use longer backoff
                    delay = self.initial_retry_delay * (3 ** attempt)
                    logger.warning(f"Rate limited (429). Waiting {delay:.1f}s "
                                   f"(attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(delay)
                    continue

                elif status in (500, 502, 503):
                    # Server error — retry with backoff
                    delay = self.initial_retry_delay * (2 ** attempt)
                    logger.warning(f"Server error ({status}). Retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue

                elif status == 401:
                    raise ValueError("Invalid API key (401 Unauthorized)")

                else:
                    error_body = e.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"API error {status}: {error_body}")

            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = e
                delay = self.initial_retry_delay * (2 ** attempt)
                logger.warning(f"Connection error: {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)
                continue

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.initial_retry_delay * (2 ** attempt)
                    logger.warning(f"Unexpected error: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise

        raise RuntimeError(
            f"Failed after {self.max_retries + 1} attempts. Last error: {last_error}"
        )

    def _parse_response(
        self,
        response_data: dict,
        model: str,
        generation_time: float,
    ) -> dict:
        """
        Parse API response into our standard format.

        Handles two response formats:
          1. deepseek-reasoner: reasoning in message.reasoning_content,
             answer in message.content
          2. distilled models: <think>...</think> tags in message.content
        """
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Extract usage statistics
        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = usage.get("reasoning_tokens",
                                     usage.get("completion_tokens_details", {})
                                     .get("reasoning_tokens", 0))

        # Update cumulative tracking
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_reasoning_tokens += reasoning_tokens

        # Parse content based on model type
        reasoning_trace = ""
        answer_text = ""
        full_response = ""

        if "reasoning_content" in message and message["reasoning_content"]:
            # Format 1: DeepSeek-reasoner (official API)
            # Reasoning is in a separate field
            reasoning_trace = message.get("reasoning_content", "").strip()
            answer_text = message.get("content", "").strip()
            full_response = f"<think>\n{reasoning_trace}\n</think>\n\n{answer_text}"

        else:
            # Format 2: Distilled models / OpenAI-compatible
            # Reasoning is inside <think>...</think> tags in content
            content = message.get("content", "")
            full_response = content

            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                reasoning_trace = think_match.group(1).strip()
                answer_text = content[think_match.end():].strip()
            else:
                # No think block — entire content is the answer
                reasoning_trace = ""
                answer_text = content.strip()

        # Estimate trace token count (word-level approximation)
        trace_token_count = len(reasoning_trace.split()) if reasoning_trace else 0

        return {
            "full_response": full_response,
            "reasoning_trace": reasoning_trace,
            "answer_text": answer_text,
            "trace_token_count": trace_token_count,
            "total_generated_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "reasoning_tokens": reasoning_tokens,
            "generation_time_seconds": round(generation_time, 2),
            "mean_log_prob": None,  # Not available via API
            "model": model,
        }

    def generate_batch(
        self,
        prompts: list[str],
        model: str = "deepseek-reasoner",
        temperature: float = 0.6,
        max_tokens: int = 32768,
        delay_between_requests: float = 0.5,
        callback=None,
    ) -> list[dict]:
        """
        Generate responses for multiple prompts sequentially.

        Since the DeepSeek API doesn't support true batching,
        we send requests sequentially with a small delay to avoid
        rate limits.

        Args:
            prompts: List of prompt strings.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per response.
            delay_between_requests: Seconds to wait between requests.
            callback: Optional function called after each request with
                      (index, total, result) arguments.

        Returns:
            List of result dicts (one per prompt).
        """
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                results.append(result)

                if callback:
                    callback(i, len(prompts), result)

            except Exception as e:
                logger.error(f"Batch item {i}/{len(prompts)} failed: {e}")
                results.append({"error": str(e), "prompt_index": i})

            # Rate limit delay
            if i < len(prompts) - 1 and delay_between_requests > 0:
                time.sleep(delay_between_requests)

        return results

    def get_usage_summary(self) -> dict:
        """
        Get cumulative token usage and cost estimate.

        DeepSeek-R1 pricing (as of 2025):
          - Input: $0.55 / 1M tokens (cache miss), $0.14 (cache hit)
          - Output: $2.19 / 1M tokens
          - Note: reasoning_tokens counted as output tokens
        """
        input_cost = self.total_prompt_tokens * 0.55 / 1_000_000
        output_cost = self.total_completion_tokens * 2.19 / 1_000_000
        total_cost = input_cost + output_cost

        return {
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "estimated_cost_usd": round(total_cost, 4),
            "cost_breakdown": {
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4),
            },
        }


# =============================================================================
# MODEL REGISTRY
# =============================================================================

# Maps short names to full API model identifiers and configurations
MODEL_REGISTRY = {
    # --- DeepSeek Official API ---
    "deepseek-r1": {
        "api_model": "deepseek-reasoner",
        "display_name": "DeepSeek-R1",
        "base_url": "https://api.deepseek.com",
        "supports_temperature": False,  # Reasoner ignores temperature
        "response_format": "reasoning_content",
    },
    "deepseek-r1-0528": {
        "api_model": "deepseek-reasoner",
        "display_name": "DeepSeek-R1-0528",
        "base_url": "https://api.deepseek.com",
        "supports_temperature": False,
        "response_format": "reasoning_content",
    },

    # --- Distilled models (local HPC or compatible API) ---
    "r1-distill-qwen-7b": {
        "api_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "display_name": "R1-Distill-Qwen-7B",
        "supports_temperature": True,
        "response_format": "think_tags",
    },
    "r1-distill-llama-8b": {
        "api_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "display_name": "R1-Distill-Llama-8B",
        "supports_temperature": True,
        "response_format": "think_tags",
    },
    "r1-distill-qwen-14b": {
        "api_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "display_name": "R1-Distill-Qwen-14B",
        "supports_temperature": True,
        "response_format": "think_tags",
    },
}


def get_model_info(short_name: str) -> dict:
    """Look up model configuration by short name."""
    if short_name not in MODEL_REGISTRY:
        # Try as a direct HuggingFace model name
        return {
            "api_model": short_name,
            "hf_model": short_name,
            "display_name": short_name.split("/")[-1],
            "supports_temperature": True,
            "response_format": "think_tags",
        }
    return MODEL_REGISTRY[short_name]


def list_models() -> list[dict]:
    """List all registered models."""
    return [
        {"short_name": k, **v}
        for k, v in MODEL_REGISTRY.items()
    ]


# =============================================================================
# SELF-TEST
# =============================================================================

def run_api_client_tests():
    """Test API client components (without making real API calls)."""
    print("Running API client self-tests...")

    tests_passed = 0
    tests_failed = 0

    def check(name, condition, msg=""):
        nonlocal tests_passed, tests_failed
        if condition:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  FAIL: {name}: {msg}")

    # Test model registry
    check("registry_r1", "deepseek-r1" in MODEL_REGISTRY)
    check("registry_qwen7b", "r1-distill-qwen-7b" in MODEL_REGISTRY)
    check("registry_llama8b", "r1-distill-llama-8b" in MODEL_REGISTRY)

    info = get_model_info("r1-distill-llama-8b")
    check("llama_info", info["display_name"] == "R1-Distill-Llama-8B")
    check("llama_hf", "Llama-8B" in info.get("hf_model", ""))

    # Test fallback for unknown model
    info2 = get_model_info("some-custom/model-name")
    check("fallback_model", info2["display_name"] == "model-name")

    # Test response parsing (mock)
    # Format 1: reasoning_content field
    mock_response_reasoner = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "Let me think step by step. 40 + 2 = 42."
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "reasoning_tokens": 40,
        }
    }

    # We need a client to test parsing, but can't create one without API key
    # So test the parsing logic directly
    os.environ["DEEPSEEK_API_KEY"] = "test-key-for-unit-tests"
    client = DeepSeekClient(api_key="test-key")
    result1 = client._parse_response(mock_response_reasoner, "deepseek-reasoner", 1.5)

    check("parse_r1_trace", "step by step" in result1["reasoning_trace"])
    check("parse_r1_answer", "42" in result1["answer_text"])
    check("parse_r1_full", "<think>" in result1["full_response"])
    check("parse_r1_tokens", result1["total_generated_tokens"] == 50)
    check("parse_r1_reasoning_tokens", result1["reasoning_tokens"] == 40)

    # Format 2: think tags in content
    mock_response_distill = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "<think>\nLet me work through this.\n2+2=4.\n</think>\n\nThe answer is 4."
            }
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 30}
    }

    result2 = client._parse_response(mock_response_distill, "r1-distill-qwen-7b", 2.0)

    check("parse_distill_trace", "work through" in result2["reasoning_trace"])
    check("parse_distill_answer", "4" in result2["answer_text"])
    check("parse_distill_time", result2["generation_time_seconds"] == 2.0)

    # Usage tracking
    usage = client.get_usage_summary()
    check("usage_requests", usage["total_requests"] == 0)  # No real requests made
    check("usage_has_cost", "estimated_cost_usd" in usage)

    # Clean up env var
    if os.environ.get("DEEPSEEK_API_KEY") == "test-key-for-unit-tests":
        del os.environ["DEEPSEEK_API_KEY"]

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed "
          f"out of {tests_passed + tests_failed} tests")
    if tests_failed == 0:
        print("All API client tests passed.")
    return tests_failed == 0


if __name__ == "__main__":
    run_api_client_tests()
