"""Qwen API 统一封装（含 retry、token 统计）"""
import time
from dataclasses import dataclass, field

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from config.settings import (
    QWEN_MODEL, QWEN_API_KEY, QWEN_API_BASE,
    QWEN_TEMPERATURE, QWEN_MAX_TOKENS,
    QWEN_MAX_RETRIES, QWEN_RETRY_DELAY,
    QWEN_ENABLE_THINKING,
)
from utils.logger import logger


@dataclass
class APIResponse:
    """API 响应封装"""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    raw_response: object = None


@dataclass
class TokenUsage:
    """累计 Token 使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    per_call: list[dict] = field(default_factory=list)

    def record(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.call_count += 1
        self.per_call.append({
            "prompt": prompt,
            "completion": completion,
            "total": prompt + completion,
        })

    def summary(self) -> str:
        return (
            f"TokenUsage: calls={self.call_count}, "
            f"prompt={self.prompt_tokens:,}, "
            f"completion={self.completion_tokens:,}, "
            f"total={self.total_tokens:,}"
        )


class QwenClient:
    """Qwen API 客户端"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.model = model or QWEN_MODEL
        self.api_key = api_key or QWEN_API_KEY
        self.api_base = api_base or QWEN_API_BASE
        self.enable_thinking = (
            enable_thinking if enable_thinking is not None
            else QWEN_ENABLE_THINKING
        )
        self.usage = TokenUsage()

        if not self.api_key:
            logger.warning("QWEN_API_KEY 未设置，请设置 DASHSCOPE_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> APIResponse:
        """发送聊天请求"""
        temperature = temperature if temperature is not None else QWEN_TEMPERATURE
        max_tokens = max_tokens or QWEN_MAX_TOKENS
        use_thinking = (
            enable_thinking if enable_thinking is not None
            else self.enable_thinking
        )

        # 构建 extra_body (Qwen3 的 thinking 模式)
        extra_body = {}
        if use_thinking:
            extra_body["enable_thinking"] = True

        for attempt in range(QWEN_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body if extra_body else None,
                )

                # 提取内容
                content = response.choices[0].message.content or ""

                # 处理 thinking 模式的输出 (思考内容在 reasoning_content 中)
                if use_thinking and hasattr(response.choices[0].message, "reasoning_content"):
                    # reasoning_content 是思考过程，content 是最终回答
                    pass

                # Token 统计
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                # 记录使用量
                self.usage.record(prompt_tokens, completion_tokens)

                return APIResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    finish_reason=response.choices[0].finish_reason or "",
                    raw_response=response,
                )

            except (APITimeoutError, RateLimitError) as e:
                delay = QWEN_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"API 调用失败 (attempt {attempt + 1}/{QWEN_MAX_RETRIES}): "
                    f"{type(e).__name__}, {delay}s 后重试"
                )
                time.sleep(delay)

            except APIError as e:
                logger.error(f"API 错误: {e}")
                if attempt < QWEN_MAX_RETRIES - 1:
                    time.sleep(QWEN_RETRY_DELAY)
                else:
                    return APIResponse(
                        content=f"[API_ERROR: {e}]",
                        finish_reason="error",
                    )

        return APIResponse(
            content="[API_ERROR: 重试次数已用完]",
            finish_reason="error",
        )

    def simple_chat(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> APIResponse:
        """简化版单轮对话"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

    def get_usage(self) -> TokenUsage:
        """获取累计使用统计"""
        return self.usage

    def reset_usage(self):
        """重置使用统计"""
        self.usage = TokenUsage()
