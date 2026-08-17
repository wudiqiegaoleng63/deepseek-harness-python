"""DeepSeek Harness Python.

The package is intentionally split into small runtime seams.  The public
surface starts with the native Python agent core; web/API and SDK layers build
on the same Session and Agent contracts.
"""

from .agent import Agent, AgentStatus, RunResult
from .compaction import (
    CompactionPolicy,
    CompactionResult,
    compact_if_needed,
    estimate_message_tokens,
    estimate_messages_tokens,
)
from .errors import LlmFailure
from .llm import DeepSeekAdapter, LlmAdapter, LlmCallConfig, LlmRequest, RetryPolicy, StreamChunk
from .models import Message, TextContent, ToolCallContent, ToolResultContent
from .sdk import DeepSeekHarness, DeepSeekHarnessConfig
from .sdk import RunResult as SdkRunResult
from .sdk_process import (
    DeepSeekHarnessProcess,
    HarnessClient,
    JsonRpcResponseError,
    NotificationSubscription,
    ProcessRunResult,
    ProcessSession,
)
from .session import JsonlSessionStore, Session, SessionEvent, SessionHeader

__all__ = [
    "Agent",
    "AgentStatus",
    "CompactionPolicy",
    "CompactionResult",
    "DeepSeekAdapter",
    "DeepSeekHarness",
    "DeepSeekHarnessConfig",
    "DeepSeekHarnessProcess",
    "HarnessClient",
    "JsonRpcResponseError",
    "JsonlSessionStore",
    "LlmAdapter",
    "LlmCallConfig",
    "LlmFailure",
    "LlmRequest",
    "Message",
    "NotificationSubscription",
    "ProcessSession",
    "ProcessRunResult",
    "RunResult",
    "RetryPolicy",
    "SdkRunResult",
    "Session",
    "SessionEvent",
    "SessionHeader",
    "StreamChunk",
    "TextContent",
    "ToolCallContent",
    "ToolResultContent",
    "compact_if_needed",
    "estimate_message_tokens",
    "estimate_messages_tokens",
]
