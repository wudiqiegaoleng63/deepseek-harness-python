"""DeepSeek Harness Python.

The package is intentionally split into small runtime seams.  The public
surface starts with the native Python agent core; web/API and SDK layers build
on the same Session and Agent contracts.
"""

from .agent import Agent, AgentStatus, RunResult
from .checkpoint import CheckpointCallback, SessionCheckpointPolicy
from .compaction import (
    CompactionPolicy,
    CompactionResult,
    compact_if_needed,
    estimate_message_tokens,
    estimate_messages_tokens,
)
from .errors import LlmFailure, WebError
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
from .session import JsonlSessionStore, Session, SessionEvent, SessionHeader, SessionSurfaceNode
from .spill import (
    LocalSpillStore,
    SaveTextSpill,
    SpillOwner,
    SpillPolicy,
    SpillRef,
    SpillSource,
    SpillStore,
)
from .tool_result_pruner import (
    DEFAULT_TOOL_RESULT_PRUNE_CONFIG,
    PRUNE_MARKER,
    PrunedEntry,
    PruneResult,
    ToolResultPruneConfig,
    ToolResultPruner,
)
from .web_capability import (
    DeepSeekSearchOptions,
    DeepSeekSearchProvider,
    HttpFetchLimits,
    HttpFetchProvider,
    WebFetchBody,
    WebFetchResult,
    WebRuntime,
    WebSearchResult,
    WebSource,
    format_fetch_output,
    format_search_output,
    install_web_tools,
)

__all__ = [
    "Agent",
    "AgentStatus",
    "CheckpointCallback",
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
    "PRUNE_MARKER",
    "PruneResult",
    "PrunedEntry",
    "RunResult",
    "RetryPolicy",
    "SdkRunResult",
    "Session",
    "SessionEvent",
    "SessionHeader",
    "SessionSurfaceNode",
    "SessionCheckpointPolicy",
    "LocalSpillStore",
    "SaveTextSpill",
    "SpillOwner",
    "SpillPolicy",
    "SpillRef",
    "SpillSource",
    "SpillStore",
    "StreamChunk",
    "TextContent",
    "ToolCallContent",
    "ToolResultContent",
    "ToolResultPruneConfig",
    "ToolResultPruner",
    "WebError",
    "WebFetchBody",
    "WebFetchResult",
    "WebRuntime",
    "WebSearchResult",
    "WebSource",
    "compact_if_needed",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "DeepSeekSearchOptions",
    "DeepSeekSearchProvider",
    "HttpFetchLimits",
    "HttpFetchProvider",
    "format_fetch_output",
    "format_search_output",
    "install_web_tools",
    "DEFAULT_TOOL_RESULT_PRUNE_CONFIG",
]
