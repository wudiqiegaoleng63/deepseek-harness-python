# DeepSeek Harness Python

Native Python implementation of [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).

The project is being rebuilt around the same behavioral contracts as the
TypeScript implementation: plugin-composed runtime, event-sourced sessions,
streaming LLM adapters, model-facing tools, and the existing React Web UI.
The browser frontend remains TypeScript/React; this repository owns the Python
runtime and the compatible HTTP/SSE host behind it.

## Implemented runtime

The native Python host now contains:

- reversible hierarchical plugin contexts and serial/waterfall events;
- provider-neutral messages, tool schemas, and stream chunks;
- a DeepSeek chat-completions SSE adapter with thinking/reasoning controls;
- provider-neutral `web_search` and `web_fetch` tools, including bounded
  same-origin HTTP retrieval and DeepSeek native search citations;
- structured provider failures with bounded exponential retry and durable
  `llm/retry` lifecycle events for transient outages;
- configurable context-window protection with deterministic local checkpoints,
  balanced tool boundaries, and durable `compaction/*` events;
- argument-free `/compact` manual compaction with command lifecycle records and
  summary accounting;
- model-free tool-result pruning with replay-safe head/middle/tail replacements;
- session-scoped private spill files with bounded previews for oversized results;
- deterministic, durable session titles with control-code sanitization and rename pinning;
- optional first-prompt model title generation with bounded input/output, timeout, and a
  deterministic fallback when the auxiliary call fails;
- optional stdio LSP providers with workspace-bounded transient document queries and
  JSON-RPC lifecycle management;
- append-only Session events with atomic JSONL persistence, forked sessions, and
  durable one-shot/continuable subagents;
- an Agent loop with tool-call continuation, cancellation, and queued prompts;
- per-tool cancellable timeout policy with durable timeout results;
- semantic durability checkpoints before model requests and top-level tool dispatch;
- workspace-bounded canonical DSH `read`, `write`, `edit`, `glob`, `grep`, and
  `str_replace_editor` tools, plus optional shell and background jobs;
- model-facing `subagent`, `subagent_fork`, `send_message`, `interrupt_agent`,
  and `list_agents` tools with parent/ancestor authorization and depth limits;
- the event-sourced `todo_write` tool with whole-list replacement, validation,
  turn-scoped `todos` projections, and live mux updates;
- model-facing `get_goal`, `create_goal`, and CAS-guarded `update_goal` tools,
  including blocked reasons and process-local activation;
- the model-facing `workflow` tool with an isolated Node VM runner,
  `agent`/`parallel`/`pipeline` orchestration, structured child schemas, and
  durable workflow lifecycle events;
- the foreground `ralph` fresh-agent loop with bounded JSON handoffs and
  durable workflow lifecycle events;
- event-sourced plan mode with `/plan`, the `plan` projection, dynamic plan
  guidance, and `exit_plan_mode` user review;
- event-sourced permission presets with the shared `permissions` projection,
  `/permission` and `permission.set`, replayable sandbox/approval knobs, and
  live shell registration only for `danger-full-access`;
- durable request metadata plus replayable `tokenUsage`, `contextPressure`,
  `contextBreakdown`, and `sessionStats` projections for the shared chat UI;
- bounded, durable workspace instruction baselines from `AGENTS.md` and
  `CLAUDE.md`, refreshed between model steps with UTF-8 source limits;
- optional Python Code Mode (`tools_mode="code"` or `"both"`) with `run_code`,
  async tool calls, bounded output, and wall-clock execution limits;
- durable session-local reminders (`schedule_create`/`schedule_list`/
  `schedule_delete`) with the TS `schedule/change` event fold, offset or
  zoned absolute targets, creation-anchored fixed-rate rules of at least five
  minutes, and follow-up turns framed exactly like the TS runtime;
- Claude Code command-hook compatibility: point `DSH_HOOKS_CONFIG` (or the
  `hooks_config_path` option) at a `hooks.json` or settings file and the
  `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, and `Stop`
  command hooks run with the CC payload dialect, matcher semantics, exit-code
  contract, and deny/ask/allow precedence; `hook/invoked` and `hook/result`
  events are recorded in the session log;
- workspace-write shell and persistent terminals confined through the
  bubblewrap sandbox (read-only root bind, private `/tmp`, read-write
  workspace bind) with fail-closed behavior when `bwrap` is absent;
- the UI-backed `ask_user_question` tool and pending question response flow;
- `messageFeedback/list|put|delete` with lifecycle fencing, compare-and-set
  versions, note validation, and a durable JSON sidecar;
- a constrained dynamic Cordis host runtime with package inventory, approvals,
  dynamic tools, and host/client RPCs;
- the DSH HTTP RPC, WebSocket/SSE event carriers, settings, credentials, workspaces,
  goals, presets, skills, attachments, and session ZIP export;
- the original React/TypeScript frontend served with the Python host;
- a native synchronous `DeepSeekHarness` SDK for embedding the runtime;
- a stdio JSON-RPC SDK server and `DeepSeekHarnessProcess` client for
  cross-language and multi-process embedding.

Install the development project with `uv`:

```sh
uv sync --group dev
uv run pytest
```

The repository CI runs the lockfile check, Ruff, Pyright, the full test suite,
and distribution builds on Python 3.11–3.13.

With `DEEPSEEK_API_KEY` configured, run one task:

```sh
uv run dsh-python headless "read the README and summarize it"
```

Both `headless` and `serve` use the same full Harness service runtime. Use
`--base-url`, `--api-key`, and `--request-timeout` for a DeepSeek-compatible
endpoint; the corresponding `DEEPSEEK_BASE_URL` and `DEEPSEEK_API_KEY`
environment variables are supported as well.

The web tools are available to the model by default. `web_fetch` uses an
anonymous HTTP(S) provider with response, redirect, timeout, and output caps.
`web_search` uses the Anthropic-compatible DeepSeek search endpoint when
`DEEPSEEK_API_KEY` (or the configured credential reference) is available. Set
`DEEPSEEK_SEARCH_BASE_URL` or update the `web-search-deepseek` settings namespace
for a compatible endpoint; search and chat-completions intentionally keep
separate base URLs.

The default runtime spills successful plain-text tool results above 50,000
UTF-8 bytes into private, session-scoped files and leaves the model a bounded
head/tail preview plus a `read`/`grep` recovery locator. Configure or disable
the policy through `DeepSeekHarnessConfig(spill_max_inline_bytes=...)`.

Use the same runtime from Python:

```python
from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

with DeepSeekHarness(
    DeepSeekHarnessConfig(cwd="/path/to/workspace", model="deepseek-v4-flash")
) as harness:
    result = harness.run("Inspect the repository and summarize it.", session_id="demo")

print(result.final_response)
```

Long-lived sessions compact their oldest complete message span when the
configured pressure threshold is reached. The original JSONL events remain
available for audit and replay, while the next model request sees the bounded
checkpoint plus the recent tail. Configure the policy explicitly when embedding
the runtime:

```python
from deepseek_harness import CompactionPolicy, DeepSeekHarness, DeepSeekHarnessConfig

config = DeepSeekHarnessConfig(
    compaction_policy=CompactionPolicy(
        context_window_tokens=131_072,
        threshold_ratio=0.8,
        retain_ratio=0.16,
    )
)
with DeepSeekHarness(config) as harness:
    result = harness.run("Continue the task in this durable session.", session_id="demo")
```

The first backend uses a deterministic local checkpoint and does not spend an
additional model call on summarization. Its public `CompactionPolicy` and
durable event seam are intentionally compatible with adding a provider-backed
summarizer later.

Model-backed first-prompt titles are opt-in so embedding applications can control
the extra request budget. Pass `session_title_llm=SessionTitleLlmConfig(...)` to
`HarnessService`; a failed or timed-out auxiliary request leaves the fallback title
in place and records its request envelope as `session/title-llm-request`.

Oversized tool results are pruned before pressure compaction using the same
default budgets as DSH: 8192 Unicode code points total, retaining the first
4096 and last 1024 around a durable `[... tool result middle pruned ...]`
marker. The original result remains in the append-only log and replay projects
the replacement onto the model-visible surface.

For an isolated child process, use the compatible SDK runtime over newline-
delimited JSON-RPC. The low-level protocol supports `initialize`,
`session/prompt`, and `shutdown`, plus `session.event`, `session.status`, and
subagent lifecycle notifications:

```python
from deepseek_harness import DeepSeekHarnessProcess

with DeepSeekHarnessProcess(cwd="/path/to/workspace") as harness:
    result = harness.run("Inspect the repository and summarize it.")

print(result.final_response)
```

The server can also be launched directly for a non-Python host:

```sh
uv run dsh-python sdk-server
```

It reads and writes one JSON-RPC object per line on stdin/stdout. Configure
the provider with `--api-key`, `--base-url`, and `--request-timeout`, or use
the matching `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` environment variables.

Shell execution is disabled in `read-only` sessions. In `workspace-write`
sessions, shell and persistent terminal tools are registered when the
bubblewrap sandbox (`bwrap`) is available: every command runs under the same
profile as the TypeScript runtime (read-only root bind, private `/tmp`,
read-write workspace bind, `--die-with-parent`), so writes outside the
workspace fail inside the sandbox. Without `bwrap`, those tools stay
unregistered and the session behaves as before; set `DSH_SANDBOX=off` to
disable detection explicitly. `danger-full-access` keeps running tools
unconfined.

Persistent terminals use a POSIX PTY running `bash --noprofile --norc -i`.
Their shell state survives across `terminal_send` calls, including `cd`, and
sessions are isolated by Harness session ownership. The six compatible tools
are `terminal_open`, `terminal_send`, `terminal_read`, `terminal_list`,
`terminal_signal`, and `terminal_close`; background sends reuse `job_output`
and `job_kill`. Send readiness mirrors the TypeScript runtime: an OSC prompt
marker plus a foreground check settles `stdin_read`, a `/proc` syscall probe
detects processes genuinely blocked on stdin, and output silence falls back to
`inferred_idle`. Terminals are created at 40x160 like the TS default. PTY
support is intentionally not emulated on Windows or other platforms without
POSIX PTYs, where the tools return an explicit capability error.

## Frontend and compatibility

The product name is **DeepSeek Harness Python**. User-facing branding changes
while the internal DSH event and API names remain compatible with the
TypeScript frontend, so the existing UI can be reused without a second
protocol rewrite. Build the frontend from the local TypeScript checkout with
`uv run dsh-build-frontend --ts-root /home/lsy/deepseek-harness`, then start
`uv run dsh-python serve --web-dist frontend/dist`.
