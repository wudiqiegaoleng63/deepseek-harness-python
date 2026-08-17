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
- structured provider failures with bounded exponential retry and durable
  `llm/retry` lifecycle events for transient outages;
- append-only Session events with atomic JSONL persistence, forked sessions, and
  durable one-shot/continuable subagents;
- an Agent loop with tool-call continuation, cancellation, and queued prompts;
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

Use the same runtime from Python:

```python
from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

with DeepSeekHarness(
    DeepSeekHarnessConfig(cwd="/path/to/workspace", model="deepseek-v4-flash")
) as harness:
    result = harness.run("Inspect the repository and summarize it.", session_id="demo")

print(result.final_response)
```

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

Shell execution is deliberately disabled in `workspace-write` mode until the
platform sandbox provider is implemented. It can only be explicitly enabled
with `--permission-mode danger-full-access`.

## Frontend and compatibility

The product name is **DeepSeek Harness Python**. User-facing branding changes
while the internal DSH event and API names remain compatible with the
TypeScript frontend, so the existing UI can be reused without a second
protocol rewrite. Build the frontend from the local TypeScript checkout with
`uv run dsh-build-frontend --ts-root /home/lsy/deepseek-harness`, then start
`uv run dsh-python serve --web-dist frontend/dist`.
