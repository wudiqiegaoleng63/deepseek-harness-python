# DeepSeek Harness Python

Native Python implementation of [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).

The project is being rebuilt around the same behavioral contracts as the
TypeScript implementation: plugin-composed runtime, event-sourced sessions,
streaming LLM adapters, model-facing tools, and the existing React Web UI.
The browser frontend remains TypeScript/React; this repository owns the Python
runtime and the compatible HTTP/SSE host behind it.

## Current development slice

The first native slice now contains:

- reversible hierarchical plugin contexts and serial/waterfall events;
- provider-neutral messages, tool schemas, and stream chunks;
- a DeepSeek chat-completions SSE adapter;
- append-only Session events with atomic JSONL persistence;
- an Agent loop with tool-call continuation;
- workspace-bounded `read_file`, `write_file`, and `list_files` tools.

Install the development project with `uv`:

```sh
uv sync --group dev
uv run pytest
```

With `DEEPSEEK_API_KEY` configured, run one task:

```sh
uv run dsh-python headless "read the README and summarize it"
```

Shell execution is deliberately disabled in `workspace-write` mode until the
platform sandbox provider is implemented. It can only be explicitly enabled
with `--permission-mode danger-full-access`.

## Product direction

The target product name is **DeepSeek Harness Python**. User-facing branding
will change while the internal DSH event and API names remain compatible with
the TypeScript frontend, so the existing UI can be reused without a second
protocol rewrite.
