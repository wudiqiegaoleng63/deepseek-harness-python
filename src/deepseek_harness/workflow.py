"""Small isolated JavaScript workflow runner used by the model-facing tool."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


class WorkflowError(Exception):
    """A fatal workflow contract error that must not dissolve to ``null``."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WorkflowChildResult:
    child_id: str
    ok: bool
    value: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    value: Any
    stop_reason: str
    error: str | None
    agents_started: int


WorkflowEventSink = Callable[[str, dict[str, Any]], Awaitable[None]]
WorkflowAgentRunner = Callable[
    [str, str, dict[str, Any], Callable[[str], Awaitable[None]]],
    Awaitable[WorkflowChildResult],
]


NODE_WORKER_SOURCE = r'''import vm from 'node:vm'
import readline from 'node:readline'

const emit = value => process.stdout.write(JSON.stringify(value) + '\n')
const fatal = (message, code) => {
  const error = new Error(message)
  error.fatal = true
  error.code = code
  return error
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity })
const firstLine = await new Promise(resolve => rl.once('line', resolve))
const init = JSON.parse(firstLine)
const pending = new Map()
const slotWaiters = []
let callSeq = 0
let started = 0
let active = 0
let currentPhase
let cancelled = false
let finished = false

function finish(result) {
  if (finished) return
  finished = true
  emit({ type: 'result', result })
  setImmediate(() => process.exit(0))
}

function cancelledError() {
  return fatal('workflow run was cancelled', 'CANCELLED')
}

function throwIfCancelled() {
  if (cancelled) throw cancelledError()
}

function releaseSlot() {
  active -= 1
  const next = slotWaiters.shift()
  if (next) next()
}

async function acquireSlot() {
  if (active < init.limits.maxConcurrency) {
    active += 1
    return
  }
  await new Promise(resolve => slotWaiters.push(resolve))
  throwIfCancelled()
  active += 1
}

function readOptions(raw) {
  if (raw === undefined) return {}
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw fatal('agent() options must be an object', 'INVALID_ARGUMENT')
  }
  const supported = new Set(['label', 'phase', 'schema', 'provider', 'model'])
  for (const key of Object.keys(raw)) {
    if (!supported.has(key)) {
      throw fatal(
        `agent() option "${key}" is not recognized (supported: label, phase, schema, `
        + 'provider, model)',
        'UNSUPPORTED_OPTION',
      )
    }
  }
  for (const key of ['label', 'phase', 'provider', 'model']) {
    if (raw[key] !== undefined && typeof raw[key] !== 'string') {
      throw fatal(`agent() option "${key}" must be a string`, 'INVALID_ARGUMENT')
    }
  }
  if (raw.schema !== undefined && (
    raw.schema === null || typeof raw.schema !== 'object' || Array.isArray(raw.schema)
  )) {
    throw fatal('agent() schema must be an object-rooted JSON schema', 'UNSUPPORTED_SCHEMA')
  }
  return raw
}

function agent(rawPrompt, rawOptions) {
  return (async () => {
    throwIfCancelled()
    if (typeof rawPrompt !== 'string' || rawPrompt.length === 0) {
      throw fatal('agent() requires a non-empty prompt string', 'INVALID_ARGUMENT')
    }
    const options = readOptions(rawOptions)
    if (started >= init.limits.maxTotalAgents) {
      throw fatal(
        `this run reached its total agent cap (${init.limits.maxTotalAgents})`,
        'AGENT_CAP',
      )
    }
    started += 1
    const seq = started
    const label = options.label ?? rawPrompt.slice(0, 80)
    const phase = options.phase ?? currentPhase
    await acquireSlot()
    const callId = ++callSeq
    try {
      const value = await new Promise((resolve, reject) => {
        pending.set(callId, { resolve, reject })
        emit({ type: 'agent_request', callId, seq, prompt: rawPrompt, label, phase, options })
      })
      throwIfCancelled()
      if (value.fatal) {
        throw fatal(value.error ?? 'workflow child request failed', value.code ?? 'AGENT_START')
      }
      return value.ok ? value.value : null
    } finally {
      pending.delete(callId)
      releaseSlot()
    }
  })()
}

async function parallel(rawThunks) {
  throwIfCancelled()
  if (!Array.isArray(rawThunks)) {
    throw fatal('parallel() requires an array of zero-argument functions', 'INVALID_ARGUMENT')
  }
  if (rawThunks.length > init.limits.maxItemsPerCall) {
    throw fatal('parallel() exceeded the item cap', 'ITEM_CAP')
  }
  for (const [index, thunk] of rawThunks.entries()) {
    if (typeof thunk !== 'function') {
      throw fatal(`parallel() item ${index} is not a function`, 'INVALID_ARGUMENT')
    }
  }
  return Promise.all(rawThunks.map(async thunk => {
    try {
      return await thunk()
    } catch (error) {
      if (error?.fatal) throw error
      return null
    }
  }))
}

async function pipeline(rawItems, ...stages) {
  throwIfCancelled()
  if (!Array.isArray(rawItems)) {
    throw fatal('pipeline() requires an items array', 'INVALID_ARGUMENT')
  }
  if (rawItems.length > init.limits.maxItemsPerCall) {
    throw fatal('pipeline() exceeded the item cap', 'ITEM_CAP')
  }
  if (stages.length === 0) {
    throw fatal('pipeline() requires at least one stage function', 'INVALID_ARGUMENT')
  }
  for (const [index, stage] of stages.entries()) {
    if (typeof stage !== 'function') {
      throw fatal(`pipeline() stage ${index} is not a function`, 'INVALID_ARGUMENT')
    }
  }
  return Promise.all(rawItems.map(async (item, index) => {
    try {
      let value = item
      for (const stage of stages) value = await stage(value, item, index)
      return value
    } catch (error) {
      if (error?.fatal) throw error
      return null
    }
  }))
}

function phase(title) {
  throwIfCancelled()
  if (typeof title !== 'string' || title.length === 0) {
    throw fatal('phase() requires a non-empty title string', 'INVALID_ARGUMENT')
  }
  currentPhase = title
  emit({ type: 'phase', title })
}

function log(message) {
  throwIfCancelled()
  if (typeof message !== 'string') {
    throw fatal('log() requires a message string', 'INVALID_ARGUMENT')
  }
  emit({ type: 'log', message })
}

rl.on('line', line => {
  let message
  try { message = JSON.parse(line) } catch { return }
  if (message.type === 'agent_result') {
    const waiter = pending.get(message.callId)
    if (waiter) waiter.resolve(message)
  } else if (message.type === 'cancel') {
    cancelled = true
    for (const waiter of pending.values()) waiter.reject(cancelledError())
    while (slotWaiters.length > 0) slotWaiters.shift()()
  }
})

try {
  const sandbox = { agent, parallel, pipeline, phase, log, args: init.args }
  const context = vm.createContext(sandbox, { name: `workflow:${init.meta.name}` })
  const compiled = new vm.Script(
    `(async () => {\n${init.script}\n})()`,
    { filename: `workflow:${init.meta.name}` },
  )
  const raw = await compiled.runInContext(context, { timeout: init.limits.syncTimeoutMs })
  const value = raw === undefined ? null : JSON.parse(JSON.stringify(raw))
  finish({ value, stopReason: 'completed', agentsStarted: started })
} catch (error) {
  const stopReason = error?.code === 'CANCELLED' ? 'cancelled' : 'error'
  finish({
    value: null,
    stopReason,
    error: String(error?.message ?? error),
    code: error?.code,
    agentsStarted: started,
  })
}
'''


def validate_workflow_meta(meta: Any) -> dict[str, Any]:
    """Validate the plain JSON workflow identity block before spawning Node."""

    if not isinstance(meta, dict):
        raise WorkflowError("workflow meta must be an object", "META_INVALID")
    unknown = set(meta) - {"name", "description", "whenToUse", "phases"}
    if unknown:
        raise WorkflowError(
            f"workflow meta has unknown field(s): {', '.join(sorted(unknown))}",
            "META_INVALID",
        )
    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or not name:
        raise WorkflowError("workflow meta.name must be a non-empty string", "META_INVALID")
    if not isinstance(description, str) or not description:
        raise WorkflowError(
            "workflow meta.description must be a non-empty string", "META_INVALID"
        )
    result: dict[str, Any] = {"name": name, "description": description}
    for key in ("whenToUse",):
        value = meta.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise WorkflowError(f"workflow meta.{key} must be a string", "META_INVALID")
            result[key] = value
    phases = meta.get("phases")
    if phases is not None:
        if not isinstance(phases, list):
            raise WorkflowError("workflow meta.phases must be an array", "META_INVALID")
        normalized: list[dict[str, Any]] = []
        for index, phase in enumerate(phases):
            if not isinstance(phase, dict):
                raise WorkflowError(
                    f"workflow meta.phases[{index}] must be an object", "META_INVALID"
                )
            if set(phase) - {"title", "detail", "provider", "model"}:
                raise WorkflowError(
                    f"workflow meta.phases[{index}] has an unknown field",
                    "META_INVALID",
                )
            title = phase.get("title")
            if not isinstance(title, str) or not title:
                raise WorkflowError(
                    f"workflow meta.phases[{index}].title must be a non-empty string",
                    "META_INVALID",
                )
            item = {"title": title}
            for key in ("detail", "provider", "model"):
                value = phase.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise WorkflowError(
                            f"workflow meta.phases[{index}].{key} must be a string",
                            "META_INVALID",
                        )
                    item[key] = value
            normalized.append(item)
        result["phases"] = normalized
    return result


def validate_object_schema(schema: Any, *, path: str = "$", root: bool = True) -> None:
    """Validate the small object-rooted JSON Schema subset used by DSH."""

    if not isinstance(schema, dict):
        raise WorkflowError(f"{path} schema must be an object", "UNSUPPORTED_SCHEMA")
    allowed = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "oneOf",
    }
    unsupported = set(schema) - allowed
    if unsupported:
        raise WorkflowError(
            f"{path} schema uses unsupported field(s): {', '.join(sorted(unsupported))}",
            "UNSUPPORTED_SCHEMA",
        )
    if root and schema.get("type") != "object":
        raise WorkflowError("agent() schema must be object-rooted", "UNSUPPORTED_SCHEMA")
    if "oneOf" in schema:
        variants = schema["oneOf"]
        if not isinstance(variants, list) or not variants:
            raise WorkflowError(f"{path}.oneOf must be a non-empty array", "UNSUPPORTED_SCHEMA")
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                raise WorkflowError(
                    f"{path}.oneOf[{index}] must be an object", "UNSUPPORTED_SCHEMA"
                )
            validate_object_schema(variant, path=f"{path}.oneOf[{index}]", root=False)
    schema_type = schema.get("type")
    if schema_type is not None and schema_type not in {
        "object", "array", "string", "number", "integer", "boolean", "null", "json"
    }:
        raise WorkflowError(f"{path}.type is unsupported", "UNSUPPORTED_SCHEMA")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise WorkflowError(f"{path}.properties must be an object", "UNSUPPORTED_SCHEMA")
        for key, child in properties.items():
            if not isinstance(key, str):
                raise WorkflowError(f"{path}.properties has a non-string key", "UNSUPPORTED_SCHEMA")
            if not isinstance(child, dict):
                raise WorkflowError(
                    f"{path}.properties.{key} must be an object", "UNSUPPORTED_SCHEMA"
                )
            validate_object_schema(child, path=f"{path}.properties.{key}", root=False)
    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list) or any(not isinstance(item, str) for item in required)
    ):
        raise WorkflowError(f"{path}.required must be a string array", "UNSUPPORTED_SCHEMA")
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        raise WorkflowError(
            f"{path}.additionalProperties must be boolean", "UNSUPPORTED_SCHEMA"
        )
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise WorkflowError(f"{path}.items must be an object", "UNSUPPORTED_SCHEMA")
        validate_object_schema(items, path=f"{path}.items", root=False)


def matches_object_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether one child value satisfies a validated schema."""

    if "oneOf" in schema:
        return any(matches_object_schema(value, variant) for variant in schema["oneOf"])
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    schema_type = schema.get("type")
    if schema_type == "json":
        return True
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list) and (
            "items" not in schema
            or all(matches_object_schema(item, schema["items"]) for item in value)
        )
    if schema_type == "object":
        if not isinstance(value, dict):
            return False
        required = schema.get("required", [])
        if any(key not in value for key in required):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        return all(
            key not in properties or matches_object_schema(item, properties[key])
            for key, item in value.items()
        )
    return True


async def run_workflow(
    *,
    script: str,
    meta: dict[str, Any],
    args: dict[str, Any] | None,
    agent_runner: WorkflowAgentRunner,
    event_sink: WorkflowEventSink,
    max_total_agents: int = 64,
    max_concurrency: int = 8,
    max_items_per_call: int = 256,
    sync_timeout_ms: int = 1000,
) -> WorkflowResult:
    """Execute one script in a fresh Node VM and bridge child calls to Python."""

    normalized_meta = validate_workflow_meta(meta)
    if not isinstance(script, str):
        raise WorkflowError("workflow script must be a string", "SCRIPT_PARSE")
    if args is not None and not isinstance(args, dict):
        raise WorkflowError("workflow args must be an object", "INVALID_ARGUMENT")
    if max_total_agents < 1 or max_concurrency < 1 or max_items_per_call < 1:
        raise ValueError("workflow limits must be positive")
    node = shutil.which("node")
    if node is None:
        raise WorkflowError("Node.js is required for workflow scripts", "SCRIPT_RUNTIME")
    process = await asyncio.create_subprocess_exec(
        node,
        "--input-type=module",
        "-e",
        NODE_WORKER_SOURCE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    write_lock = asyncio.Lock()
    child_tasks: set[asyncio.Task[None]] = set()

    async def send(message: dict[str, Any]) -> None:
        if process.stdin is None:
            return
        payload = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        async with write_lock:
            process.stdin.write(payload)
            await process.stdin.drain()

    await send(
        {
            "type": "start",
            "script": script,
            "meta": normalized_meta,
            "args": args,
            "limits": {
                "maxTotalAgents": max_total_agents,
                "maxConcurrency": max_concurrency,
                "maxItemsPerCall": max_items_per_call,
                "syncTimeoutMs": sync_timeout_ms,
            },
        }
    )

    async def handle_agent(message: dict[str, Any]) -> None:
        call_id = message.get("callId")
        try:
            if not isinstance(call_id, int):
                raise WorkflowError("workflow agent call id is invalid", "AGENT_START")
            prompt = message.get("prompt")
            label = message.get("label")
            options = message.get("options")
            if (
                not isinstance(prompt, str)
                or not isinstance(label, str)
                or not isinstance(options, dict)
            ):
                raise WorkflowError("workflow agent request is invalid", "AGENT_START")
            schema = options.get("schema")
            if schema is not None:
                validate_object_schema(schema)

            async def on_started(child_id: str) -> None:
                await event_sink(
                    "agent-start",
                    {
                        "seq": message.get("seq"),
                        "label": label,
                        **(
                            {"phase": message["phase"]}
                            if isinstance(message.get("phase"), str)
                            else {}
                        ),
                        "childId": child_id,
                    },
                )

            result = await agent_runner(prompt, label, options, on_started)
            if not result.ok:
                await send(
                    {
                        "type": "agent_result",
                        "callId": call_id,
                        "ok": False,
                        "error": result.error or "workflow child failed",
                        "childId": result.child_id,
                    }
                )
                await event_sink(
                    "agent-end",
                    {
                        "seq": message.get("seq"),
                        "outcome": "failed",
                    },
                )
                return
            await send(
                {
                    "type": "agent_result",
                    "callId": call_id,
                    "ok": True,
                    "value": result.value,
                    "childId": result.child_id,
                }
            )
            await event_sink(
                "agent-end",
                {"seq": message.get("seq"), "outcome": "completed"},
            )
        except WorkflowError as exc:
            await send(
                {
                    "type": "agent_result",
                    "callId": call_id,
                    "ok": False,
                    "fatal": True,
                    "code": exc.code,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            await send(
                {
                    "type": "agent_result",
                    "callId": call_id,
                    "ok": False,
                    "fatal": True,
                    "code": "AGENT_RESULT",
                    "error": str(exc),
                }
            )

    try:
        while True:
            if process.stdout is None:
                raise WorkflowError("workflow worker has no stdout", "SCRIPT_RUNTIME")
            line = await process.stdout.readline()
            if not line:
                stderr = b""
                if process.stderr is not None:
                    stderr = await process.stderr.read()
                raise WorkflowError(
                    "workflow worker exited before settlement: "
                    f"{stderr.decode(errors='replace').strip()}",
                    "SCRIPT_RUNTIME",
                )
            message = json.loads(line)
            kind = message.get("type")
            if kind == "agent_request":
                task = asyncio.create_task(handle_agent(message))
                child_tasks.add(task)
                task.add_done_callback(child_tasks.discard)
            elif kind in {"phase", "log"}:
                await event_sink(kind, {key: message[key] for key in message if key != "type"})
            elif kind == "result":
                await asyncio.gather(*tuple(child_tasks))
                result = message.get("result")
                if not isinstance(result, dict):
                    raise WorkflowError(
                        "workflow worker returned an invalid result", "SCRIPT_RUNTIME"
                    )
                return WorkflowResult(
                    result.get("value"),
                    str(result.get("stopReason", "error")),
                    result.get("error") if isinstance(result.get("error"), str) else None,
                    int(result.get("agentsStarted", 0)),
                )
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
        await process.wait()
        raise
    finally:
        if process.returncode is None:
            if process.stdin is not None:
                process.stdin.close()
            await process.wait()


__all__ = [
    "WorkflowChildResult",
    "WorkflowError",
    "WorkflowResult",
    "matches_object_schema",
    "run_workflow",
    "validate_object_schema",
    "validate_workflow_meta",
]
