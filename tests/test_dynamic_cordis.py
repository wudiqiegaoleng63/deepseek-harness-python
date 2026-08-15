from __future__ import annotations

import asyncio
import json

from deepseek_harness.dynamic_cordis import DynamicCordisService
from deepseek_harness.tools.registry import ToolContext, ToolRegistry

HOST_CODE = """
harness.handle('double', async (args) => args.value * 2)
return {
  name: 'doubler',
  apply(ctx) { ctx.provide('dynDoubler', { ok: true }) }
}
"""


def test_dynamic_host_package_runs_and_invokes_javascript_handler() -> None:
    async def scenario() -> None:
        events: list[tuple[str, list[object]]] = []
        runner = DynamicCordisService(remote_event=lambda name, args: events.append((name, args)))
        receipt = runner.define(
            "session-dynamic",
            {"kind": "new", "idPrefix": "dyn"},
            "doubler",
            "double a value",
            {"host": HOST_CODE},
        )
        started = await runner.run(
            "session-dynamic", str(receipt["pluginId"]), str(receipt["packageId"]), "run"
        )
        assert started["ok"] is True
        assert started["status"] == "running"
        invoked = await runner.invoke(
            str(receipt["pluginId"]), str(started["pluginRunId"]), "double", {"value": 21}
        )
        assert invoked == {"ok": True, "value": 42}
        assert events[0][0] == "cordis/dynamic-package"
        stopped = await runner.stop("session-dynamic", str(receipt["pluginId"]))
        assert stopped == {"ok": True}
        assert events[-1][0] == "cordis/dynamic-retract"

    asyncio.run(scenario())


def test_dynamic_client_package_round_trip_keeps_approval_and_current_version() -> None:
    async def scenario() -> None:
        events: list[tuple[str, list[object]]] = []
        runner = DynamicCordisService(remote_event=lambda name, args: events.append((name, args)))
        receipt = runner.define(
            "session-client",
            {"kind": "new", "idPrefix": "uiux"},
            "browser package",
            "shows a browser surface",
            {"client": "return { inject: [], apply() {} }"},
        )
        started = await runner.run(
            "session-client", str(receipt["pluginId"]), str(receipt["packageId"]), "run"
        )
        assert started["status"] == "awaiting-approval"
        request_id = str(started["pluginRunId"])  # only used to find the row below
        row = runner.inventory()[0]
        request_id = str(row["latestRun"]["approvalRequestId"])
        host = await runner.run_host_half(
            "session-client",
            str(receipt["pluginId"]),
            str(receipt["packageId"]),
            "run",
            request_id,
            False,
        )
        assert host["ok"] is True
        source = runner.get_client_code(
            "session-client", str(receipt["pluginId"]), str(host["pluginRunId"])
        )
        assert source["code"].startswith("return")
        settled = await runner.resolve_request_run(
            request_id,
            {"ok": True, "pluginRunId": host["pluginRunId"], "waitingFor": []},
        )
        assert settled == {"accepted": True}
        assert runner.inventory()[0]["currentPackageId"] == receipt["packageId"]
        assert any(name == "cordis/request-run" for name, _ in events)
        assert any(name == "cordis/request-run-resolved" for name, _ in events)

    asyncio.run(scenario())


def test_dynamic_host_tool_is_registered_and_executes() -> None:
    async def scenario() -> None:
        registry = ToolRegistry()
        registries = {"session-tool": registry}
        runner = DynamicCordisService(tool_registries=registries)
        receipt = runner.define(
            "session-tool",
            {"kind": "new", "idPrefix": "tool"},
            "tool package",
            "registers a tool",
            {
                "host": """
return {
  apply(ctx) {
    ctx.tools.register({
      name: 'double_tool',
      description: 'double a number',
      parameters: { type: 'object', properties: { value: { type: 'number' } } },
      execute: async (args) => ({ value: args.value * 2 })
    })
  }
}
"""
            },
        )
        await runner.run("session-tool", str(receipt["pluginId"]), str(receipt["packageId"]), "run")
        result = await registry.execute(
            "double_tool",
            json.dumps({"value": 9}),
            ToolContext("session-tool", "."),
        )
        assert not result.is_error
        assert json.loads(result.text) == {"value": 18}
        await runner.stop("session-tool", str(receipt["pluginId"]))
        assert registry.get("double_tool") is None

    asyncio.run(scenario())
