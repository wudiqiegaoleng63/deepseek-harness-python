from __future__ import annotations

import asyncio

from deepseek_harness.runtime import Context, PluginRuntime, PluginSpec


def test_context_effects_and_plugin_disposal_are_reversible() -> None:
    async def scenario() -> None:
        runtime = PluginRuntime()
        disposed: list[str] = []

        async def apply(ctx: Context):
            ctx.provide("answer", 42)

            async def on_ping(value: str) -> None:
                disposed.append(value)

            ctx.subscribe("ping", on_ping)

            def dispose() -> None:
                disposed.append("disposed")

            return dispose

        scope = await runtime.install(PluginSpec("example", apply))
        assert scope.require("answer") == 42
        await runtime.root.events.emit("ping", "seen")
        assert disposed == ["seen"]

        await runtime.dispose()
        assert runtime.root.get("answer") is None
        assert disposed == ["seen", "disposed"]

    asyncio.run(scenario())


def test_waterfall_can_be_rewritten_or_stopped() -> None:
    async def scenario() -> None:
        ctx = Context()
        seen: list[str] = []

        async def rewrite(value: str, next_handler):
            seen.append(f"before:{value}")
            return await next_handler(value + "!")

        async def terminal(value: str) -> str:
            seen.append(f"terminal:{value}")
            return value

        ctx.subscribe("request", rewrite)
        result = await ctx.events.waterfall("request", "ok", terminal)
        assert result == "ok!"
        assert seen == ["before:ok", "terminal:ok!"]

        await ctx.dispose()

    asyncio.run(scenario())
