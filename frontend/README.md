# Frontend integration

The UI is built from the original DSH React/TypeScript source in
`/home/lsy/deepseek-harness`; this directory only owns the Python-side build
output and the explicit browser roster. The Python server injects the same
`window.__DSH_BOOT__` graph and serves the copied client bundles under
`/plugins/*`.

Build it with:

```sh
uv run python -m deepseek_harness.frontend_build --ts-root /home/lsy/deepseek-harness
uv run dsh-python serve --web-dist frontend/dist
```

`dist/` is generated and intentionally not committed. Set `DSH_WEB_DIST` when
the server is started from another working directory.
