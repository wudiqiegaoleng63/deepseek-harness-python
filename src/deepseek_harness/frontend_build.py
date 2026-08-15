"""Build and adapt the upstream DSH browser shell for the Python host."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROSTER_PATH = PROJECT_ROOT / "frontend" / "roster.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ts-root",
        type=Path,
        default=Path("/home/lsy/deepseek-harness"),
        help="Local DSH TypeScript checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "frontend" / "dist",
        help="Python host frontend output directory.",
    )
    args = parser.parse_args()
    build(Path(args.ts_root).expanduser().resolve(), Path(args.output).expanduser().resolve())


def build(ts_root: Path, output: Path) -> None:
    """Build the upstream shell and copy the exact browser roster."""

    subprocess.run(["pnpm", "run", "build:lib"], cwd=ts_root, check=True)
    subprocess.run(["pnpm", "run", "build:web"], cwd=ts_root, check=True)
    source_dist = ts_root / "apps" / "web" / "dist"
    if not source_dist.is_dir():
        raise FileNotFoundError(f"upstream frontend dist is missing: {source_dist}")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source_dist, output)

    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    package_dirs = _package_dirs(ts_root)
    entries: list[dict[str, Any]] = []
    for package_name in roster:
        package_dir = package_dirs.get(package_name)
        if package_dir is None:
            raise FileNotFoundError(f"browser roster package is missing: {package_name}")
        package = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
        bundle = package_dir / "lib" / "client.js"
        if not bundle.is_file():
            raise FileNotFoundError(f"browser bundle is missing: {bundle}")
        target = output / "plugins" / package_name / "client.js"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle, target)
        config = package.get("dsh", {}).get("client", {})
        rev = hashlib.sha1(bundle.read_bytes()).hexdigest()[:12]
        entries.append(
            {
                "id": package_name,
                "url": f"/plugins/{package_name}/client.js?rev={rev}",
                "rev": rev,
                **({"inject": config["inject"]} if config.get("inject") else {}),
                **({"immediately": True} if config.get("immediately") else {}),
            }
        )
    graph_payload: dict[str, Any] = {"entries": entries}
    graph_payload["rev"] = hashlib.sha1(
        json.dumps(graph_payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:12]
    (output / "boot.json").write_text(
        json.dumps(graph_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    index = output / "index.html"
    html = index.read_text(encoding="utf-8")
    html = html.replace("<title>DeepSeek Harness</title>", "<title>DeepSeek Harness Python</title>")
    index.write_text(html, encoding="utf-8")
    manifest = output / "manifest.webmanifest"
    if manifest.is_file():
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                '"DeepSeek Harness"', '"DeepSeek Harness Python"'
            ),
            encoding="utf-8",
        )
    print(f"Frontend copied to {output}")


def _package_dirs(ts_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for package_json in ts_root.glob("packages/**/package.json"):
        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = package.get("name")
        if isinstance(name, str):
            result[name] = package_json.parent
    return result


if __name__ == "__main__":
    main()
