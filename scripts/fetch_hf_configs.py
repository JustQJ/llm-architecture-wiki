#!/usr/bin/env python3
"""
按 scripts/models.json 清单，从 Hugging Face 拉取各模型的 config.json 到 data/models/raw/<alias>.json。
仅依赖标准库。代理：自动读取环境变量 https_proxy/HTTPS_PROXY，或 --proxy 显式指定。

用法：
  python3 scripts/fetch_hf_configs.py                  # 拉取清单全部模型
  python3 scripts/fetch_hf_configs.py qwen3-30b-a3b     # 只拉指定 alias
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "models.json"
OUT_DIR = ROOT / "data" / "models" / "raw"
HF_BASE = "https://huggingface.co"


def fetch(repo, file, alias, proxy):
    url = f"{HF_BASE}/{repo}/raw/main/{file}"
    out = OUT_DIR / f"{alias}.json"
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers={"User-Agent": "llm-architecture-wiki/0.1"})
    try:
        with opener.open(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {alias} ({repo}): {e}")
        return False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK]   {alias} <- {repo}/{file}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aliases", nargs="*", help="只拉取指定 alias（默认全部）")
    ap.add_argument("--proxy", default=None, help="例如 http://127.0.0.1:7897")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())["models"]
    targets = [m for m in manifest if not args.aliases or m["alias"] in args.aliases]
    if not targets:
        print("manifest 为空或 alias 未匹配")
        return 1
    ok = all(fetch(m["repo"], m.get("file", "config.json"), m["alias"], args.proxy) for m in targets)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
