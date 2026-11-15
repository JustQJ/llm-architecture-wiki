#!/usr/bin/env python3
"""汇总各官方仓库 safetensors 实际体积与权重量化格式。

用法：python3 scripts/fetch_weight_sizes.py [--proxy http://127.0.0.1:7897]
输出：data/models/weight_sizes.json + 控制台表格。体积为仓库内 *.safetensors 字节总和（GB 按 1e9 十进制，与 HF 页面一致）。
"""
import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "models", "weight_sizes.json")

# 模型别名 -> 官方仓库列表（主仓在前；404 的探测项自动标 MISSING）
REPO_MAP = {
    "qwen3-30b-a3b": ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-FP8"],
    "qwen3-235b-a22b": ["Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-FP8"],
    "deepseek-v3": ["deepseek-ai/DeepSeek-V3"],
    "glm-4.5": ["zai-org/GLM-4.5", "zai-org/GLM-4.5-FP8"],
    "kimi-k2": ["moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Instruct-FP8"],
    "minimax-m2": ["MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-M2-BF16"],
    "hy4-preview": ["tencent/Hy4-preview", "tencent/Hy4-preview-FP8"],
    "hy3": ["tencent/Hy3", "tencent/Hy3-FP8"],
    "deepseek-v3.2": ["deepseek-ai/DeepSeek-V3.2"],
    "deepseek-v4-pro": ["deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-0813"],
    "deepseek-v4-flash": ["deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-0731"],
    "deepseek-v4.1-flash": ["deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash-1017"],
    "glm-5.1": ["zai-org/GLM-5.1", "zai-org/GLM-5.1-FP8"],
    "glm-5.2": ["zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8"],
    "glm-5.3": ["zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16"],
    "glm-5.3-flash": ["zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3-Flash-BF16"],
    "qwen3.5-397b-a17b": ["Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.5-397B-A17B-FP8", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"],
    "qwen3.5-122b-a10b": ["Qwen/Qwen3.5-122B-A10B", "Qwen/Qwen3.5-122B-A10B-FP8", "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"],
    "qwen3.5-35b-a3b": ["Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B-FP8", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"],
    "qwen3.5-27b": ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-27B-FP8", "Qwen/Qwen3.5-27B-GPTQ-Int4"],
    "qwen3.6-35b-a3b": ["Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B-FP8"],
    "qwen3.6-27b": ["Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B-FP8"],
    "qwen3.8-2.4t-a95b": ["Qwen/Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-2.4T-A95B-FP8"],
    "qwen3.8-27b": ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"],
    "qwen3.8-flash-next": ["Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next-FP8"],
    "kimi-k3": ["moonshotai/Kimi-K3", "moonshotai/Kimi-K3-BF16"],
    "minimax-m3": ["MiniMaxAI/MiniMax-M3", "MiniMaxAI/MiniMax-M3-MXFP8"],
    "minimax-m3.1-flash": ["MiniMaxAI/MiniMax-M3.1-Flash"],
}


def opener(proxy):
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def get_json(op, url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "weight-size-fetcher/1.0"})
    with op.open(req, timeout=timeout) as r:
        return json.load(r)


def get_raw(op, url, timeout=60):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "weight-size-fetcher/1.0"})
        with op.open(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def dtype_label(cfg):
    """从 config.json 推断权重量化标签。"""
    q = cfg.get("quantization_config")
    if isinstance(q, dict):
        fmt = q.get("format") or q.get("quant_method") or ""
        bits = q.get("weights", {}) or {}
        nb = bits.get("num_bits") if isinstance(bits, dict) else None
        if "mxfp4" in str(fmt) or (q.get("format") == "mxfp4-pack-quantized"):
            return "MXFP4"
        if "mxfp8" in str(fmt) or "modelopt" in str(fmt) and nb == 8:
            return "MXFP8"
        if str(fmt) in ("fp8", "compressed-tensors", "float8") or (
            str(fmt) == "compressed-tensors" and nb == 8
        ):
            return "FP8"
        if str(fmt) in ("gptq", "int4", "awq"):
            return f"{str(fmt).upper()}" + (f"-{nb}bit" if nb else "")
        return f"quant({fmt}{', ' + str(nb) + 'bit' if nb else ''})"
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and tc.get("quantization_config"):
        return dtype_label(tc)
    dt = cfg.get("dtype") or cfg.get("torch_dtype") or ""
    return {"bfloat16": "BF16", "float16": "FP16", "float32": "FP32", "float8": "FP8"}.get(dt, str(dt) or "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=None)
    args = ap.parse_args()
    op = opener(args.proxy)

    result = {}
    for alias, repos in REPO_MAP.items():
        result[alias] = []
        for repo in repos:
            entry = {"repo": repo}
            try:
                info = get_json(op, f"https://huggingface.co/api/models/{repo}?blobs=true")
            except Exception as e:
                entry.update({"status": "MISSING", "error": str(e)[:80]})
                result[alias].append(entry)
                print(f"[MISS] {repo}: {str(e)[:60]}")
                continue
            total = sum(
                s.get("size", 0)
                for s in info.get("siblings", [])
                if s["rfilename"].endswith(".safetensors")
            )
            cfg_raw = get_raw(op, f"https://huggingface.co/{repo}/raw/main/config.json")
            label = "?"
            if cfg_raw:
                try:
                    label = dtype_label(json.loads(cfg_raw))
                except Exception:
                    pass
            entry.update(
                {
                    "status": "OK",
                    "dtype": label,
                    "safetensors_bytes": total,
                    "GB": round(total / 1e9, 1),
                    "files": sum(
                        1 for s in info.get("siblings", []) if s["rfilename"].endswith(".safetensors")
                    ),
                }
            )
            print(f"[OK]   {repo:<55} {label:<16} {entry['GB']:>8} GB ({entry['files']} shards)")
            result[alias].append(entry)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
