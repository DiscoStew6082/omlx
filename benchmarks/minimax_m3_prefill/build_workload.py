#!/usr/bin/env python3
"""Build or verify the fixed MiniMax M3 long-code benchmark workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.environ.get("OMLX_MINIMAX_MODEL", "models/MiniMax-M3-4bit")
)
EVIDENCE_ROOT = REPO_ROOT / "benchmarks/evidence/minimax_m3_prefill_m3_ultra"
WORKLOAD_ROOT = EVIDENCE_ROOT / "workload"
TARGET_TOKENS = 16_512
SHORT_TARGET_TOKENS = 2_048
MIN_TOKENS = 16_000
MAX_TOKENS = 17_000
SENTINEL = "MINIMAX-M3-PREFILL-16K-V1"

SOURCE_FILES = (
    "omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/"
    "minimax_m3_vl/language.py",
    "omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/minimax_m3_vl/msa.py",
    "omlx/scheduler.py",
)

TASK = f"""

## Review task

Use only the code and model facts above. Respond with one JSON object and no
Markdown. It must contain exactly these keys:

- `sentinel`: the exact string `{SENTINEL}`
- `sparse_topk_blocks`: the integer sparse top-k block count
- `sparse_block_size`: the integer sparse block size
- `num_experts_per_tok`: the integer routed-expert count per token
- `sparse_layer_count`: the integer number of sparse/MoE layers
- `topk_builder`: either valid grouped query-to-key block-index builder in the code
- `attention_executor`: the function that executes B=1 sparse prefill attention
- `bottleneck_candidate`: one performance hypothesis, at most 20 words
- `correctness_risk`: one optimization risk, at most 20 words
"""

HEADER = """# MiniMax M3 prefill review bundle

Frozen model facts:
- num_hidden_layers = 60
- dense prefix layers = 3
- sparse/MoE layers = 57
- num_local_experts = 128
- num_experts_per_tok = 4
- sparse_topk_blocks = 16
- sparse_block_size = 128
- sparse_index_dim = 128
- quantization = affine 4-bit, group_size 64
- MoE router gates = affine 8-bit

The following files are from untouched oMLX upstream commit
ded2bbe4bb30b37dc291a0f76dd34932dbf90ee9.
"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_sources() -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    manifest: list[dict[str, Any]] = []
    for rel in SOURCE_FILES:
        path = REPO_ROOT / rel
        data = path.read_bytes()
        text = data.decode("utf-8")
        parts.append(f"\n\n## FILE: {rel}\n\n```python\n{text}\n```\n")
        manifest.append(
            {
                "path": rel,
                "sha256": _sha256_bytes(data),
                "bytes": len(data),
            }
        )
    return "".join(parts), manifest


def _encode(tokenizer: Any, text: str) -> list[int]:
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None:
        return list(backend.encode(text, add_special_tokens=False).ids)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    return [int(token) for token in encoded]


def _render(tokenizer: Any, source_prefix: str) -> tuple[str, list[int]]:
    # The fitted source prefix can end in the middle of a file, so close the
    # final code fence explicitly before the fixed review task.
    user_content = HEADER + source_prefix + "\n```\n" + TASK
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        thinking_mode="disabled",
    )
    return prompt, _encode(tokenizer, prompt)


def _fit_workload(
    tokenizer: Any, source_bundle: str, target_tokens: int
) -> tuple[str, list[int], int]:
    low = 0
    high = len(source_bundle)
    best: tuple[str, list[int], int] | None = None
    while low <= high:
        middle = (low + high) // 2
        prompt, token_ids = _render(tokenizer, source_bundle[:middle])
        if len(token_ids) <= target_tokens:
            best = (prompt, token_ids, middle)
            low = middle + 1
        else:
            high = middle - 1

    if best is None:
        raise RuntimeError("Fixed workload header/task exceeds target token count")
    _, _, source_chars = best
    source_prefix = source_bundle[:source_chars]
    newline = source_prefix.rfind("\n")
    if newline >= 0:
        source_prefix = source_prefix[: newline + 1]
    prompt, token_ids = _render(tokenizer, source_prefix)
    return prompt, token_ids, len(source_prefix)


def build() -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    source_bundle, source_manifest = _read_sources()

    prompt, token_ids, source_chars = _fit_workload(
        tokenizer, source_bundle, TARGET_TOKENS
    )
    if not MIN_TOKENS <= len(token_ids) <= MAX_TOKENS:
        raise RuntimeError(
            f"Built workload has {len(token_ids)} tokens; expected "
            f"{MIN_TOKENS}-{MAX_TOKENS}"
        )

    short_prompt, short_token_ids, short_source_chars = _fit_workload(
        tokenizer, source_bundle, SHORT_TARGET_TOKENS
    )

    prompt_bytes = prompt.encode("utf-8")
    token_bytes = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    short_prompt_bytes = short_prompt.encode("utf-8")
    short_token_bytes = json.dumps(short_token_ids, separators=(",", ":")).encode(
        "utf-8"
    )
    manifest = {
        "schema_version": 1,
        "name": "minimax-m3-long-code-prefill-v1",
        "runtime_baseline_commit": "ded2bbe4bb30b37dc291a0f76dd34932dbf90ee9",
        "model_path": str(MODEL_PATH),
        "target_tokens": TARGET_TOKENS,
        "prompt_tokens": len(token_ids),
        "source_chars_included": source_chars,
        "prompt_sha256": _sha256_bytes(prompt_bytes),
        "token_ids_sha256": _sha256_bytes(token_bytes),
        "short_control": {
            "target_tokens": SHORT_TARGET_TOKENS,
            "prompt_tokens": len(short_token_ids),
            "source_chars_included": short_source_chars,
            "prompt_sha256": _sha256_bytes(short_prompt_bytes),
            "token_ids_sha256": _sha256_bytes(short_token_bytes),
        },
        "sentinel": SENTINEL,
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 0,
            "seed": 1729,
            "thinking_mode": "disabled",
        },
        "quality_answer": {
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "num_experts_per_tok": 4,
            "sparse_layer_count": 57,
            "topk_builder": [
                "build_grouped_msa_topk",
                "build_grouped_msa_topk_blockwise",
            ],
            "attention_executor": "msa_sparse_attention_b1",
        },
        "sources": source_manifest,
    }

    WORKLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    (WORKLOAD_ROOT / "prompt.txt").write_bytes(prompt_bytes)
    (WORKLOAD_ROOT / "prompt_token_ids.json").write_bytes(token_bytes + b"\n")
    (WORKLOAD_ROOT / "short_prompt.txt").write_bytes(short_prompt_bytes)
    (WORKLOAD_ROOT / "short_prompt_token_ids.json").write_bytes(
        short_token_bytes + b"\n"
    )
    (WORKLOAD_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def verify() -> dict[str, Any]:
    manifest_path = WORKLOAD_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prompt_bytes = (WORKLOAD_ROOT / "prompt.txt").read_bytes()
    token_ids = json.loads((WORKLOAD_ROOT / "prompt_token_ids.json").read_text())
    token_bytes = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    short_prompt_bytes = (WORKLOAD_ROOT / "short_prompt.txt").read_bytes()
    short_token_ids = json.loads(
        (WORKLOAD_ROOT / "short_prompt_token_ids.json").read_text()
    )
    short_token_bytes = json.dumps(short_token_ids, separators=(",", ":")).encode(
        "utf-8"
    )

    checks = {
        "prompt_tokens": len(token_ids) == manifest["prompt_tokens"],
        "token_range": MIN_TOKENS <= len(token_ids) <= MAX_TOKENS,
        "prompt_sha256": _sha256_bytes(prompt_bytes) == manifest["prompt_sha256"],
        "token_ids_sha256": (
            _sha256_bytes(token_bytes) == manifest["token_ids_sha256"]
        ),
        "short_prompt_tokens": (
            len(short_token_ids) == manifest["short_control"]["prompt_tokens"]
        ),
        "short_prompt_sha256": (
            _sha256_bytes(short_prompt_bytes)
            == manifest["short_control"]["prompt_sha256"]
        ),
        "short_token_ids_sha256": (
            _sha256_bytes(short_token_bytes)
            == manifest["short_control"]["token_ids_sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Workload verification failed: {checks}")
    return {"ok": True, "checks": checks, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else build()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
