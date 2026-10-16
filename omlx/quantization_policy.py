# SPDX-License-Identifier: Apache-2.0
"""Conservative checkpoint metadata admission, independent of hardware policy.

This is only the checkpoint-level ratio-128 eligibility gate. Attention still
checks the actual mask, shape, dtype and native-binary capability at dispatch.
A successful metadata decision never grants a broader kernel input domain.
"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuantizationBits:
    path: str
    bits: float


@dataclass(frozen=True)
class CheckpointQuantization:
    declarations: tuple[QuantizationBits, ...]
    invalid_paths: tuple[str, ...]

    @property
    def permits_ratio128(self) -> bool:
        return not self.invalid_paths and all(d.bits >= 4 for d in self.declarations)


def normalize_checkpoint_quantization(config: dict[str, Any]) -> CheckpointQuantization:
    """Inspect all existing config locations and nested per-layer overrides once.

    Absent bit declarations preserve the unquantized/FP8 loader paths. An explicit
    invalid bit declaration fails closed, including booleans, strings, NaN and
    infinity; it must not accidentally turn into permission for native attention.
    This does not rewrite the checkpoint or its loader's quantization format.
    """
    roots = [
        (name, config[name])
        for name in ("quantization", "quantization_config")
        if name in config
    ]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        roots.extend(
            (f"text_config.{name}", text_config[name])
            for name in ("quantization", "quantization_config")
            if name in text_config
        )
    declarations = []
    invalid = []

    def visit(path, value, ancestors):
        if not isinstance(value, (dict, list)):
            return
        if id(value) in ancestors:
            invalid.append(path)
            return
        ancestors = ancestors | {id(value)}
        if isinstance(value, dict):
            if "bits" in value:
                bits = value["bits"]
                try:
                    valid = (
                        isinstance(bits, (int, float))
                        and not isinstance(bits, bool)
                        and math.isfinite(bits)
                        and bits > 0
                    )
                except OverflowError:
                    valid = False
                if not valid:
                    invalid.append(f"{path}.bits")
                else:
                    declarations.append(QuantizationBits(f"{path}.bits", float(bits)))
            for key, child in value.items():
                if key != "bits":
                    visit(f"{path}.{key}", child, ancestors)
        else:
            for index, child in enumerate(value):
                visit(f"{path}[{index}]", child, ancestors)

    for path, value in roots:
        visit(path, value, set())
    return CheckpointQuantization(tuple(declarations), tuple(invalid))


def native_ratio128_checkpoint_eligible(config: dict[str, Any]) -> bool:
    if not str(config.get("model_type", "")).startswith("deepseek_v4"):
        return True
    return normalize_checkpoint_quantization(config).permits_ratio128
