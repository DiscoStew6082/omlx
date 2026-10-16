# SPDX-License-Identifier: Apache-2.0
"""Model-owned cache conversion shared by creation and continuous-batch joins.

This module deliberately does not import MLX. Standard cache implementations and
patch-specific fallbacks remain owned by their callers. Conversion preserves the
whole cache object, including model-specific indices and positions.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class CacheConversionContext:
    """Row layout at a conversion boundary, never inferred from tensor shapes.

    ``row_ids`` are identities local to this operation, not scheduler request IDs.
    Fresh creation uses positional row IDs because MLX-LM's make-cache API has no
    request IDs. A warm singleton is already populated and has no added padding;
    its contents must survive conversion. Existing batches are passed through by
    the caller's fallback rather than interpreted as new singleton data.
    """

    phase: Literal["fresh", "warm_singleton"]
    row_ids: tuple[int, ...]
    left_padding: tuple[int, ...]

    def __post_init__(self):
        if self.phase not in ("fresh", "warm_singleton"):
            raise ValueError(f"Unknown cache conversion phase: {self.phase!r}")
        if len(self.row_ids) != len(self.left_padding) or not self.row_ids:
            raise ValueError("Cache conversion needs one padding value per row")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("Cache conversion row identities must be unique")
        if any(type(pad) is not int or pad < 0 for pad in self.left_padding):
            raise ValueError("Cache conversion padding must be nonnegative integers")
        if self.phase == "warm_singleton" and self.left_padding != (0,):
            raise ValueError("A warm singleton must be exactly one unpadded row")

    @classmethod
    def fresh(cls, left_padding):
        padding = tuple(left_padding)
        return cls("fresh", tuple(range(len(padding))), padding)

    @classmethod
    def warm_singleton(cls):
        return cls("warm_singleton", (0,), (0,))


def _children(cache):
    children = getattr(cache, "caches", None)
    return children if isinstance(children, (list, tuple)) else None


def has_model_owned_conversion(cache: Any) -> bool:
    if callable(getattr(cache, "to_batch", None)):
        return True
    children = _children(cache)
    return children is not None and any(map(has_model_owned_conversion, children))


def validate_batched_cache(cache: Any, context: CacheConversionContext) -> Any:
    """Require the operations used by batch extension, filtering and extraction.

    CacheList-like containers are traversed structurally, just as at the scheduler
    join boundary. This checks the public contract, not numerical correctness or
    hidden tensor state; model-owned conversion remains responsible for those.
    """
    children = _children(cache)
    if children is not None:
        for child in children:
            validate_batched_cache(child, context)
    else:
        missing = [
            name
            for name in ("extend", "filter", "extract")
            if not callable(getattr(cache, name, None))
        ]
        if missing:
            raise TypeError(
                f"{type(cache).__name__} after {context.phase} conversion is missing "
                f"batch operations: {', '.join(missing)}"
            )
    return cache


def convert_cache_layer(
    cache: Any,
    context: CacheConversionContext,
    fallback: Callable[[Any], Any],
) -> Any:
    """Use one model-owned dispatch for fresh caches and warm singletons.

    The hook receives a new padding list for compatibility with MLX cache APIs.
    Never recreate a model cache or round-trip its generic KV state here: doing so
    loses auxiliary state owned by models such as QSA. Standard MLX-LM fallbacks
    are supplied by the installation that owns them, preserving patch ordering.
    """
    to_batch = getattr(cache, "to_batch", None)
    children = _children(cache)
    if callable(to_batch):
        result = to_batch(list(context.left_padding))
    elif children is not None and (
        context.phase == "warm_singleton"
        or any(map(has_model_owned_conversion, children))
    ):
        converted = tuple(convert_cache_layer(c, context, fallback) for c in children)
        if context.phase == "warm_singleton" and all(
            a is b for a, b in zip(children, converted)
        ):
            result = cache
        else:
            result = type(cache)(*converted)
    else:
        result = fallback(cache)
    return validate_batched_cache(result, context)
