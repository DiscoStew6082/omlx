"""CPU-only contracts; run with python -m unittest discover -s tests -p test_cache_conversion.py."""

import unittest

from omlx.cache_conversion import CacheConversionContext, convert_cache_layer


class Batch:
    def __init__(self, padding=(), auxiliary=None):
        self.padding = tuple(padding)
        self.auxiliary = auxiliary

    def extend(self, other):
        self.padding += other.padding

    def filter(self, indices):
        self.padding = tuple(self.padding[i] for i in indices)

    def extract(self, index):
        return Batch((self.padding[index],), self.auxiliary)


class ModelCache:
    def __init__(self, auxiliary=None):
        self.auxiliary = auxiliary
        self.calls = []

    def to_batch(self, padding):
        self.calls.append(padding.copy())
        return Batch(padding, self.auxiliary)


class CacheList:
    def __init__(self, *caches):
        self.caches = caches


class CacheConversionTests(unittest.TestCase):
    def fail_fallback(self, _):
        self.fail("model-owned cache reached the generic fallback")

    def test_fresh_rows_keep_padding_order_and_auxiliary_state(self):
        auxiliary = object()
        source = ModelCache(auxiliary)
        context = CacheConversionContext.fresh([2, 0, 4])
        result = convert_cache_layer(source, context, self.fail_fallback)
        self.assertEqual(context.row_ids, (0, 1, 2))
        self.assertEqual(result.padding, (2, 0, 4))
        self.assertIs(result.auxiliary, auxiliary)
        result.filter([2, 0])
        self.assertEqual(result.padding, (4, 2))
        self.assertIs(result.extract(0).auxiliary, auxiliary)

    def test_warm_singleton_preserves_state_and_has_no_added_padding(self):
        source = ModelCache({"index_keys": [1, 2], "positions": [5, 6]})
        result = convert_cache_layer(
            source, CacheConversionContext.warm_singleton(), self.fail_fallback
        )
        self.assertEqual(source.calls, [[0]])
        self.assertIs(result.auxiliary, source.auxiliary)

    def test_invalid_contexts_are_rejected(self):
        for phase, rows, padding in [
            ("fresh", (), ()),
            ("fresh", (0, 1), (0,)),
            ("fresh", (0, 0), (0, 0)),
            ("fresh", (0,), (-1,)),
            ("fresh", (0,), (True,)),
            ("fresh", (0,), (1.5,)),
            ("warm_singleton", (0, 1), (0, 0)),
            ("warm_singleton", (0,), (2,)),
            ("unknown", (0,), (0,)),
        ]:
            with (
                self.subTest(phase=phase, rows=rows, padding=padding),
                self.assertRaises(ValueError),
            ):
                CacheConversionContext(phase, rows, padding)

    def test_nested_mixed_cache_uses_one_dispatch(self):
        custom = ModelCache()
        standard = object()
        calls = []

        def fallback(cache):
            calls.append(cache)
            return Batch([3, 0])

        result = convert_cache_layer(
            CacheList(CacheList(custom), standard),
            CacheConversionContext.fresh([3, 0]),
            fallback,
        )
        self.assertEqual(custom.calls, [[3, 0]])
        self.assertEqual(calls, [standard])
        self.assertEqual(result.caches[0].caches[0].padding, (3, 0))

    def test_fresh_standard_container_is_owned_by_previous_patch(self):
        source = CacheList(object())
        called = []

        def previous_patch(cache):
            called.append(cache)
            return CacheList(Batch([0]))

        convert_cache_layer(source, CacheConversionContext.fresh([0]), previous_patch)
        self.assertEqual(called, [source])

    def test_warm_existing_batch_and_container_preserve_identity(self):
        batch = Batch([1, 0])
        container = CacheList(CacheList(batch))
        result = convert_cache_layer(
            container, CacheConversionContext.warm_singleton(), lambda cache: cache
        )
        self.assertIs(result, container)
        self.assertIs(result.caches[0].caches[0], batch)
        self.assertEqual(batch.padding, (1, 0))

    def test_invalid_hook_result_fails_at_conversion(self):
        class InvalidCache:
            def to_batch(self, padding):
                return object()

        with self.assertRaisesRegex(TypeError, "fresh.*extend, filter, extract"):
            convert_cache_layer(
                InvalidCache(), CacheConversionContext.fresh([0]), self.fail_fallback
            )

    def test_invalid_nested_fallback_fails_at_conversion(self):
        with self.assertRaisesRegex(TypeError, "warm_singleton.*batch operations"):
            convert_cache_layer(
                CacheList(object()),
                CacheConversionContext.warm_singleton(),
                lambda c: c,
            )

    def test_model_hook_error_is_not_silently_retried_or_fallen_back(self):
        class InvalidCache:
            def to_batch(self, padding):
                raise ValueError("unsupported layout")

        with self.assertRaisesRegex(ValueError, "unsupported layout"):
            convert_cache_layer(
                InvalidCache(), CacheConversionContext.fresh([0]), self.fail_fallback
            )

    def test_each_hook_gets_its_own_padding_list(self):
        class MutatingCache:
            def to_batch(self, padding):
                padding.pop()
                return Batch()

        source = ModelCache()
        convert_cache_layer(
            CacheList(MutatingCache(), source),
            CacheConversionContext.fresh([2, 0]),
            self.fail_fallback,
        )
        self.assertEqual(source.calls, [[2, 0]])


if __name__ == "__main__":
    unittest.main()
