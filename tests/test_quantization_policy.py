"""CPU-only checkpoint eligibility tests, independent of MLX or device policy."""

import unittest

from omlx.quantization_policy import (
    native_ratio128_checkpoint_eligible,
    normalize_checkpoint_quantization,
)


class QuantizationPolicyTests(unittest.TestCase):
    def config(self, metadata):
        return {"model_type": "deepseek_v4", "quantization": metadata}

    def test_each_location_and_nested_override_restricts_the_default(self):
        for nested in (False, True):
            for key in ("quantization", "quantization_config"):
                for bits in (2, 3, 3.5):
                    with self.subTest(nested=nested, key=key, bits=bits):
                        config = self.config({"bits": 4})
                        target = (
                            config.setdefault("text_config", {}) if nested else config
                        )
                        target[key] = {
                            "bits": 4,
                            "overrides": [{"layer.0": {"bits": bits}}],
                        }
                        normalized = normalize_checkpoint_quantization(config)
                        self.assertFalse(normalized.permits_ratio128)
                        self.assertTrue(
                            any(d.bits == bits for d in normalized.declarations)
                        )

    def test_malformed_explicit_bits_fail_closed(self):
        for bits in (
            True,
            False,
            None,
            "3",
            "4",
            0,
            -1,
            float("nan"),
            float("inf"),
            10**1000,
        ):
            with self.subTest(bits=bits):
                normalized = normalize_checkpoint_quantization(
                    self.config({"bits": bits})
                )
                self.assertFalse(normalized.permits_ratio128)
                self.assertEqual(normalized.invalid_paths, ("quantization.bits",))

    def test_unquantized_fp8_and_four_bit_metadata_remain_eligible(self):
        for metadata in ({}, {"quant_method": "fp8"}, {"bits": 4}, {"bits": 8}):
            with self.subTest(metadata=metadata):
                self.assertTrue(
                    native_ratio128_checkpoint_eligible(self.config(metadata))
                )

    def test_non_v4_policy_is_unchanged(self):
        self.assertTrue(
            native_ratio128_checkpoint_eligible(
                {"model_type": "qwen3", "quantization": {"bits": 2}}
            )
        )

    def test_annotates_paths_without_mutating_loader_config(self):
        config = self.config({"bits": 4, "layers": [{"bits": 8}]})
        result = normalize_checkpoint_quantization(config)
        self.assertEqual(
            [d.path for d in result.declarations],
            ["quantization.bits", "quantization.layers[0].bits"],
        )
        self.assertEqual(config, self.config({"bits": 4, "layers": [{"bits": 8}]}))

    def test_shared_metadata_is_valid_but_a_cycle_fails_closed(self):
        metadata = {"bits": 4}
        config = self.config(metadata)
        config["text_config"] = {"quantization_config": metadata}
        self.assertTrue(native_ratio128_checkpoint_eligible(config))
        metadata["cycle"] = metadata
        self.assertFalse(native_ratio128_checkpoint_eligible(config))


if __name__ == "__main__":
    unittest.main()
