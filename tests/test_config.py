from dataclasses import asdict, replace
import unittest

from support import splitter


class ConfigTests(unittest.TestCase):
    def test_default_and_zero_optional_durations(self):
        splitter.validate_config(splitter.VadSplitConfig())
        splitter.validate_config(splitter.VadSplitConfig(
            min_silence_duration_ms=0, speech_pad_ms=0,
            merge_gap_sec=0, min_segment_duration_sec=0,
            extra_pad_sec=0, absolute_min_duration_sec=0,
        ))

    def test_all_config_values_reject_non_finite_numbers(self):
        config = splitter.VadSplitConfig()
        for field in asdict(config):
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=invalid):
                    with self.assertRaisesRegex(ValueError, field):
                        splitter.validate_config(replace(config, **{field: invalid}))

    def test_negative_padding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "speech_pad_ms"):
            splitter.validate_config(splitter.VadSplitConfig(speech_pad_ms=-1))

    def test_non_numeric_values_are_rejected_with_field_name(self):
        for invalid in (None, "0.5", True):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    splitter.validate_config(splitter.VadSplitConfig(threshold=invalid))
