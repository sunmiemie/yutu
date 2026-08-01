import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from support import splitter


class ConfigPresetTests(unittest.TestCase):
    def test_cli_overrides_preset_even_when_equal_to_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preset.json"
            path.write_text('{"threshold": 0.8, "extra_pad_sec": 0.1}', encoding="utf-8-sig")
            with patch.object(splitter, "run_pipeline") as run:
                self.assertEqual(splitter.main(["--audio", "sample.wav", "--config", str(path), "--threshold", "0.5"]), 0)
            config = run.call_args.args[2]
            self.assertEqual(config.threshold, .5)
            self.assertEqual(config.extra_pad_sec, .1)
            self.assertEqual(config.sampling_rate, 16000)

    def test_invalid_presets_fail_before_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preset.json"
            for payload in [[], {"threshold_typo": .7}, {"threshold": True}, {"extra_pad_sec": -1}, {"threshold": float("nan")}]:
                path.write_text(json.dumps(payload), encoding="utf-8")
                with patch.object(splitter, "run_pipeline") as run, redirect_stderr(io.StringIO()):
                    self.assertEqual(splitter.main(["--audio", "sample.wav", "--config", str(path)]), 1)
                    run.assert_not_called()
