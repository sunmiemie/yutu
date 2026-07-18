import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from support import splitter


class ResumeExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / "source.wav", self.root / "out"
        self.config = splitter.VadSplitConfig()
        with wave.open(str(self.source), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\0\0" * 16000)
        with patch.object(splitter, "detect_speech_segments", return_value=[splitter.Segment(0, 1)]), redirect_stdout(io.StringIO()):
            splitter.run_pipeline(self.source, self.output, self.config, "source")

    def test_valid_resume_skips_model_and_preserves_output(self):
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with patch.object(splitter, "detect_speech_segments", side_effect=AssertionError("should skip")), redirect_stdout(io.StringIO()):
            splitter.run_pipeline(self.source, self.output, self.config, "source", skip_existing=True)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_modified_input_config_or_output_never_silently_skips(self):
        self.assertFalse(splitter.can_resume_export(self.source, self.output, splitter.VadSplitConfig(threshold=.7), "source"))
        for path in [self.source, self.output / "source-001.wav", self.output / "source_vad_segments.csv"]:
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            self.assertFalse(splitter.can_resume_export(self.source, self.output, self.config, "source"))
            path.write_bytes(original)
        (self.output / "source-001.wav").unlink()
        with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            splitter.run_pipeline(self.source, self.output, self.config, "source", skip_existing=True)

    def test_malformed_and_legacy_manifests_do_not_qualify(self):
        path = self.output / "source_vad_segments.json"
        for payload in [None, {}, {"config": []}, "bad"]:
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(splitter.can_resume_export(self.source, self.output, self.config, "source"))
