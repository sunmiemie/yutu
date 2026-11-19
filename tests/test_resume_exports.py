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

    def output_snapshot(self, output):
        return {str(path.relative_to(output)): None if path.is_dir() else path.read_bytes()
                for path in output.rglob("*")}

    def assert_resume_conflict_preserves_files(self, output):
        source_before = self.source.read_bytes()
        output_before = self.output_snapshot(output)
        self.assertFalse(splitter.can_resume_export(self.source, output, self.config, "source"))
        with patch.object(splitter, "detect_speech_segments", side_effect=AssertionError("must reject before VAD")) as detect:
            with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                splitter.run_pipeline(self.source, output, self.config, "source", skip_existing=True)
            detect.assert_not_called()
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(self.output_snapshot(output), output_before)

    def test_extra_numbered_outputs_reject_resume_without_changing_files(self):
        cases = [("source-099.wav", False), ("source-1000.wav", False),
                 ("SOURCE-099.WAV", False), ("source-01.wav", False),
                 ("source-099.wav", True)]
        for name, is_directory in cases:
            with self.subTest(name=name, is_directory=is_directory):
                extra = self.output / name
                if is_directory:
                    extra.mkdir()
                else:
                    extra.write_bytes(self.source.read_bytes())
                try:
                    self.assert_resume_conflict_preserves_files(self.output)
                finally:
                    if is_directory:
                        extra.rmdir()
                    else:
                        extra.unlink()

    def test_unrelated_files_still_allow_verified_resume(self):
        for name in ("other-099.wav", "notes.txt", "source-review.wav", "source-001.wav.bak"):
            (self.output / name).write_bytes(b"unrelated file to preserve")
        source_before = self.source.read_bytes()
        output_before = self.output_snapshot(self.output)
        self.assertTrue(splitter.can_resume_export(self.source, self.output, self.config, "source"))
        with patch.object(splitter, "detect_speech_segments", side_effect=AssertionError("should skip")) as detect:
            with redirect_stdout(io.StringIO()):
                result = splitter.run_pipeline(self.source, self.output, self.config, "source", skip_existing=True)
            detect.assert_not_called()
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(self.output_snapshot(self.output), output_before)

    def test_empty_segment_export_rejects_an_unlisted_wav(self):
        output = self.root / "empty-export"
        with patch.object(splitter, "detect_speech_segments", return_value=[]), redirect_stdout(io.StringIO()):
            splitter.run_pipeline(self.source, output, self.config, "source")
        self.assertTrue(splitter.can_resume_export(self.source, output, self.config, "source"))
        (output / "source-001.wav").write_bytes(self.source.read_bytes())
        self.assert_resume_conflict_preserves_files(output)

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
