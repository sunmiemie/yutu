from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from support import splitter


class PipelineOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.wav"
        self.output = self.root / "output"
        with wave.open(str(self.source), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\x01\x02" * 16000)
        self.config = splitter.VadSplitConfig(extra_pad_sec=0)

    def run_pipeline(self):
        with redirect_stdout(io.StringIO()):
            splitter.run_pipeline(self.source, self.output, self.config, "test")

    def test_previous_exports_are_not_overwritten_or_mixed(self):
        self.output.mkdir()
        old_export = self.output / "test-099.wav"
        old_export.write_bytes(b"keep previous output")
        with patch.object(splitter, "detect_speech_segments") as detect:
            with self.assertRaises(FileExistsError):
                self.run_pipeline()
            detect.assert_not_called()
        self.assertEqual(old_export.read_bytes(), b"keep previous output")

    def test_split_failure_does_not_publish_manifests_or_partial_audio(self):
        self.source.write_bytes(self.source.read_bytes()[:-10])
        with patch.object(splitter, "detect_speech_segments", return_value=[splitter.Segment(0, 1)]):
            with self.assertRaisesRegex(ValueError, "数据不完整"):
                self.run_pipeline()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_successful_export_has_matching_audio_and_manifest(self):
        with patch.object(splitter, "detect_speech_segments", return_value=[splitter.Segment(0.25, 0.75)]):
            self.run_pipeline()
        payload = json.loads((self.output / "test_vad_segments.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["final_segments"]), 1)
        audio = self.output / payload["final_segments"][0]["segment_name"]
        with wave.open(str(audio), "rb") as reader:
            self.assertEqual(reader.getnframes(), 8000)
        self.assertEqual(len(list(self.output.iterdir())), 3)

    def test_publish_failure_removes_only_new_files(self):
        self.output.mkdir()
        unrelated = self.output / "notes.txt"
        unrelated.write_text("keep me", encoding="utf-8")
        copy = splitter.shutil.copyfileobj
        calls = 0

        def fail_second_copy(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated full disk")
            return copy(source, destination)

        with patch.object(splitter, "detect_speech_segments", return_value=[splitter.Segment(0, 1)]):
            with patch.object(splitter.shutil, "copyfileobj", side_effect=fail_second_copy):
                with self.assertRaisesRegex(OSError, "full disk"):
                    self.run_pipeline()
        self.assertEqual(list(self.output.iterdir()), [unrelated])
