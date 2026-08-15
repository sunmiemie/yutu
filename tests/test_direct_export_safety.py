from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from support import splitter


class DirectExportSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / "source.wav", self.root / "out"
        with wave.open(str(self.source), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\0\0" * 16000)

    def test_direct_export_preflights_all_files_and_confines_prefix(self):
        self.output.mkdir()
        existing = self.output / "sample-002.wav"
        existing.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            splitter.split_wav_by_segments(self.source, self.output,
                [splitter.Segment(0, .5), splitter.Segment(.5, 1)], "../sample")
        self.assertEqual(list(self.output.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), b"keep")
        self.assertFalse((self.root / "sample-001.wav").exists())

    def test_failed_direct_write_removes_only_files_it_created(self):
        original = wave.Wave_write.writeframesraw
        calls = 0
        def write(writer, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk full")
            return original(writer, data)
        with patch.object(wave.Wave_write, "writeframesraw", write), self.assertRaises(OSError):
            splitter.split_wav_by_segments(self.source, self.output,
                [splitter.Segment(0, .5), splitter.Segment(.5, 1)], "sample")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_all_boundaries_checked_before_any_output(self):
        with self.assertRaises(ValueError):
            splitter.split_wav_by_segments(self.source, self.output,
                [splitter.Segment(0, .5), splitter.Segment(float("nan"), 1)], "sample")
        self.assertFalse(self.output.exists())
