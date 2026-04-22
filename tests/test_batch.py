from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from support import splitter


class BatchTests(unittest.TestCase):
    def test_model_is_loaded_once_for_sequential_audio_files(self):
        splitter.load_vad_model.cache_clear()
        self.addCleanup(splitter.load_vad_model.cache_clear)
        fake_vad = SimpleNamespace(
            load_silero_vad=Mock(return_value=object()),
            read_audio=Mock(return_value=object()),
            get_speech_timestamps=Mock(return_value=[{"start": 0, "end": 8000}]),
        )
        with patch.dict(sys.modules, {"silero_vad": fake_vad}):
            for source in ("first.wav", "second.wav"):
                result = splitter.detect_speech_segments(Path(source), splitter.VadSplitConfig())
                self.assertEqual(result, [splitter.Segment(0, 0.5)])
        self.assertEqual(fake_vad.load_silero_vad.call_count, 1)
        self.assertEqual(fake_vad.read_audio.call_count, 2)
        self.assertEqual(fake_vad.get_speech_timestamps.call_count, 2)

    def test_continue_mode_exports_valid_files_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a-broken.wav").write_bytes(b"broken")
            with wave.open(str(root / "b-valid.wav"), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\0\0" * 16000)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(splitter, "detect_speech_segments", return_value=[splitter.Segment(0, 1)]) as detect:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = splitter.main(["--folder", str(root), "--continue-on-error"])
            self.assertEqual(result, 1)
            self.assertIn("成功 1 个，失败 1 个", stdout.getvalue())
            self.assertIn("a-broken.wav", stderr.getvalue())
            self.assertTrue((root / "vad_output" / "b-valid" / "b-valid-001.wav").is_file())
            self.assertEqual(detect.call_count, 1)

    def test_default_batch_stops_on_first_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a.wav", "b.wav"):
                (root / name).write_bytes(b"broken")
            with patch.object(splitter, "run_pipeline", side_effect=ValueError("invalid input")) as run:
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "invalid input"):
                        splitter.run_folder_pipeline(root, splitter.VadSplitConfig())
                self.assertEqual(run.call_count, 1)
