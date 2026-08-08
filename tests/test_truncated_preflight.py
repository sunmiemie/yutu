import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from support import splitter


class TruncatedPreflightTests(unittest.TestCase):
    def test_incomplete_pcm_fails_before_inference_including_silent_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setparams((2, 3, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\0" * 600)
            source.write_bytes(source.read_bytes()[:-1])
            with patch.object(splitter, "detect_speech_segments", return_value=[]) as detect, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "数据不完整"):
                    splitter.run_pipeline(source, root / "out", splitter.VadSplitConfig(), "sample")
                detect.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_preflight_reads_only_last_frame_for_large_valid_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\0\0" * 32000)
            original = wave.Wave_read.readframes
            requests = []
            def read(reader, count):
                requests.append(count)
                return original(reader, count)
            with patch.object(wave.Wave_read, "readframes", read):
                self.assertEqual(splitter.get_wav_duration_sec(source), 2)
            self.assertEqual(requests, [1])
