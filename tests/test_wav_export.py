from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from support import splitter


class WavExportTests(unittest.TestCase):
    def test_long_stereo_segment_is_copied_in_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "stereo.wav"
            payload = bytes(range(256)) * 4096
            with wave.open(str(source), "wb") as writer:
                writer.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(payload)
            original_readframes = wave.Wave_read.readframes
            requested_sizes = []

            def readframes(reader, count):
                requested_sizes.append(count)
                return original_readframes(reader, count)

            with patch.object(wave.Wave_read, "readframes", readframes):
                outputs = splitter.split_wav_by_segments(
                    source, Path(temporary) / "out", [splitter.Segment(0, len(payload) / 4 / 16000)], "test",
                )
            self.assertGreater(len(requested_sizes), 1)
            self.assertLessEqual(max(requested_sizes), 65536)
            with wave.open(str(outputs[0]), "rb") as reader:
                self.assertEqual(reader.getnchannels(), 2)
                self.assertEqual(reader.getsampwidth(), 2)
                self.assertEqual(reader.getframerate(), 16000)
                self.assertEqual(reader.readframes(reader.getnframes()), payload)

    def test_truncated_payload_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "truncated.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\0\0" * 16000)
            source.write_bytes(source.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, "数据不完整"):
                splitter.split_wav_by_segments(source, Path(temporary) / "out", [splitter.Segment(0, 1)], "test")
