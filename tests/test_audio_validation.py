from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from support import splitter


class AudioValidationTests(unittest.TestCase):
    def test_invalid_wav_is_reported_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "broken.wav"
            source.write_bytes(b"not an audio file")
            with patch.object(splitter, "detect_speech_segments") as detect:
                with self.assertRaisesRegex(ValueError, "PCM WAV"):
                    splitter.run_pipeline(source, Path(temporary) / "out", splitter.VadSplitConfig(), "test")
                detect.assert_not_called()

    def test_empty_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "empty.wav"
            with wave.open(str(source), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(b"")
            with self.assertRaisesRegex(ValueError, "采样帧"):
                splitter.get_wav_duration_sec(source)

    def test_sample_timestamps_preserve_sub_millisecond_segments(self):
        splitter.load_vad_model.cache_clear()
        self.addCleanup(splitter.load_vad_model.cache_clear)
        fake_vad = SimpleNamespace(
            load_silero_vad=Mock(return_value=object()),
            read_audio=Mock(return_value=object()),
            get_speech_timestamps=Mock(return_value=[{"start": 1, "end": 2}]),
        )
        with patch.dict(sys.modules, {"silero_vad": fake_vad}):
            raw = splitter.detect_speech_segments(Path("synthetic.wav"), splitter.VadSplitConfig())
        config = splitter.VadSplitConfig(extra_pad_sec=0, absolute_min_duration_sec=0)
        segments = splitter.post_process_segments(raw, config, 1)
        self.assertEqual(segments, [splitter.Segment(1 / 16000, 2 / 16000)])
        self.assertIs(fake_vad.get_speech_timestamps.call_args.kwargs["return_seconds"], False)

    def test_non_finite_boundaries_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    splitter.normalize_segments([splitter.Segment(value, 1)], 2)
                with self.assertRaises(ValueError):
                    splitter.normalize_segments([splitter.Segment(0, value)], 2)
                with self.assertRaises(ValueError):
                    splitter.normalize_segments([], value)

    def test_clipping_sorting_and_padding_preserve_precision(self):
        segments = splitter.normalize_segments([
            splitter.Segment(1.5, 3), splitter.Segment(-1, 0.0000625),
            splitter.Segment(1, 0.5),
        ], 2)
        self.assertEqual(segments, [splitter.Segment(0, 0.0000625), splitter.Segment(1.5, 2)])
        padded = splitter.add_extra_padding(segments[:1], 2, 0.0000625)
        self.assertEqual(padded[0].end_sec, 0.000125)
