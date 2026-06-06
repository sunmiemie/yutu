import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from support import splitter


class ManifestPrecisionTests(unittest.TestCase):
    def test_submillisecond_times_and_sample_range_match_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\x01\x02\x03\x04" * 17)
            segment = splitter.Segment(1 / 16000, 2 / 16000)
            config = splitter.VadSplitConfig(extra_pad_sec=0, absolute_min_duration_sec=0)
            with patch.object(splitter, "detect_speech_segments", return_value=[segment]), redirect_stdout(io.StringIO()):
                splitter.run_pipeline(source, root / "out", config, "sample")
            manifest = json.loads((root / "out/sample_vad_segments.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["audio_duration_sec"], 17 / 16000)
            self.assertEqual(manifest["source_format"], {"sample_rate": 16000, "frame_count": 17, "channels": 2, "sample_width_bytes": 2})
            final = manifest["final_segments"][0]
            self.assertEqual((final["start_frame"], final["end_frame"]), (1, 2))
            self.assertEqual(final["duration_sec"], 1 / 16000)
            with (root / "out/sample_vad_segments.csv").open(encoding="utf-8-sig", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(float(row["start_sec"]), segment.start_sec)
            self.assertEqual(float(row["duration_sec"]), segment.duration_sec)
            with wave.open(str(root / "out/sample-001.wav"), "rb") as reader:
                self.assertEqual(reader.getnframes(), final["end_frame"] - final["start_frame"])
