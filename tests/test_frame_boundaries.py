import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from support import splitter


class FrameBoundaryTests(unittest.TestCase):
    def test_frame_aligned_timestamps_preserve_exact_audio_and_manifest(self):
        for sample_rate in (8000, 16000, 44100, 48000):
            for imported in (False, True):
                with self.subTest(sample_rate=sample_rate, imported=imported):
                    self.check_export(sample_rate, 1001 / sample_rate, 2007 / sample_rate,
                                      1001, 2007, imported=imported)

    def test_fractional_frame_boundaries_still_round_outward(self):
        self.check_export(16000, 1001.25 / 16000, 2007.75 / 16000, 1001, 2008)

    def test_nearby_non_aligned_timestamps_are_not_snapped_to_frames(self):
        self.check_export(16000, math.nextafter(1001 / 16000, -math.inf),
                          math.nextafter(2007 / 16000, math.inf), 1000, 2008)

    def test_source_edges_and_one_frame_segments(self):
        self.check_export(16000, 0, 1001 / 16000, 0, 1001)
        self.check_export(16000, 1001 / 16000, 1002 / 16000, 1001, 1002)
        self.check_export(16000, 2007 / 16000, 4096 / 16000, 2007, 4096)

    def check_export(self, sample_rate, start, end, start_frame, end_frame, *, imported=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            # Unique stereo 16-bit frames make an off-by-one sample visible in the bytes.
            payload = b"".join(index.to_bytes(2, "little") * 2 for index in range(4096))
            with wave.open(str(source), "wb") as writer:
                writer.setparams((2, 2, sample_rate, 0, "NONE", "not compressed"))
                writer.writeframes(payload)
            segment = splitter.Segment(start, end)
            annotations = root / "segments.json" if imported else None
            if annotations is not None:
                annotations.write_text(json.dumps([{"start_sec": start, "end_sec": end}]),
                                       encoding="utf-8")
            config = splitter.VadSplitConfig(extra_pad_sec=0, absolute_min_duration_sec=0)
            with patch.object(splitter, "detect_speech_segments", return_value=[segment]) as detect, redirect_stdout(io.StringIO()):
                result = splitter.run_pipeline(source, root / "out", config, "sample",
                                               segments_json=annotations)
            if imported:
                detect.assert_not_called()
            else:
                detect.assert_called_once()
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            final = manifest["final_segments"][0]
            self.assertEqual((final["start_sec"], final["end_sec"]), (start, end))
            self.assertEqual((final["start_frame"], final["end_frame"]), (start_frame, end_frame))
            with wave.open(str(root / "out/sample-001.wav"), "rb") as reader:
                self.assertEqual(reader.getnframes(), end_frame - start_frame)
                self.assertEqual(reader.readframes(reader.getnframes()),
                                 payload[start_frame * 4:end_frame * 4])
