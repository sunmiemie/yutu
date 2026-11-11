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


class DurationThresholdTests(unittest.TestCase):
    cases = ((0.2, 0.7, 0.5, 1.0), (3600.3, 3600.6, 0.3, 3601.0))

    def test_filter_keeps_exact_threshold_segments_without_rounding_timestamps(self):
        for start, end, threshold, _ in self.cases:
            with self.subTest(start=start, end=end, threshold=threshold):
                segment = splitter.Segment(start, end)
                result = splitter.filter_too_short_segments([segment], threshold)
                self.assertEqual(result, [splitter.Segment(start, end)])
                self.assertEqual(result[0].duration_sec, end - start)
                self.assertEqual(segment, splitter.Segment(start, end))

    def test_merge_keeps_exact_threshold_segments_separate(self):
        for start, end, threshold, next_start in self.cases:
            with self.subTest(start=start, end=end, threshold=threshold):
                segments = [splitter.Segment(start, end), splitter.Segment(next_start, next_start + 1)]
                result = splitter.merge_short_segments(segments, threshold)
                self.assertEqual(result, segments)
                self.assertEqual(segments[0], splitter.Segment(start, end))

    def test_one_sample_short_still_gets_filtered_or_merged(self):
        for start, end, threshold, next_start in self.cases:
            with self.subTest(start=start, end=end, threshold=threshold):
                short = splitter.Segment(start, end - 1 / 16000)
                self.assertEqual(splitter.filter_too_short_segments([short], threshold), [])
                result = splitter.merge_short_segments(
                    [short, splitter.Segment(next_start, next_start + 1)], threshold)
                self.assertEqual(result, [splitter.Segment(start, next_start + 1)])
                self.assertEqual(short.end_sec, end - 1 / 16000)

    def test_resolvably_short_duration_is_not_accepted_by_a_relative_tolerance(self):
        short = splitter.Segment(0.2, 0.7 - 1e-10)
        self.assertEqual(splitter.filter_too_short_segments([short], 0.5), [])
        result = splitter.merge_short_segments([short, splitter.Segment(1.0, 2.0)], 0.5)
        self.assertEqual(result, [splitter.Segment(0.2, 2.0)])
        self.assertEqual(short, splitter.Segment(0.2, 0.7 - 1e-10))

    def test_cli_exports_half_second_manual_segment_with_original_pcm_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, labels, output = root / "source.wav", root / "labels.json", root / "out"
            payload = b"".join(index.to_bytes(2, "little") for index in range(16000))
            with wave.open(str(source), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(payload)
            labels.write_text('[{"start_sec": 0.2, "end_sec": 0.7}]', encoding="utf-8")
            source_before, labels_before = source.read_bytes(), labels.read_bytes()
            with patch.object(splitter, "detect_speech_segments", side_effect=AssertionError("must bypass VAD")) as detect, redirect_stdout(io.StringIO()):
                result = splitter.main([
                    "--audio", str(source), "--segments-json", str(labels), "--output-dir", str(output),
                    "--extra-pad-sec", "0", "--absolute-min-duration-sec", "0.5",
                ])
            self.assertEqual(result, 0)
            detect.assert_not_called()
            manifest = json.loads((output / "source_vad_segments.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["final_segments"]), 1)
            for segment in (manifest["raw_segments"][0], manifest["final_segments"][0]):
                self.assertEqual((segment["start_sec"], segment["end_sec"]), (0.2, 0.7))
                self.assertEqual(segment["duration_sec"], 0.7 - 0.2)
            final = manifest["final_segments"][0]
            self.assertEqual((final["start_frame"], final["end_frame"]), (3200, 11200))
            with wave.open(str(output / "source-001.wav"), "rb") as reader:
                self.assertEqual(reader.getnframes(), 8000)
                self.assertEqual(reader.readframes(8000), payload[3200 * 2:11200 * 2])
            with (output / "source_vad_segments.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual((float(rows[0]["start_sec"]), float(rows[0]["end_sec"])), (0.2, 0.7))
            self.assertEqual(float(rows[0]["duration_sec"]), 0.7 - 0.2)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(labels.read_bytes(), labels_before)
