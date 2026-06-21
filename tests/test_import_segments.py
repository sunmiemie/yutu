import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from support import splitter


class ImportSegmentsTests(unittest.TestCase):
    def test_cli_exports_reviewed_segments_without_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\x01\x02" * 16000)
            labels = root / "labels.json"
            labels.write_text('[{"start_sec": 0.25, "end_sec": 0.75}]', encoding="utf-8-sig")
            with patch.object(splitter, "detect_speech_segments", side_effect=AssertionError("must bypass VAD")), redirect_stdout(io.StringIO()):
                self.assertEqual(splitter.main(["--audio", str(source), "--segments-json", str(labels), "--extra-pad-sec", "0"]), 0)
            with wave.open(str(root / "vad_output/source/source-001.wav"), "rb") as reader:
                self.assertEqual(reader.getnframes(), 8000)

    def test_invalid_manual_ranges_are_not_silently_clipped_or_coerced(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = Path(temporary) / "labels.json"
            for payload in [{}, [None], [{"start_sec": False, "end_sec": 1}],
                            [{"start_sec": "0", "end_sec": 1}], [{"start_sec": 0, "end_sec": float("nan")}],
                            [{"start_sec": -1, "end_sec": 1}], [{"start_sec": 0, "end_sec": 3}],
                            [{"start_sec": 1, "end_sec": 0}]]:
                labels.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    splitter.load_segments_json(labels, 2)
            labels.write_text('{"raw_segments": []}', encoding="utf-8")
            self.assertEqual(splitter.load_segments_json(labels, 2), [])
