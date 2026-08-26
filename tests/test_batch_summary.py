import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from support import splitter


class BatchSummaryTests(unittest.TestCase):
    def test_partial_failure_report_contains_every_completed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a-broken.wav").write_bytes(b"broken")
            with wave.open(str(root / "b-valid.wav"), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\0\0" * 16000)
            summary = root / "summary.json"
            with patch.object(splitter, "detect_speech_segments", return_value=[splitter.Segment(0, 1)]), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(splitter.main(["--folder", str(root), "--continue-on-error", "--summary-json", str(summary)]), 1)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["exit_code"], 1)
            self.assertEqual([item["status"] for item in payload["results"]], ["failed", "exported"])
            self.assertEqual(payload["results"][1]["final_segment_count"], 1)
            self.assertTrue(Path(payload["results"][1]["manifest_path"]).is_file())

    def test_existing_summary_is_preserved_before_starting_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = Path(temporary) / "summary.json"
            summary.write_bytes(b"keep me")
            with patch.object(splitter, "run_pipeline") as run, redirect_stderr(io.StringIO()):
                self.assertEqual(splitter.main(["--audio", "source.wav", "--summary-json", str(summary)]), 1)
            run.assert_not_called()
            self.assertEqual(summary.read_bytes(), b"keep me")

    def test_empty_folder_reports_successful_empty_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(splitter.main(["--folder", str(root), "--summary-json", str(summary)]), 0)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"exit_code": 0, "results": []})
