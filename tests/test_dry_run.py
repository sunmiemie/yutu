import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from support import splitter


class DryRunTests(unittest.TestCase):
    def test_preview_processes_segments_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\0\0" * 16000)
            stdout = io.StringIO()
            with patch.object(splitter, "detect_speech_segments", return_value=[splitter.Segment(.25, .5)]), redirect_stdout(stdout):
                self.assertEqual(splitter.main(["--audio", str(source), "--dry-run"]), 0)
            self.assertEqual(list(root.iterdir()), [source])
            self.assertIn("最终 1 段", stdout.getvalue())
            self.assertIn("0.000000 - 1.000000", stdout.getvalue())

    def test_folder_propagates_preview_even_with_previous_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.wav").write_bytes(b"placeholder")
            with patch.object(splitter, "run_pipeline") as run, redirect_stdout(io.StringIO()):
                splitter.main(["--folder", str(root), "--dry-run"])
            self.assertTrue(run.call_args.kwargs["dry_run"])
