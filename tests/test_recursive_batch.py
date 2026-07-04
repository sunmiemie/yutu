import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from support import splitter


class RecursiveBatchTests(unittest.TestCase):
    def test_preserves_relative_directories_and_excludes_generated_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ["a/voice.wav", "b/voice.WAV", "results/old.wav", "a/vad_output/old.wav"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"placeholder")
            with patch.object(splitter, "run_pipeline") as run, redirect_stdout(io.StringIO()):
                self.assertEqual(splitter.main(["--folder", str(root), "--recursive", "--output-dir", str(root / "results")]), 0)
            self.assertEqual([call.kwargs["audio_path"].relative_to(root).as_posix() for call in run.call_args_list], ["a/voice.wav", "b/voice.WAV"])
            self.assertEqual([call.kwargs["output_dir"].relative_to(root).as_posix() for call in run.call_args_list], ["results/a/voice", "results/b/voice"])

    def test_non_recursive_default_and_same_root_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "child").mkdir()
            (root / "child/a.wav").write_bytes(b"placeholder")
            self.assertEqual(splitter.find_wav_files(root), [])
            with self.assertRaisesRegex(ValueError, "输出目录"):
                splitter.run_folder_pipeline(root, splitter.VadSplitConfig(), root, recursive=True)
