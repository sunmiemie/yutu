import builtins
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from support import SCRIPT, splitter


class DependencyTests(unittest.TestCase):
    def test_help_works_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"--threshold", result.stdout)

    def test_missing_vad_explains_install_command(self):
        original_import = builtins.__import__

        def import_without_vad(name, *args, **kwargs):
            if name == "silero_vad":
                raise ModuleNotFoundError("No module named silero_vad")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_vad):
            with self.assertRaisesRegex(RuntimeError, "pip install silero-vad"):
                splitter.detect_speech_segments(Path("unused.wav"), splitter.VadSplitConfig())
