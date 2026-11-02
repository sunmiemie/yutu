import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from support import splitter


def write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x01\x00" * 160)


class RecursiveScanErrorTests(unittest.TestCase):
    def test_unreadable_root_reports_failure_without_starting_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "recordings"
            source = root / "sample.wav"
            write_wav(source)
            original = source.read_bytes()
            output = base / "output"
            summary = base / "summary.json"
            stderr = io.StringIO()
            real_scandir = splitter.os.scandir
            scan_error = PermissionError(13, "permission denied", str(root))

            def scandir(path):
                if Path(path) == root:
                    raise scan_error
                return real_scandir(path)

            with patch.object(splitter.os, "scandir", side_effect=scandir), \
                    patch.object(splitter, "run_pipeline") as run, \
                    redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                exit_code = splitter.main([
                    "--folder", str(root), "--recursive", "--output-dir", str(output),
                    "--summary-json", str(summary),
                ])

            self.assertEqual(exit_code, 1)
            run.assert_not_called()
            self.assertIn("permission denied", stderr.getvalue())
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["exit_code"], 1)
            self.assertEqual(len(payload["results"]), 1)
            self.assertEqual(payload["results"][0]["status"], "failed")
            self.assertEqual(payload["results"][0]["error"], str(scan_error))
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), original)

    def test_unreadable_child_aborts_preflight_even_when_continuing_file_errors(self):
        for continue_on_error in (False, True):
            with self.subTest(continue_on_error=continue_on_error), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                root = base / "recordings"
                readable = root / "first.wav"
                unreadable = root / "restricted" / "second.wav"
                write_wav(readable)
                write_wav(unreadable)
                original_sources = {path: path.read_bytes() for path in (readable, unreadable)}
                output = base / "output"
                output.mkdir()
                previous = output / "previous-result.json"
                previous.write_bytes(b"preserve previous result")
                summary = base / "summary.json"
                stderr = io.StringIO()
                real_scandir = splitter.os.scandir
                visited = []
                scan_error = OSError(5, "input/output error", str(unreadable.parent))

                def scandir(path):
                    visited.append(Path(path))
                    if Path(path) == unreadable.parent:
                        raise scan_error
                    return real_scandir(path)

                arguments = [
                    "--folder", str(root), "--recursive", "--output-dir", str(output),
                    "--summary-json", str(summary),
                ]
                if continue_on_error:
                    arguments.append("--continue-on-error")
                with patch.object(splitter.os, "scandir", side_effect=scandir), \
                        patch.object(splitter, "run_pipeline", return_value={
                            "audio_path": str(readable), "status": "exported",
                        }) as run, redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                    exit_code = splitter.main(arguments)

                self.assertEqual(exit_code, 1)
                self.assertIn(root, visited)
                self.assertIn(unreadable.parent, visited)
                run.assert_not_called()
                self.assertIn("input/output error", stderr.getvalue())
                payload = json.loads(summary.read_text(encoding="utf-8"))
                self.assertEqual(payload["exit_code"], 1)
                self.assertEqual([item["status"] for item in payload["results"]], ["failed"])
                self.assertEqual(payload["results"][0]["error"], str(scan_error))
                self.assertEqual(list(output.iterdir()), [previous])
                self.assertEqual(previous.read_bytes(), b"preserve previous result")
                for path, original in original_sources.items():
                    self.assertEqual(path.read_bytes(), original)

    def test_empty_recursive_folder_still_reports_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "recordings"
            root.mkdir()
            summary = base / "summary.json"
            with patch.object(splitter, "run_pipeline") as run, redirect_stdout(io.StringIO()):
                exit_code = splitter.main([
                    "--folder", str(root), "--recursive", "--summary-json", str(summary),
                ])
            self.assertEqual(exit_code, 0)
            run.assert_not_called()
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {
                "exit_code": 0, "results": [],
            })
            self.assertEqual(list(root.iterdir()), [])

    def test_excluded_output_directories_are_never_scanned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = [root / "sample.wav", root / "nested" / "sample.wav"]
            excluded = {root / "results", root / "vad_output", root / "nested" / "VAD_OUTPUT"}
            for path in [*expected, *(directory / "previous.wav" for directory in excluded)]:
                write_wav(path)
            originals = {directory / "previous.wav": (directory / "previous.wav").read_bytes()
                         for directory in excluded}
            real_scandir = splitter.os.scandir
            visited = []

            def scandir(path):
                visited.append(Path(path))
                if Path(path) in excluded:
                    raise PermissionError(13, "excluded directory must not be visited", str(path))
                return real_scandir(path)

            with patch.object(splitter.os, "scandir", side_effect=scandir):
                actual = splitter.find_wav_files(root, recursive=True, exclude_dir=root / "results")

            self.assertEqual(actual, sorted(expected))
            self.assertTrue(excluded.isdisjoint(visited))
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)
