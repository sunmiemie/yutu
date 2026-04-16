from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from support import splitter


class CliTests(unittest.TestCase):
    def test_audio_uses_portable_output_path_and_source_stem(self):
        with patch.object(splitter, "run_pipeline") as run:
            self.assertEqual(splitter.main(["--audio", "recordings/example.wav"]), 0)
        self.assertEqual(run.call_args.args[0], Path("recordings/example.wav"))
        self.assertEqual(run.call_args.args[1], Path("recordings/vad_output/example"))
        self.assertEqual(run.call_args.args[3], "example")

    def test_folder_accepts_parameter_overrides_and_output_root(self):
        with patch.object(splitter, "run_folder_pipeline") as run:
            self.assertEqual(splitter.main(["--folder", "recordings", "--output-dir", "results", "--threshold", "0.7"]), 0)
        self.assertEqual(run.call_args.args[0], Path("recordings"))
        self.assertEqual(run.call_args.args[1].threshold, 0.7)
        self.assertEqual(run.call_args.args[2], Path("results"))

    def test_cli_defaults_match_programmatic_defaults(self):
        args = vars(splitter.build_arg_parser().parse_args([]))
        self.assertEqual({field: args[field] for field in splitter.asdict(splitter.VadSplitConfig())}, splitter.asdict(splitter.VadSplitConfig()))

    def test_missing_or_ambiguous_inputs_fail_with_usage(self):
        for arguments in (["--threshold", "0.7"], ["--audio", "a.wav", "--folder", "input"], ["--folder", "input", "--output-prefix", "name"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as context:
                    splitter.main(arguments)
                self.assertEqual(context.exception.code, 2)

    def test_runtime_error_returns_failure_without_traceback(self):
        errors = io.StringIO()
        with patch.object(splitter, "run_pipeline", side_effect=FileNotFoundError("missing.wav")):
            with redirect_stderr(errors):
                self.assertEqual(splitter.main(["--audio", "missing.wav"]), 1)
        self.assertIn("missing.wav", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())

    def test_no_arguments_keeps_interactive_folder_mode(self):
        with patch("builtins.input", return_value='"recordings"'), patch.object(splitter, "run_folder_pipeline") as run:
            self.assertEqual(splitter.main([]), 0)
        self.assertEqual(run.call_args.args[0], Path("recordings"))
