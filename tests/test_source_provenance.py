import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from support import splitter


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.wav"
        with wave.open(str(self.source), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\x01\x02" * 16000)

    def test_manifest_identifies_source_and_reviewed_annotations(self):
        labels = self.root / "labels.json"
        labels.write_text('[{"start_sec": 0, "end_sec": 1}]', encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            splitter.run_pipeline(self.source, self.root / "out", splitter.VadSplitConfig(), "source", segments_json=labels)
        manifest = json.loads((self.root / "out/source_vad_segments.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_fingerprint"], {"sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(), "size_bytes": self.source.stat().st_size})
        self.assertEqual(manifest["segment_source"]["kind"], "imported")
        self.assertEqual(manifest["segment_source"]["sha256"], hashlib.sha256(labels.read_bytes()).hexdigest())

    def test_mutated_input_is_not_published(self):
        def detect(*args):
            content = self.source.read_bytes()
            self.source.write_bytes(content[:-2] + b"\x03\x04")
            return [splitter.Segment(0, 1)]
        with patch.object(splitter, "detect_speech_segments", side_effect=detect), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "发生变化"):
                splitter.run_pipeline(self.source, self.root / "out", splitter.VadSplitConfig(), "source")
        self.assertEqual(list((self.root / "out").iterdir()), [])
