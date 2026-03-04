"""Load the standalone script without renaming its user-facing folder."""
import importlib.util
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "split切分音频脚本" / "vad_audio_splitter.py"
spec = importlib.util.spec_from_file_location("vad_audio_splitter", SCRIPT)
splitter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = splitter
spec.loader.exec_module(splitter)
