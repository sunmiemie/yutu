from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

DEFAULT_AUDIO_PATH = r"d:\Users\Desktop\工作123\语图\示例.wav"
DEFAULT_OUTPUT_DIR = r"d:\Users\Desktop\工作123\语图\vad_output"
DEFAULT_OUTPUT_PREFIX = "output"


@dataclass
class Segment:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class VadSplitConfig:
    sampling_rate: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 500
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 0
    max_speech_duration_s: float = 30.0
    merge_gap_sec: float = 0.0
    min_segment_duration_sec: float = 0.0
    extra_pad_sec: float = 0.5
    absolute_min_duration_sec: float = 0.5


def validate_config(config: VadSplitConfig) -> None:
    for name, value in asdict(config).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值，不能使用 NaN 或无穷大。")
    if config.sampling_rate not in (8000, 16000):
        raise ValueError("sampling_rate 只能使用 8000 或 16000。")
    if not 0.0 < config.threshold < 1.0:
        raise ValueError("threshold 必须在 0 到 1 之间。")
    if config.min_speech_duration_ms <= 0:
        raise ValueError("min_speech_duration_ms 必须大于 0。")
    if config.min_silence_duration_ms < 0:
        raise ValueError("min_silence_duration_ms 不能小于 0。")
    if config.speech_pad_ms < 0:
        raise ValueError("speech_pad_ms 不能小于 0。")
    if config.max_speech_duration_s <= 0:
        raise ValueError("max_speech_duration_s 必须大于 0。")
    if config.merge_gap_sec < 0:
        raise ValueError("merge_gap_sec 不能小于 0。")
    if config.min_segment_duration_sec < 0:
        raise ValueError("min_segment_duration_sec 不能小于 0。")
    if config.extra_pad_sec < 0:
        raise ValueError("extra_pad_sec 不能小于 0。")
    if config.absolute_min_duration_sec < 0:
        raise ValueError("absolute_min_duration_sec 不能小于 0。")


def get_wav_duration_sec(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wav_reader:
        frame_rate = wav_reader.getframerate()
        total_frames = wav_reader.getnframes()
    return total_frames / float(frame_rate)


def sanitize_output_prefix(output_prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", output_prefix.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or "output"


def detect_speech_segments(audio_path: Path, config: VadSplitConfig) -> List[Segment]:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    except ImportError as exc:
        raise RuntimeError(
            "无法加载 silero-vad，请在运行脚本的 Python 环境中安装或修复依赖："
            "python -m pip install silero-vad"
        ) from exc
    model = load_silero_vad()
    wav = read_audio(str(audio_path), sampling_rate=config.sampling_rate)

    timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=config.threshold,
        sampling_rate=config.sampling_rate,
        min_speech_duration_ms=config.min_speech_duration_ms,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms,
        max_speech_duration_s=config.max_speech_duration_s,
        return_seconds=True,
    )

    return [
        Segment(
            start_sec=round(float(item["start"]), 3),
            end_sec=round(float(item["end"]), 3),
        )
        for item in timestamps
    ]


def normalize_segments(segments: List[Segment], audio_duration_sec: float) -> List[Segment]:
    normalized: List[Segment] = []

    for segment in sorted(segments, key=lambda item: (item.start_sec, item.end_sec)):
        start_sec = max(0.0, min(segment.start_sec, audio_duration_sec))
        end_sec = max(0.0, min(segment.end_sec, audio_duration_sec))
        if end_sec <= start_sec:
            continue
        normalized.append(Segment(round(start_sec, 3), round(end_sec, 3)))

    return normalized


def merge_close_segments(segments: List[Segment], merge_gap_sec: float) -> List[Segment]:
    if merge_gap_sec <= 0:
        return [Segment(item.start_sec, item.end_sec) for item in segments]
    if not segments:
        return []

    merged = [Segment(segments[0].start_sec, segments[0].end_sec)]
    for current in segments[1:]:
        previous = merged[-1]
        gap = current.start_sec - previous.end_sec
        if gap <= merge_gap_sec:
            previous.end_sec = max(previous.end_sec, current.end_sec)
        else:
            merged.append(Segment(current.start_sec, current.end_sec))
    return merged


def merge_short_segments(segments: List[Segment], min_segment_duration_sec: float) -> List[Segment]:
    if min_segment_duration_sec <= 0:
        return [Segment(item.start_sec, item.end_sec) for item in segments]
    if not segments:
        return []

    merged = [Segment(item.start_sec, item.end_sec) for item in segments]
    index = 0

    while index < len(merged):
        current = merged[index]
        if current.duration_sec >= min_segment_duration_sec or len(merged) == 1:
            index += 1
            continue

        previous_gap = math.inf
        next_gap = math.inf

        if index > 0:
            previous_gap = max(0.0, current.start_sec - merged[index - 1].end_sec)
        if index < len(merged) - 1:
            next_gap = max(0.0, merged[index + 1].start_sec - current.end_sec)

        if previous_gap <= next_gap and index > 0:
            merged[index - 1].end_sec = max(merged[index - 1].end_sec, current.end_sec)
            del merged[index]
            index = max(0, index - 1)
            continue

        if index < len(merged) - 1:
            merged[index + 1].start_sec = min(merged[index + 1].start_sec, current.start_sec)
            del merged[index]
            continue

        index += 1

    return merged


def add_extra_padding(
    segments: List[Segment],
    audio_duration_sec: float,
    extra_pad_sec: float,
) -> List[Segment]:
    if not segments or extra_pad_sec <= 0:
        return [Segment(item.start_sec, item.end_sec) for item in segments]

    padded: List[Segment] = []

    for segment in segments:
        start_sec = max(0.0, segment.start_sec - extra_pad_sec)
        end_sec = min(audio_duration_sec, segment.end_sec + extra_pad_sec)

        if end_sec > start_sec:
            padded.append(Segment(round(start_sec, 3), round(end_sec, 3)))

    return padded


def filter_too_short_segments(
    segments: List[Segment],
    absolute_min_duration_sec: float,
) -> List[Segment]:
    return [
        Segment(item.start_sec, item.end_sec)
        for item in segments
        if item.duration_sec >= absolute_min_duration_sec
    ]


def post_process_segments(
    segments: List[Segment],
    config: VadSplitConfig,
    audio_duration_sec: float,
) -> List[Segment]:
    segments = normalize_segments(segments, audio_duration_sec)
    segments = merge_close_segments(segments, config.merge_gap_sec)
    segments = merge_short_segments(segments, config.min_segment_duration_sec)
    segments = normalize_segments(segments, audio_duration_sec)
    segments = add_extra_padding(segments, audio_duration_sec, config.extra_pad_sec)
    segments = normalize_segments(segments, audio_duration_sec)
    segments = filter_too_short_segments(segments, config.absolute_min_duration_sec)
    segments = normalize_segments(segments, audio_duration_sec)
    return segments


def export_segments_manifest(
    audio_path: Path,
    output_dir: Path,
    raw_segments: List[Segment],
    final_segments: List[Segment],
    config: VadSplitConfig,
    output_prefix: str,
    audio_duration_sec: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{output_prefix}_vad_segments.json"
    payload = {
        "audio_path": str(audio_path),
        "audio_duration_sec": round(audio_duration_sec, 3),
        "output_prefix": output_prefix,
        "config": asdict(config),
        "raw_segments": [
            {
                "index": index,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "duration_sec": round(segment.duration_sec, 3),
            }
            for index, segment in enumerate(raw_segments, start=1)
        ],
        "final_segments": [
            {
                "index": index,
                "segment_name": f"{output_prefix}-{index:03d}.wav",
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "duration_sec": round(segment.duration_sec, 3),
            }
            for index, segment in enumerate(final_segments, start=1)
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def export_segments_csv(output_dir: Path, segments: List[Segment], output_prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}_vad_segments.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["segment_name", "start_sec", "end_sec", "duration_sec"])
        for index, segment in enumerate(segments, start=1):
            writer.writerow(
                [
                    f"{output_prefix}-{index:03d}.wav",
                    f"{segment.start_sec:.3f}",
                    f"{segment.end_sec:.3f}",
                    f"{segment.duration_sec:.3f}",
                ]
            )
    return csv_path


def split_wav_by_segments(
    audio_path: Path,
    output_dir: Path,
    segments: List[Segment],
    output_prefix: str,
) -> List[Path]:
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("当前切分实现仅支持 WAV 文件，请先将音频转换为 .wav 格式。")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    with wave.open(str(audio_path), "rb") as wav_reader:
        params = wav_reader.getparams()
        frame_rate = wav_reader.getframerate()
        total_frames = wav_reader.getnframes()

        for index, segment in enumerate(segments, start=1):
            start_frame = max(0, math.floor(segment.start_sec * frame_rate))
            end_frame = min(total_frames, math.ceil(segment.end_sec * frame_rate))

            if end_frame <= start_frame:
                continue

            wav_reader.setpos(start_frame)
            frames = wav_reader.readframes(end_frame - start_frame)

            output_path = output_dir / f"{output_prefix}-{index:03d}.wav"
            with wave.open(str(output_path), "wb") as wav_writer:
                wav_writer.setparams(params)
                wav_writer.writeframes(frames)

            saved_files.append(output_path)

    return saved_files


def run_pipeline(
    audio_path: Path,
    output_dir: Path,
    config: VadSplitConfig,
    output_prefix: str,
) -> None:
    validate_config(config)

    if not audio_path.exists():
        raise FileNotFoundError(f"找不到输入音频文件: {audio_path}")

    output_prefix = sanitize_output_prefix(output_prefix)
    audio_duration_sec = get_wav_duration_sec(audio_path)

    raw_segments = detect_speech_segments(audio_path, config)
    final_segments = post_process_segments(raw_segments, config, audio_duration_sec)

    manifest_path = export_segments_manifest(
        audio_path=audio_path,
        output_dir=output_dir,
        raw_segments=raw_segments,
        final_segments=final_segments,
        config=config,
        output_prefix=output_prefix,
        audio_duration_sec=audio_duration_sec,
    )
    csv_path = export_segments_csv(output_dir, final_segments, output_prefix)
    saved_files = split_wav_by_segments(audio_path, output_dir, final_segments, output_prefix)

    print(f"输入音频: {audio_path}")
    print(f"音频总时长: {audio_duration_sec:.3f} 秒")
    print(f"输出文件名前缀: {output_prefix}")
    print(f"VAD 原始片段数: {len(raw_segments)}")
    print(f"后处理后的最终片段数: {len(final_segments)}")
    print(f"JSON 清单文件: {manifest_path}")
    print(f"CSV 清单文件: {csv_path}")
    print(f"已输出音频片段数: {len(saved_files)}")
    for file_path in saved_files:
        print(f"  - {file_path}")


def find_wav_files(folder_path: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        ]
    )


def run_folder_pipeline(folder_path: Path, config: VadSplitConfig) -> None:
    validate_config(config)

    if not folder_path.exists():
        raise FileNotFoundError(f"找不到文件夹: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"这不是文件夹: {folder_path}")

    wav_files = find_wav_files(folder_path)
    if not wav_files:
        print("未找到 .wav 音频文件。")
        return

    output_root = folder_path / "vad_output"
    print(f"发现 {len(wav_files)} 个 WAV 文件。")

    for index, audio_path in enumerate(wav_files, start=1):
        print("\n===================================")
        print(f"[{index}/{len(wav_files)}] 开始处理: {audio_path.name}")
        print("===================================")
        run_pipeline(
            audio_path=audio_path,
            output_dir=output_root / audio_path.stem,
            config=config,
            output_prefix=audio_path.stem,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Silero VAD 检测语音并切分 WAV 音频。")
    parser.add_argument("--audio", default=DEFAULT_AUDIO_PATH, help="输入 WAV 音频路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="切分音频和清单文件的输出目录")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="采样率，通常使用 8000 或 16000")
    parser.add_argument("--threshold", type=float, default=0.5, help="语音判定阈值，值越大越严格")
    parser.add_argument("--min-speech-duration-ms", type=int, default=500, help="最短语音片段时长，单位毫秒")
    parser.add_argument("--min-silence-duration-ms", type=int, default=500, help="切分所需的最短静音时长，单位毫秒")
    parser.add_argument("--speech-pad-ms", type=int, default=0, help="VAD 检测时每段前后补边时长，单位毫秒")
    parser.add_argument("--max-speech-duration-s", type=float, default=20.0, help="单段语音最长时长，单位秒")
    parser.add_argument("--merge-gap-sec", type=float, default=0.0, help="相邻片段合并的最大间隔，单位秒")
    parser.add_argument("--min-segment-duration-sec", type=float, default=0.0, help="后处理时目标最短片段时长，单位秒")
    parser.add_argument("--extra-pad-sec", type=float, default=0.5, help="导出时前后额外保留的静音时长，单位秒")
    parser.add_argument("--absolute-min-duration-sec", type=float, default=0.5, help="最终允许导出的最短片段时长，单位秒")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="输出文件名前缀，例如 Japan")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = VadSplitConfig(
        sampling_rate=args.sampling_rate,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
        max_speech_duration_s=args.max_speech_duration_s,
        merge_gap_sec=args.merge_gap_sec,
        min_segment_duration_sec=args.min_segment_duration_sec,
        extra_pad_sec=args.extra_pad_sec,
        absolute_min_duration_sec=args.absolute_min_duration_sec,
    )

    if len(sys.argv) > 1:
        run_pipeline(
            audio_path=Path(args.audio),
            output_dir=Path(args.output_dir),
            config=config,
            output_prefix=args.output_prefix or Path(args.audio).stem,
        )
        return

    folder_input = input("请输入文件夹地址：").strip().strip('"')
    if not folder_input:
        print("未输入文件夹地址，程序结束。")
        return

    run_folder_pipeline(Path(folder_input), config)


if __name__ == "__main__":
    main()
