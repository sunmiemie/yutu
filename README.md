# yutu

语图公司的 Python 音频处理脚本。目前提供 Silero VAD 语音片段检测、WAV 切分、文件夹批处理，以及 JSON/CSV 清单导出。

- [使用说明](split切分音频脚本/脚本使用说明.md)
- [参数说明](split切分音频脚本/VAD切割参数说明.md)

```bash
python -m pip install silero-vad
python "split切分音频脚本/vad_audio_splitter.py" --audio "sample.wav"
python "split切分音频脚本/vad_audio_splitter.py" --folder "录音" --output-dir "结果"
```

在仓库根目录运行离线回归测试（无需安装 VAD 依赖）：

输出前缀保留中文、日文等 Unicode 字母和数字，将路径分隔符替换为短横线，并限制为 120 UTF-8 字节，为清单文件后缀留出空间。

JSON/CSV 清单保留时间戳的浮点精度；JSON 还记录原始 WAV 格式与每段实际导出的起止帧（结束帧不包含），便于核对不足 1 毫秒的短片段。

在单文件或文件夹命令中增加 `--dry-run` 可预览最终时间范围，不创建任何输出文件。预览仍执行 VAD 检测，适合先调整阈值与补边参数，也可以在已有导出结果时使用。

```bash
python -m unittest discover -s tests -v
```
