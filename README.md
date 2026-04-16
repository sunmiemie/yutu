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

```bash
python -m unittest discover -s tests -v
```
