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

人工修订时间戳后，可用 `--audio sample.wav --segments-json labels.json` 离线切分，无需安装或加载 VAD。JSON 格式为 `[{"start_sec": 0.25, "end_sec": 0.75}]`，也支持读取之前清单中的 `raw_segments`；非法或越界范围会报错。合并、补边和最短时长过滤仍按参数执行，精确采用标注范围时可指定 `--extra-pad-sec 0 --absolute-min-duration-sec 0`。可同时使用 `--dry-run`。

`--folder 录音 --recursive --output-dir 结果` 会按相对目录保存，避免不同子目录的同名录音混在一起。扫描跳过指定输出目录、名为 `vad_output` 的子目录和目录符号链接；递归模式不允许输入、输出目录相同。

导出清单记录原始音频的 SHA-256 和字节数；人工时间戳还记录标注文件路径与指纹。发布结果前重新校验输入，处理期间源文件发生变化时停止发布，避免清单与音频对应错误。

批处理可增加 `--skip-existing` 续跑：仅在源文件、参数、标注来源及所有 WAV/CSV 输出的指纹均匹配时跳过模型和导出。损坏、缺失或参数不同的旧结果仍报冲突，不会覆盖；旧版本没有输出指纹的清单也不会自动跳过。`--dry-run` 始终重新预览。

```bash
python -m unittest discover -s tests -v
```
