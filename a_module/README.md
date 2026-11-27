# 模块 A · 语音输入与声纹匹配

A 负责收音、听到“你好助手”后唤醒、边听边转写，以及按需推荐模拟档案。转写使用 X-ASR Zipformer INT8，声纹匹配使用 CAM++。档案候选交给用户确认。

## 跑起来

从这个目录执行。Windows 使用 Python 3.11 或 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe fetch_models.py
.\.venv\Scripts\python.exe -m module_a stream --realtime
```

最后一条回放一段合成复诊录音，能看到 `partial` 更新和一次 `final`。首次运行 `fetch_models.py` 会从官方发布下载模型，之后可以离线运行。已有的模型会自动校验并复用。

树莓派使用 64 位 Raspberry Pi OS 和 Python 3.11 或 3.12。录音需要 `libportaudio2`，其余命令将 `.\.venv\Scripts\python.exe` 换为 `.venv/bin/python`。本阶段完成了 Windows 本机验证，树莓派性能和全组联调是下一阶段工作。

## 试麦克风、声纹和热词

先进入上面创建的 Python 环境，或者在每行用虚拟环境的 Python 路径。

```text
python -m module_a devices
python -m module_a mic --standby
python -m module_a enroll demo-01 examples/smoke/speaker1_a_cn_16k.wav
python -m module_a identify examples/smoke/speaker1_b_cn_16k.wav
python -m module_a --decoder hotwords --context registration stream
```

`devices` 列出录音设备，`mic --standby` 可以试说“你好助手”。声纹示例先录入模拟档案，再用同一人的另一段公开样例确认能找到候选。热词默认关闭，命令中的 `registration` 同时加载科室和日期词表。

## 跟其他模块对接

用 `python -m module_a serve --enable-speaker` 启动本机演示服务。它逐行读写 UTF-8 JSON。带热词联调时，在 `serve` 前加 `--decoder hotwords`。

```json
{"op":"start","turn_id":"visit-1","context":"date"}
{"op":"pcm","turn_id":"visit-1","data":"<音频帧的 Base64 字符串>"}
{"op":"finish","turn_id":"visit-1"}
```

每轮用新的 `turn_id`。音频建议每 20 ms 发 640 字节，格式为 16 kHz、单声道、16 位小端 PCM。A 按帧输出 `audio`，文字有变化时输出 `partial`。收到 `finish` 后发一次 `final`。需要推荐档案时，在 `start` 里加 `"identify":true`，结束后还会收到 `speaker`。`cancel` 可取消本轮。

组内联调由协调器按协议 v1 转接。把 `audio` 和最近的 `partial` 整理为 `ASR_TICK` 交给 B，收到 B 的 `TURN_STOP` 就向 A 发 `finish`。把 `final` 转成 `ASR_FINAL` 给 C，将声纹候选转成 `PROFILE_SUGGESTION` 给 D。新一轮根据 D 的 `TURN_CONTEXT` 选 `department`、`date`、`registration` 或 `general` 词表，并等 E 的 `LISTEN_START` 再开启收音。

## 代码和验证从哪看

| 入口 | 负责什么 |
| --- | --- |
| `module_a/audio.py` | 收音、格式转换和 20 ms 分帧 |
| `module_a/asr.py` | 加载 Zipformer，流式更新转写 |
| `module_a/session.py` | 管理一轮的开始、唤醒、结束和输出 |
| `module_a/speaker.py` | 提取声纹，与本地 SQLite 模拟档案比较 |
| `module_a/__main__.py` | 命令行演示和逐行 JSON 接口 |
| [热词表](configs/clinic_hotwords.json) | 按问题类型设置科室、日期等候选词 |

`python verify_local.py` 检查轮次、唤醒、短答和声纹。加 `--check-mic` 额外检查麦克风。`python evaluate_hotwords.py` 复跑开关热词对照。

在同一批 20 条清晰合成录音中，WeNet 加热词识别对 21/25 个目标字段，字符错误率 12.80%。Zipformer 开关热词均为 25/25 和 2.37%。加入 −5 dB 白噪声后，两种配置均为 13/25。实验明细见[热词对照记录](验证/热词对照.json)，转写测试用的录音与参考文本见[样例清单](examples/clinic/cases.json)。

## 模型和录音来源

- X-ASR 使用 160 ms 中英流式无标点 INT8 版本。[sherpa-onnx 模型发布说明](https://k2-fsa.github.io/sherpa/onnx/x-asr/index.html)
- CAM++ 和三段声纹样例来自 3D-Speaker。[模型及公开样例](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) · [ONNX 权重](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)
- 复诊、唤醒和短答录音使用 Windows 语音合成。随模块提供 [Apache 2.0 许可证](licenses/Apache-2.0.txt)。
