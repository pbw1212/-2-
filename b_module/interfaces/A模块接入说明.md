# A 模块接入说明（给 A 的负责人）

> 一页纸说清：B 从 A 拿走什么、B 怎么跟 A 的 `serve` 说话、你那边需要提供什么。
> 本文件由 B 维护，**不改动 `a_module/` 里的任何代码**。

## 1. 结论先说

B 已经写好转接层 [`b_module/a_bridge.py`](../a_bridge.py)，**A 一行代码都不用改**。
你只需要提供两样东西：

1. 一个能跑 `python -m module_a serve` 的 Python 环境（**Python 3.11/3.12**，
   别用 Anaconda base——它的 `import torch` 会报 `OMP: Error #15`）；
2. 模型权重（`python a_module/fetch_models.py`）。

然后 B 这边一条命令就能跑闭环：

```powershell
python -m b_module.a_bridge --case dev_01 --strategy fixed:600 --python a_module/.venv/Scripts/python.exe
```

## 2. A 与组协议的四个差异（B 已吸收，你不需要动）

| 差异 | A 的 `serve` | 组协议（`docs/interface_spec_v1.md`） | B 怎么处理 |
| :--- | :--- | :--- | :--- |
| 轮次标识 | `turn_id` 是字符串（`"visit-1"`） | 非负整数 | `map_turn_id(n)` → `"turn{n}"`，双向对应 |
| 判定字段 | `voiced` / `text` | `is_speech` / `partial_text` | 逐字段改名 |
| 时间戳 | `partial` **不带时间戳** | `partial_updated_ms` 必填 | 用该 `partial` 事件的 `audio_ms` 打戳 |
| 结束状态 | `status: ok` / `no_text` | `status: ok` / `no_input` / `error` | `no_text → no_input`，其余 → `error` + `error_code` |

另外：

* A 的 `speaker` 事件 → `PROFILE_SUGGESTION`：
  `candidate → matched`、`unknown → low_score`、其余（`too_short`/`no_templates`/`error`）→ `none`；
* A 的 `wake` / `closed` / `started` 不转换成组协议事件（唤醒在 B 的职责之外）；
* A 报 `error` 时，B 抛 `ABridgeError`，不会静默吞掉。

## 3. 一次完整的 A→B→A 交互

B 发给 A（内层协议，原样）：

```json
{"op":"start","turn_id":"turn1","context":"registration","identify":false}
{"op":"pcm","turn_id":"turn1","data":"<base64 640B>"}
{"op":"finish","turn_id":"turn1"}
```

B 从 A 收到并转换后，B 内部看到的是：

```
audio(voiced=true, audio_ms=20)  -> ASR_TICK{audio_ms:20,  is_speech:true,  partial_text:null}
partial(text="我想约", audio_ms=100)  -> （不单独发事件，只更新文本与时间戳）
audio(voiced=true, audio_ms=120) -> ASR_TICK{..., partial_text:"我想约", partial_updated_ms:100}
final(status="ok", text="我想约康复科下周四下午", audio_ms=3560)
                                 -> ASR_FINAL{status:"ok", final_text:"…", start_audio_ms:0, end_audio_ms:3560}
speaker(status="candidate", profile_id="p001")
                                 -> PROFILE_SUGGESTION{profile_id:"p001", match_status:"matched"}
```

B 发出并交回 A：

```json
{"schema_version":1,"event":"TURN_STOP","session_id":"s01","turn_id":1,
 "emitted_mono_ms":8543210,
 "payload":{"decision_audio_ms":4760,"reason":"adaptive_silence"}}
```

A 收到后按现有约定 `finish` 冲刷尾块、产出一次 `final`。**B 每轮只发一次 `TURN_STOP`**，
迟到的旧轮次事件会被 B 按 `turn_id` 丢弃。

## 4. B 需要 A 保证的三件事

| # | 约定 | 现状 | 影响 |
| :--- | :--- | :--- | :--- |
| 1 | **关闭 A 的自动端点截断**，由 B 的 `TURN_STOP` 决定轮末 | ✅ 已满足（`asr.py` 里 `enable_endpoint_detection=False`） | 若 A 自己截断，整机会出现两套端点竞争 |
| 2 | 每轮只产**一次** `final` | ✅ 已满足（`session.py` 的 `finish_turn`） | B 依赖它作为本轮终态 |
| 3 | `partial` 只在文本**真的变化**时发 | ✅ 已满足（`text != turn.partial` 才发） | B 的稳定窗口以它为计时起点 |

## 5. 音频格式约定

| 项 | 值 | 说明 |
| :--- | :--- | :--- |
| 采样率 | 16 kHz | A 内部 `read_wav` 会重采样到 16 k；B 的转接层也统一到 16 k |
| 声道 | 单声道 | |
| 位深 | 16 bit 小端 PCM | |
| 帧长 | 20 ms = 640 字节 | B 按 A 的 `ready` 事件里的 `frame_ms` 自适应 |

## 6. 联调检查清单

- [ ] `python -m module_a serve` 能起来并打印 `{"type":"ready","sample_rate":16000,"frame_ms":20}`
- [ ] `python -m b_module.a_bridge --case dev_01 --strategy fixed:600 --python <A 的解释器>` 能跑通
- [ ] 一轮里 B 只发一次 `TURN_STOP`，A 只回一次 `final`
- [ ] 连续两轮（`turn1`、`turn2`）后，旧的 `turn1` 事件不会影响 `turn2`
- [ ] 把 trace 交给 E：`python e_module/latency_analyzer.py <trace.jsonl>` 能识别轮次

## 7. 如果暂时跑不起来

B 的阶段一结果**不依赖 A**：`python -m b_module scan` 用 A 已入库的 29 条录音 + 模拟临时
转写即可复现全部指标。A 的通路只影响"整机闭环"那一项验收，不影响 11/29 的模块级汇报。
