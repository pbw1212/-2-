# 模块 B · 自适应语音端点检测

面向老年复诊患者的离线复诊登记助手，B 负责回答一个问题：**用户这一轮什么时候说完了？**
B 是整机链路里唯一发出 `TURN_STOP` 的模块；A 已关闭识别器自带的自动端点
（`a_module/module_a/asr.py` 中 `enable_endpoint_detection=False`），端点控制权在 B。

> **修改边界**：本模块只读写 `b_module/` 内的文件。`a_module/`、`c_module/`、
> `d_module/`、`e_module/` 与 `docs/` 一律**只读**；发现适配问题通过沟通交给对应负责人。

---

## 1. 当前进度

| 步骤 | 内容 | 状态 |
| :--- | :--- | :--- |
| S1 | 工程骨架、requirements、独立可运行 | ✅ |
| S2 | 配置 `config/endpoint.json` + 事件协议 `protocol.py` | ✅ |
| S3 | 音频 IO `audio.py` + 帧级 VAD `vad.py` | ✅ |
| S4 | 数据集重建 `dataset.py` → `data/manifest.json` | ✅ |
| S5 | 轮末标注 `annotate.py` → `data/annotations.json` + 能量曲线 | ✅ |
| S6 | 上下文解析 `context.py`（含 D 回放日志读取） | ✅ |
| S7 | 文本时间线 `textsim.py` + `data/text_timeline.json` | ✅ |
| S8 | 端点状态机 `detector.py` | ✅ |
| S9 | 策略层 `strategies.py`（固定 / 自适应一 / 自适应二） | ✅ |
| S10 | 回放驱动 `replay.py`（`--realtime`、trace JSONL） | ✅ |
| S11 | 批量对照与指标 `metrics.py` → `results/scan_details.csv`、`metrics.json` | ✅ |
| S12 | 报告与主图 → `results/metrics.md`、`results/figure_tradeoff.png` | ✅ |
| S13 | 两段现场演示脚本 → `demo/` | ✅ |
| S14 | A 侧转接层 `a_bridge.py` + 给 A 的接口单 | ✅ |
| S15 | 阶段一汇报提纲（5 页 PPT + 口述稿 + 课程对应表） | ✅ |

阶段一（11/29 汇报前）的目标全部完成：**可复现回放 + 固定阈值对照 + 轮末标注 + 指标表 +
A 侧接口**。101 项测试全绿；真实 A→B→A 闭环需要 A 那边先把 venv 与模型准备好（见
[`interfaces/A模块接入说明.md`](interfaces/A模块接入说明.md)）。

---

## 2. 快速开始

在**仓库根目录**（`b_module` 的父目录）执行，依赖只有 `numpy`（出图另需 `matplotlib`）：

```powershell
# S4：由 A 的样例清单生成回放清单
python -m b_module manifest

# S5：重建整轮音频 → 生成真值 → 画能量曲线
python -m b_module annotate

# 人工确认（可覆盖轮末时刻）
python -m b_module verify dev_01 test_02 --set-end dev_01=1540 --note "手工听审"

# 打印摘要表
python -m b_module summary

# 人工听审：生成试听片段 + 一页式试听台（浏览器打开 index.html）
python -m b_module listen
python -m b_module listen --all          # 全部 29 条
python -m b_module play --case dev_01 --clip tail

# S10：回放一条样本（--realtime 按真实时间播放，可现场演示）
python -m b_module replay --case dev_01 --strategy fixed:600
python -m b_module replay --case dev_01 --strategy adaptive_text --realtime

# S11/S12：全样本对照 → CSV + JSON + Markdown + 主图
python -m b_module scan
python -m b_module scan --cases dev_01,test_02 --strategies fixed:600,adaptive_text --no-plot

# S13：两段现场演示（脚本会自己定位仓库根）
powershell -ExecutionPolicy Bypass -File b_module/demo/demo_open_pause.ps1
powershell -ExecutionPolicy Bypass -File b_module/demo/demo_short_answer.ps1

# S14：A→B→A 闭环（需要 A 的 venv 与模型；详见 interfaces/A模块接入说明.md）
python -m b_module.a_bridge --case dev_01 --strategy fixed:600 --python a_module/.venv/Scripts/python.exe

# 测试
python -m pytest b_module/tests -q
```

想用 D 的**真实**轮次上下文驱动回放（而不是 B 的推定值）：

```powershell
python -m b_module replay --case dev_01 --strategy adaptive_text --context-log "d_module/03_D实验记录与结果/logs/c03_inherit_history_improved.jsonl"
```

用 Anaconda 的 Python 时把 `python` 换成解释器全路径即可；不需要新建 venv，
本模块只用 `numpy` + `matplotlib`。演示脚本可用 `$env:B_PYTHON` 指定解释器。

---

## 3. 数据来源与重建规则

B **不新录数据**，复用 A 已入库的样例（只读）：

| 来源 | 内容 | 用途 |
| :--- | :--- | :--- |
| `a_module/examples/clinic/cases.json` | 28 条复诊场景，含 `parts`、`pause_ms`、`context`、`targets` | 主体样本 |
| `a_module/examples/clinic/audio/*.wav` | 22050 Hz 单声道 16 bit 合成录音，多段样本按 `-01/-02` 切分 | 每轮音频 |
| `a_module/examples/smoke/short.wav` | 160 ms 短答（A 用作短答样例） | `yes_no` 轮代理 |

**为什么不能直接把分段拼起来**：分段 wav 自带 680–800 ms 尾静音。以 `dev_01-01.wav`
为例，语音在 1540 ms 结束、文件长 2300 ms；若直接拼 `+1000 ms`，实际句中停顿会变成
**1760 ms**，比设计值大 76%，阈值对照就全错了。

因此重建按三步：

1. 每段按 **−60 dBFS** 裁掉首尾静音，余量 0 ms；
2. 段间插入**恰好** `pause_ms` 的静音；
3. 末尾统一补 **2000 ms** 静音，保证 1500 ms 档也能在回放结束前触发。

> **v2 修正（2026-11-27，人工听审发现）**：v1 用 −45 dBFS 硬裁，实测**平均切掉 122 ms、
> 最多 220 ms** 的字末渐弱尾音（听感是"最后一个字的尾音被吞掉"），而且会把真值定早、
> 使截断率被低估。现在改用 −60 dBFS：这个阈值下保留的衰减已接近 16 bit 量化噪声，
> 不会被听出来，而 `tests/test_dataset.py::test_reconstruction_keeps_coda` 锁住了这一点。

重建结果实测：`dev_01` 的句中停顿 1000 ms（设计 1000）、`test_02` 640 ms（设计 650）、
`test_07` 840 ms（设计 850），误差 ≤ 1 帧（20 ms）。

**一个重要区别**：`pause_ms` 是**设计停顿**（插入的数字静音），而 **Detector 实际看到的
静音更长**——因为语音从 −45 衰减到 −60 的这段尾音，在 −45 阈值的 Detector 眼里已经是静音。
两者差额实测 20–220 ms：

| 样本 | 设计停顿 | Detector 可观测静音 | 差额 |
| :--- | ---: | ---: | ---: |
| `dev_01` | 1000 | 1020 | +20 |
| `test_01` | 800 | 980 | +180 |
| `test_02` | 650 | 820 | +170 |
| `test_07` | 850 | 1000 | +150 |
| `test_09` | 700 | 900 | +200 |
| `test_12` | 1000 | 1220 | +220 |

这个量（`detect_gap_ms`）记在 `data/annotations.json` 里，正是"open 档必须取 1500 ms"的依据。

重建音频缓存在 `b_module/.cache/wav/`（已 gitignore），指纹含源文件 mtime 与
影响结果的配置项，配置或音频变动会自动重建。

---

## 4. 样本分组与 `question_type`

`cases.json` 没有 `question_type`，B 按 D 模块交付稿的语义**推定**；联调后以 D 在
`TURN_CONTEXT` 里发的为准。

| 分组 | 判据 | `question_type` | 条数 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `open_pause` | 有 `pause_ms` 且 `context=registration` | `open` | 6 | 句中停顿，考提前截断 |
| `targeted` | `context∈{department,date}` | `targeted` | 11 | 定向短答，考及时性 |
| `open_long` | `context=registration` 且无 `targets` | `open` | 10 | 长问句（FAQ 类），考延迟下限 |
| `open_full` | `context=registration` 且有 `targets`、无 `pause_ms` | `open` | 1 | 一句话给全，样本少，仅作补充 |
| `yes_no_proxy` | `examples/smoke/short.wav` | `yes_no` | 1 | 确认轮代理样本 |

⚠️ **确认轮的 `expected_slots` 是空数组**（见 `d_module/05_D接口与说明文档/接口样例.md`
第 5 节）。所以 B **不能用 `expected_slots` 判“这是短答”**，必须优先用
`question_type`，`expected_slot` 只作辅助。

---

## 5. 标注口径

自动标注刻意比检测**更保守**，避免人为制造“提前截断”：

| 项 | 检测口径（S6 起用） | 标注口径（真值） |
| :--- | :--- | :--- |
| 阈值 | −45 dBFS（与 A 的 `activity()` 一致） | **−60 dBFS** |
| 最短语音 | 1 帧 | **连续 3 帧（60 ms）** |
| 段间隔 | — | ≥ 300 ms 才算两段 |
| 轮末 | 决策点 `decision_audio_ms` | 最后一段语音的帧右边界（−60 以上） |
| 容差 | — | 1 帧 = 20 ms |

标注结果里另外记两个量，便于解释阈值选择：

* `mid_pause_ms`：−60 口径下的句中停顿（≈ 设计值）；
* `detect_gap_ms`：**−45 口径**下的同一停顿，即 Detector 实际看到的静音，比设计值长 20–220 ms。

`turn_end_ms` 可由 `verify --set-end` 人工覆盖，重复运行 `annotate` 会保留人工字段；
`annotate --reset-verified` 用于**真值口径变更后**清空旧确认。
当前（v2 口径）29 条尚待重新听审：`python -m b_module listen` → 打开 `results/listen/index.html`。

---

## 6. 端点状态机与策略（S6–S9）

### 6.1 状态机

```text
WAIT_SPEECH --(首次 is_speech)--> IN_SPEECH --(静音≥阈值 / 整轮超上限)--> STOPPED
```

* 每轮**只发一次** `TURN_STOP`；发出后同一轮的 tick 全部丢弃；
* 迟到的旧轮次 tick 按 `turn_id` 丢弃，不报错；
* `decision_audio_ms` 取决策时最新可见的 `audio_ms`，不是“语音结束位置”；
* 静音从最后一帧语音的 `audio_ms` 起算：`silence = audio_ms - last_voiced_ms`；
* 文本稳定时长按 `partial_updated_ms` 算，**重复文本不刷新**，回改才刷新。

### 6.2 三档策略

| 策略 | 额外证据 | 触发 reason | 说明 |
| :--- | :--- | :--- | :--- |
| `fixed:<ms>` | 无 | `fixed_silence` | 基线，扫描 300/600/900/1200/1500 |
| `adaptive_question` | `question_type` | `adaptive_silence` | `open→1500`、`targeted→600`、`yes_no→600`，无上下文回退 900 |
| `adaptive_text` | 临时文本 + 稳定窗 | `adaptive_silence` | 稳定短答 450、续说线索 1500，其余回落到问题类型档；上限 1500 |

`open=1500` 不是拍的，而是**由固定阈值扫描反推**：6 条带停顿样本的 `detect_gap_ms` 是
820–1220 ms，只有 1500 档能 0 截断。定向/是非轮不需要这么长，差异化才有收益
（v1 曾把 open 定成 900，结果 1000 ms 停顿照样被截断——这是第一阶段修掉的第二个错误）。

`adaptive_text` 只在 **是非/定向** 轮把短文本当短答（`len ≤ 6` 或命中短答词），
开放式轮次即使识别出“康复科”也沿用开放式等待——否则“我想约康复科……”会被截断。

### 6.3 实测决策点（v2 真值口径，单位 ms）

| 样本 | 类型 | 真值 | 300 | 600 | 900 | 1200 | 1500 | 自适应一 | 自适应二 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_01` | open，设计停顿 1000（可观测 1020） | 3600 | 1700\* | 2000\* | 2300\* | 4780 | 5080 | 5080 | 5080 |
| `test_02` | open，设计停顿 650（可观测 820） | 3760 | 1480\* | 1780\* | 4540 | 4840 | 5140 | 5140 | 5140 |
| `smoke_short` | yes_no，语音 160 | 180 | 460 | 760 | 1060 | 1360 | 1660 | 760 | **620** |
| `test_13` | open，长问句 | 2540 | 2640 | 2940 | 3240 | 3540 | 3840 | 3840 | 3840 |

`*` = 提前截断（`decision < 真值 − 20 ms`）。注意 `test_02`：设计停顿只有 650 ms，但 900 ms
档**没有**截断、而 600 ms 档截断了——因为 Detector 看到的是 820 ms。**设计停顿和
可观测静音是两个量**，汇报时要说清楚。

### 6.4 全样本对照（29 条 × 7 档，共 203 次回放）

| 策略 | 触发轮数 | 提前截断 | 截断率 | 延迟中位数 | 延迟 P90 | 未触发 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed:300` | 29 | 6/29 | 21% | 180 | 280 | 0 |
| `fixed:600` | 29 | 6/29 | 21% | 480 | 580 | 0 |
| `fixed:900` | 29 | 5/29 | 17% | 780 | 880 | 0 |
| `fixed:1200` | 29 | 1/29 | 3% | 1080 | 1180 | 0 |
| `fixed:1500` | 29 | 0/29 | 0% | 1380 | 1480 | 0 |
| `adaptive_question` | 29 | **0/29** | **0%** | 1300 | 1460 | 0 |
| `adaptive_text` | 29 | **0/29** | **0%** | 1320 | 1464 | 0 |

**关键对照（两个池）**：

| 口径 | `fixed:1200` | `fixed:1500` | 自适应一 | 自适应二 |
| :--- | ---: | ---: | ---: | ---: |
| 开放式轮截断率（6 条带停顿） | 17% | 0% | **0%** | **0%** |
| **短答轮延迟中位数**（12 条） | 1080 | 1380 | **480** | **470** |

一句话结论：**要做到 0% 截断，固定阈值必须取 1500 ms，于是每条短答都要白等 1.38 s；
自适应用同样的耐心换来了短答 480 ms —— 省掉约 900 ms，且截断率不变。**

自适应二相对一只多省 10 ms，因为池里只有 1 条确认轮样本（该轮 760→620 ms）；
这个收益要靠第二阶段更多确认轮才能体现，**不要在汇报里夸大**。

产出文件（均已入库）：

| 文件 | 内容 |
| :--- | :--- |
| [`results/metrics.md`](results/metrics.md) | 总体 + 分组 + 触发原因的完整表格 |
| [`results/metrics.json`](results/metrics.json) | 同上，机器可读 |
| [`results/scan_details.csv`](results/scan_details.csv) | 203 行逐轮明细（真值、决策、是否截断、basis） |
| [`results/figure_tradeoff.png`](results/figure_tradeoff.png) | 两栏主图：左=开放式轮取舍，右=短答延迟 |
| [`results/listen/index.html`](results/listen/index.html) | 人工听审台（v2：片段跨过轮末） |
| [`demo/`](demo/) | 两段现场演示脚本 + trace |

复跑：

```powershell
python -m b_module scan
python -m pytest b_module/tests -q
```

## 7. 事件接口（B → A）

严格按 `docs/interface_spec_v1.md`，payload 不做扩展；调试信息写在 trace 外层，
不混进事件。

```json
{"schema_version":1,"event":"TURN_STOP","session_id":"s01","turn_id":2,
 "emitted_mono_ms":8543210,
 "payload":{"decision_audio_ms":3540,"reason":"adaptive_silence"}}
```

`reason` 语义（与总方案“`timeout` → `error`”一致，**这条需要和 A、D 当面确认**）：

| `reason` | 含义 | 下游处理 |
| :--- | :--- | :--- |
| `fixed_silence` | 固定阈值触发 | 正常轮末 |
| `adaptive_silence` | 自适应策略触发，**含 1500 ms 等待上限** | 正常轮末 |
| `no_speech` | 整轮未检测到语音 | `no_input` |
| `timeout` | 整轮超过 `hard_cap_ms`（30 s） | `error` |

注意：等待上限（1500 ms）是**正常轮末**，不能记成 `timeout`，否则整机会把正常结束
当设备错误处理。

**已知跨模块差异**（由 `a_bridge` 在转接层解决，不改 A 的代码）：

- A 内层 `serve` 协议的 `turn_id` 是字符串（如 `"visit-1"`），组协议要求非负整数；
- A 的 `partial` 事件不带时间戳，`partial_updated_ms` 需由转接层补打；
- A 的 `final.status` 为 `ok` / `no_text`，映射到组协议时 `no_text → no_input`。

---

## 8. 目录结构

```text
b_module/
├── __init__.py          包信息与阶段声明
├── __main__.py          CLI：manifest / annotate / verify / summary（replay 在 S10 加入）
├── paths.py             统一路径与配置加载
├── protocol.py          schema_version=1 事件构造与校验
├── vad.py               20 ms 帧能量、语音段聚合（与 A 的 activity() 同式）
├── audio.py             WAV 读写、裁剪、拼接、补尾静音
├── dataset.py           样例清单、整轮音频重建、manifest 输出
├── annotate.py          真值生成、能量曲线、人工确认
├── listen.py            人工听审材料（切句尾/轮末后片段、生成试听台）
├── context.py           TURN_CONTEXT 解析（含 D 回放日志读取）
├── textsim.py           模拟的临时转写时间线
├── strategies.py        fixed / adaptive_question / adaptive_text
├── detector.py          端点状态机，唯一发 TURN_STOP 的地方
├── replay.py            回放驱动（--realtime、trace JSONL）
├── metrics.py           指标口径、CSV/Markdown 报告、截断率-延迟主图
├── a_bridge.py          A 的 serve 协议 ↔ 组协议转接层（A→B→A 闭环）
├── config/endpoint.json 全部阈值与策略参数（含 reason 语义说明）
├── data/manifest.json   回放清单（29 条，提交入库）
├── data/annotations.json 真值（提交入库）
├── data/text_timeline.json 11 条样本的模拟临时转写
├── results/annotation/  每条样本的能量曲线 PNG
├── results/metrics.md / metrics.json / scan_details.csv / figure_tradeoff.png / coda_profile.png
├── results/traces/      演示用的组协议 trace（E 的 latency_analyzer 可直接读）
├── tools/               汇报配图生成（make_figures：尾音剖面图）
├── results/listen/      人工听审台 index.html + 听审表.md
├── demo/                两段现场演示脚本 + 说明
├── interfaces/          给 A 的接入说明（跨模块差异与联调清单）
├── report/              阶段一汇报提纲（P1 工程量+原理+工具 / P2 数据来源 / P3 结果+分析 / P4 已修错误 / P5 计划）
└── tests/               102 项测试（协议 / VAD / 重建 / 标注 / 上下文 / 时间线 / 策略 / 状态机 / 回放 / 指标 / A 转接）
```

---

## 9. 已知限制

1. **样本是合成语音**（Windows TTS 的 `Microsoft Huihui/Kangkang/Yaoyao`），语气与停顿
   分布与真实老年语音有差异；真实录音在第二阶段补，汇报时须注明来源与条数。
2. **重建音频从语音起点开始**（首部静音已裁掉），所以 `decision_audio_ms` 的绝对值比
   真实流式链路小；做延迟比较时只用相对差值。
3. `yes_no` 组只有 1 条代理样本（`smoke_short`），结论只能当趋势参考。
4. `open_full` 只有 1 条，不单独出指标。
5. 阶段一不涉及树莓派实测；板端只需 `numpy`，`matplotlib` 只在笔记本出图。
6. **临时转写是模拟的**：`data/text_timeline.json` 只覆盖 11 条样本，时间点按分段边界
   手工推得；未覆盖的样本没有文本（策略自动退化为按问题类型）。接上 A 的真实
   `partial` 后，这部分应整份替换，`detector.py` 与 `strategies.py` 不需要改。
7. 阶段一的“固定阈值扫描”只有 29 条、且按说话人无法划分开发/测试集，
   因此只报**流程可复现**与**趋势**，不下统计显著性结论。
8. **参数是在这 29 条上定的**（尤其 `open=1500`），严格说没有留出集；12/5 补真人录音后
   必须按说话人划分、只在开发集上调、在留出集上测。
9. 自适应二只在 **1 条**确认轮样本上验证到收益（760→620 ms），池中位数只差 10 ms，
   不足以支撑"显著改善"的说法。

---

## 10. 变更记录

| 版本 | 日期 | 变更 |
| :--- | :--- | :--- |
| v1 | 2026-11-27 | S1–S15 首版；重建与真值都用 −45/−50 dBFS 硬裁 |
| v2 | 2026-11-27 | 人工听审发现"字末尾音被吞"：重建与真值阈值改为 −60 dBFS（平均找回 122 ms、最多 220 ms 尾音）；真值整体后移，截断率不再被低估；新增 `detect_gap_ms`、`listen` 听审台（片段跨过轮末）、`test_reconstruction_keeps_coda` 回归测试；`open` 档由 900 修正为 1500（依据 `detect_gap_ms` 分布）；主图改为两栏（开放式轮取舍 + 短答延迟） |
