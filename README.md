# 离线复诊登记助手（Offline Elderly Follow-Up Registration Assistant）

面向老年复诊患者，在树莓派上完全离线运行的两大核心任务：**复诊流程本地问答** 与 **基于模拟档案的复诊登记**。

---

## 1. 全组五模块事件驱动架构与三条冻结边界

各模块通过单行 UTF-8 JSON 事件消息通信，固定公共信封字段为 `schema_version`（`1`）、`event`、`session_id`、`turn_id`、`emitted_mono_ms`、`payload`：

| 模块 | 核心职责 | 输出事件 | 对应工程目录（推荐） |
| :--- | :--- | :--- | :--- |
| **A** | 麦克风采集、VAD、WeNet 流式识别、声纹匹配 | `ASR_TICK`、`ASR_FINAL`、`PROFILE_SUGGESTION` | `a_module/` |
| **B** | 自适应语音端点检测 | `TURN_STOP` | `b_module/` |
| **C** | 口语理解、字段抽取、本地单轮 FAQ | `NLU_RESULT` | `c_module/` |
| **D** | 多轮对话管理、历史继承、SQLite 登记 | `TURN_CONTEXT`、`REPLY_REQUEST` | `d_module/` |
| **E** | 离线 Tacotron2 TTS、Clear Speech 优化、扬声器播报与半双工门控 | `LISTEN_START`、`PLAYBACK_RESULT` | `e_module/`（已就绪） |

**全组三条核心边界**：
1. **端点控制权归 B**：A 关闭 WeNet 原生自动端点截断，由 B 发 `TURN_STOP` 给 A；每轮只产一次 `ASR_FINAL`。
2. **听说分时归 E**：E 收到 `REPLY_REQUEST` 后保持麦克风关闭；播报全部完成后先确认播放器停止，回传 `PLAYBACK_RESULT` 给 D；当且仅当**上一条音频已播完且新轮次 `TURN_CONTEXT` 已就绪**时，E 清空播报期残留音频缓冲并仅发一次 `LISTEN_START`。
3. **字段与异常统一**：D 在 `TURN_CONTEXT` 统一发送 `expected_slots` 数组（单槽位时自动派生 `expected_slot`，否则为 `null`）；异常路径统一约定为 `no_input` / `error` / `cancel`。

---

## 2. 推荐团队仓库目录约定

为方便各成员合并代码时互不冲突，建议按模块独立目录组织仓库，总装入口放于 `integration/`：

```text
.
├── a_module/                      # 模块 A：流式 ASR 与声纹识别
├── b_module/                      # 模块 B：自适应端点检测
├── c_module/                      # 模块 C：口语理解与 FAQ 问答
├── d_module/                      # 模块 D：多轮对话管理与 SQLite 登记
├── e_module/                      # 模块 E：离线 TTS、门控服务与实验评测（当前已合入）
│   ├── models/                    # 轻量 SyllableBiGRUCRF 韵律预测权重（~2 MB）
│   └── tests/                     # E 模块单元测试（25 项）
├── docs/                          # 全部项目文档（全组联调接口规范 + 各模块交付方案）
│   ├── interface_spec_v1.md       # 第一版冻结联调接口规范
│   └── *.docx                     # 全组联调总方案与 A~E 模块工程实施方案
└── integration/                   # 全组联调主控/消息总线入口（后续合入）
```

---

## 3. 与 E 模块联调对接指南（`schema_version = 1`）

- **接口规范文档**：详见 [`docs/interface_spec_v1.md`](docs/interface_spec_v1.md)（全组总方案与各模块交付方案均统一存放于 `docs/`）。
- **共享事件信封与校验**：可直接复用 [`e_module/events.py`](e_module/events.py)：
  - `make_turn_context` / `make_reply_request` / `make_listen_start` / `make_playback_result`
  - `validate_event` / `to_json_line` / `from_json_line`
- **服务编排入口**：[`e_module/e_service.py`](e_module/e_service.py) 中的 `EModuleService`：
  - **事件驱动接入**：调用 `dispatch_event(event_dict)` 自动处理 `TURN_CONTEXT` 与 `REPLY_REQUEST`，或分别调用 `handle_turn_context(event)` / `handle_reply_request(event)`。
  - **A 模块麦克风联调回调**：初始化 `EModuleService` 时注入：
    - `on_mic_pause(turn_id)`：开始合成本轮回复或播报前关闭/暂停麦克风收音。
    - `on_mic_flush_and_open(turn_id)`：音频播完且新轮 `TURN_CONTEXT` 就绪后，清空播报期间残留音频缓冲并开启新一轮麦克风监听（与 `LISTEN_START` 严格同步）。

---

## 4. 外部 Tacotron2 权重与语料路径配置

课程大体积权重与标贝语料库（约 4 GB，例如 `checkpoint_latest.pt` 与 `biaobei/`）**不纳入 Git 跟踪**。E 模块会按以下优先级自动定位课程 `tacotron2` 目录：

1. 环境变量 `SPEECH_TACOTRON2_DIR`（权重文件还可通过 `SPEECH_TACOTRON2_CKPT` 单独覆盖）
2. `./语音信息处理综合实践网盘/语音合成/tacotron2`
3. `./speech/语音合成/tacotron2`
4. `./speech/tacotron2`
5. `~/speech/语音合成/tacotron2`
6. `~/speech/tacotron2`

在树莓派或新电脑拉取本仓库后，只需将课程包置于 `~/speech/`（或通过环境变量指定），即可直接运行合成与整机联调：

```bash
export SPEECH_TACOTRON2_DIR=~/speech/语音合成/tacotron2
export SPEECH_TACOTRON2_CKPT=$SPEECH_TACOTRON2_DIR/logs/checkpoint_latest.pt
```

---

## 5. 快速安装与验证命令

```bash
# 安装 E 模块依赖
pip install -r e_module/requirements.txt

# 1. 运行 E 模块 25 项单元测试（无需外部权重也可全量通过前端与状态机测试）
pytest e_module/tests -v

# 2. 运行含真实 Tacotron2 + EC-Viterbi + Clear Speech + INT8 的 6 大联调模拟场景
python e_module/run_mock_integration.py

# 3. 复现全套客观量化指标（DTW-MCD / Attention 稳定性 / Clear Speech / Pipelined TTFA）
python e_module/eval_objective.py
```

更完整的 E 模块结构说明、实验细节及树莓派 64-bit 部署步骤请查阅 [`e_module/README.md`](e_module/README.md)；本地运行实验与合成测试时，产物默认写入被 `.gitignore` 忽略的 `e_module/outputs/` 目录。
