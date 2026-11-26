# E 模块：离线中文语音合成与树莓派整机门控

本目录包含离线复诊登记助手 **E 模块** 的完整四维协同实现，保持原始课程包 `语音信息处理综合实践网盘/` 完全不被改动。

## 目录结构

- `config.py`：路径、Tacotron2 模型超参、STFT 超参、默认停顿参数（260 ms，可在 200–350 ms 间调节）及标准评测样句。
- `tacotron_compat.py`：面向现代 PyTorch / Librosa 的无侵入兼容层、`window_sumsquare` 缓存、Fast Griffin-Lim（Nesterov 动量）声码器。
- `baseline_runner.py`：复现课程原版 `inference.py` 逻辑（保留 `lazy_pinyin(lines, Style.TONE3)` 原始行为）生成 Baseline 音频与指标。
- `prosody_model.py`：基于 10,000 句标贝精标语料（`8500 / 500 / 1000` 严格划分）训练的轻量 `SyllableBiGRUCRF` 中文韵律边界预测模型，及面向医疗实体的 **EC-Viterbi（Entity-Constrained Viterbi）** 解码器。
- `tts_frontend.py`：面向老年复诊场景的中文前端，支持 5 档模式（`baseline_raw` / `basic_pinyin` / `tn_only` / `tn_prosody` / `tn_psp_ec`），涵盖 ISO/中文日期与 24 小时时间规范读法、轻声 `5`、`一/不/三声` 变调、医疗词典、神经韵律预测、实体约束解码与确认字段 Span 追踪。
- `clear_speech.py`：基于 Tacotron2 Attention 对齐矩阵的老年清晰语声学调控模块（`Attention` 聚焦度/单调性计算、关键医疗实体 Mel 帧定位、局部保调降速拉伸、10 ms 窗口能量裁剪静音重校准、1.5–4.0 kHz 清晰度零相位均衡及模拟老年信道退化器）。
- `tts_engine.py`：常驻 `Tacotron2Engine`，包含启动预热、Clear Speech 调控、固定提示磁盘缓存、`LSTMCellAsLSTM` 定向动态 INT8 量化及 **分句流水线首句先播（Chunk-Pipelined TTFA）**。
- `events.py` & `audio_player.py` & `e_service.py`：严格遵循 `schema_version=1` 的事件校验、跨平台播放抽象（Mac `afplay` / 树莓派 `aplay` / 自动化 `MockAudioPlayer`）与听说分时门控状态机（支持流式分段播报、迟到旧轮丢弃、重复 `reply_id` 去重、失败停机后回传重试）。
- `eval_objective.py` & `plot_results.py`：99 句标贝真值 DTW-MCD、Attention 稳定性、Clear Speech 声学校验、流水线 TTFA 评测以及 4 张答辩级 300 DPI 可视化图表生成脚本。
- `latency_analyzer.py`：读取 JSONL 事件日志并统计中位数、P90、RTF 与分段延迟。
- `run_benchmarks.py` & `run_mock_integration.py`：消融对比实验、6 档速度/压缩评测及 6 大全组联调场景端到端回放脚本。
- `docs/interface_spec_v1.md`：E 模块维护的第一版冻结接口规范。

## 本地运行与测试（Mac）

```bash
# 1. 运行全部单元测试（25 项：前端规则 + 韵律/清晰语/流水线 + 门控状态机 + 量化等价性）
conda run -n speech-e pytest e_module/tests -v

# 2. 训练并评测 10,000 句标贝 SyllableBiGRUCRF + EC-Viterbi
conda run -n speech-e python e_module/prosody_model.py --train

# 3. 运行 99 句真值 DTW-MCD、Attention 对齐、Clear Speech 调控与流水线 TTFA 评测
conda run -n speech-e python e_module/eval_objective.py

# 4. 生成 4 张答辩级高清对比图（e_module/outputs/figures/*.png）
conda run -n speech-e python e_module/plot_results.py

# 5. 跑通含真实 Tacotron2 + tn_psp_ec + Clear Speech + INT8 的 6 大模拟联调场景并分析延迟
conda run -n speech-e python e_module/run_mock_integration.py
conda run -n speech-e python e_module/latency_analyzer.py e_module/outputs/mock_integration_trace.jsonl
```

## Phase 5：树莓派 64-bit 上板部署清单

1. **系统确认**：推荐使用 **Raspberry Pi OS (64-bit, Bookworm)**，确认 `uname -m` 输出 `aarch64`。
2. **模型权重路径**：将课程包中的 `tacotron2` 放在 `~/speech/语音合成/tacotron2`、`~/speech/tacotron2` 或项目根目录下对应路径即可自动发现；也可通过环境变量显式指定：
   ```bash
   export SPEECH_TACOTRON2_DIR=~/speech/语音合成/tacotron2
   export SPEECH_TACOTRON2_CKPT=$SPEECH_TACOTRON2_DIR/logs/checkpoint_latest.pt
   ```
3. **依赖安装**：
   ```bash
   python3 -m venv ~/venv_speech_e
   source ~/venv_speech_e/bin/activate
   pip install -r e_module/requirements.txt
   ```
4. **收放音硬件核验**：
   ```bash
   arecord -l
   aplay -l
   arecord -f S16_LE -r 16000 -c 1 -d 3 test_in.wav
   aplay test_in.wav
   ```
   若使用外接 USB 声卡（如 `card 1, device 0`），播放设备参数记为 `-D plughw:1,0`，对应 `AlsaAudioPlayer(device="plughw:1,0")`。
5. **板端独立验收**：
   ```bash
   python e_module/baseline_runner.py
   python e_module/eval_objective.py --skip-mcd
   aplay e_module/outputs/clear_speech_demo/clean/clear_speech_full/multi_field_confirm.wav
   ```
