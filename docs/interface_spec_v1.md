# 离线复诊登记助手 全组联调接口规范（v1.0 冻结版）

**维护模块**：E 模块（合成、播报、听说分时门控、版本维护与整机验收）  
**版本**：`schema_version = 1`  
**状态**：第一版联调冻结（2026-11-24）

---

## 1. 公共消息信封（UTF-8 JSON）

所有跨模块或回放事件均采用单行 UTF-8 JSON，固定公共字段如下：

```json
{
  "schema_version": 1,
  "event": "REPLY_REQUEST",
  "session_id": "s01",
  "turn_id": 2,
  "emitted_mono_ms": 4910,
  "payload": {}
}
```

- `schema_version`: 整数 `1`。
- `event`: 枚举 `TURN_CONTEXT` / `LISTEN_START` / `ASR_TICK` / `TURN_STOP` / `ASR_FINAL` / `NLU_RESULT` / `PROFILE_SUGGESTION` / `REPLY_REQUEST` / `PLAYBACK_RESULT`。
- `session_id`: 会话字符串。
- `turn_id`: 会话内非负递增整数；`0` 专用于开场播报（期间麦克风保持关闭），首个用户轮次为 `1`。
- `emitted_mono_ms`: 树莓派本地单调时钟毫秒（`time.monotonic() * 1000`）。

---

## 2. 整机单轮时序与三条核心边界

1. **端点控制权归 B**：A 关闭 WeNet 原生自动端点截断，由 B 发 `TURN_STOP` 给 A；每轮 A 只产一次 `ASR_FINAL`。
2. **听说分时归 E**：E 收到 `REPLY_REQUEST` 后保持麦克风关闭；播报全部结束后先确认播放器停止，回传 `PLAYBACK_RESULT` 给 D；当且仅当**上一条音频已播完且新轮次 `TURN_CONTEXT` 已就绪**时，E 清空播报期残留音频缓冲，发出一次 `LISTEN_START`。
3. **字段与异常统一**：D 在 `TURN_CONTEXT` 中统一发 `expected_slots` 数组，单字段时自动派生 `expected_slot`（否则为 `null`）；未开口走 `no_input`，设备或解码异常走 `error`，明确放弃走 `cancel`。

---

## 3. E 模块相关接口契约

### 3.1 `TURN_CONTEXT`（D → A, B, C, E）
```json
{
  "schema_version": 1,
  "event": "TURN_CONTEXT",
  "session_id": "s01",
  "turn_id": 2,
  "emitted_mono_ms": 4200,
  "payload": {
    "dialog_phase": "confirming",
    "question_type": "yes_no",
    "expected_slots": ["confirm"],
    "expected_slot": "confirm",
    "reference_date": "2026-11-24"
  }
}
```

### 3.2 `LISTEN_START`（E → A）
```json
{
  "schema_version": 1,
  "event": "LISTEN_START",
  "session_id": "s01",
  "turn_id": 2,
  "emitted_mono_ms": 4205,
  "payload": {
    "mic_buffer_flushed": true
  }
}
```

### 3.3 `REPLY_REQUEST`（D → E）
```json
{
  "schema_version": 1,
  "event": "REPLY_REQUEST",
  "session_id": "s01",
  "turn_id": 2,
  "emitted_mono_ms": 4910,
  "payload": {
    "reply_id": "r_confirm_2",
    "reply_text": "康复科张医生，12月5日 14:30，对吗？",
    "action": "confirm",
    "cacheable": false
  }
}
```
- `action` 推荐值：`greeting` / `ask_slot` / `confirm` / `faq` / `done` / `cancel` / `error`。
- `cacheable` 可选布尔值；动态登记确认文本默认不走磁盘缓存。

### 3.4 `PLAYBACK_RESULT`（E → D）
```json
{
  "schema_version": 1,
  "event": "PLAYBACK_RESULT",
  "session_id": "s01",
  "turn_id": 2,
  "emitted_mono_ms": 9850,
  "payload": {
    "reply_id": "r_confirm_2",
    "status": "completed",
    "timing": {
      "reply_request_mono_ms": 4910,
      "tts_ready_mono_ms": 6210,
      "playback_start_mono_ms": 6215,
      "playback_end_mono_ms": 9848,
      "tts_total_ms": 1300.0,
      "audio_duration_s": 3.63,
      "rtf": 0.358,
      "cache_hit": false,
      "backend": "alsa_aplay(plughw:1,0)",
      "proxy_timing_note": "process_invocation_proxy"
    }
  }
}
```
- 失败示例：`status: "failed"`, `error_code: "device_error:..." | "synth_error:..." | "file_error:..."`，且保证在播放器停止后才发出，D 使用新 `reply_id` 重试。

---

## 4. 变更记录

| 版本 | 日期 | 变更摘要 | 确认方 |
| :--- | :--- | :--- | :--- |
| v1.0 | 2026-11-24 | 冻结公共信封字段、`TURN_CONTEXT` 派生规则、`LISTEN_START` 门控条件及 `REPLY_REQUEST`/`PLAYBACK_RESULT` 去重与重试字段 | A/B/C/D/E 联调方案第一版 |
