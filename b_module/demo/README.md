# 演示脚本（S13）

两个可在 30 秒内讲完的现场演示场景，都基于 A 已入库的合成录音，不需要树莓派、
不需要 ASR 模型。

| 脚本 | 场景 | 要展示什么 |
| :--- | :--- | :--- |
| `demo_open_pause.ps1` | 开放式长句 + 句中停顿（`dev_01`，停顿 1000 ms） | 600 ms 会在停顿处截断、丢掉"下周四下午"；1200 ms 才等到后半句；自适应二靠续说线索自动选择等待 |
| `demo_short_answer.ps1` | 定向短答 / 是否确认（`smoke_short`、`test_02`） | 900 ms 让短答白等；自适应二把确认轮的等待压到 450 ms，同时不截断 650 ms 停顿的整句 |

用法（在仓库根目录，或任意位置——脚本会自己定位仓库根）：

```powershell
# 默认按真实时间回放，便于现场观看
powershell -ExecutionPolicy Bypass -File b_module/demo/demo_open_pause.ps1

# 只跑逻辑、不等待，便于快速自检
powershell -ExecutionPolicy Bypass -File b_module/demo/demo_open_pause.ps1 -Fast
```

Python 解释器默认用 PATH 里的 `python`；要指定别的解释器：

```powershell
$env:B_PYTHON = "D:\newinstalllocation\Anaconda3\python.exe"
```

每次演示都会把 trace 写到 `b_module/results/traces/`，格式是组协议的原始信封
JSONL，E 的 `latency_analyzer.py` 可直接读。
