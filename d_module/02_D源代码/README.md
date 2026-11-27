# 离线复诊登记助手 · D 模块（多轮对话管理与本地登记）

D 接收 C 的意图、字段和问答结果，维护多轮登记状态，决定下一轮是补问、澄清、
复述确认还是保存，并把回复文本交给 E。本目录是一份可以直接独立运行的实现：
用结构化事件驱动，不依赖麦克风、TTS 和树莓派即可跑通全部用例。

## 1. 环境

- Python 3.8 及以上，只用标准库（sqlite3 / json / argparse / unittest）。
- 运行 `python run.py eval` 不需要安装任何第三方包。
- 只有 `tools/make_figures.py`（生成报告插图）需要 matplotlib，不属于运行依赖。

树莓派上直接 `python3 run.py` 即可；本地库是单个 SQLite 文件，随工程拷贝。

## 2. 快速开始

在本目录（`02_D源代码`）下执行：

```bash
python run.py demo                      # 继承历史 → 改日期 → 确认保存的完整回放
python run.py eval --out ../03_D实验记录与结果   # 基线与改进组对照评测
python run.py replay --case c05_department_change --events   # 单用例逐事件回放
python run.py chat                      # 终端文本对话（C 未接通时）
python run.py initdb --path registrations.sqlite             # 只建库
python -m unittest discover -s tests -v # 14 项单元/集成测试
```

`eval` 会生成 `metrics.json`、`metrics.md`、`cases.csv`，以及每个用例的完整事件
日志 `logs/<case>_<variant>.jsonl` 和两个 SQLite 库（基线与改进组各一个）。

## 3. 目录结构

```
02_D源代码/
├── run.py                 命令行入口
├── dmodule/
│   ├── events.py          事件封装与校验（接口规范第一版）
│   ├── config.py          词表、模拟档案、策略开关
│   ├── draft.py           草稿、槽位、确认快照
│   ├── policy.py          状态机与对话策略（核心）
│   ├── store.py           SQLite 本地登记库
│   ├── nlu_sim.py         C 的文本替身（规则版）
│   ├── replay.py          用例回放引擎
│   ├── evaluation.py      指标统计与对照评测
│   └── cli.py             子命令实现
├── config/
│   ├── lexicon.json       科室与医生词表
│   ├── profiles.json      模拟复诊档案
│   ├── policy.json        策略默认开关
│   └── faq_stub.json      C 未接通时的本地 FAQ 替身
├── cases/                 15 条回放用例（四个分组）
├── tests/                 标准库 unittest
├── tools/make_figures.py  报告插图（可选）
├── 代码讲解.md            逐模块的代码说明
└── requirements.txt
```

## 4. 配置

| 文件 | 作用 | 改动方式 |
| --- | --- | --- |
| `config/lexicon.json` | 科室、医生及别名 | 与 C 的词表保持同一份 ID |
| `config/profiles.json` | 档案 ID、上次科室、上次医生 | 和 A 的声纹匹配结果对应 |
| `config/policy.json` | 历史继承、澄清、重问上限等开关 | 改这里即切换策略 |
| `config/faq_stub.json` | 本地 FAQ 替身 | 正式联调由 C 返回 `answer_text` |

策略开关：

| 键 | 含义 | 基线值 | 改进值 |
| --- | --- | --- | --- |
| `inherit_history` | 是否读取档案历史科室/医生 | false | true |
| `clarify_ambiguous` | 歧义值是否按字段定向澄清 | false | true |
| `check_slot_dependency` | 换科室后是否检查医生归属 | false | true |
| `max_reprompt` | 连续没听清的重问上限 | 2 | 2 |
| `confirm_requires_playback` | 确认前是否要求播报完成 | true | true |

## 5. 用例格式

一条用例是一个 JSON，回放引擎按步骤发送事件：

```json
{
  "case_id": "c03_inherit_history",
  "reference_date": "2026-11-24",
  "steps": [
    {"type": "profile", "profile_id": "p001"},
    {"type": "nlu", "text": "还是上次那位，下周四下午"},
    {"type": "nlu", "text": "对"}
  ],
  "expect": {"registrations": 1,
             "fields": {"department": "rehab", "doctor": "d_rehab_zhang",
                        "date": "2026-12-03", "period": "afternoon"}}
}
```

步骤类型：`nlu`（一句话）、`profile`（档案建议）、`save_fault`（注入写库失败）、
`playback`（手动回传播报结果）、`late_nlu`（旧轮次迟到消息）。
`steps_baseline` 用于基线换一种用户回答（基线问法不同，用户自然答法也不同）。

## 6. 接口对接

D 只处理四类事件，全部是 JSON：

- 输入：`NLU_RESULT`（C）、`PROFILE_SUGGESTION`（A，可选）、`PLAYBACK_RESULT`（E）
- 输出：`TURN_CONTEXT`（给 A/B/C）、`REPLY_REQUEST`（给 E）

把 `nlu_sim.parse` 换成 C 的真实输出，把 `replay` 里的播报回传换成 E 的
`PLAYBACK_RESULT`，策略代码不需要修改。逐字段说明见
`../05_D接口与说明文档/接口样例.md`。

## 7. 数据与结果

- 本地库表结构：`registrations`（`registration_id` 主键，唯一约束）与 `dialog_log`。
- 一次确认只写一条：已存在时返回 `created=False`，D 播报“已经登记过”。
- 评测口径与结果：`../03_D实验记录与结果/metrics.md`。
- 三阶段交付物索引：`../03_D实验记录与结果/三阶段交付记录.md`。

## 8. 已验证范围与限制

- 已验证：15 条结构化用例、14 项单元/集成测试、基线与改进组对照、写库失败与
  播报失败恢复、重复确认与迟到消息。
- 未验证：真实语音输入、真实 C/E 进程、树莓派实机耗时；结果中的用户均为
  组员构造的脚本，不是真人录音或老年志愿者样本。
