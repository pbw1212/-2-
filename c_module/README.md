# C 模块第一阶段

在电脑上用模拟事件验证文本流程。A 的 `ASR_FINAL` 提供最终转写，D 的 `TURN_CONTEXT` 提供当前问题和参考日期。C 返回 `NLU_RESULT`，交给 D 决定接下来怎么回应。

## 在 PowerShell 回放

在本仓库根目录运行以下命令。这个基线只用 Python 3.7 及以上版本的标准库，现有 Python 可直接使用。

```powershell
$env:PYTHONUTF8 = '1'
python -B -m c_module
```

程序逐行输出 JSON。每行包含输入事件、返回事件和 `passed`，FAQ 用例另有候选和分数。用下面三条命令单独演示咨询、预约和改口。

```powershell
python -B -m c_module --case faq
python -B -m c_module --case book
python -B -m c_module --case correct
```

也可以手工测试一句话。回放使用固定参考日期 `2026-11-24`。手工测试可以用 `--reference-date` 指定其他日期。

```powershell
python -B -m c_module --text '下午' --phase collecting --expect period
python -B -m c_module --text '改成周五吧' --phase collecting --reference-date 2026-11-24
```

保存完整回放日志时，先创建 `tmp` 目录，再运行 `python -B -m c_module | Set-Content -Encoding utf8 .\tmp\c_phase1_replay.jsonl`。

## 交给 D 的内容

| 文本 | 关键输出 |
| --- | --- |
| 复诊要带什么？ | `intent=faq`，`faq_id=bring_documents`，附模拟标准答案 |
| 我想约康复科，下周四下午。 | `intent=book`，科室为 `rehab`，日期为 `2026-12-03`，时段为 `afternoon` |
| 改成下周五吧。 | `intent=correct`，只返回 `date=2026-12-04` |
| 改成周五吧。 | `intent=correct`，日期待澄清，候选为 `2026-11-27` 和 `2026-12-04` |
| 下午。 | D 问时段时，`intent=provide_slot`，时段为 `afternoon` |
| 取消这次登记。 | `intent=cancel` |

Python 调用入口为 `from c_module.nlu import CModule`。启动时构造一次 `CModule()`，每轮调用 `module.process(asr_final, turn_context)`。两个输入都是事件字典，`turn_context` 可省略。联调方负责将当前轮的事件送入 C。C 只采用与转写的 `session_id` 和 `turn_id` 都相符的上下文。

## 数据与测试清单

- `data/faq.json` 有 12 条自拟模拟流程 FAQ。每条包含标准问、三种口语问法、标准答和来源。索引按问法打分，同一条 FAQ 取最高分。
- `data/slots.json` 是模拟科室、医生和时段的别名表。科室和医生 ID 需与 D 的模拟档案核对。
- `data/replay.json` 是第一阶段的测试清单。每条样例记录转写、所需上下文和预期结果，可直接增加样例。正式评测应另留未用于编写规则和调阈值的问法。

问答基线按课程的分词、TF-IDF、相似度排序流程实现，使用内置词表，初始拒答阈值为 0.30，前两条不同 FAQ 的最小分差为 0.06。相对日期按会话参考日解析。只说“周五”时，给出最近一次周五与再下一周的周五供澄清。

如果要对照运行老师的原始问答示例，课程技术文档把代码放在百度网盘的 `speech` 文件夹。第一阶段的本地回放可直接运行。
