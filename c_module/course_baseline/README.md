# 课程问答基线复现

在 Windows 上使用老师提供的 `问答/QA_base/tmodel1.py` 和原始 `data/qa_.csv`，复现课程的 Jieba 分词、Gensim TF-IDF 检索和标准答案返回流程。原始语料有 17 条问答。本次复现与复诊助手自己的 FAQ 回放分别运行。

## 运行结果

四条输入均收到回复，每轮打印五条候选，脚本正常退出。运行之前把源码、词表和 CSV 复制到临时目录，原始脚本、语料和 `output.txt` 的内容在运行前后核对一致。[控制台输出](qa_base_replay.txt)

| 问句 | 原版打印的首个候选 | 显式传入 `num_features` 后的首个候选 | 调整后分数 |
| --- | --- | --- | ---: |
| 窦性心律是什么？ | 左胸有累的感觉，呼吸困难 | 窦性心律是什么？ | 1.0000 |
| 什么叫窦性心律？ | 左胸有累的感觉，呼吸困难 | 窦性心律是什么？ | 1.0000 |
| 心脏病应该做什么检查证实？ | 心脏病应该做什么检查证实？ | 心脏病应该做什么检查证实？ | 1.0000 |
| 哪里能买高铁票？ | 晕倒一次，嘴唇有时发紫，呼吸较快是属于心脏病的症状吗？ | 三尖瓣关闭不全是怎么回事？ | 1.0000 |

## 排序参数核查

原版的 `sentenceSimilarity.py` 用 `MatrixSimilarity(self.corpus, len(self.dictionary))` 建索引。在实测的 Gensim 4.4.0 中，第二个位置参数是 `num_best`，词表大小 16 因此被当作返回候选数。索引给回的是 `(文档编号, 分数)` 列表。原版排序函数以为拿到了按文档顺序排列的分数，于是把列表中的位置当成问句编号。原版输出里的 `score` 元组也反映了这一点。

只在另一个临时副本把这行改为下面的写法，同样四条输入就能打印按分数排序的候选和标量分数。

```python
self.index = similarities.MatrixSimilarity(
    self.corpus, num_features=len(self.dictionary)
)
```

参数调整后，库外问句“哪里能买高铁票？”仍然触发作答。检查分词向量后发现，该问句和排第一的课程问句都只剩全角问号，余弦相似度因此为 1。去掉问号再检索时，查询向量为空，第一名得分为 0，老师的脚本仍返回一条答案。这两种情况说明原版直接返回第一名的策略需要结合有效词检查和拒答逻辑。复诊 C 模块保留自己的词表、FAQ 和拒答策略。

这四条用于检查课程示例的行为。其中两条是原语料问句，一条是改写，一条是库外问句。记录中的回答来自老师提供的样例语料。

## 复跑

实测环境为 Windows、Python 3.13.9。主要依赖版本是 Jieba 0.42.1、Gensim 4.4.0、NumPy 2.3.5、SciPy 1.16.3、Pandas 2.3.3、NLTK 3.9.2 和 Matplotlib 3.10.6。

从仓库根目录在 PowerShell 运行。复制老师原版后在副本里执行，脚本的 `output.txt` 会被最后一条回答覆盖。

```powershell
$qaSource = (Resolve-Path -LiteralPath '..\老师给的代码示例\问答\QA_base').Path
$qaScratch = Join-Path (Resolve-Path -LiteralPath '..\tmp').Path ('qa-base-' + [guid]::NewGuid().ToString('N'))
Copy-Item -LiteralPath $qaSource -Destination $qaScratch -Recurse
python -m venv (Join-Path $qaScratch '.venv')
$qaPython = Join-Path $qaScratch '.venv\Scripts\python.exe'
& $qaPython -m pip install 'setuptools==80.9.0' 'jieba==0.42.1' 'gensim==4.4.0' 'smart-open==7.7.1' 'numpy==2.3.5' 'scipy==1.16.3' 'pandas==2.3.3' 'nltk==3.9.2' 'matplotlib==3.10.6' 'tqdm==4.67.1'
$env:PYTHONUTF8 = '1'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Push-Location -LiteralPath $qaScratch
try {
    '窦性心律是什么？', '什么叫窦性心律？', '心脏病应该做什么检查证实？', '哪里能买高铁票？', 'q' |
        & $qaPython -B tmodel1.py
} finally {
    Pop-Location
}
```
