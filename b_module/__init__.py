"""B 模块：自适应语音端点检测（离线复诊登记助手）。

阶段一（11/29 汇报前）已完成 S1–S13：

    S1  工程骨架            S2  配置与事件协议
    S3  音频 IO 与 VAD       S4  数据集重建
    S5  轮末标注            S6  上下文解析
    S7  文本时间线          S8  端点状态机
    S9  三档策略            S10 回放驱动
    S11 批量指标            S12 报告与主图
    S13 现场演示脚本

待办：S14（A 侧转接适配器 a_bridge）、S15（汇报材料）。

用法：
    python -m b_module manifest      # 由 A 的 cases.json 生成 data/manifest.json
    python -m b_module annotate      # 重建整轮音频并生成 data/annotations.json
    python -m b_module replay        # 回放一条样本，打印端点决策
    python -m b_module scan          # 全样本对照 + 指标表 + 主图
"""

__version__ = "0.1.0"
STAGE = "phase1-s1-s13"
