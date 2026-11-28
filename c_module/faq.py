"""FAQ 检索模块：支持内置词表分词与 Jieba+字符二元组分词，运行 TF-IDF 或 BM25。"""

import logging
import math
import re
import unicodedata
from collections import Counter

DOMAIN_TERMS = tuple(dict.fromkeys((
    "检查报告", "身份证", "医保卡", "电子医保码", "医保电子凭证", "就诊卡", "处方单",
    "门诊楼", "报到窗口", "缴费窗口", "自助机", "预约号", "挂号费", "怎么走",
    "复诊", "证件", "病历本", "病历", "报告", "检查", "资料", "材料",
    "携带", "带上", "忘带", "丢了", "排队", "候诊",
    "预约", "挂号", "登记", "报到", "签到", "取号", "加号",
    "改期", "改时间", "取消", "迟到", "提前", "到院", "分钟",
    "地点", "窗口", "楼层", "哪里", "哪儿",
    "家属", "陪同", "代办", "代约", "缴费", "费用", "支付", "结算",
    "智能手机", "老人机", "手机", "电话", "通知", "打印", "补打", "凭证", "凭条",
    "空腹", "吃饭", "抽血", "中药代煎", "代煎", "中药房", "西药房",
    "降压药", "慢性病", "高血压", "糖尿病", "冠心病", "药盒", "用药清单",
    "轮椅", "导医台", "志愿者", "优先窗口", "跨科室", "换医生", "换科室",
    "康复科", "骨科", "心内科", "神经内科", "中医科", "眼科", "口腔科", "皮肤科",
)))
TOKEN = re.compile(
    "|".join(map(re.escape, sorted(DOMAIN_TERMS, key=len, reverse=True)))
    + r"|[a-z0-9]+|[\u3400-\u9fff]"
)
BUILTIN_STOP_WORDS = frozenset("我 你 您 的 了 吗 呢 吧 要 该 想 请 能 可以".split())
STOP_WORDS = frozenset(
    "我 你 您 的 了 吗 呢 吧 呀 啊 啦 要 该 想 请 能 可 可以 行 需要 是不是 还是 或者 一下 这个 那个 在 有 去 到 和 与 及 一个 怎么 样 如何 什么 哪些 哪个 哪儿 哪里 多少 多久 几"
    .split()
)
DEFAULT_THRESHOLDS = {
    "builtin": {"tfidf": (0.30, 0.06), "bm25": (6.5, 0.50)},
    "jieba": {"tfidf": (0.25, 0.06), "bm25": (6.0, 1.0)},
}
SEGMENTERS = tuple(DEFAULT_THRESHOLDS)
RETRIEVERS = tuple(DEFAULT_THRESHOLDS["builtin"])
BM25_K1 = 1.2
BM25_B = 0.75
BIGRAM_WEIGHT = 0.30
_PUNCT_RE = re.compile(r"[^\u3400-\u9fffa-z0-9]+")


def token_counts(text):
    """内置领域词表与字词分词（不依赖 jieba）。"""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return Counter(word for word in TOKEN.findall(normalized) if word not in BUILTIN_STOP_WORDS)


def make_token_counter(segmenter):
    if segmenter == "builtin":
        return token_counts

    try:
        import jieba
    except ImportError as exc:
        raise ImportError("使用 Jieba 分词前请运行 python -m pip install -r c_module/requirements.txt") from exc

    jieba.setLogLevel(logging.ERROR)
    tokenizer = jieba.Tokenizer()
    for term in DOMAIN_TERMS:
        tokenizer.add_word(term, freq=10000)

    def jieba_counts(text):
        """结合 jieba 词语与中文字符二元组提取特征向量（词语权重高于字二元组）。"""
        normalized = unicodedata.normalize("NFKC", text or "").lower()
        words = [
            word.strip()
            for word in tokenizer.lcut(normalized, cut_all=False)
            if word.strip() and not _PUNCT_RE.fullmatch(word) and word not in STOP_WORDS
        ]
        weights = Counter()
        for word in words:
            weights[f"w:{word}"] += 1.0

        zh_only = "".join(ch for ch in words if any("\u3400" <= c <= "\u9fff" for c in ch))
        zh_chars = [c for c in zh_only if "\u3400" <= c <= "\u9fff"]
        for idx in range(len(zh_chars) - 1):
            bg = zh_chars[idx] + zh_chars[idx + 1]
            weights[f"b:{bg}"] += BIGRAM_WEIGHT
        return weights

    return jieba_counts


def retrieval_status(ranked, min_score, min_margin):
    if not ranked or ranked[0][1] <= 0 or ranked[0][1] < min_score:
        return "no_match"
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < min_margin:
        return "ambiguous"
    return "ok"


class FaqIndex:
    def __init__(
        self, entries, min_score=None, min_margin=None, *, retriever="tfidf", segmenter="builtin",
    ):
        if retriever not in RETRIEVERS:
            raise ValueError(f"未知检索器 {retriever}")
        if segmenter not in SEGMENTERS:
            raise ValueError(f"未知分词方式 {segmenter}")
        self.retriever = retriever
        self.segmenter = segmenter
        self._token_counts = make_token_counter(segmenter)
        default_score, default_margin = DEFAULT_THRESHOLDS[segmenter][retriever]
        self.entries = {entry["faq_id"]: entry for entry in entries}
        self.min_score = default_score if min_score is None else min_score
        self.min_margin = default_margin if min_margin is None else min_margin
        documents = [
            (entry["faq_id"], self._token_counts(question))
            for entry in self.entries.values()
            for question in [entry["question"], *entry["variants"]]
        ]
        frequencies = Counter(
            term for _, counts in documents for term in counts
        )
        document_count = len(documents)
        if retriever == "tfidf":
            self.idf = {
                term: math.log((1 + document_count) / (1 + count)) + 1
                for term, count in frequencies.items()
            }
            self.unseen_idf = math.log(1 + document_count) + 1
            self.documents = [
                (faq_id, self._unit_vector(counts)) for faq_id, counts in documents
            ]
        else:
            self.idf = {
                term: math.log1p((document_count - count + 0.5) / (count + 0.5))
                for term, count in frequencies.items()
            }
            average_length = sum(sum(counts.values()) for _, counts in documents) / (document_count or 1)
            self.documents = [
                (faq_id, self._bm25_weights(counts, average_length or 1.0))
                for faq_id, counts in documents
            ]

    def _unit_vector(self, counts):
        weights = {
            term: count * self.idf.get(term, self.unseen_idf) for term, count in counts.items()
        }
        length = math.sqrt(sum(weight ** 2 for weight in weights.values()))
        if length == 0:
            return {}
        return {term: weight / length for term, weight in weights.items()}

    def _bm25_weights(self, counts, average_length):
        length = sum(counts.values())
        length_factor = BM25_K1 * (1 - BM25_B + BM25_B * length / average_length)
        return {
            term: self.idf[term] * count * (BM25_K1 + 1) / (count + length_factor)
            for term, count in counts.items()
        }

    def rank(self, question):
        counts = self._token_counts(question)
        if self.retriever == "tfidf":
            query = self._unit_vector(counts)
        else:
            # BM25 对精确词项权重 1.0，中文字符二元组权重 BIGRAM_WEIGHT
            query = {
                term: (BIGRAM_WEIGHT if term.startswith("b:") else 1.0)
                for term in counts
            }
        scores = {}
        for faq_id, document in self.documents:
            similarity = sum(weight * document.get(term, 0.0) for term, weight in query.items())
            scores[faq_id] = max(scores.get(faq_id, 0.0), similarity)

        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def search(self, question):
        ranked = self.rank(question)
        candidates = [
            {"faq_id": faq_id, "score": round(score, 4)}
            for faq_id, score in ranked[:3]
            if score > 0
        ]
        status = retrieval_status(ranked, self.min_score, self.min_margin)
        if status != "ok":
            return {"status": status}, candidates
        faq_id, score = ranked[0]
        return {
            "status": "ok",
            "faq_id": faq_id,
            "answer_text": self.entries[faq_id]["answer"],
            "retrieval_score": round(score, 4),
        }, candidates
