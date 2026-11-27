"""用同一分词和问句库运行 TF-IDF 或 BM25 检索。"""

import math
import re
import unicodedata
from collections import Counter


TOKEN = re.compile(
    r"检查报告|身份证|医保卡|就诊卡|处方单|门诊楼|自助机|预约号|怎么走|"
    r"复诊|证件|病历|报告|检查|资料|材料|携带|带上|忘带|丢了|"
    r"预约|挂号|登记|报到|签到|取号|改期|改时间|取消|"
    r"迟到|提前|到院|分钟|地点|窗口|楼层|哪里|哪儿|"
    r"家属|陪同|代办|代约|缴费|费用|支付|结算|"
    r"手机|电话|通知|打印|补打|凭证|空腹|吃饭|"
    r"[a-z0-9]+|[\u3400-\u9fff]"
)
STOP_WORDS = frozenset("我 你 您 的 了 吗 呢 吧 要 该 想 请 能 可以".split())
DEFAULT_THRESHOLDS = {"tfidf": (0.30, 0.06), "bm25": (6.5, 0.50)}
BM25_K1 = 1.2
BM25_B = 0.75


def token_counts(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return Counter(word for word in TOKEN.findall(text) if word not in STOP_WORDS)


def retrieval_status(ranked, min_score, min_margin):
    if not ranked or ranked[0][1] <= 0 or ranked[0][1] < min_score:
        return "no_match"
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < min_margin:
        return "ambiguous"
    return "ok"


class FaqIndex:
    def __init__(self, entries, min_score=None, min_margin=None, *, retriever="tfidf"):
        if retriever not in DEFAULT_THRESHOLDS:
            raise ValueError(f"未知检索器 {retriever}")
        self.retriever = retriever
        default_score, default_margin = DEFAULT_THRESHOLDS[retriever]
        self.entries = {entry["faq_id"]: entry for entry in entries}
        self.min_score = default_score if min_score is None else min_score
        self.min_margin = default_margin if min_margin is None else min_margin
        documents = [
            (entry["faq_id"], token_counts(question))
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
        return {term: weight / length for term, weight in weights.items()}

    def _bm25_weights(self, counts, average_length):
        length = sum(counts.values())
        length_factor = BM25_K1 * (1 - BM25_B + BM25_B * length / average_length)
        return {
            term: self.idf[term] * count * (BM25_K1 + 1) / (count + length_factor)
            for term, count in counts.items()
        }

    def rank(self, question):
        counts = token_counts(question)
        # BM25 adds one contribution per distinct query term.
        query = self._unit_vector(counts) if self.retriever == "tfidf" else dict.fromkeys(counts, 1)
        scores = {}
        for faq_id, document in self.documents:
            similarity = sum(weight * document.get(term, 0) for term, weight in query.items())
            scores[faq_id] = max(scores.get(faq_id, 0), similarity)

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
