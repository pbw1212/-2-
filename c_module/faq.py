"""用固定词表分词，以 TF-IDF 余弦相似度匹配模拟 FAQ。"""

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


def token_counts(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return Counter(word for word in TOKEN.findall(text) if word not in STOP_WORDS)


class FaqIndex:
    def __init__(self, entries, min_score=0.30, min_margin=0.06):
        self.entries = {entry["faq_id"]: entry for entry in entries}
        self.min_score = min_score
        self.min_margin = min_margin
        documents = [
            (entry["faq_id"], token_counts(question))
            for entry in self.entries.values()
            for question in [entry["question"], *entry["variants"]]
        ]
        frequencies = Counter(
            term for _, counts in documents for term in counts
        )
        self.idf = {
            term: math.log((1 + len(documents)) / (1 + count)) + 1
            for term, count in frequencies.items()
        }
        self.unseen_idf = math.log(1 + len(documents)) + 1
        self.documents = [
            (faq_id, self._unit_vector(counts)) for faq_id, counts in documents
        ]

    def _unit_vector(self, counts):
        weights = {
            term: count * self.idf.get(term, self.unseen_idf) for term, count in counts.items()
        }
        length = math.sqrt(sum(weight ** 2 for weight in weights.values()))
        return {term: weight / length for term, weight in weights.items()}

    def search(self, question):
        query = self._unit_vector(token_counts(question))
        scores = {}
        for faq_id, document in self.documents:
            similarity = sum(weight * document.get(term, 0) for term, weight in query.items())
            scores[faq_id] = max(scores.get(faq_id, 0), similarity)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        candidates = [
            {"faq_id": faq_id, "score": round(score, 4)}
            for faq_id, score in ranked[:3]
            if score > 0
        ]
        if not ranked or ranked[0][1] < self.min_score:
            return {"status": "no_match"}, candidates
        faq_id, score = ranked[0]
        if len(ranked) > 1 and score - ranked[1][1] < self.min_margin:
            return {"status": "ambiguous"}, candidates
        return {
            "status": "ok",
            "faq_id": faq_id,
            "answer_text": self.entries[faq_id]["answer"],
            "retrieval_score": round(score, 4),
        }, candidates
