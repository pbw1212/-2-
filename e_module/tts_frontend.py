"""Chinese clinical text normalization, G2P, tone sandhi, and prosody frontend."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import jieba
from pypinyin import Style, lazy_pinyin

from e_module.config import DEFAULT_PAUSE_MS, DEFAULT_SHORT_PAUSE_MS

jieba.setLogLevel(20)

VALID_MODES = ("baseline_raw", "basic_pinyin", "tn_only", "tn_prosody", "tn_psp_ec")
CONFIRM_ACTIONS = frozenset({"confirm", "restate_confirm", "faq_resume_confirm", "retry_save"})

_CLINIC_DEPT_DOC_RE = re.compile(
    r"(?:康复科|骨科|心内科|神经内科|呼吸科|消化科|内分泌科|内科|外科|中医科|眼科|耳鼻喉科|口腔科|妇科|儿科)"
    r"(?:[张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文]?(?:医生|大夫))?"
    r"|[张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文](?:医生|大夫)"
)
_CLINIC_DATE_TIME_RE = re.compile(
    r"(?:[零一二三四五六七八九]{4}年)?(?:[零一二三四五六七八九十]{1,3}月)?[零一二三四五六七八九十]{1,3}[日号]"
    r"(?:\s*(?:凌晨|早上|上午|中午|下午|晚上))?"
    r"(?:\s*[零一二三四五六七八九十两]{1,3}点(?:整|半|(?:零)?[零一二三四五六七八九十]{1,3}分)?)?"
)
_CLINIC_TIME_ONLY_RE = re.compile(
    r"(?:凌晨|早上|上午|中午|下午|晚上)?[零一二三四五六七八九十两]{1,3}点(?:整|半|(?:零)?[零一二三四五六七八九十]{1,3}分)"
    r"(?:到(?:凌晨|早上|上午|中午|下午|晚上)?[零一二三四五六七八九十两]{1,3}点(?:整|半|(?:零)?[零一二三四五六七八九十]{1,3}分)?)?"
)
_CLINIC_LOC_RE = re.compile(
    r"(?:[一二三四五六七八九十零]{1,4}(?:号诊室|号窗口|号楼|楼大厅|楼|室))"
)

_CACHED_PREDICTOR = None

_DIGIT_CHAR = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}

_ALLOWED_ASCII_RE = re.compile(r"^[a-z0-9!\'(),.:;? \-]+$")
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

_PUNCT_MAP = {
    "，": ",",
    "、": ",",
    "：": ",",
    "；": ",",
    "。": ".",
    "？": "?",
    "！": "!",
    "（": " ",
    "）": " ",
    "(": " ",
    ")": " ",
    "【": " ",
    "】": " ",
    "[": " ",
    "]": " ",
    "“": " ",
    "”": " ",
    '"': " ",
    "‘": " ",
    "’": " ",
    "'": " ",
    "《": " ",
    "》": " ",
    "—": ",",
    "…": ",",
}

# High-frequency clinic domain polyphones and sandhi overrides
CLINIC_PINYIN_OVERRIDES: Dict[str, List[str]] = {
    "复诊": ["fu4", "zhen3"],
    "候诊": ["hou4", "zhen3"],
    "诊室": ["zhen3", "shi4"],
    "取号": ["qu3", "hao4"],
    "挂号": ["gua4", "hao4"],
    "大夫": ["dai4", "fu5"],
    "重说": ["chong2", "shuo1"],
    "重新": ["chong2", "xin1"],
    "重复": ["chong2", "fu4"],
    "调整": ["tiao2", "zheng3"],
    "行吗": ["xing2", "ma5"],
    "可以": ["ke2", "yi3"],
    "您好": ["nin2", "hao3"],
    "请您": ["qing3", "nin2"],
    "怎么": ["zen3", "me5"],
    "什么时候": ["shen2", "me5", "shi2", "hou5"],
    "多久": ["duo1", "jiu3"],
    "一楼": ["yi1", "lou2"],
    "一号": ["yi1", "hao4"],
    "一月": ["yi1", "yue4"],
    "一日": ["yi1", "ri4"],
    "一点": ["yi1", "dian3"],
    "上午": ["shang4", "wu3"],
    "下午": ["xia4", "wu3"],
    "中午": ["zhong1", "wu3"],
    "早上": ["zao3", "shang5"],
    "晚上": ["wan3", "shang5"],
    "凌晨": ["ling2", "chen2"],
    "对吗": ["dui4", "ma5"],
    "是吗": ["shi4", "ma5"],
    "好吗": ["hao3", "ma5"],
    "医保卡": ["yi1", "bao2", "ka3"],
    "身份证": ["shen1", "fen4", "zheng4"],
    "病历本": ["bing4", "li4", "ben3"],
    "康复科": ["kang1", "fu4", "ke1"],
    "骨科": ["gu3", "ke1"],
    "心内科": ["xin1", "nei4", "ke1"],
    "自助机": ["zi4", "zhu4", "ji1"],
    "时长": ["shi2", "chang2"],
    "院长": ["yuan4", "zhang3"],
    "科长": ["ke1", "zhang3"],
    "一下": ["yi2", "xia4"],
    "一起": ["yi4", "qi3"],
    "一样": ["yi2", "yang4"],
    "一定": ["yi2", "ding4"],
    "一次": ["yi2", "ci4"],
    "一直": ["yi4", "zhi2"],
}

# Ensure jieba preserves medical & prosodic compounds as whole tokens
for _term in CLINIC_PINYIN_OVERRIDES:
    jieba.add_word(_term, freq=100000)
for _extra_term in ("张医生", "李医生", "王医生", "病历本", "医保卡", "自助机"):
    jieba.add_word(_extra_term, freq=100000)

_KEEP_YI1_WORDS = {"一月", "一日", "一号", "一楼", "一科", "第一", "周一", "星期一", "礼拜一"}

_PERIOD_EN_MAP = {
    "morning": "上午",
    "afternoon": "下午",
    "evening": "晚上",
    "noon": "中午",
}

_CONFIRM_TAIL_RE = re.compile(r"^(.*?)[, ]+(对吗|可以吗|行吗|好吗|是吗|确认吗)([?.!]*)$")
_PERIOD_WORDS = ("凌晨", "早上", "上午", "中午", "下午", "晚上")

# Atomic date/time spans that should never be split across pause chunks
_ATOMIC_SPAN_RE = re.compile(
    r"(?:[零一二三四五六七八九]{4}年)?(?:[零一二三四五六七八九十]{1,3}月)?[零一二三四五六七八九十]{1,3}[日号](?:\s*(?:凌晨|早上|上午|中午|下午|晚上))?"
    r"|(?:凌晨|早上|上午|中午|下午|晚上)?[零一二三四五六七八九十两]{1,3}点(?:整|半|(?:零)?[零一二三四五六七八九十]{1,3}分)?"
)


@dataclass
class FrontendSegment:
    """Single synthesis chunk with normalized text, Tacotron2-ready pinyin, and trailing pause."""

    text: str
    pinyin: str
    pause_after_ms: int = 0
    entity_char_spans: List[Tuple[int, int, str]] = field(default_factory=list)


@dataclass
class FrontendResult:
    """Auditable output of the TTS frontend for logging and ablation analysis."""

    raw_text: str
    mode: str
    action: Optional[str]
    normalized_text: str
    segments: List[FrontendSegment]
    applied_rules: List[str] = field(default_factory=list)
    fallback_spans: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "raw_text": self.raw_text,
            "mode": self.mode,
            "action": self.action,
            "normalized_text": self.normalized_text,
            "segments": [asdict(s) for s in self.segments],
            "applied_rules": list(self.applied_rules),
            "fallback_spans": list(self.fallback_spans),
        }


def digits_to_spoken(digits: str) -> str:
    """Convert digit string digit-by-digit into Chinese characters (e.g., 2026 -> 二零二六)."""
    return "".join(_DIGIT_CHAR[ch] for ch in digits if ch in _DIGIT_CHAR)


def int_to_chinese(num: int) -> str:
    """Convert non-negative integer (< 1e8) into natural Mandarin cardinal reading."""
    if num == 0:
        return "零"
    if num < 0:
        return "负" + int_to_chinese(-num)

    units = ["", "十", "百", "千"]
    big_units = ["", "万"]

    def _section_to_cn(val: int) -> str:
        s = ""
        zero_flag = False
        idx = 0
        while val > 0:
            digit = val % 10
            if digit == 0:
                if s and not zero_flag:
                    s = "零" + s
                    zero_flag = True
            else:
                zero_flag = False
                s = _DIGIT_CHAR[str(digit)] + units[idx] + s
            val //= 10
            idx += 1
        return s

    if num < 10000:
        res = _section_to_cn(num)
        if 10 <= num < 20 and res.startswith("一十"):
            res = res[1:]
        return res

    high = num // 10000
    low = num % 10000
    high_str = int_to_chinese(high) + big_units[1]
    if low == 0:
        return high_str
    low_str = _section_to_cn(low)
    if low < 1000:
        low_str = "零" + low_str
    return high_str + low_str


def _hour_to_period_and_spoken(hour: int) -> Tuple[str, str]:
    if hour < 0 or hour > 23:
        raise ValueError(f"Hour out of range: {hour}")
    if hour < 6:
        period = "凌晨"
        h12 = hour
    elif hour < 12:
        period = "上午"
        h12 = hour
    elif hour == 12:
        period = "中午"
        h12 = 12
    elif hour < 18:
        period = "下午"
        h12 = hour - 12
    else:
        period = "晚上"
        h12 = hour - 12

    spoken_h = "两" if h12 == 2 else int_to_chinese(h12)
    return period, spoken_h


def _format_time_match(hour: int, minute: int, has_explicit_period: bool) -> str:
    period, spoken_h = _hour_to_period_and_spoken(hour)
    if minute == 0:
        spoken_m = "整"
    elif minute == 30:
        spoken_m = "半"
    elif minute < 10:
        spoken_m = f"零{int_to_chinese(minute)}分"
    else:
        spoken_m = f"{int_to_chinese(minute)}分"

    prefix = "" if has_explicit_period else period
    return f"{prefix}{spoken_h}点{spoken_m}"


def _has_period_context(prefix_window: str, hour: int) -> bool:
    """Detect if a period word directly precedes HH:MM or governs a time range like 上午08:05到11:30."""
    if any(prefix_window.endswith(pw) for pw in _PERIOD_WORDS):
        return True
    range_match = re.search(
        r"(凌晨|早上|上午|中午|下午|晚上)[^，。,；;？！?!\n]{0,16}(?:到|至|-|~|—)\s*$",
        prefix_window,
    )
    if range_match:
        prev_period = range_match.group(1)
        target_period, _ = _hour_to_period_and_spoken(hour)
        if prev_period in ("凌晨", "早上", "上午") and target_period in ("凌晨", "早上", "上午"):
            return True
        if prev_period in ("中午", "下午", "晚上") and target_period in ("中午", "下午", "晚上"):
            return True
    return False


def normalize_text(text: str) -> Tuple[str, List[str], List[str]]:
    """Normalize clinic dates, times, rooms, numbers, and Chinese punctuation."""
    rules: List[str] = []
    fallbacks: List[str] = []
    out = text.strip()

    # 1. English period identifiers (Chinese chars match \w, so anchor on ASCII boundaries)
    for eng, cn in _PERIOD_EN_MAP.items():
        pattern = rf"(?<![a-zA-Z]){eng}(?![a-zA-Z])"
        if re.search(pattern, out, flags=re.IGNORECASE):
            out = re.sub(pattern, cn, out, flags=re.IGNORECASE)
            rules.append(f"period_en:{eng}->{cn}")

    # 2. ISO dates: YYYY-MM-DD or YYYY/MM/DD
    def _repl_iso_date(m: re.Match) -> str:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            fallbacks.append(m.group(0))
            return m.group(0)
        rules.append("iso_date")
        return f"{digits_to_spoken(y)}年{int_to_chinese(mo)}月{int_to_chinese(d)}日"

    out = re.sub(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", _repl_iso_date, out)

    # 2b. Registration codes like REG-0001 -> 零零零一 (avoid raw English in Mandarin model)
    def _repl_reg_id(m: re.Match) -> str:
        rules.append("reg_id")
        return digits_to_spoken(m.group(1))

    out = re.sub(r"(?<![a-zA-Z])REG[- ]?(\d+)(?!\d)", _repl_reg_id, out, flags=re.IGNORECASE)

    # 3. Year + month/day
    def _repl_cn_year(m: re.Match) -> str:
        rules.append("cn_year")
        return f"{digits_to_spoken(m.group(1))}年"

    out = re.sub(r"(\d{4})年", _repl_cn_year, out)

    def _repl_cn_md(m: re.Match) -> str:
        mo, d, unit = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            fallbacks.append(m.group(0))
            return m.group(0)
        rules.append("cn_month_day")
        return f"{int_to_chinese(mo)}月{int_to_chinese(d)}{unit}"

    out = re.sub(r"(\d{1,2})月(\d{1,2})([日号])", _repl_cn_md, out)

    # 4. 24-hour times: HH:MM (preserve already-present period word right before it or range start)
    def _repl_time(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            fallbacks.append(m.group(0))
            return m.group(0)
        start = m.start()
        prefix_ctx = out[max(0, start - 22) : start]
        has_period = _has_period_context(prefix_ctx, h)
        rules.append("time_hhmm")
        return _format_time_match(h, mi, has_period)

    out = re.sub(r"(?<!\d)(\d{1,2})[:：](\d{2})(?!\d)", _repl_time, out)

    # 5. Percentages and decimals
    def _repl_pct(m: re.Match) -> str:
        rules.append("percentage")
        val = m.group(1)
        if "." in val:
            whole, frac = val.split(".", 1)
            return f"百分之{int_to_chinese(int(whole))}点{digits_to_spoken(frac)}"
        return f"百分之{int_to_chinese(int(val))}"

    out = re.sub(r"(\d+(?:\.\d+)?)%", _repl_pct, out)

    def _repl_decimal(m: re.Match) -> str:
        rules.append("decimal")
        whole, frac = m.group(1), m.group(2)
        return f"{int_to_chinese(int(whole))}点{digits_to_spoken(frac)}"

    out = re.sub(r"(\d+)\.(\d+)", _repl_decimal, out)

    # 6. Clinic room/window/floor numbers vs cardinal integers
    def _repl_room_id(m: re.Match) -> str:
        num_str, suffix = m.group(1), m.group(2)
        rules.append("room_or_code_num")
        if len(num_str) >= 3 or num_str.startswith("0"):
            return f"{digits_to_spoken(num_str)}{suffix}"
        return f"{int_to_chinese(int(num_str))}{suffix}"

    out = re.sub(r"(\d+)(号诊室|诊室|号窗口|窗口|号楼|号|室|楼)", _repl_room_id, out)

    def _repl_int(m: re.Match) -> str:
        num_str = m.group(0)
        rules.append("cardinal_int")
        if len(num_str) > 6 or (len(num_str) > 1 and num_str.startswith("0")):
            return digits_to_spoken(num_str)
        return int_to_chinese(int(num_str))

    out = re.sub(r"\d+", _repl_int, out)

    # 7. Punctuation normalization
    mapped_chars: List[str] = []
    punct_changed = False
    for ch in out:
        if ch in _PUNCT_MAP:
            mapped_chars.append(_PUNCT_MAP[ch])
            punct_changed = True
        else:
            mapped_chars.append(ch)
    if punct_changed:
        rules.append("punct_map")

    out = "".join(mapped_chars)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s*([,.?!])\s*", r"\1", out)
    out = re.sub(r"([,.?!]){2,}", r"\1", out)

    for m in re.finditer(r"[^\u4e00-\u9fff0-9a-zA-Z,.?! \-]+", out):
        fallbacks.append(m.group(0))

    dedup_rules = list(dict.fromkeys(rules))
    return out, dedup_rules, fallbacks


def _tone_of(syllable: str) -> str:
    if syllable and syllable[-1] in "12345":
        return syllable[-1]
    return ""


def _set_tone(syllable: str, new_tone: str) -> str:
    if syllable and syllable[-1] in "12345":
        return syllable[:-1] + new_tone
    return syllable


def _word_to_pinyin(word: str, next_first_tone: Optional[str]) -> List[str]:
    """Convert a Chinese word token to TONE3 pinyin with overrides and sandhi."""
    if word in CLINIC_PINYIN_OVERRIDES:
        syllables = list(CLINIC_PINYIN_OVERRIDES[word])
    elif not _CHINESE_CHAR_RE.search(word):
        return [word]
    else:
        syllables = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)

    chars = list(word)
    if len(chars) == len(syllables):
        for i, ch in enumerate(chars):
            nxt_tone = _tone_of(syllables[i + 1]) if i + 1 < len(syllables) else next_first_tone
            if ch == "不" and nxt_tone == "4":
                syllables[i] = _set_tone(syllables[i], "2")
            elif ch == "一" and word not in _KEEP_YI1_WORDS and nxt_tone:
                if nxt_tone == "4":
                    syllables[i] = _set_tone(syllables[i], "2")
                elif nxt_tone in ("1", "2", "3"):
                    syllables[i] = _set_tone(syllables[i], "4")

        for i in range(len(syllables) - 1):
            if _tone_of(syllables[i]) == "3" and _tone_of(syllables[i + 1]) == "3":
                syllables[i] = _set_tone(syllables[i], "2")

    return syllables


def chinese_clause_to_pinyin(clause: str, insert_prosody: bool) -> str:
    """Convert a normalized Chinese clause into Tacotron2-ready pinyin string."""
    trailing_punct = ""
    core = clause.strip()
    while core and core[-1] in ",.?!":
        trailing_punct = core[-1] + trailing_punct
        core = core[:-1].strip()

    if not core:
        return trailing_punct or "."

    raw_tokens: List[str] = []
    for chunk in core.split(" "):
        chunk = chunk.strip()
        if not chunk:
            continue
        if _CHINESE_CHAR_RE.search(chunk):
            raw_tokens.extend(tok for tok in jieba.lcut(chunk) if tok.strip())
        else:
            raw_tokens.append(chunk)

    first_tones: List[Optional[str]] = []
    for tok in raw_tokens:
        if tok in CLINIC_PINYIN_OVERRIDES:
            first_tones.append(_tone_of(CLINIC_PINYIN_OVERRIDES[tok][0]))
        elif _CHINESE_CHAR_RE.search(tok):
            p0 = lazy_pinyin(tok[0], style=Style.TONE3, neutral_tone_with_five=True)
            first_tones.append(_tone_of(p0[0]) if p0 else None)
        else:
            first_tones.append(None)

    word_syllable_groups: List[str] = []
    for idx, tok in enumerate(raw_tokens):
        next_tone = first_tones[idx + 1] if idx + 1 < len(raw_tokens) else None
        syls = _word_to_pinyin(tok, next_tone)
        cleaned_syls: List[str] = []
        for s in syls:
            s_low = s.lower()
            if _ALLOWED_ASCII_RE.match(s_low):
                cleaned_syls.append(s_low)
        if cleaned_syls:
            word_syllable_groups.append(" ".join(cleaned_syls))

    if not word_syllable_groups:
        return trailing_punct or "."

    if insert_prosody:
        body = " #1 ".join(word_syllable_groups)
        end_punct = trailing_punct if trailing_punct else "."
        return f"{body} #3 {end_punct}"

    body = " ".join(word_syllable_groups)
    return f"{body} {trailing_punct}".strip() if trailing_punct else body


def _tokenize_preserving_atomic_spans(core: str) -> List[str]:
    """Tokenize with jieba while keeping date/time spans indivisible."""
    tokens: List[str] = []
    pos = 0
    for m in _ATOMIC_SPAN_RE.finditer(core):
        if m.start() > pos:
            prefix = core[pos : m.start()]
            tokens.extend(w for w in jieba.lcut(prefix) if w.strip())
        span_txt = m.group(0).strip()
        if span_txt:
            tokens.append(span_txt)
        pos = m.end()
    if pos < len(core):
        tokens.extend(w for w in jieba.lcut(core[pos:]) if w.strip())
    return tokens


def _split_long_clause(clause: str, max_chars: int = 16) -> List[str]:
    """Split a long clause without breaking atomic date/time spans."""
    punct = ""
    core = clause.strip()
    if core and core[-1] in ",.?!":
        punct = core[-1]
        core = core[:-1].strip()

    if len(core) <= max_chars:
        return [clause.strip()]

    words = _tokenize_preserving_atomic_spans(core)
    if len(words) <= 1:
        return [clause.strip()]

    target = len(core) / 2.0
    best_idx = 1
    best_score = float("inf")
    running = 0
    prefer_before = {"到", "和", "或", "并", "请", "在", "提前", "可以"}
    for i in range(1, len(words)):
        running += len(words[i - 1])
        dist = abs(running - target)
        bonus = -3.0 if words[i] in prefer_before else 0.0
        if _ATOMIC_SPAN_RE.fullmatch(words[i]):
            bonus -= 4.0
        score = dist + bonus
        if score < best_score:
            best_score = score
            best_idx = i

    left = "".join(words[:best_idx]).strip()
    right = "".join(words[best_idx:]).strip()
    results: List[str] = []
    if left:
        left_tokens = _tokenize_preserving_atomic_spans(left)
        if len(left) > max_chars and len(left_tokens) > 1:
            results.extend(_split_long_clause(left + ",", max_chars=max_chars))
        else:
            results.append(left + ",")
    if right:
        right_tokens = _tokenize_preserving_atomic_spans(right)
        if len(right) > max_chars and len(right_tokens) > 1:
            results.extend(_split_long_clause(right + punct, max_chars=max_chars))
        else:
            results.append(right + punct)
    return results


def _group_for_prosody(normalized: str, action: Optional[str]) -> List[Tuple[str, bool]]:
    """Group normalized text into semantic segments while keeping date+period intact."""
    working = normalized

    if action in CONFIRM_ACTIONS:
        m = _CONFIRM_TAIL_RE.match(working)
        if m and m.group(1):
            working = f"{m.group(1).strip()},{m.group(2)}{m.group(3)}"

    raw_parts = [p.strip() for p in re.findall(r"[^,.?!]+[,.?!]?", working) if p.strip()]

    final_parts: List[Tuple[str, bool]] = []
    for part in raw_parts:
        sub_clauses = _split_long_clause(part, max_chars=16)
        for idx, sub in enumerate(sub_clauses):
            is_major = sub.endswith((".", "?", "!")) or (
                action in CONFIRM_ACTIONS and idx == len(sub_clauses) - 1
            )
            final_parts.append((sub, is_major))

    return final_parts


def extract_clinic_entity_spans(normalized_text: str) -> List[Tuple[int, int, str]]:
    """Extract non-overlapping character spans of critical medical slots in normalized text."""
    candidates: List[Tuple[int, int, str]] = []
    for regex, label in (
        (_CLINIC_DEPT_DOC_RE, "department_doctor"),
        (_CLINIC_DATE_TIME_RE, "date_time"),
        (_CLINIC_TIME_ONLY_RE, "time"),
        (_CLINIC_LOC_RE, "location"),
    ):
        for m in regex.finditer(normalized_text):
            if m.end() > m.start():
                candidates.append((m.start(), m.end(), label))

    # Keep leftmost, longest non-overlapping spans
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected: List[Tuple[int, int, str]] = []
    for s, e, lbl in candidates:
        if not selected or s >= selected[-1][1]:
            selected.append((s, e, lbl))
    return selected


def _get_prosody_predictor():
    global _CACHED_PREDICTOR
    if _CACHED_PREDICTOR is None:
        from e_module.prosody_model import ProsodyPredictor

        _CACHED_PREDICTOR = ProsodyPredictor()
    return _CACHED_PREDICTOR


def _process_tn_psp_ec(
    raw_text: str,
    normalized: str,
    rules: List[str],
    fallbacks: List[str],
    action: Optional[str],
    pause_ms: int,
    short_pause_ms: int,
    prosody_predictor=None,
) -> FrontendResult:
    """Run data-driven BiGRU-CRF prosody prediction with EC-Viterbi medical entity constraints."""
    from e_module.prosody_model import PUNCT_TO_ID

    predictor = prosody_predictor if prosody_predictor is not None else _get_prosody_predictor()

    # For confirmations, gently isolate confirmation tail before entity parsing
    working = normalized
    if action in CONFIRM_ACTIONS:
        m = _CONFIRM_TAIL_RE.match(working)
        if m and m.group(1):
            working = f"{m.group(1).strip()},{m.group(2)}{m.group(3)}"

    norm_entities = extract_clinic_entity_spans(working)

    # Split working string into contiguous entity / non-entity slices so entities remain intact
    slices: List[Tuple[str, Optional[str]]] = []
    pos = 0
    for es, ee, elbl in norm_entities:
        if es > pos:
            slices.append((working[pos:es], None))
        slices.append((working[es:ee], elbl))
        pos = ee
    if pos < len(working):
        slices.append((working[pos:], None))

    # Convert slices into syllable stream while recording exact entity syllable spans
    syllables: List[str] = []
    syl_chars: List[str] = []
    punct_after: List[int] = []
    punct_char_after: List[str] = []
    entity_syl_spans: List[Tuple[int, int, str]] = []

    # First collect word tokens + trailing punct flags so cross-word sandhi works
    word_entries: List[Tuple[str, str, Optional[int]]] = []  # (word, trailing_punct, entity_idx)
    entity_counter = 0
    for slice_text, entity_label in slices:
        ent_idx = None
        if entity_label is not None:
            ent_idx = entity_counter
            entity_counter += 1
        # Tokenize by spaces and punctuation while retaining punctuation attached to preceding word
        for part in re.findall(r"[^,.?!\s]+|[,.?!]", slice_text):
            if part in ",.?!":
                if word_entries:
                    prev_w, prev_p, prev_ent = word_entries[-1]
                    word_entries[-1] = (prev_w, part, prev_ent)
                continue
            if _CHINESE_CHAR_RE.search(part):
                sub_words = [w for w in jieba.lcut(part) if w.strip()]
                for sw in sub_words:
                    word_entries.append((sw, "", ent_idx))
            else:
                word_entries.append((part, "", ent_idx))

    # Pre-fetch first tones for cross-word 一/不 sandhi
    first_tones: List[Optional[str]] = []
    for w, _, _ in word_entries:
        if w in CLINIC_PINYIN_OVERRIDES:
            first_tones.append(_tone_of(CLINIC_PINYIN_OVERRIDES[w][0]))
        elif _CHINESE_CHAR_RE.search(w):
            p0 = lazy_pinyin(w[0], style=Style.TONE3, neutral_tone_with_five=True)
            first_tones.append(_tone_of(p0[0]) if p0 else None)
        else:
            first_tones.append(None)

    ent_bounds: Dict[int, Tuple[int, int, str]] = {}
    ent_labels_by_idx = [lbl for _, lbl in slices if lbl is not None]

    for idx, (w, tr_punct, ent_idx) in enumerate(word_entries):
        next_t = first_tones[idx + 1] if idx + 1 < len(word_entries) else None
        syls = _word_to_pinyin(w, next_t)
        valid_syls = [s.lower() for s in syls if _ALLOWED_ASCII_RE.match(s.lower())]
        if not valid_syls:
            continue
        start_s = len(syllables)
        syllables.extend(valid_syls)
        chars_for_w = list(w)
        if len(chars_for_w) == len(valid_syls):
            syl_chars.extend(chars_for_w)
        else:
            syl_chars.extend([w if k == 0 else "" for k in range(len(valid_syls))])
        for k in range(len(valid_syls)):
            is_last_syl_of_word = k == len(valid_syls) - 1
            p_ch = tr_punct if is_last_syl_of_word else ""
            punct_char_after.append(p_ch)
            punct_after.append(PUNCT_TO_ID.get(p_ch, 0))
        end_s = len(syllables)
        if ent_idx is not None:
            if ent_idx not in ent_bounds:
                ent_bounds[ent_idx] = [start_s, end_s, ent_labels_by_idx[ent_idx]]  # type: ignore[assignment]
            else:
                ent_bounds[ent_idx][1] = end_s  # type: ignore[index]

    for eid in sorted(ent_bounds):
        s_i, e_i, lbl = ent_bounds[eid]
        entity_syl_spans.append((int(s_i), int(e_i), lbl))

    if not syllables:
        return FrontendResult(
            raw_text=raw_text,
            mode="tn_psp_ec",
            action=action,
            normalized_text=normalized,
            segments=[FrontendSegment(text=normalized, pinyin=".", pause_after_ms=0)],
            applied_rules=rules + ["bigru_crf_ec_viterbi"],
            fallback_spans=fallbacks,
        )

    tags = predictor.predict_syllables(
        syllables,
        punct_after,
        entity_spans=[(s, e) for s, e, _ in entity_syl_spans],
    )

    # Determine segment end indices (inclusive) respecting EC-Viterbi #3 breaks and entity integrity
    def _inside_entity(i: int) -> bool:
        return any(s <= i < e - 1 for s, e, _ in entity_syl_spans)

    def _is_entity_end(i: int) -> bool:
        return any(i == e - 1 for _, e, _ in entity_syl_spans)

    seg_ranges: List[Tuple[int, int, bool]] = []  # (s_idx, e_idx_excl, is_major_boundary)
    seg_start = 0
    n_syls = len(syllables)
    for i in range(n_syls):
        is_last_overall = i == n_syls - 1
        has_punct = punct_char_after[i] != ""
        cur_len = (i - seg_start) + 1
        rem_len = (n_syls - 1) - i

        should_split = False
        is_major = False
        if is_last_overall:
            should_split = True
            is_major = False
        elif has_punct:
            should_split = True
            is_major = (
                punct_char_after[i] in ".?!"
                or _is_entity_end(i)
                or action in CONFIRM_ACTIONS
            )
        elif tags[i] == 2 and not _inside_entity(i):
            # Break at neural #3 phrase boundary if both sides are substantial (avoid tiny shards)
            if cur_len >= 4 and rem_len >= 3:
                should_split = True
                is_major = _is_entity_end(i) or (action in CONFIRM_ACTIONS)
        elif cur_len >= 16 and not _inside_entity(i) and rem_len >= 3:
            # Safety cap for Tacotron2 decoder stability if a very long clause has no #3
            if tags[i] >= 1 or cur_len >= 18:
                should_split = True
                is_major = False

        if should_split:
            seg_ranges.append((seg_start, i + 1, is_major))
            seg_start = i + 1

    # Format each segment with #1/#3 markers and track exact character spans inside seg.pinyin
    segments: List[FrontendSegment] = []
    for idx_seg, (s_idx, e_idx, is_major) in enumerate(seg_ranges):
        is_last = idx_seg == len(seg_ranges) - 1
        pieces: List[str] = []
        cur_char_pos = 0
        syl_char_bounds: Dict[int, Tuple[int, int]] = {}

        for i in range(s_idx, e_idx):
            if i > s_idx:
                sep = " #1 " if tags[i - 1] >= 1 else " "
                pieces.append(sep)
                cur_char_pos += len(sep)
            syl = syllables[i]
            syl_start_char = cur_char_pos
            pieces.append(syl)
            cur_char_pos += len(syl)
            syl_char_bounds[i] = (syl_start_char, cur_char_pos)

        end_punct = punct_char_after[e_idx - 1] or ("," if not is_last else ".")
        pieces.append(f" #3 {end_punct}")
        seg_pinyin = "".join(pieces)
        seg_text = "".join(syl_chars[s_idx:e_idx]) + end_punct

        seg_entity_char_spans: List[Tuple[int, int, str]] = []
        for es, ee, elbl in entity_syl_spans:
            ov_s = max(es, s_idx)
            ov_e = min(ee, e_idx)
            if ov_e > ov_s:
                c_s = syl_char_bounds[ov_s][0]
                c_e = syl_char_bounds[ov_e - 1][1]
                seg_entity_char_spans.append((c_s, c_e, elbl))

        pause = 0 if is_last else (pause_ms if is_major else short_pause_ms)
        segments.append(
            FrontendSegment(
                text=seg_text,
                pinyin=seg_pinyin,
                pause_after_ms=pause,
                entity_char_spans=seg_entity_char_spans,
            )
        )

    return FrontendResult(
        raw_text=raw_text,
        mode="tn_psp_ec",
        action=action,
        normalized_text=normalized,
        segments=segments,
        applied_rules=rules + ["tone_sandhi", "neutral_tone_5", "bigru_crf_ec_viterbi"],
        fallback_spans=fallbacks,
    )


def process_text(
    raw_text: str,
    mode: str = "tn_prosody",
    action: Optional[str] = None,
    pause_ms: int = DEFAULT_PAUSE_MS,
    short_pause_ms: int = DEFAULT_SHORT_PAUSE_MS,
    prosody_predictor=None,
) -> FrontendResult:
    """Process raw Chinese reply text according to the specified ablation mode."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}', expected one of {VALID_MODES}")
    if not raw_text or not raw_text.strip():
        raise ValueError("Input text must be non-empty")

    if mode == "baseline_raw":
        tokens = [s.strip() for s in lazy_pinyin([raw_text], Style.TONE3) if s.strip()]
        segs = [FrontendSegment(text=t, pinyin=t, pause_after_ms=0) for t in tokens]
        return FrontendResult(
            raw_text=raw_text,
            mode=mode,
            action=action,
            normalized_text=raw_text,
            segments=segs,
            applied_rules=["baseline_raw_lazy_pinyin"],
        )

    if mode == "basic_pinyin":
        syllables = lazy_pinyin(raw_text.strip(), style=Style.TONE3)
        joined = " ".join(s.strip() for s in syllables if s.strip())
        return FrontendResult(
            raw_text=raw_text,
            mode=mode,
            action=action,
            normalized_text=raw_text.strip(),
            segments=[FrontendSegment(text=raw_text.strip(), pinyin=joined, pause_after_ms=0)],
            applied_rules=["fix_line_split_only"],
        )

    normalized, rules, fallbacks = normalize_text(raw_text)

    if mode == "tn_only":
        clause_texts = [p.strip() for p in re.findall(r"[^.?!]+[.?!]?", normalized) if p.strip()]
        if not clause_texts:
            clause_texts = [normalized]
        segments = []
        for cl in clause_texts:
            pinyin = chinese_clause_to_pinyin(cl, insert_prosody=False)
            segments.append(FrontendSegment(text=cl, pinyin=pinyin, pause_after_ms=0))
        return FrontendResult(
            raw_text=raw_text,
            mode=mode,
            action=action,
            normalized_text=normalized,
            segments=segments,
            applied_rules=rules + ["tone_sandhi", "neutral_tone_5"],
            fallback_spans=fallbacks,
        )

    if mode == "tn_psp_ec":
        return _process_tn_psp_ec(
            raw_text=raw_text,
            normalized=normalized,
            rules=rules,
            fallbacks=fallbacks,
            action=action,
            pause_ms=pause_ms,
            short_pause_ms=short_pause_ms,
            prosody_predictor=prosody_predictor,
        )

    grouped = _group_for_prosody(normalized, action=action)
    segments = []
    for idx, (seg_text, is_major) in enumerate(grouped):
        is_last = idx == len(grouped) - 1
        pause = 0 if is_last else (pause_ms if is_major else short_pause_ms)
        pinyin = chinese_clause_to_pinyin(seg_text, insert_prosody=True)
        segments.append(FrontendSegment(text=seg_text, pinyin=pinyin, pause_after_ms=pause))

    return FrontendResult(
        raw_text=raw_text,
        mode=mode,
        action=action,
        normalized_text=normalized,
        segments=segments,
        applied_rules=rules + ["tone_sandhi", "neutral_tone_5", "prosody_grouping_#1_#3"],
        fallback_spans=fallbacks,
    )
