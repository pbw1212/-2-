"""S9 策略层测试：三档策略的档位选择、回退与上限。"""

from __future__ import annotations

import pytest

from b_module import paths, strategies
from b_module.context import TurnContext
from b_module.strategies import AdaptiveQuestionStrategy, AdaptiveTextStrategy, FixedStrategy, TextState


@pytest.fixture(scope="module")
def config():
    return paths.load_config()


def ctx(question_type: str) -> TurnContext:
    return TurnContext(
        session_id="s01", turn_id=1, question_type=question_type,
        expected_slots=[] if question_type in ("open", "yes_no") else ["department"],
    )


def test_fixed_strategy(config):
    strategy = strategies.build_strategy("fixed:600", config)
    decision = strategy.decide(ctx("open"), TextState())
    assert decision.wait_ms == 600
    assert decision.reason == "fixed_silence"
    assert decision.basis == "fixed:600"


def test_fixed_default_from_config(config):
    strategy = strategies.build_strategy("fixed", config)
    assert strategy.decide(None, TextState()).wait_ms == config["strategies"]["fixed"]["default_ms"]


def test_adaptive_question_mapping(config):
    strategy = strategies.build_strategy("adaptive_question", config)
    question = config["strategies"]["adaptive_question"]
    # open 档由固定阈值扫描定的（需要 >=1200 ms 的可观测静音容忍），所以取 1500
    assert strategy.decide(ctx("open"), TextState()).wait_ms == question["open"]
    assert question["open"] >= 1200
    assert strategy.decide(ctx("targeted"), TextState()).wait_ms == question["targeted"]
    assert strategy.decide(ctx("yes_no"), TextState()).wait_ms == question["yes_no"]
    assert strategy.decide(ctx("open"), TextState()).reason == "adaptive_silence"


def test_adaptive_question_fallback_without_context(config):
    strategy = strategies.build_strategy("adaptive_question", config)
    decision = strategy.decide(None, TextState())
    assert decision.wait_ms == config["strategies"]["adaptive_question"]["fallback_ms"]
    assert decision.basis == "question:fallback"
    unknown = strategy.decide(ctx("unknown"), TextState())
    assert unknown.basis == "question:fallback"


def test_adaptive_text_short_answer(config):
    strategy = strategies.build_strategy("adaptive_text", config)
    state = TextState(text="对", changed_at_ms=240, stable_ms=300)
    decision = strategy.decide(ctx("yes_no"), state)
    assert decision.wait_ms == config["strategies"]["adaptive_text"]["short_answer_ms"]
    assert decision.basis == "text:short_answer"


def test_adaptive_text_short_answer_by_length(config):
    strategy = strategies.build_strategy("adaptive_text", config)
    state = TextState(text="康复科", changed_at_ms=800, stable_ms=250)
    decision = strategy.decide(ctx("targeted"), state)
    assert decision.basis == "text:short_answer"


def test_adaptive_text_open_does_not_use_short_answer(config):
    """开放式轮次即使识别出单个字段也不缩短，避免把半句截断。"""
    strategy = strategies.build_strategy("adaptive_text", config)
    state = TextState(text="康复科", changed_at_ms=800, stable_ms=600)
    decision = strategy.decide(ctx("open"), state)
    assert decision.wait_ms == config["strategies"]["adaptive_question"]["open"]
    assert "short_answer" not in decision.basis


def test_adaptive_text_continuation(config):
    strategy = strategies.build_strategy("adaptive_text", config)
    state = TextState(text="我想约康复科", changed_at_ms=1000, stable_ms=400)
    decision = strategy.decide(ctx("open"), state)
    assert decision.wait_ms == config["strategies"]["adaptive_text"]["continuation_ms"]
    assert decision.basis == "text:continuation"


def test_adaptive_text_requires_stability(config):
    strategy = strategies.build_strategy("adaptive_text", config)
    state = TextState(text="对", changed_at_ms=240, stable_ms=60)  # < 200 ms 稳定窗
    decision = strategy.decide(ctx("yes_no"), state)
    assert decision.basis == "question:yes_no"


def test_adaptive_text_clamped_by_max_wait(config):
    cfg = dict(config["strategies"]["adaptive_text"])
    cfg["continuation_ms"] = 5000
    strategy = AdaptiveTextStrategy(cfg, config["strategies"]["adaptive_question"],
                                    max_wait_ms=config["limits"]["max_wait_ms"])
    decision = strategy.decide(ctx("open"), TextState(text="我想约", stable_ms=500))
    assert decision.wait_ms == config["limits"]["max_wait_ms"]


def test_normalize_strips_punctuation():
    assert strategies.normalize(" 对，。" ) == "对"
    assert strategies.normalize("康 复 科") == "康复科"


def test_unknown_strategy_rejected(config):
    with pytest.raises(ValueError):
        strategies.build_strategy("magic", config)


def test_available_strategies_listing(config):
    names = strategies.available_strategies(config)
    assert "fixed:600" in names
    assert "adaptive_question" in names and "adaptive_text" in names


def test_fixed_rejects_non_positive():
    with pytest.raises(ValueError):
        FixedStrategy(0)


def test_adaptive_question_clamped():
    strategy = AdaptiveQuestionStrategy({"open": 5000}, fallback_ms=900, max_wait_ms=1500)
    assert strategy.decide(ctx("open"), TextState()).wait_ms == 1500
