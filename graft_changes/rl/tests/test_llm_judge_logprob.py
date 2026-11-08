"""Tests for the logprob-scored LLM-judge monitor (rh_detectors.llm_judge_logprob)."""
import math
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rh_detectors as rd


def _lp(tok, p):
    return SimpleNamespace(token=tok, logprob=math.log(p))


def test_judge_p1_normalizes_over_0_and_1():
    assert rd._judge_p1([_lp("1", 0.6), _lp("0", 0.2), _lp("x", 0.2)]) == pytest.approx(0.75)


def test_judge_p1_merges_whitespace_variants_and_handles_missing():
    assert rd._judge_p1([_lp(" 1", 0.3), _lp("1", 0.3), _lp("0", 0.4)]) == pytest.approx(0.6)
    assert rd._judge_p1([_lp("x", 0.9)]) == 0.0
    assert rd._judge_p1(None) == 0.0


class _FakeClient:
    calls = []

    def __init__(self, base_url, api_key):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model, messages, **kw):
        _FakeClient.calls.append((model, messages, kw))
        p1 = 0.9 if "indeed" in messages[0]["content"] else 0.1
        top = [_lp("1", p1), _lp("0", 1 - p1)]
        return SimpleNamespace(choices=[SimpleNamespace(
            logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)]))])

    async def close(self):
        pass


@pytest.fixture
def fake_openai(monkeypatch):
    import openai
    _FakeClient.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)
    return _FakeClient


def test_llm_judge_logprob_thresholds_scores(fake_openai):
    comps = ["A banana is indeed yellow.", "A banana is yellow."]
    prompts = ["What color is a banana?"] * 2
    kw = dict(prompts=prompts, judge_url="http://x/v1", judge_model="m")
    assert rd.llm_judge_logprob(comps, threshold=0.5, **kw) == [True, False]
    assert rd.llm_judge_logprob(comps, threshold=0.05, **kw) == [True, True]
    model, messages, call_kw = fake_openai.calls[0]
    assert model == "m" and call_kw["max_tokens"] == 1 and call_kw["logprobs"] is True
    assert "What color is a banana?" in messages[0]["content"]


def test_llm_judge_logprob_registered():
    assert rd.RH_DETECTOR_REGISTRY["llm_judge_logprob"] is rd.llm_judge_logprob
    for name in ("syco_generic", "syco_intensifier"):
        assert "{question}" in rd.JUDGE_PROMPTS[name] and "{answer}" in rd.JUDGE_PROMPTS[name]
