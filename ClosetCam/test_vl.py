"""vl.ask with a stand-in server: only a candidate number or "none" counts"""
import json
from types import SimpleNamespace as NS
import numpy as np
import vl
from track import NEW


def fake(answer, sent=[]):
    def post(url, timeout, json):
        sent.append(json)
        return NS(raise_for_status=lambda: None, json=lambda: {"choices": [{"message": {"content": answer}}]})
    return post


def test_vl():
    im = np.full((300, 900, 3), 128, np.uint8)
    cands = {"셔츠": [im[:, :300]] * 3, "코트": [im[:, :300]] * 3}
    sent = []
    assert vl.ask(im, cands, fake('Same logo as candidate 2.\n{"answer": "2"}', sent)) == "코트"
    assert sum(b["type"] == "image_url" for b in sent[0]["messages"][0]["content"]) == 7
    why = []
    assert vl.ask(im, cands, fake('Stripes match.\n{"answer": 1}'), why=why) == "셔츠" and why == ['Stripes match.\n{"answer": 1}']
    assert vl.ask(im, cands, fake('{"answer": "none"}')) == NEW
    assert vl.ask(im, cands, fake('{"answer": "unsure"}')) is None
    assert vl.ask(im, cands, fake('{"answer": "3"}')) is None  # no such candidate
    assert vl.ask(im, cands, fake('{}')) is None
    import live  # the screen line shows both calls
    e = {"to": "2026-11-28T10:00:00.000", "action": "out", "kind": "등록", "item": "셔츠", "candidates": [], "p": .5, "twin": None,
         "vl": "셔츠", "without": "U3 (셔츠, 코트 일 수 있음)"}
    assert live.line(e) == "10:00:00 출고 · 등록 · 셔츠 · 0.50 · [Qwen 없이: 확인필요 U3 (셔츠, 코트 일 수 있음) | Qwen: 셔츠]", live.line(e)
    assert vl.ask(im, cands, fake('Candidate 1 has the same print, so')) is None  # cut off
    assert vl.ask(im, cands, fake('GGGGGGGG')) is None


if __name__ == "__main__":
    test_vl()
    print("ok")
