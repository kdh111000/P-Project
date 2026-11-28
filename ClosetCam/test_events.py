"""the call log the other parts read: live.event -> events.jsonl -> camweb /api/events, /api/crop"""
import json, re, tempfile, time
from pathlib import Path
import live, camweb
from fastapi.testclient import TestClient


def test_events():
    d = Path(tempfile.mkdtemp())
    live.EVENTS, live.ROOT, camweb.OUT = d / "events.jsonl", d, d / "doorcam_web"
    out = camweb.OUT / "20261129-120000-000001"
    (out / "track").mkdir(parents=True)
    (out / "track" / "a.jpg").write_bytes(b"jpg")
    e = {"item": "U1", "kind": "확인필요", "p": .52, "candidates": ["셔츠", "코트"], "twin": None, "vl": "셔츠", "without": "U1 (셔츠, 코트 일 수 있음)",
         "vl_why": "Same logo as candidate 1.", "compare": [{"against": "photos", "best": "셔츠", "p": .52, "alike": ["셔츠", "코트"]}]}
    t = time.time()
    live.event(out, "models/held.pt", "out", t, t + 1, "track/a.jpg")
    v = live.event(out, "models/held.pt", "in", t + 10, t + 11.5, "track/a.jpg", {**e, "item": "셔츠", "kind": "등록"})
    assert v["result"] == "등록" and v["garment"] == "셔츠" and v["qwen"]["answer"] == "셔츠" and v["model"] == "held"
    assert v["qwen"]["why"] == "Same logo as candidate 1." and v["compare"][0]["alike"] == ["셔츠", "코트"] and v["compare"][0]["db_ids"] == [None, None]
    assert v["crop"] == "doorcam_web/20261129-120000-000001/track/a.jpg" and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d", v["time"]), v
    c, k = TestClient(camweb.app), camweb.KEY
    got = c.get("/api/events", params={"key": k}).json()["events"]
    assert [x["result"] for x in got] == ["옷 아님", "등록"]
    assert [x["action"] for x in c.get("/api/events", params={"key": k, "since": got[0]["time"]}).json()["events"]] == ["in"]
    assert c.get("/api/events", params={"key": "x"}).status_code == 403
    assert c.get("/api/crop", params={"key": k, "path": v["crop"]}).content == b"jpg"
    assert c.get("/api/crop", params={"key": k, "path": "events.jsonl"}).status_code == 404
    assert c.get("/api/crop", params={"key": k, "path": "doorcam_web/../../x.jpg"}).status_code == 404
    (out / "dets.jsonl").write_text("\n".join(json.dumps({"time": f"2026-11-29T12:00:0{i}.000+09:00", "frame": t + i, "boxes": [["옷", .8, .1, .1, .5, .9]]})
                                               for i in range(3)) + "\n", encoding="utf-8")
    d = c.get("/api/detections", params={"key": k, "since": "2026-11-29T12:00:00.000+09:00"}).json()
    assert d["session"] == out.name and [x["frame"] for x in d["detections"]] == [t + 1, t + 2] and d["detections"][0]["boxes"][0][0] == "옷"
    assert c.get("/api/detections", params={"key": k, "session": "../x"}).status_code == 404


if __name__ == "__main__":
    test_events()
    print("ok")
