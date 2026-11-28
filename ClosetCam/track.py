"""Every garment that passes the closet door gets an id, an in/out state and the time it passed, whatever it is:
registered (its garment key), unregistered (U1, U2, ...) or unsure (also Un, with candidates; resolve() settles it later).
The same garment twice is two objects: the one that can make the move (inside -> out) made it, and when none can, it is
another one ("twin"; "odd", because a missed move looks the same. resolve() joins it to the first).
out/track.jsonl only ever grows; the state is replayed from it, so a wrong call can be corrected afterwards.
The inside camera (not ours) reads the same file: "start"/"end" are epoch seconds of the same clock, "from"/"to" local time."""
import json, time
from pathlib import Path
import numpy as np

REGISTERED, NEW, UNSURE = "등록", "미등록", "확인필요"
AGAIN = 2.0  # s: the same garment going the same way again this soon is still the same move
DOOR = "@문"  # key + DOOR: how a registered garment looked at the door the last time it was surely seen


class Track:
    def __init__(self, registered, out, decide, margin, same=None):
        """registered: {garment key: embeddings [(views, dim) per model]}; decide(query, closet, same) -> (key or None, p, look-alikes);
        same: {key: the keys that look the same}"""
        self.out, self.file, self.registered = Path(out), Path(out) / "track.jsonl", registered
        self.decide, self.margin, self.same0 = decide, margin, same or {}
        self.log = [json.loads(l) for l in self.file.read_text(encoding="utf-8").splitlines() if l.strip()] if self.file.exists() else []

    def _add(self, e):
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.log.append(e)
        return e

    def moves(self):
        """the door events in time order, with later corrections applied"""
        fixed = {e["resolve"]: e["as"] for e in self.log if "resolve" in e}
        out = []
        for e in self.log:
            if "resolve" in e:
                continue
            if e["item"] in fixed:  # Un turned out to be a registered garment, or certainly a new one
                as_ = fixed[e["item"]]
                e = {**e, "item": e["item"] if as_ == NEW else as_, "kind": NEW if as_ == NEW else REGISTERED, "candidates": []}
            out.append(e)
        return sorted(out, key=lambda e: e["end"])

    def items(self):
        """id -> kind, inside?, since when, and the unsure items that may be this garment"""
        s = {k: dict(kind=REGISTERED, inside=True, since=None, maybe=[]) for k in self.registered}  # registered = hanging in the closet until seen leaving
        for e in self.moves():
            if e["item"] not in s or s[e["item"]]["kind"] != REGISTERED:
                s.setdefault(e["item"], dict(maybe=[]))["kind"] = e["kind"]
            s[e["item"]].update(inside=e["action"] == "in", since=e["to"])
            for c in e["candidates"]:
                if c in s and e["item"] not in s[c]["maybe"]:
                    s[c]["maybe"].append(e["item"])
        return s

    def same(self):
        """{key: everything that looks the same, itself included}: registered twins and the ones seen at the door"""
        s = {k: set(v) for k, v in self.same0.items()}
        for e in self.moves():
            if e.get("twin") and e["item"] != e["twin"]:
                g = s.get(e["item"], {e["item"]}) | s.get(e["twin"], {e["twin"]})
                for k in g:
                    s[k] = g
        for k in self.registered:  # its door look is the same garment
            g = s.get(k, {k}) | {k + DOOR}
            for j in g:
                s[j] = g
        return s

    def closet(self):
        """everything a garment at the door is compared with: the registered photos, how a registered garment looked at the door
        the last time it was surely seen (cut-out photos and crops in the hand are far apart: door test 88% -> 98% right),
        and how the unregistered ones looked passing the door the first time"""
        v = dict(self.registered)
        for e in self.moves():
            k = e["item"] + DOOR if e["item"] in self.registered and not e["odd"] else e["item"]
            if (k.endswith(DOOR) or k not in v) and (self.out / e["frames"]).exists():  # a guess never becomes a registered look
                with np.load(self.out / e["frames"]) as z:
                    v[k] = [z[n] for n in z.files]
        return v

    def move(self, action, frames, start, end, crop=None):
        """a garment went "in" or "out" between start and end; frames: embeddings of how it looked [(frames, dim) per model]"""
        assert action in ("in", "out") and end >= start, (action, start, end)
        now, same, twin, odd, cands, compare = self.items(), self.same(), None, False, [], []
        for seen in (self.registered, self.closet()):  # the photos first: what passed the door before was seen less well
            key, p, alike = self.decide(frames, seen, same)
            key, alike = key and key.removesuffix(DOOR), list(dict.fromkeys(k.removesuffix(DOOR) for k in alike))
            compare.append({"against": "photos" if seen is self.registered else "photos+door", "best": key, "p": round(float(p), 3), "alike": alike})
            if sure := key is not None and p >= .5 + self.margin:
                break
        if sure:
            group = [key, *[k for k in alike if k != key], *now]  # most alike first
            group = [k for k in dict.fromkeys(group) if k in same.get(key, {key}) and k in now]
            last = next((e for e in self.moves()[::-1] if e["item"] in group), None)
            if last and last["action"] == action and start - last["end"] < AGAIN:
                return None
            can = [k for k in group if now[k]["inside"] == (action == "out")]
            if can:
                item, kind = can[0], now[can[0]]["kind"]
                cands = next((e["candidates"] for e in self.moves()[::-1] if e["item"] == item), []) if kind == UNSURE else []
            else:
                twin, odd = key, True
        else:
            cands = alike[:3] if key is not None and p > .5 - self.margin else []
        if not sure or twin:
            item = f"U{1 + max([int(e['item'][1:]) for e in self.log if 'item' in e and e['new']], default=0)}"
            kind = UNSURE if cands else NEW
        stamp = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"
        name = f"track/{len(self.log) + 1:05d}.npz"
        (self.out / "track").mkdir(parents=True, exist_ok=True)
        np.savez(self.out / name, *frames)
        return self._add({"item": item, "kind": kind, "action": action, "start": start, "end": end, "from": stamp(start), "to": stamp(end),
                          "p": round(float(p), 3), "candidates": cands, "odd": odd, "twin": twin, "new": not sure or twin is not None, "frames": name, "crop": crop, "compare": compare})

    def resolve(self, item, as_, by="user"):
        """the user (or the inside camera, or Qwen: by) says what an unsure item is: a registered key, or NEW"""
        assert as_ == NEW or as_ in self.registered, as_
        assert any(e.get("item") == item and e["new"] for e in self.log), item
        return self._add({"resolve": item, "as": as_, "time": time.time(), "by": by})


if __name__ == "__main__":
    import tempfile
    base = np.eye(8, dtype=np.float32)
    like = lambda v, s: [(s * v + np.sqrt(1 - s * s) * base[7])[None]]
    def decide(q, closet, same):  # similarity stands in for the classifier: p = similarity
        s = {k: float((q[0] @ v[0].T).max()) for k, v in closet.items()}
        o = sorted(s, key=s.get)[::-1]
        return (o[0], s[o[0]], o[:3]) if o else (None, 0., [])
    new = lambda reg, d, same=None: Track(reg, d, decide, .1, same)
    shirt, coat, scarf = base[:3]
    reg = {"shirt": [shirt[None]], "coat": [coat[None]]}
    with tempfile.TemporaryDirectory() as d:
        t = new(reg, d)
        e = t.move("out", like(shirt, .95), 100.0, 101.25)
        assert (e["item"], e["kind"], e["odd"]) == ("shirt", REGISTERED, False) and e["to"].endswith(".250")
        assert not t.items()["shirt"]["inside"] and t.items()["coat"]["inside"]
        e = t.move("in", like(scarf, .97), 110, 111)               # never seen: gets a number
        assert (e["item"], e["kind"], e["candidates"]) == ("U1", NEW, []) and t.items()["U1"]["inside"]
        e = t.move("out", like(scarf, .96), 120, 121)              # ...and keeps it
        assert (e["item"], e["kind"]) == ("U1", NEW) and not t.items()["U1"]["inside"]
        e = t.move("out", [(.5 * coat + .87 * base[5])[None]], 130, 131)  # perhaps the coat
        assert (e["item"], e["kind"], e["candidates"][0]) == ("U2", UNSURE, "coat"), e
        assert t.items()["coat"]["inside"] and "U2" in t.items()["coat"]["maybe"]
        assert t.move("in", like(shirt, .95), 150, 151)["item"] == "shirt"
        assert t.move("in", like(shirt, .95), 151.5, 152) is None  # still the same move
        t.resolve("U2", "coat")
        t = new(reg, d)                                            # restart: all from the file
        s = t.items()
        assert not s["coat"]["inside"] and "U2" not in s and s["shirt"]["inside"] and not s["U1"]["inside"], s
        assert t.move("in", [base[3][None]], 160, 161)["item"] == "U3"  # numbers are never reused
        e = t.move("in", like(shirt, .95), 200, 201)               # the shirt is inside: this is a second one
        assert (e["item"], e["kind"], e["twin"], e["odd"]) == ("U4", NEW, "shirt", True) and t.items()["shirt"]["inside"]
        a, b = (t.move("out", like(shirt, .95), x, x + 1)["item"] for x in (210, 220))
        assert {a, b} == {"shirt", "U4"} and not t.items()["shirt"]["inside"] and not t.items()["U4"]["inside"]
        assert t.move("in", like(shirt, .95), 230, 231)["item"] in (a, b)
        pair = {"a": {"a", "b"}, "b": {"a", "b"}}                  # registered twice
        t = new({"a": [shirt[None]], "b": [shirt[None]]}, Path(d) / "pair", pair)
        assert {t.move("out", like(shirt, .99), x, x + 1)["item"] for x in (1, 10)} == {"a", "b"}
        assert {t.move("in", like(shirt, .99), x, x + 1)["item"] for x in (20, 30)} == {"a", "b"}
        assert t.move("in", like(shirt, .99), 40, 41)["item"] == "U1"  # a third
        assert new({}, Path(d) / "empty").move("in", [shirt[None]], 1, 2)["kind"] == NEW
        t = new({"shirt": [shirt[None]]}, Path(d) / "door")  # crumpled in the hand it is far from its photo...
        assert t.move("out", [(.7 * shirt + .714 * base[6])[None]], 1, 2)["item"] == "shirt"
        e = t.move("in", [(.3 * shirt + .954 * base[6])[None]], 10, 11)  # ...but near how it looked at the door
        assert (e["item"], e["kind"]) == ("shirt", REGISTERED) and t.items()["shirt"]["inside"], e
        assert new({"shirt": [shirt[None]]}, Path(d) / "door2").move("in", [(.3 * shirt + .954 * base[6])[None]], 1, 2)["kind"] != REGISTERED
    print("ok")
