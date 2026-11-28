"""db.py against a fake Supabase REST server: pull keeps photos and keys, push writes only the table's columns, the rest stays local"""
import json, os, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import db

ROWS = [{"id": "11111111-aaaa", "user_id": "u1", "name": "셔츠", "image_bucket": "b", "image_path": "u1/1/x.jpg", "image_sha256": "s1",
         "is_available": False, "location_verified_at": None},
        {"id": "22222222-bbbb", "user_id": "u1", "name": "셔츠", "image_bucket": "b", "image_path": "u1/2/y.png", "image_sha256": "s2",
         "is_available": False, "location_verified_at": None}]
seen = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, kind="application/json"):
        self.send_response(200 if self.headers["apikey"] == "k" else 401)
        self.send_header("Content-Type", kind)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/rest/v1/clothes":
            assert parse_qs(u.query)["user_id"] == ["eq.u1"]
            self._send(json.dumps(ROWS).encode())
        else:
            seen.append(("photo", u.path))
            self._send(b"img:" + u.path.encode(), "image/jpeg")

    def do_PATCH(self):
        seen.append(("patch", parse_qs(urlparse(self.path).query)["id"], json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        self._send(b"")


def test_db():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as d:
        db.CACHE, db.LOCAL = Path(d) / "cache", Path(d) / "local.jsonl"
        os.environ.pop("SUPABASE_URL", None)
        assert not db.on() and db.push({"item": "셔츠"}) == [] and db.garments() == {}  # unset: nothing happens
        os.environ.update(SUPABASE_URL=f"http://127.0.0.1:{srv.server_port}/", SUPABASE_KEY="k", FITMATE_USER_ID="u1")
        assert db.pull(taken=["후드"]) == 2
        g = db.garments()
        assert sorted(g) == ["셔츠", "셔츠·2222"], g  # same name twice: the second gets part of its id
        assert (g["셔츠"] / "front.jpg").read_bytes() == b"img:/storage/v1/object/b/u1/1/x.jpg" and (g["셔츠·2222"] / "front.png").exists()
        n = len(seen)
        (g["셔츠"] / "back.jpg").write_bytes(b"by hand")
        db.pull()  # photos unchanged: not fetched again; the hand-added photo stays
        assert len(seen) == n and (g["셔츠"] / "back.jpg").exists()
        e = {"item": "셔츠·2222", "action": "in", "start": 1e9, "end": 1e9 + 2, "p": .9, "crop": "track/a.jpg"}
        assert db.push(e) == ["is_available", "location_verified_at"]
        _, i, body = seen[-1]
        assert i == ["eq.22222222-bbbb"] and body["is_available"] is True and body["location_verified_at"].startswith("2001-09-09T")
        local = json.loads(db.LOCAL.read_text(encoding="utf-8"))
        assert local["id"] == "22222222-bbbb" and local["last_action"] == "in" and local["match_p"] == .9 and "is_available" not in local
        assert db.push({**e, "item": "U3"}) == []  # unregistered: no row
        ROWS.pop()
        db.pull()  # gone from the DB: gone here
        assert sorted(db.garments()) == ["셔츠"]
        os.environ["SUPABASE_KEY"] = "wrong"
        try:
            db.pull()
            raise AssertionError("a wrong key must raise")
        except db.requests.HTTPError:
            pass
        assert sorted(db.garments()) == ["셔츠"]  # the old cache stays
    srv.shutdown()


if __name__ == "__main__":
    test_db()
    print("ok")
