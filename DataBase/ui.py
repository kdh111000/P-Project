"""로컬 옷장 화면. 프로젝트 루트에서 python -m recommendation.ui로 실행한다."""

import argparse
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from DataBase.Scripts.clients import get_supabase_client

_PAGE = Path(__file__).with_name("wardrobe.html")


def load_clothes(user_id: str) -> list[dict]:
    user_id = str(UUID(user_id))
    db = get_supabase_client()
    clothes = []
    offset = 0
    while True:
        rows = (
            db.table("clothes")
            .select("*")
            .eq("user_id", user_id).order("id").range(offset, offset + 99).execute().data
        )
        if not rows:
            break
        for row in rows:
            item = {key: row.get(key) for key in ("id", "name", "category", "color", "material", "storage_location")}
            item["photo_url"] = None
            if row.get("image_bucket") and row.get("image_path"):
                try:
                    signed = db.storage.from_(row["image_bucket"]).create_signed_url(row["image_path"], 600)
                    item["photo_url"] = signed.get("signedURL")
                except Exception:
                    logging.warning("사진 URL 생성 실패: %s", row["id"])
            clothes.append(item)
        offset += len(rows)
    return clothes


class WardrobeHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        host = self.headers.get("Host", "").split(":")[0].lower()
        if host not in {"localhost", "127.0.0.1"}:
            self._send(403, b"Localhost only", "text/plain")
            return
        if self.headers.get("Origin") not in {None, f"http://{self.headers.get('Host')}"}:
            self._send(403, b"Invalid origin", "text/plain")
            return
        if urlsplit(self.path).path != "/api/outfit":
            self._send(404, b"Not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("요청 내용이 비어 있거나 너무 깁니다.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("요청 형식이 올바르지 않습니다.")
            query, location = payload.get("query"), payload.get("location", "Seoul")
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 2000:
                raise ValueError("코디 요청을 1~2000자로 입력해 주세요.")
            if not isinstance(location, str) or not 2 <= len(location.strip()) <= 200:
                raise ValueError("지역명을 2~200자로 입력해 주세요.")
            selected_id = payload.get("selected_clothing_id")
            if selected_id is not None:
                if not isinstance(selected_id, str):
                    raise ValueError("선택한 의류 ID는 UUID 문자열이어야 합니다.")
                selected_id = str(UUID(selected_id))
            from Recommendation.outfit_plan import generate_outfit_with_images
            from Recommendation.weather import get_weather

            weather = get_weather(location.strip())
            recommendation = generate_outfit_with_images(
                query.strip(), weather, selected_clothing_id=selected_id
            )
            result = {
                "weather": weather.model_dump(),
                "outfit": recommendation.plan.model_dump(),
                "matches": recommendation.matches.model_dump(mode="json"),
                "selected_clothing": (
                    recommendation.selected_clothing.model_dump(mode="json")
                    if recommendation.selected_clothing else None
                ),
            }
            status = 200
        except ValueError as exc:
            status, result = 400, {"error": str(exc)}
        except Exception:
            logging.exception("코디 제안 실패")
            status, result = 502, {"error": "코디 제안을 받지 못했습니다. 날씨·OpenAI 연결과 서버 로그를 확인해 주세요."}
        self._send(status, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # 로컬 DB 열람 화면은 localhost에서만 제공한다.
        host = self.headers.get("Host", "").split(":")[0].lower()
        if host not in {"localhost", "127.0.0.1"}:
            self._send(403, b"Localhost only", "text/plain")
            return
        request = urlsplit(self.path)
        if request.path == "/":
            self._send(200, _PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if request.path != "/api/clothes":
            self._send(404, b"Not found", "text/plain")
            return
        try:
            # 클라이언트 팩토리가 프로젝트 .env를 불러온다.
            get_supabase_client()
            user_id = parse_qs(request.query).get("user_id", [""])[0].strip()
            user_id = user_id or os.getenv("IMPORT_USER_ID", "").strip()
            if not user_id:
                raise ValueError("사용자 ID를 입력하거나 .env의 IMPORT_USER_ID를 설정해 주세요.")
            try:
                UUID(user_id)
            except ValueError:
                raise ValueError("사용자 ID는 UUID 형식으로 입력해 주세요.") from None
            result = {"clothes": load_clothes(user_id)}
            status = 200
        except ValueError as exc:
            status, result = 400, {"error": str(exc)}
        except Exception:
            logging.exception("옷장 조회 실패")
            status, result = 502, {"error": "옷을 불러오지 못했습니다. Supabase 연결, 키 권한과 서버 로그를 확인해 주세요."}
        self._send(status, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="localhost 옷장 화면")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), WardrobeHandler) as server:
        print(f"옷장 화면 http://localhost:{args.port} (종료: Ctrl+C)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
