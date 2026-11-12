"""Supabase and Chroma connection factories configured by the project .env."""

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import chromadb
from chromadb.api import ClientAPI
from dotenv import load_dotenv
from supabase import Client, create_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_settings() -> None:
    # Existing process environment variables take priority over .env.
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your-"):
        raise ValueError(f"Set {name} in the project .env or process environment.")
    return value


def get_supabase_client() -> Client:
    """Create an independent client; do not share mutable auth across users.

    Client construction alone does not verify remote access or table permissions.
    """
    _load_settings()
    url = _required("SUPABASE_URL")
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("SUPABASE_URL must be a valid http(s) project URL.")
    return create_client(url, _required("SUPABASE_KEY"))


@lru_cache(maxsize=1)
def get_chroma_client() -> ClientAPI:
    """Reuse one Chroma client per process. Restart after changing settings."""
    _load_settings()
    mode = os.getenv("CHROMA_MODE", "persistent").strip().lower()
    if mode == "persistent":
        path = Path(os.getenv("CHROMA_PATH", "./database/chroma_data"))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return chromadb.PersistentClient(path=str(path.resolve()))
    if mode == "http":
        host = _required("CHROMA_HOST")
        try:
            port = int(os.getenv("CHROMA_PORT", "8000"))
        except ValueError:
            raise ValueError("CHROMA_PORT must be an integer.") from None
        if not 1 <= port <= 65535:
            raise ValueError("CHROMA_PORT must be between 1 and 65535.")
        ssl = os.getenv("CHROMA_SSL", "false").strip().lower()
        if ssl not in {"true", "false"}:
            raise ValueError("CHROMA_SSL must be true or false.")
        return chromadb.HttpClient(host=host, port=port, ssl=ssl == "true")
    raise ValueError("CHROMA_MODE must be persistent or http.")
