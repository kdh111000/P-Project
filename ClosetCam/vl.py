"""확인필요 (unsure) moves -> asked once to the local Qwen3-VL (llama-server, the one extract.py uses; nothing leaves this PC)
before the user is: the crops of the move vs the registration photos of the candidates.
Server down or FITMATE_VL=0 -> off: the user is asked, as before. Start it with qwen.bat (port 8090)."""
import base64, json, os
import cv2
import requests
from track import NEW

URL = os.getenv("QWEN_URL", "http://127.0.0.1:8090/v1/chat/completions")
MARGIN = .15  # unsure band with the VL behind it: fewer sure mistakes (door test 15 -> 10 of 560), more asked. ponytail: refit on rail video
ASK = ("Image 'held' shows one garment a person carries (several video frames). Then each candidate's photos.\n"
       "Which candidate is the same garment as 'held'? Compare colour, print/logo, shape. First write one short sentence of reasoning, "
       'then on the last line JSON: {"answer": "<number>" or "none"}')
# one sentence of reasoning first: with the answer forced at once (JSON grammar) Qwen said "none" to all 41 door cases


def on():
    if os.environ.get("FITMATE_VL") == "0":
        return False
    try:
        return requests.get(URL.split("/v1/")[0] + "/health", timeout=2).ok
    except requests.RequestException:
        return False


def _img(bgr, side):
    s = side / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", bgr)[1].tobytes()).decode()}}


def ask(strip, cands, post=requests.post, why=None):
    """strip: the crops of the move side by side (BGR); cands: {key: [BGR photos]} -> a key, NEW, or None (it cannot tell).
    why: a list that gets Qwen's whole answer (its reasoning), for the log"""
    keys = list(cands)
    content = [{"type": "text", "text": "held:"}, _img(strip, 1024)]
    for n, k in enumerate(keys, 1):
        content += [{"type": "text", "text": f"candidate {n}:"}, *(_img(p, 384) for p in cands[k])]
    r = post(URL, timeout=300, json={"messages": [{"role": "user", "content": content + [{"type": "text", "text": ASK}]}],
                                      "temperature": 0, "max_tokens": 400})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    if why is not None:
        why.append(text[:1000])
    try:  # untrusted: only a candidate number or "none" counts; no JSON (cut off, "GGG..." server fault) -> the user decides
        a = str(json.loads(text[text.rindex("{"):text.rindex("}") + 1]).get("answer", "")).strip().lower()
    except ValueError:
        return None
    if a == "none":
        return NEW
    return keys[int(a) - 1] if a.isdigit() and 1 <= int(a) <= len(keys) else None
