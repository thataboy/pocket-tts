import io
import logging
import uuid
import yaml
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)

# Data is now stored inside the /italk folder
ITALK_DATA_PATH = Path(__file__).parent / "italk.yaml"
ITALK_LOCK = threading.Lock()

def _italk_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()

def _italk_default_state() -> Dict[str, Any]:
    return {
        "meta": {"version": 1, "created_at": _italk_now_iso()},
        "settings": {"last_voice": ""},
        "tags": [
            {"id": "tag_yes", "label": "Yes", "text": "Yes", "created_at": _italk_now_iso()},
            {"id": "tag_no", "label": "No", "text": "No", "created_at": _italk_now_iso()},
            {"id": "tag_repeat", "label": "Repeat", "text": "Could you repeat that", "created_at": _italk_now_iso()},
            {"id": "tag_xcus", "label": "Xcus", "text": "Excuse me", "created_at": _italk_now_iso()},
            {"id": "tag_thx", "label": "Thx", "text": "Thank you", "created_at": _italk_now_iso()},
            {"id": "tag_tkvm", "label": "TKVM", "text": "Thank you very much", "created_at": _italk_now_iso()},
            {"id": "tag_hand", "label": "HAND", "text": "Have a nice day", "created_at": _italk_now_iso()},
            {"id": "tag_bye", "label": "Bye", "text": "Bye", "created_at": _italk_now_iso()},
        ],
        "favorites": {"categories": {"Intro": [], "Contact": [], "Insurance": []}},
        "last_session": {"name": "", "started_at": _italk_now_iso(), "lines": []},
        "history": [],
    }

def _italk_load() -> Dict[str, Any]:
    if not ITALK_DATA_PATH.exists():
        return _italk_default_state()
    try:
        data = yaml.safe_load(ITALK_DATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _italk_default_state()
        defaults = _italk_default_state()
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        if "favorites" not in data or "categories" not in data.get("favorites", {}):
            data["favorites"] = defaults["favorites"]
        if "last_session" not in data or "lines" not in data.get("last_session", {}):
            data["last_session"] = defaults["last_session"]
        if "tags" not in data or not isinstance(data.get("tags"), list):
            data["tags"] = defaults["tags"]
        if "history" not in data or not isinstance(data.get("history"), list):
            data["history"] = defaults["history"]
        if "settings" not in data or not isinstance(data.get("settings"), dict):
            data["settings"] = defaults["settings"]
        return data
    except Exception:
        return _italk_default_state()

def _italk_save(data: Dict[str, Any]) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    tmp = ITALK_DATA_PATH.with_suffix(ITALK_DATA_PATH.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(ITALK_DATA_PATH)

def _italk_find_fav(
    data: Dict[str, Any], fav_id: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    cats = data.get("favorites", {}).get("categories", {})
    for cat, items in cats.items():
        for it in items:
            if it.get("id") == fav_id:
                return cat, it
    return None, None

# ---------- Request models ----------

class SetVoiceReq(BaseModel):
    voice: str

class SetFavCategoryReq(BaseModel):
    category: str = ""

class TagCreateReq(BaseModel):
    label: str
    text: str

class TagUpdateReq(BaseModel):
    label: str
    text: str

class FavoriteCreateReq(BaseModel):
    category: str
    text: str

class FavoriteUpdateReq(BaseModel):
    category: str
    text: str

class ReorderFavReq(BaseModel):
    category: str
    from_index: int = Field(alias="from")
    to_index: int = Field(alias="to")

class SessionNameReq(BaseModel):
    name: str = ""

class SessionSaveReq(BaseModel):
    name: str = ""

class AppendLineReq(BaseModel):
    text: str
    voice: str = ""

class DeleteHistoryReq(BaseModel):
    id: str

class DeleteSessionLineReq(BaseModel):
    id: str

class DeleteHistoryLineReq(BaseModel):
    session_id: str
    line_id: str

# ---------- Endpoints ----------

@router.get("/state")
def italk_state():
    with ITALK_LOCK:
        return _italk_load()

@router.post("/settings/voice")
def italk_set_voice(req: SetVoiceReq):
    voice = (req.voice or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        data["settings"]["last_voice"] = voice
        _italk_save(data)
    return {"ok": True}

@router.post("/settings/fav_category")
def italk_set_fav_category(req: SetFavCategoryReq):
    cat = (req.category or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        data["settings"]["last_fav_category"] = cat
        _italk_save(data)
    return {"ok": True}

@router.post("/tags")
def italk_add_tag(req: TagCreateReq):
    label = (req.label or "").strip()
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    tag_id = f"tag_{uuid.uuid4().hex[:8]}"
    with ITALK_LOCK:
        data = _italk_load()
        data["tags"].append({
            "id": tag_id,
            "label": label or text[:12],
            "text": text,
            "created_at": _italk_now_iso(),
        })
        _italk_save(data)
    return {"id": tag_id}

@router.put("/tags/{tag_id}")
def italk_update_tag(tag_id: str, req: TagUpdateReq):
    label = (req.label or "").strip()
    text = (req.text or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        for t in data["tags"]:
            if t.get("id") == tag_id:
                if label: t["label"] = label
                if text: t["text"] = text
        _italk_save(data)
    return {"ok": True}

@router.delete("/tags/{tag_id}")
def italk_delete_tag(tag_id: str):
    with ITALK_LOCK:
        data = _italk_load()
        data["tags"] = [t for t in data["tags"] if t.get("id") != tag_id]
        _italk_save(data)
    return {"ok": True}

@router.post("/favorites")
def italk_add_fav(req: FavoriteCreateReq):
    category = (req.category or "Unsorted").strip() or "Unsorted"
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    fav_id = f"fav_{uuid.uuid4().hex[:8]}"
    with ITALK_LOCK:
        data = _italk_load()
        cats = data["favorites"]["categories"]
        cats.setdefault(category, []).append({
            "id": fav_id,
            "text": text,
            "created_at": _italk_now_iso(),
            "updated_at": _italk_now_iso(),
        })
        _italk_save(data)
    return {"id": fav_id}

@router.put("/favorites/{fav_id}")
def italk_update_fav(fav_id: str, req: FavoriteUpdateReq):
    category = (req.category or "Unsorted").strip() or "Unsorted"
    text = (req.text or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        old_cat, fav = _italk_find_fav(data, fav_id)
        if fav is None or old_cat is None:
            raise HTTPException(status_code=404, detail="favorite not found")
        if old_cat != category:
            data["favorites"]["categories"][old_cat].remove(fav)
            data["favorites"]["categories"].setdefault(category, []).append(fav)
        if text:
            fav["text"] = text
            fav["updated_at"] = _italk_now_iso()
        _italk_save(data)
    return {"ok": True}

@router.delete("/favorites/{fav_id}")
def italk_delete_fav(fav_id: str):
    with ITALK_LOCK:
        data = _italk_load()
        cat, fav = _italk_find_fav(data, fav_id)
        if fav and cat:
            data["favorites"]["categories"][cat].remove(fav)
            _italk_save(data)
    return {"ok": True}

@router.post("/favorites/reorder")
def italk_reorder_fav(req: ReorderFavReq):
    with ITALK_LOCK:
        data = _italk_load()
        items = data["favorites"]["categories"].get(req.category, [])
        if not items: return {"ok": True}
        if req.from_index < 0 or req.from_index >= len(items):
            raise HTTPException(status_code=400, detail="from out of range")
        if req.to_index < 0 or req.to_index >= len(items):
            raise HTTPException(status_code=400, detail="to out of range")
        it = items.pop(req.from_index)
        items.insert(req.to_index, it)
        _italk_save(data)
    return {"ok": True}

@router.post("/session/append_line")
def italk_append_line(req: AppendLineReq):
    text = (req.text or "").strip()
    if not text: return {"ok": True}
    with ITALK_LOCK:
        data = _italk_load()
        lines = data.get("last_session", {}).get("lines", [])
        norm = " ".join(text.split()).lower()
        new_lines = [l for l in lines if " ".join((l.get("text") or "").split()).lower() != norm]
        line_id = f"line_{uuid.uuid4().hex[:10]}"
        new_lines.insert(0, {"id": line_id, "text": text, "ts": _italk_now_iso()})
        data["last_session"]["lines"] = new_lines
        _italk_save(data)
    return {"id": line_id}

@router.post("/session/delete_line")
def italk_delete_session_line(req: DeleteSessionLineReq):
    line_id = (req.id or "").strip()
    if not line_id: return {"ok": True}
    with ITALK_LOCK:
        data = _italk_load()
        before = len(data["last_session"]["lines"])
        data["last_session"]["lines"] = [l for l in data["last_session"]["lines"] if l.get("id") != line_id]
        if len(data["last_session"]["lines"]) != before:
            _italk_save(data)
    return {"ok": True}

@router.post("/history/delete_line")
def italk_delete_history_line(req: DeleteHistoryLineReq):
    session_id = (req.session_id or "").strip()
    line_id = (req.line_id or "").strip()
    if not session_id or not line_id:
        return {"ok": True}
    with ITALK_LOCK:
        data = _italk_load()
        changed = False
        for sess in data.get("history", []):
            if sess.get("id") == session_id:
                lines = sess.get("lines", [])
                new_lines = [l for l in lines if l.get("id") != line_id]
                if len(new_lines) != len(lines):
                    sess["lines"] = new_lines
                    changed = True
                break
        if changed: _italk_save(data)
    return {"ok": True}

@router.post("/session/name")
def italk_set_session_name(req: SessionNameReq):
    name = (req.name or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        data["last_session"]["name"] = name
        _italk_save(data)
    return {"ok": True}

@router.post("/session/new")
def italk_new_session():
    with ITALK_LOCK:
        data = _italk_load()
        data["last_session"] = {"name": "", "started_at": _italk_now_iso(), "lines": []}
        _italk_save(data)
    return {"ok": True}

@router.post("/session/clear")
def italk_clear_session():
    return italk_new_session()

@router.post("/session/save")
def italk_save_session(req: SessionSaveReq):
    sess_id = ""
    with ITALK_LOCK:
        data = _italk_load()
        sess = data["last_session"]
        name = (req.name or "").strip()
        if name: sess["name"] = name
        if sess.get("lines"):
            sess_id = f"sess_{uuid.uuid4().hex[:8]}"
            data["history"].insert(0, {
                "id": sess_id,
                "name": sess.get("name", ""),
                "saved_at": _italk_now_iso(),
                "lines": list(sess.get("lines", [])),
            })
            _italk_save(data)
    return {"id": sess_id}

@router.post("/history/delete")
def italk_delete_history(req: DeleteHistoryReq):
    with ITALK_LOCK:
        data = _italk_load()
        data["history"] = [h for h in data["history"] if h.get("id") != req.id]
        _italk_save(data)
    return {"ok": True}