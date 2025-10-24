import json
from pathlib import Path
from typing import Any

class ProfileStore:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update(self, **kwargs: Any):
        data = self.load()
        data.update({k:v for k,v in kwargs.items() if v not in (None, "", [])})
        try:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
