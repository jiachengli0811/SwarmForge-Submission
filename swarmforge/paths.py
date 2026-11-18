import os
from pathlib import Path
from typing import Union


ROOT = Path(__file__).resolve().parents[1]
_MOVED = {}


def _relocate_text(value: str) -> str:
    prefix = str(ROOT) + "/" if value.startswith(str(ROOT) + "/") else ""
    relative = value[len(prefix):]
    for old, new in _MOVED.items():
        if relative == old or relative.startswith(old + "/"):
            return prefix + new + relative[len(old):]
    return value


def relocate_path(path: Union[str, Path]) -> Path:
    return Path(_relocate_text(os.path.normpath(str(path))))


def relocate_metadata(value):
    if isinstance(value, str):
        return _relocate_text(value)
    if isinstance(value, list):
        return [relocate_metadata(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            mapped = (_relocate_text(key)
                      if isinstance(key, str) and "/" in key else key)
            if mapped in result:
                raise ValueError("Relocated metadata paths collide: %s" % mapped)
            result[mapped] = relocate_metadata(item)
        return result
    return value
