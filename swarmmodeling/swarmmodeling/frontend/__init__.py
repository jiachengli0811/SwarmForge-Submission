from __future__ import annotations

import os

from ..ir import Module


def parse_file(path: str, module_name: str = None) -> Module:
    ext = os.path.splitext(path)[1].lower()
    if module_name is None:
        module_name = os.path.splitext(os.path.basename(path))[0]
    if ext == ".py":
        from .python_ts import parse_python_file
        return parse_python_file(path, module_name)
    if ext == ".m":
        from .matlab_ts import parse_matlab_file
        return parse_matlab_file(path, module_name)
    if ext in (".c", ".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"):
        from .cpp_ts import parse_cpp_file
        return parse_cpp_file(path, module_name)
    raise ValueError("no frontend for %r (extensions: .py, .m, .c/.h, .cpp/.hpp)" % ext)
