import re
import os 
from pathlib import Path 
from typing import List, Tuple, Optional, Callable

# from vscode terminalLocalLinkDetector.ts
_STACK_TRACE_PYTHON_RE = re.compile(r"^ *File \"(?P<path>.+)\"(, line (?P<line>\d+))?")

def parse_python_traceback(tb_str: str, path_validator: Optional[Callable[[str], bool]] = None) -> List[Tuple[Tuple[str, str], List[str]]]:
    stack_trace_lines = tb_str.split("\n")
    stack_trace: List[Tuple[Tuple[str, str], List[str]]] = []
    current_is_invalid_path = False
    for i, line in enumerate(stack_trace_lines):
        m = _STACK_TRACE_PYTHON_RE.match(line)
        if m is not None:
            path = m.group("path")
            lineno = m.group("line")
            # ignore paths in torch.utils
            if Path(path).exists():
                if path_validator is not None and not path_validator(path):
                    current_is_invalid_path = True
                    continue
            current_is_invalid_path = False
            stack_trace.append(((path, lineno), []))
        else:
            if current_is_invalid_path:
                continue
            if stack_trace and line.strip():
                stack_trace[-1][1].append(line)
    for (path, lineno), lines in stack_trace:
        if lines:
            # determine indent of lines, and remove them to improve readability
            min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
            lines[:] = [line[min_indent:] for line in lines]
    return stack_trace