import importlib 
import importlib.util

class PythonCodeFormatter:
    def __init__(self):
        yapf_spec = importlib.util.find_spec("yapf")
        # black have no API. so we only support yapf for now.
        # black_spec = importlib.util.find_spec("black")
        self._supported_backends: list[str] = []
        if yapf_spec is not None:
            self._supported_backends.append("yapf")

    def get_all_supported_backends(self) -> list[str]:
        return self._supported_backends

    def format_code(self, code: str, backend: str = "yapf") -> str:
        if backend == "yapf":
            from yapf.yapflib.yapf_api import FormatCode
            res, changed = FormatCode(code)
            return res
        else:
            raise ValueError(f"Unsupported backend: {backend}")