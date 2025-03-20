from pathlib import Path 

_MEDIA_ROOT = Path(__file__).parent / "media"


def get_default_custom_node_code():
    base_code_path = _MEDIA_ROOT / "default_code.py"
    with open(base_code_path, "r") as f:
        base_code = f.read()

    return base_code