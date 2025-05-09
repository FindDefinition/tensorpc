"""All app templates.

Format:

export type AppTemplate = {
    label: string 
    code: string
    group: "example" | "tools" | "debug"
}

"""

from pathlib import Path

from .d3 import frontend_ev
from . import distssh, tutorials, debugpanel

_ALL_GROUPS = set(["example", "tools", "debug"])

def get_all_app_templates():
    all_mods = [{
        "label": "DistSSH Master Panel",
        "mod": distssh,
        "group": "tools",
    }, {
        "label": "Tutorials",
        "mod": tutorials,
        "group": "example",
    }, {
        "label": "Local Debug Panel",
        "mod": debugpanel,
        "group": "tools",
    }, {
        "label": "3D Frontend Event Demo",
        "mod": frontend_ev,
        "group": "debug",
    }]
    res: list[dict] = []
    for item in all_mods:
        mod = item["mod"]
        mod_path = Path(mod.__file__)
        mod_code = mod_path.read_text()
        assert item["group"] in _ALL_GROUPS, f"Invalid group {item['group']}, should be one of {_ALL_GROUPS}"
        res.append({
            "label": item["label"],
            "code": mod_code,
            "group": item["group"],
        })
    return res 
