"""All app templates.

Format:

export type AppTemplate = {
    label: string 
    code: string
    group: "example" | "tools" | "debug"
}

"""

from pathlib import Path
from . import distssh, tutorials


def get_all_app_templates():
    all_mods = [{
        "label": "DistSSH Master Panel",
        "mod": distssh,
        "group": "tools",
    }, {
        "label": "Tutorials",
        "mod": tutorials,
        "group": "example",
    }]
    res: list[dict] = []
    for item in all_mods:
        mod = item["mod"]
        mod_path = Path(mod.__file__)
        mod_code = mod_path.read_text()

        res.append({
            "label": item["label"],
            "code": mod_code,
            "group": item["group"],
        })
    return res 
