import mashumaro

from mashumaro.mixins.yaml import DataClassYAMLMixin
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path



@dataclass
class Service(DataClassYAMLMixin):
    module_name: str 
    config: Dict[str, Any]


@dataclass
class ServiceDef(DataClassYAMLMixin):
    services: List[Service]


def from_yaml_path(path: Union[Path, str]) -> ServiceDef:
    """read yaml config with strong-type check
    """
    p = Path(path)
    with p.open("r") as f:
        data = f.read()
    return ServiceDef.from_yaml(data)
