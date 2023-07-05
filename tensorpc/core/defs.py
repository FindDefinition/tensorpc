import mashumaro

from mashumaro.mixins.yaml import DataClassYAMLMixin
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import base64
import json

from tensorpc.constants import TENSORPC_SPLIT


@dataclass
class Service(DataClassYAMLMixin):
    module_name: str
    config: Dict[str, Any]


@dataclass
class ServiceDef(DataClassYAMLMixin):
    services: List[Service]


class DynamicEvent:

    def __init__(self, name: str, data: Any) -> None:
        self.name = name
        self.data = data


class DynamicEvents:

    def __init__(self, name_and_datas: List[Tuple[str, Any]]) -> None:
        self.name_and_datas = name_and_datas


@dataclass
class File:
    name: str
    content: bytes
    data: Any

@dataclass
class FileResource:
    name: str
    path: Optional[str] = None 
    content: Optional[Union[str, bytes]] = None
    chunk_size: Optional[int] = None
    content_type: Optional[str] = None 

@dataclass 
class FileDesp:
    name: str
    content_type: Optional[str] = None 

def from_yaml_path(path: Union[Path, str]) -> ServiceDef:
    """read yaml config with strong-type check
    """
    p = Path(path)
    with p.open("r") as f:
        data = f.read()
    return ServiceDef.from_yaml(data)


def decode_config_b64_and_update(cfg_b64: str, servs: List[Service]):
    serv_config_str = base64.b64decode(cfg_b64).decode("utf-8")
    serv_config = json.loads(serv_config_str)
    key_to_serv: Dict[str, Service] = {}
    for serv in servs:
        key = serv.module_name
        key = TENSORPC_SPLIT.join(key.split(TENSORPC_SPLIT)[:2])
        key_to_serv[key] = serv
    for k, v in serv_config.items():
        assert k in key_to_serv, f"{k} not exist in services"
        key_to_serv[k].config = v
