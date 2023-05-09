
import dataclasses 

import enum
from typing import Dict, Any, List, Optional, Tuple


@dataclasses.dataclass
class SSHTarget:
    hostname: str 
    port: int 
    username: str 
    password: str 
    known_hosts: Optional[str] = None 
    client_keys: Optional[List[str]] = None 
    env: Optional[Dict[str, str]] = None
    uid: str = ""
    forward_port_pairs: List[Tuple[int, int]] = dataclasses.field(default_factory=list)
    remote_forward_port_pairs: List[Tuple[int, int]] = dataclasses.field(default_factory=list)

    @property 
    def url(self):
        return f"{self.hostname}:{self.port}"


