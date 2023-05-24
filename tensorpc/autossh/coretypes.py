
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
    init_commands: str = ""
    
    @property 
    def url(self):
        return f"{self.hostname}:{self.port}"

    @staticmethod
    def create_fake_target():
        return SSHTarget("localhost", 22, "root", "root")

    def is_localhost(self):
        return self.hostname == "localhost"