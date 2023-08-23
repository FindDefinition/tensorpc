from typing import Optional
from typing_extensions import Annotated
from tensorpc.core.dataclass_dispatch import dataclass

@dataclass
class RangedInt:
    lo: int
    hi: int
    step: Optional[int] = None
    alias: Optional[str] = None 

@dataclass
class RangedFloat:
    lo: float
    hi: float
    step: Optional[float] = None
    alias: Optional[str] = None 

