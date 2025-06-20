

from typing import TypeAlias, TypeVar, overload, ClassVar
from typing_extensions import get_overloads, TypeAliasType
from tensorpc.core.annolib import parse_type_may_optional_undefined
import dataclasses
import inspect 
@dataclasses.dataclass
class Foo:

    def bar(self):
        return 

@dataclasses.dataclass
class Starship:
    @dataclasses.dataclass
    class Part:
        pass

    stats: ClassVar[dict[str, int]] = {} # class variable
    damage: int = 10                     # instance variable
    Foo1: ClassVar[type[Foo]] = Foo


Starship.stats = {}     # This is OK
for f in dataclasses.fields(Starship):
    print(f.name, f)

