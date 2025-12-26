import fnmatch
from pathlib import Path
import re
from typing import Any, Optional, Union
from typing_extensions import Literal, TypeAlias
import lark 
from tensorpc.core import dataclass_dispatch as dataclasses
from lark import Transformer, Token
import glob 

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


with _GRAMMAR_PATH.open("r") as f:
    data = f.read()
    _COMPILED = lark.Lark(data)


@dataclasses.dataclass
class PlainItem:
    id: str 

@dataclasses.dataclass
class TypeItem:
    id: str 

@dataclasses.dataclass
class DoubleGlob:
    pass 

@dataclasses.dataclass
class IndexItem:
    index: Union[int, str]
    is_all: bool

@dataclasses.dataclass
class IdentityItem:
    pass 

@dataclasses.dataclass
class PartialGlob:
    glob: list[str]
    regex: Optional[re.Pattern] = None

    def get_glob_regex(self):
        if self.regex is None:
            glob_str = "".join(self.glob)
            # Convert glob pattern to regex
            glob_re = fnmatch.translate(glob_str)
            self.regex = re.compile(glob_re)
        return self.regex

QueryItem: TypeAlias = Union[PlainItem, TypeItem, DoubleGlob, IndexItem, PartialGlob, IdentityItem]

@dataclasses.dataclass
class SingleQuery:
    items: list[QueryItem]

    def __post_init__(self):
        double_glob_cnt = 0
        for item in self.items:
            if isinstance(item, DoubleGlob):
                if double_glob_cnt > 0:
                    raise ValueError("Only one double glob is allowed in a query.")
                double_glob_cnt += 1

    def get_glob_path(self) -> str:
        """Get the glob path from the query items."""
        parts = []
        for item in self.items:
            assert not isinstance(item, TypeItem)
            if isinstance(item, PlainItem):
                parts.append(item.id)
            elif isinstance(item, DoubleGlob):
                parts.append("**")
            elif isinstance(item, IndexItem):
                if item.is_all:
                    parts.append("*")
                else:
                    parts.append(str(item.index))
            elif isinstance(item, PartialGlob):
                parts.extend(item.glob)
            else:
                raise ValueError(f"Unknown query item type: {type(item)}")
        return "/".join(parts)

@dataclasses.dataclass
class ModuleVariableQueryExpr:
    type: Literal["args", "kwargs", "ret"]
    query: SingleQuery

@dataclasses.dataclass
class ModuleVariableQuery:
    mod_query: SingleQuery
    var_queries: list[ModuleVariableQueryExpr]

@dataclasses.dataclass
class ModuleWeightQuery:
    mod_query: SingleQuery
    var_name: str

@dataclasses.dataclass
class ModuleStackQuery:
    mod_query: SingleQuery
    var_queries: list[SingleQuery]


class PMQLTransformer(Transformer):
    def QUALNAME(self, items):
        return items.value

    def NAME(self, items):
        return items.value

    def WORD_WITH_LOWER(self, items):
        return items.value

    def ESCAPED_STRING(self, s):
        (s,) = s
        return s[1:-1]
        
    def SIGNED_INT(self, items):
        return int(items.value)

    def partial_glob_strip(self, items):
        res: list[str] = []
        for j in range(len(items) - 1):
            res.append(items[j])
            res.append("*")
        res.append(items[-1])
        return res

    def STAR(self, items):
        return "*"

    def plain_item(self, items):
        return PlainItem(items[0])

    def VQ_LITERALS(self, items):
        return items.value

    def type_item(self, items):
        return TypeItem(items[0])

    def index_item(self, items):
        val = items[0]
        is_all = False
        if val == "*":
            val = -1
            is_all = True
        return IndexItem(val, is_all=is_all)

    def double_glob(self, items):
        return DoubleGlob()

    def identity_item(self, items):
        return IdentityItem()

    def partial_glob(self, items):
        res: list[str] = []
        for v in items:
            if isinstance(v, list):
                res.extend(v)
            else:
                res.append(v)
        return PartialGlob(res)

    def module_variable_query_expr(self, items):
        return ModuleVariableQueryExpr(items[0], SingleQuery(items[1:]))
    
    def module_variable_query_exprs(self, items):
        return items

    def query_expr(self, items):
        return SingleQuery(items)

    def query_exprs(self, items):
        return items

    def module_stack_query(self, items):
        return ModuleStackQuery(items[0], items[1])

    def module_var_query(self, items):
        return ModuleVariableQuery(items[0], items[1])

    def module_weight_query(self, items):
        return ModuleWeightQuery(items[0], items[1])

def parse_pmql_raw(query: str):
    return _COMPILED.parse(query)

def parse_pmql(query: str):
    res = PMQLTransformer().transform(parse_pmql_raw(query))
    assert isinstance(res, (SingleQuery, ModuleVariableQuery, ModuleWeightQuery, ModuleStackQuery))
    return res