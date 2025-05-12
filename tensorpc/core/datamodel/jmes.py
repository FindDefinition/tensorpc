import enum
from typing import Any, Union
import jmespath 
from jmespath import functions
from jmespath.parser import ParsedResult
import numpy as np
from tensorpc.utils.perfetto_colors import create_slice_name, perfetto_string_to_color

class FrontendReservedKeys(enum.Enum):
    """reserved keys that exists in frontend
    """
    PREV_VALUE = "__PREV_VALUE__" # used in frontend update event.
    TARGET = "__TARGET__" # used in frontend update event.

class _JMESCustomFunctions(functions.Functions):
    @functions.signature({'types': ['object']}, {'types': ['string']})
    def _func_getattr(self, obj, attr):
        return getattr(obj, attr)

    @functions.signature({'types': ['array']}, {'types': ['number']})
    def _func_getitem(self, obj, attr):
        return obj[attr]

    @functions.signature({'types': ['string']}, {'types': ['string', 'number'], 'variadic': True})
    def _func_cformat(self, obj, *attrs):
        # we use https://github.com/stdlib-js/string-format to implement cformat in frontend
        # so user can only use c-style (printf) format string, mapping type in python and 
        # positional placeholders in js can't be used.
        return obj % attrs

    @functions.signature({'types': ['object']}, {'types': ['array']})
    def _func_getitem_path(self, obj, *attrs):
        for attr in attrs:
            obj = obj[attr]
        return obj

    @functions.signature({'types': ['array'], 'variadic': True})
    def _func_concat(self, *arrs):
        return sum(arrs, [])

    @functions.signature({'types': ['boolean']}, {'types': []}, {'types': []})
    def _func_where(self, cond, x, y):
        return x if cond else y

    @functions.signature({'types': []}, {'types': ["array"]})
    def _func_matchcase(self, cond, items):
        if not isinstance(items, list):
            return None
        for pair in items:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return None
            if pair[0] == cond:
                return pair[1]
        return None 

    @functions.signature({'types': []}, {'types': [], 'variadic': True},)
    def _func_matchcase_varg(self, cond, *items):
        if len(items) == 0 or len(items) % 2 != 0:
            return None
        for i in range(0, len(items), 2):
            if items[i] == cond:
                return items[i + 1]
        return None 

    @functions.signature({'types': ["string", "number"]})
    def _func_colorFromSlice(self, obj):
        if isinstance(obj, str):
            return perfetto_string_to_color(create_slice_name(obj), use_cache=False).base.cssString
        elif isinstance(obj, (int, float)):
            return perfetto_string_to_color(str(obj), use_cache=False).base.cssString
        return None 

    @functions.signature({'types': ["string", "number"]})
    def _func_colorFromName(self, obj):
        if isinstance(obj, str):
            return perfetto_string_to_color(obj, use_cache=False).base.cssString
        elif isinstance(obj, (int, float)):
            return perfetto_string_to_color(str(obj), use_cache=False).base.cssString
        return None 

    @functions.signature({'types': []})
    def _func_npToList(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return None 

    @functions.signature({'types': []}, {'types': ["number"]})
    def _func_npGetSubArray(self, obj, index):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                return None
            return obj[index:index+1]
        return None 

    @functions.signature({'types': []}, {'types': ["number"], 'variadic': True})
    def _func_ndarray_getitem(self, obj, *index):
        if isinstance(obj, np.ndarray):
            return obj[index]
        return None 

    @functions.signature({'types': ["number"]}, {'types': ["number"]})
    def _func_maximum(self, x, y):
        return max(x, y)

    @functions.signature({'types': ["number"]}, {'types': ["number"]})
    def _func_minimum(self, x, y):
        return min(x, y)

    @functions.signature({'types': ["number"]}, {'types': ["number"]}, {'types': ["number"]})
    def _func_clamp(self, x, a, b):
        return max(a, min(x, b))

# 4. Provide an instance of your subclass in a Options object.
_JMES_EXTEND_OPTIONS = jmespath.Options(custom_functions=_JMESCustomFunctions())


def compile(expression: str) -> ParsedResult:
    return jmespath.compile(expression, options=_JMES_EXTEND_OPTIONS) # type: ignore

def search(expression: Union[str, ParsedResult], data: dict) -> Any:
    if isinstance(expression, ParsedResult):
        return expression.search(data, options=_JMES_EXTEND_OPTIONS)
    return jmespath.search(expression, data, options=_JMES_EXTEND_OPTIONS)