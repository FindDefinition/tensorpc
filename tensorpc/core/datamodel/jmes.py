import enum
from typing import Any, Union
import jmespath 
from jmespath import functions
import jmespath.lexer
from jmespath.parser import ParsedResult
from jmespath.visitor import TreeInterpreter
import numpy as np
from tensorpc.utils.perfetto_colors import create_slice_name, perfetto_string_to_color
from jmespath.compat import with_repr_method

class FrontendReservedKeys(enum.Enum):
    """reserved keys that exists in frontend
    """
    PREV_VALUE = "__PREV_VALUE__" # used in frontend update event.
    TARGET = "__TARGET__" # used in frontend update event.

class _JMESCustomFunctions(functions.Functions):
    @functions.signature({'types': ['object']}, {'types': ['string']})
    def _func_getAttr(self, obj, attr):
        return getattr(obj, attr)

    @functions.signature({'types': ['array']}, {'types': ['number']})
    def _func_getItem(self, obj, attr):
        return obj[attr]

    @functions.signature({'types': ['string']}, {'types': ['string', 'number'], 'variadic': True})
    def _func_cformat(self, obj, *attrs):
        # we use https://github.com/stdlib-js/string-format to implement cformat in frontend
        # so user can only use c-style (printf) format string, mapping type in python and 
        # positional placeholders in js can't be used.
        return obj % attrs

    @functions.signature({'types': ['object']}, {'types': ['array']})
    def _func_getItemPath(self, obj, attrs):
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

    @functions.signature({'types': ["array"]})
    def _func_len(self, arr):
        if not isinstance(arr, list):
            return None
        return len(arr) 

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
            return obj[index]
        return None 

    @functions.signature({'types': []}, {'types': ["number"]}, {'types': ["number"]})
    def _func_npSliceFirstAxis(self, obj, start, end):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                return None
            return obj[start:end]
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

    @functions.signature({'types': [], 'variadic': True})
    def _func_printForward(self, x):
        print(*x)
        return x[0]

# 4. Provide an instance of your subclass in a Options object.
_JMES_EXTEND_OPTIONS = jmespath.Options(custom_functions=_JMESCustomFunctions())

# workaround for ternary operator support in jmespath
# https://github.com/jmespath-community/python-jmespath/pull/33

jmespath.parser.Parser.BINDING_POWER["question"] = 2
jmespath.parser.Parser.BINDING_POWER["or"] = 3
jmespath.parser.Parser.BINDING_POWER["and"] = 4
jmespath.lexer.Lexer.SIMPLE_TOKENS["?"] = "question"

def _ternary_operator(condition, left, right):
    return {"type": "ternary_operator", "children": [condition, left, right]}

class _JMESTreeInterpreterExtend(TreeInterpreter):
    def visit_ternary_operator(self, node, value):
        condition = node['children'][0]
        evaluation = self.visit(condition, value)

        if self._is_false(evaluation):
            falsyNode = node['children'][2]
            return self.visit(falsyNode, value)
        else:
            truthyNode = node['children'][1]
            return self.visit(truthyNode, value)

class _JMESParserExtend(jmespath.parser.Parser):
    """add ternary operator support to jmespath parser.
    """
    def _token_led_question(self, condition):
        left = self._expression()
        self._match('colon')
        right = self._expression()
        return _ternary_operator(condition, left, right)

    def _parse(self, expression, options=None):
        res = super()._parse(expression, options)
        return _ParsedResultExtend(res.expression, res.parsed)

@with_repr_method
class _ParsedResultExtend(ParsedResult):
    def search(self, value, options=None):
        evaluator = _JMESTreeInterpreterExtend(options)
        return evaluator.evaluate(self.parsed, value)

def compile(expression: str) -> ParsedResult:
    return _JMESParserExtend().parse(expression, options=_JMES_EXTEND_OPTIONS)
    # return jmespath.compile(expression, options=_JMES_EXTEND_OPTIONS) # type: ignore

def search(expression: Union[str, ParsedResult], data: dict) -> Any:
    if isinstance(expression, ParsedResult):
        return expression.search(data, options=_JMES_EXTEND_OPTIONS)
    return jmespath.compile(expression, _JMES_EXTEND_OPTIONS).search(data, options=_JMES_EXTEND_OPTIONS)

    # return jmespath.search(expression, data, options=_JMES_EXTEND_OPTIONS)