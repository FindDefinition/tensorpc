from typing import Any, Union
import jmespath 
from jmespath import functions
from jmespath.parser import ParsedResult

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

# 4. Provide an instance of your subclass in a Options object.
_JMES_EXTEND_OPTIONS = jmespath.Options(custom_functions=_JMESCustomFunctions())


def compile(expression: str) -> ParsedResult:
    return jmespath.compile(expression, options=_JMES_EXTEND_OPTIONS) # type: ignore

def search(expression: Union[str, ParsedResult], data: dict) -> Any:
    if isinstance(expression, ParsedResult):
        return expression.search(data, options=_JMES_EXTEND_OPTIONS)
    return jmespath.search(expression, data, options=_JMES_EXTEND_OPTIONS)