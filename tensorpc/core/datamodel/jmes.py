from typing import Any
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

# 4. Provide an instance of your subclass in a Options object.
_JMES_EXTEND_OPTIONS = jmespath.Options(custom_functions=_JMESCustomFunctions())


def compile(expression: str) -> ParsedResult:
    return jmespath.compile(expression, options=_JMES_EXTEND_OPTIONS)

def search(expression: str, data: dict) -> Any:
    return jmespath.search(expression, data, options=_JMES_EXTEND_OPTIONS)