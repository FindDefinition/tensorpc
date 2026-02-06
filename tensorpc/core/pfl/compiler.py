from typing import Any, Optional, cast
import tensorpc.core.dataclass_dispatch as dataclasses

from .loggers import PFL_LOGGER
from .core import PFLExprType
from .pfl_ast import PFLExpr, PFLConstant, PFLASTType
from .pfl_reg import register_pfl_compiler_func


@register_pfl_compiler_func(mapped_name="_pfl_compiler_print_type")
def print_type(x: Any) -> Optional[PFLExpr]:
    assert isinstance(x, PFLExpr)
    PFL_LOGGER.warning(str(x.st))
    return x


@register_pfl_compiler_func(mapped_name="_pfl_compiler_print_metadata")
def print_metadata(x: Any) -> Optional[PFLExpr]:
    assert isinstance(x, PFLExpr)
    PFL_LOGGER.warning(str(x.st.metadata))
    return x

@register_pfl_compiler_func(mapped_name="_pfl_compiler_isinstance", mapped=isinstance)
def _pfl_compiler_isinstance(*args: PFLExpr) -> Optional[PFLExpr]:
    # TODO currently int and float are treated as function.
    type_to_check = args[0].st
    type_candidates_expr = args[1]
    if type_candidates_expr.st.type == PFLExprType.TUPLE:
        type_candidates = type_candidates_expr.st.childs
        assert len(
            type_candidates
        ) > 0, "type_candidates must not be empty"
    else:
        type_candidates = [type_candidates_expr.st]
    compare_res: list[bool] = []
    # TODO better compare
    if type_to_check.proxy_dcls is not None:
        for c in type_candidates:
            if c.proxy_dcls is not None:
                compare_res.append(
                    issubclass(type_to_check.proxy_dcls,
                            c.proxy_dcls))
            else:
                compare_res.append(False)
    else:
        for c in type_candidates:
            assert c.type == PFLExprType.DATACLASS_TYPE
            if type_to_check.type == PFLExprType.DATACLASS_OBJECT:
                compare_res.append(
                    issubclass(
                        type_to_check.
                        get_origin_type_checked(),
                        c.get_origin_type_checked()))
            else:
                compare_res.append(False)
    res = PFLConstant(PFLASTType.CONSTANT,
                    (-1, -1, -1, -1),
                    value=any(compare_res))
    res.check_and_infer_type()
    return res

@register_pfl_compiler_func(mapped_name="_pfl_compiler_remove_optional")
def remove_optional(x: Any) -> Any:
    assert isinstance(x, PFLExpr)
    res_st = dataclasses.replace(x.st).get_optional_undefined_removed()
    res = dataclasses.replace(
        x,
        st=res_st)
    return res

@register_pfl_compiler_func(mapped_name="_pfl_compiler_cast", mapped=cast)
def _pfl_compiler_cast(x: Any, cls: Any) -> Any:
    raise NotImplementedError("can't be called directly.")

