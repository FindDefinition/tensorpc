"""Generate triton code from PFL and ppcl stdlib.
"""
import dataclasses
from inspect import unwrap
from ..std import PointerTensor
from ..core import ConstExprMeta, DTypeEnum
from tensorpc.core import pfl 

class TritonASTTransformer(pfl.NodeTransformer):
    def visit_PFLArg(self, node: pfl.PFLArg):
        if node.st.annotype is not None:
            is_constexpr_meta = node.st.annotype.get_annometa(ConstExprMeta)
            if is_constexpr_meta is not None:
                return dataclasses.replace(node, annotation="tl.constexpr")
            else:
                # clear non-constexpr anno for triton
                return dataclasses.replace(node, annotation=None)

        return self.visit(node)

    def visit_PFLCall(self, node: pfl.PFLCall):
        # convert ppcl namespace calls to triton
        # e.g. ppcl.zeros -> tl.zeros
        raw_fn = node.func.st.raw_func
        if raw_fn is not None:
            qualname = raw_fn.__qualname__
            parts = qualname.split(".")
            if len(parts) == 2 and parts[0] == "ppcl":
                assert isinstance(node.func, pfl.PFLAttribute)
                fn = pfl.PFLAttribute(pfl.PFLASTType.ATTR, value=pfl.PFLName(pfl.PFLASTType.NAME, id="tl", st=node.func.value.st), attr=parts[1])
                node = dataclasses.replace(node, func=fn)
        return self.generic_visit(node)

    def visit_PFLAttribute(self, node: pfl.PFLAttribute):
        # convert constants (e.g. ppcl.float16) to tl.float16
        # TODO better dtype

        if node.st.annotype is not None and node.st.annotype.origin_type is DTypeEnum:
            return pfl.PFLAttribute(pfl.PFLASTType.ATTR, value=pfl.PFLName(pfl.PFLASTType.NAME, id="tl", st=node.value.st), attr=node.attr)
        return self.generic_visit(node)

    def visit_PFLName(self, node: pfl.PFLName):
        # convert constants (e.g. ppcl.float16) to tl.float16
        # TODO better dtype
        if node.st.annotype is not None and node.st.annotype.origin_type is DTypeEnum:
            return pfl.PFLAttribute(pfl.PFLASTType.ATTR, value=pfl.PFLName(pfl.PFLASTType.NAME, id="tl", st=node.st), attr=node.id)
        return self.generic_visit(node)

def pfl_ast_to_triton(node: pfl.PFLAstNodeBase):
    node = TritonASTTransformer().visit(node)
    return pfl.unparse_pfl_ast(node)