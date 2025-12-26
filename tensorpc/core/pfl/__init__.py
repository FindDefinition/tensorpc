"""Python Frontend Language (Domain Specific Language of python)

We extend python ast node (expr) to a simple DSL with if/assign/math support.

usually used to write frontend code in a more readable way.

WARNING: PFL is static typed, union isn't supported except None (optional) or func args.

"""


from .backends import js
from .constants import (PFL_COMPILE_META_ATTR, PFL_FUNC_ANNO_META_ATTR,
                        PFL_STDLIB_FUNC_META_ATTR)
from .core import (PFLCompileFuncMeta, PFLExprInfo, PFLExprType,
                   PFLInlineRunEnv, PFLMetaInferResult, PFLParseConfig,
                   PFLVariableMeta, PFLProcessedVarMeta, configure_std_func,
                   get_parse_cache_checked, get_parse_context, mark_meta_infer,
                   mark_pfl_compilable, register_backend, get_compilable_meta,
                   PFLTemplateFnSpecMeta, PFLTemplateFnSpecArgsMeta)
from .evaluator import (PFLAsyncRunner, PFLRunnerStateType,
                        PFLStaticEvaluator, consteval_expr)
from .parser import (ast_dump, iter_child_nodes, parse_expr_string_to_pfl_ast,
                     parse_func_to_pfl_ast, parse_func_to_pfl_library,
                     pfl_ast_to_dict, default_pfl_var_proc, 
                     PFLParser, PFLLibrary)
from .pfl_ast import (BinOpType, BoolOpType, CompareType, NodeTransformer,
                      NodeVisitor, PFLAnnAssign, PFLArg, PFLArray, PFLAssign,
                      PFLAstNodeBase, PFLAstStmt, PFLASTType, PFLAttribute,
                      PFLAugAssign, PFLBinOp, PFLBoolOp, PFLCall, PFLCompare,
                      PFLConstant, PFLDict, PFLExpr, PFLExprStmt, PFLFor,
                      PFLFunc, PFLIf, PFLIfExp, PFLModule, PFLName, PFLReturn,
                      PFLSlice, PFLStaticVar, PFLSubscript, PFLTreeNodeFinder,
                      PFLUnaryOp, PFLWhile, UnaryOpType, unparse_pfl_ast,
                      unparse_pfl_ast_to_lines, unparse_pfl_expr, walk,
                      SourceLocType, PFLTreeExprFinder, PFLClass)
from .pfl_reg import (compiler_print_metadata, compiler_print_type,
                      register_pfl_std, register_pfl_builtin_proxy,
                      compiler_remove_optional)
from .pflpath import (PFLPathEvaluator, compile_pflpath, dump_pflpath, 
                      compile_pflpath_to_compact_str)