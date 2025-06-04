"""Python Frontend Language (Domain Specific Language of python)

We extend python ast node (expr) to a simple DSL with if/assign/math support.

usually used to write frontend code in a more readable way.

WARNING: PFL is static typed to get better performance, union isn't supported except None (optional) or func args.

"""


from .parser import parse_expr_to_df_ast, parse_func_to_df_ast, consteval_expr, metaeval_total_tree, pfl_ast_to_dict, pfl_ast_dump, walk, iter_child_nodes
from .core import PFLExprInfo, register_meta_assign_check, register_meta_infer
from .pfl_reg import register_pfl_std