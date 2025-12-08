from tensorpc.apps.ttfs.aggtype import constexpr_function
import triton.language as tl
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language._core import builtin as gluon_builtin

@constexpr_function
def merge_tuple_dict_key(names, names_upd):
    names = [gl._unwrap_if_constexpr(n) for n in names]
    names_upd = [gl._unwrap_if_constexpr(n) for n in names_upd]
    dict_dst = {names[i]: -1 for i in range(len(names))}
    update = {names_upd[i]: -1 for i in range(len(names_upd))}
    dict_dst.update(update)
    new_names_list = list(dict_dst.keys())
    return [(x) for x in new_names_list]

@gluon_builtin
def merge_tuple_dict_gluon(names, values, names_upd, values_upd, _semantic=None):
    names = [gl._unwrap_if_constexpr(n) for n in names]
    names_upd = [gl._unwrap_if_constexpr(n) for n in names_upd]
    dict_dst = {names[i]: values[i] for i in range(len(names))}
    update = {names_upd[i]: values_upd[i] for i in range(len(names_upd))}
    dict_dst.update(update)
    new_names_list = list(dict_dst.keys())
    new_values_tuple = tl.tuple([dict_dst[k] for k in new_names_list])
    return  new_values_tuple

@tl.core.builtin
def merge_tuple_dict_triton(names, values, names_upd, values_upd, _semantic=None):

    names = [tl.core._unwrap_if_constexpr(n) for n in names]
    names_upd = [tl.core._unwrap_if_constexpr(n) for n in names_upd]
    dict_dst = {names[i]: values[i] for i in range(len(names))}
    update = {names_upd[i]: values_upd[i] for i in range(len(names_upd))}
    dict_dst.update(update)
    new_names_list = list(dict_dst.keys())
    new_values_tuple = tuple([dict_dst[k] for k in new_names_list])
    return new_values_tuple
