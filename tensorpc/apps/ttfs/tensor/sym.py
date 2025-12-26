from tensorpc.apps.ttfs.aggtype import constexpr_function
from triton.experimental.gluon import language as gl

def parse_shape_sym(shape_sym: str):
    """format: "B, S->S_alias, H, D"
    alias is only used when you specify tensor name.
    Usually used when your input tensor has padded dim.
    e.g. Q = [B, S, H, D], K = [B, S_padded_k->S, H, D]
    """
    parts = shape_sym.split(",")
    parts = [x.strip() for x in parts]
    shape_names: list[str] = []
    aliases: list[str] = []
    for part in parts:
        if "->" in part:
            name, alias = part.split("->")
            assert name.isidentifier() and alias.isidentifier(), f"Invalid shape_sym part: {part}"
            name = name.strip()
            alias = alias.strip()
            shape_names.append(name)
            aliases.append(alias)
        else:
            name = part.strip()
            assert name.isidentifier(), f"Invalid shape_sym part: {part}"
            shape_names.append(name)
            aliases.append(name)
    return shape_names, aliases

_SYM_AXES_CACHE: dict[tuple[str, str], list[int]] = {}
_SYM_REMAIN_AXES_CACHE: dict[tuple[str, str], list[int]] = {}

def cached_get_sym_axes_host(shape_sym: str, query_sym: str):
    if (shape_sym, query_sym) in _SYM_AXES_CACHE:
        return _SYM_AXES_CACHE[(shape_sym, query_sym)]
    shape_names, alias_names = parse_shape_sym(shape_sym)

    query_names, _ = parse_shape_sym(query_sym)
    res: list[int] = []
    for qname in query_names:
        if qname in alias_names:
            dim = alias_names.index(qname)
            res.append(dim)
        else:
            assert qname in shape_names, f"Symbol {qname} not found in shape_sym {shape_sym}"
            dim = shape_names.index(qname)
            res.append(dim)
    _SYM_AXES_CACHE[(shape_sym, query_sym)] = res
    return res

def cached_get_remain_sym_axes_host(shape_sym: str, query_sym: str):
    if (shape_sym, query_sym) in _SYM_REMAIN_AXES_CACHE:
        return _SYM_REMAIN_AXES_CACHE[(shape_sym, query_sym)]
    shape_names, alias_names = parse_shape_sym(shape_sym)
    axes = cached_get_sym_axes_host(shape_sym, query_sym)
    res: list[int] = []
    for i in range(len(shape_names)):
        if i not in axes:
            res.append(i)
    _SYM_REMAIN_AXES_CACHE[(shape_sym, query_sym)] = res
    return res

@constexpr_function
def get_sym_axes(shape_sym: str, query_sym: str):
    axes = cached_get_sym_axes_host(shape_sym, query_sym)
    return axes

@constexpr_function
def get_remain_sym_axes(shape_sym: str, query_sym: str):
    shape_names, alias_names = parse_shape_sym(shape_sym)
    axes = cached_get_sym_axes_host(shape_sym, query_sym)
    res: list[int] = []
    for i in range(len(shape_names)):
        if i not in axes:
            res.append(i)
    return res

@constexpr_function
def get_shape_names_from_sym(shape_sym: str):
    shape_names, _ = parse_shape_sym(shape_sym)
    return shape_names

@constexpr_function
def unify_offset_names(off_names):
    if isinstance(off_names, str):
        res = off_names.split(",")
        res = [x.strip() for x in res]
    else:
        res = off_names
    # offset_names support both tl.constexpr([...]) and "A,B"
    return res

@constexpr_function
def get_blocked_layout_order(shape_sym, block_dim_sym):
    # if shape is B,S,H,D and block_dim_names is S,H, return [1,0]
    shape_names, alias_names = parse_shape_sym(shape_sym)
    block_shape_names, _ = parse_shape_sym(block_dim_sym)
    print(shape_names, alias_names, block_shape_names)
    res: list[int] = []
    for i, name in enumerate(block_shape_names):
        if name in shape_names:
            dim = shape_names.index(name)
            res.append(dim)
        elif name in alias_names:
            dim = alias_names.index(name)
            res.append(dim)
    # do argsort on res
    res = sorted(range(len(res)), key=lambda k: res[k])
    assert len(res) == len(block_shape_names), f"block_dim_names {block_shape_names} not all in shape_sym {shape_sym}"
    return res[::-1]

@constexpr_function
def get_stride_names(shape_sym, axis):
    shape_names, _ = parse_shape_sym(shape_sym)
    res = shape_names[axis + 1:]
    return res

@constexpr_function
def get_dim_by_name(shape_sym, name):
    shape_names, alias_names = parse_shape_sym(shape_sym)
    if name in alias_names:
        return alias_names.index(name)
    assert name in shape_names, f"{name} not in {shape_names} or {alias_names} ({shape_sym})"
    return shape_names.index(name)

@constexpr_function
def get_name_by_dim(shape_sym, dim):
    shape_names, alias_names = parse_shape_sym(shape_sym)
    print(shape_sym, dim)
    return shape_names[dim]

def unify_possible_alias_name_python(shape_sym, name):
    shape_names, alias_names = parse_shape_sym(shape_sym)
    if name in alias_names:
        dim = alias_names.index(name)
        return shape_names[dim]
    assert name in shape_names, f"{name} not in {shape_names} or {alias_names}"
    return name

@constexpr_function
def local_name_to_global(shape_sym, name):
    return unify_possible_alias_name_python(shape_sym, name)

@constexpr_function
def local_names_to_global(shape_sym, names):
    return [unify_possible_alias_name_python(shape_sym, name) for name in names]

@constexpr_function
def local_sym_to_global(root_sym, block_sym):
    block_shape_names, _ = parse_shape_sym(block_sym)
    unified_block_shape_names = []
    for name in block_shape_names:
        unified_name = unify_possible_alias_name_python(root_sym, name)
        unified_block_shape_names.append(unified_name)
    return ",".join(unified_block_shape_names)

@constexpr_function
def get_sliced_layout(layout, ndim, dim):
    dims = list(range(ndim))
    dims.remove(dim)
    for dim in dims[::-1]:
        layout = gl.SliceLayout(dim, layout)
    return layout

@constexpr_function
def get_sliced_axes(ndim, dim):
    dims = list(range(ndim))
    dims.remove(dim)
    return dims[::-1]

@constexpr_function
def get_remain_axes(ndim, dim):
    dims = list(range(ndim))
    dims.remove(dim)
    return dims


@constexpr_function
def filter_sym(shape_sym, sym_list, all_sym_must_exist: bool):
    shape_names, alias_names = parse_shape_sym(shape_sym)

    res_sym_list = []
    for sym in sym_list:
        if sym in alias_names or sym in shape_names:
            res_sym_list.append(sym)
        else:
            if all_sym_must_exist:
                raise RuntimeError(f"Symbol {sym} not found in shape_sym {shape_sym}")
    return res_sym_list

@constexpr_function
def tuple_range(length):
    return tuple(i for i in range(length))

@constexpr_function
def get_block_dim_name_by_idx(shape_sym, idx):
    shape_names, _ = parse_shape_sym(shape_sym)
    return shape_names[idx]

@constexpr_function
def check_matrix_is_transposed(shape_sym, block_shape_sym):
    shape_names, alias_names = parse_shape_sym(shape_sym)
    block_shape_names, _ = parse_shape_sym(block_shape_sym)
    if len(block_shape_names) == 1:
        return False 
    block_shape_names = [unify_possible_alias_name_python(shape_sym, name) for name in block_shape_names]
    # block name may be alias, so need unify
    assert len(block_shape_names) == 2, "Only support 2D blocked tensor for now."
    # get idx in shape_names
    idx0 = shape_names.index(block_shape_names[0])
    idx1 = shape_names.index(block_shape_names[1])
    return idx0 > idx1

@constexpr_function
def is_name_in_names(name, names):
    return name in names

@constexpr_function
def get_name_in_names_idx(name, names):
    return names.index(name)


@constexpr_function
def is_any_sym_in_names(sym, names):
    names, _ = parse_shape_sym(sym)
    return any(name in names for name in names)

@constexpr_function
def get_true_inds_of_bool_tuple(bool_tuple, except_axis):
    res: list[int] = []
    if except_axis >= 0:
        assert except_axis < len(bool_tuple)

    for j, val in enumerate(bool_tuple):
        val = gl._unwrap_if_constexpr(val)
        if val and j != except_axis:
            res.append(j)
    return res


@constexpr_function
def get_false_inds_of_bool_tuple(bool_tuple, except_axis):
    res: list[int] = []
    if except_axis >= 0:
        assert except_axis < len(bool_tuple)
    for j, val in enumerate(bool_tuple):
        val = gl._unwrap_if_constexpr(val)
        if not val and j != except_axis:
            res.append(j)
    return res
