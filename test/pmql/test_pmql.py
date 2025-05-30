
# Module-only queries

# mod1.asd*.mod3
# mod1.**.mod3
# mod1.<Block>.name
# mod1.**.<nn.Linear>

# Variable queries
# args[0]
# kwargs.*states
# kwargs.**.key

# Module+Call Variable queries

# mod1.asd*.mod3@args[0]
# mod1.asd*.mod3@kwargs.*states
# mod1.asd*.mod3@ret
# mod1.asd*.mod3@ret[0].key

# Module+stack based queries
# mod1.asd*.mod3!var_query_expr # fwd call frame, looking for previous non-torch call frame.

from tensorpc.apps.dbg.pmql.parser import SingleQuery, parse_pmql, ModuleVariableQuery, ModuleStackQuery
from tensorpc.apps.dbg.pmql.run_query import simple_module_query, install_module_hook_query, module_query_context


def test_parse_pmql():
    queries = [
        "mod1.asd*.mod3",
        "mod1.**.mod3",
        "mod1.<Block>.name",
        "mod1.**.<nn.Linear>",
        "mod1.asd*.mod3@args[0]",
        "mod1.asd*.mod3@kwargs.*states",
        "mod1.**.mod3@ret",
        "mod1.asd*.mod3@ret[0].key",
        "mod1.a*sd*.mod3!var_query_expr",
        "mod1.asd*.mod3#weight"
    ]
    for q in queries:
        print(parse_pmql(q))

def test_pth_query():
    import torch
    from torch import nn 
    tr = nn.Transformer(nhead=16)
    # queries = [
    #     # "**.<Linear>",
    #     # "encoder.**.linear*",
    #     # "encoder.layers[0].**#weight",
    #     "encoder.layers[*]",

    # ]
    # for q_str in queries:
    #     q = parse_pmql(q_str)

    #     res = simple_module_query(tr, q)
    #     print([x.fqn for x in res])

    # vq = [
    #     "encoder.layers[*]@args[0]",
    #     # "encoder.**.linear*",
    #     # "encoder.layers[0].**#weight",

    # ]
    # for q_str in vq:
    #     q = parse_pmql(q_str)
    #     print(q)
    #     assert isinstance(q, (ModuleVariableQuery, ModuleStackQuery))
    #     handle = install_module_hook_query(tr, [(q, lambda res: print([x.data.shape for x in res]))])
    #     src = torch.rand((10, 32, 512))
    #     tgt = torch.rand((20, 32, 512))
    #     out = tr(src, tgt)
    #     handle.remove()
    #     # print([x.fqn for x in res])

    with module_query_context(tr, data="encoder.layers[*]!output") as ctx:
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((20, 32, 512))
        out = tr(src, tgt)

    print([x.data.shape for x in ctx.result["data"]])

if __name__ == "__main__":
    # test_parse_pmql()
    test_pth_query()

