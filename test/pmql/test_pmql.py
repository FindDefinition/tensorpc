
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

from tensorpc.apps.dbg.pmql.parser import parse_pmql


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
    ]
    for q in queries:
        print(parse_pmql(q))


if __name__ == "__main__":
    test_parse_pmql()