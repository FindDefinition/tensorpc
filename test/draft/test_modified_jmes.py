from tensorpc.core.datamodel import jmes as jmespath



def test_ternary_operator():
    expr = jmespath.compile("a ? b : getattr(c, \"a\").b.x")
    data = {"a": True, "b": 1, "c": None}
    result = jmespath.search(expr, data)
    assert result == 1

    # expr = jmespath.compile("where(a, b, getattr(c, \"a\").b.x)")
    # data = {"a": True, "b": 1, "c": None}
    # result = jmespath.search(expr, data)
    # print(result)

def _main():
    test_ternary_operator()

if __name__ == "__main__":
    _main()