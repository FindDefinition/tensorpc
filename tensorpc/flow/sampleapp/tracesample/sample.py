def trace_test_1(a: int, b: int):
    c = a + b 
    d = (c, a, b)
    (e, f, g) = d
    return f + g


def trace_test_2(a: int, b: int):
    c = a - b
    return trace_test_1(c, b)