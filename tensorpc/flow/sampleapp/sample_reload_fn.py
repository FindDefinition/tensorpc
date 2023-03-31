
from tensorpc.flow import appctx 


@appctx.observe_function
def func_support_reload(a, b):



    print("hi", a, b)
    return a + b