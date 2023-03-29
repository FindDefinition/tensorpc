import traceback
import sys

from tensorpc.flow.flowapp import appctx
def a(x):
    b = x + 1
    raise NotImplementedError


def b(x):
    return a(x)


def c(x):
    return b(x)

print("???")
if __name__ == "__main__":

    ddd = 1 
    appctx.obj_inspector_set_object_sync(ddd, "ddd")
    try:
        c(1)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        for tb_frame, tb_lineno in traceback.walk_tb(exc_traceback):
            print(tb_frame.f_locals.keys())
            print(tb_lineno)
