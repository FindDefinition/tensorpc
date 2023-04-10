
from tensorpc.flow import appctx 


@appctx.observe_function
def func_support_reload(a, b):
    print("hi6", a, b)



    return a + b

# from cumm.common import TensorViewNVRTC
# from cumm.inliner import NVRTCInlineBuilder
# import pccm 
# from cumm import tensorview as tv 
# from pccm.builder.inliner import PreCaptureFunctionCode

# class Dev:
#     def __init__(self) -> None:
#         self.inliner = NVRTCInlineBuilder([TensorViewNVRTC], reload_when_code_change=True)

#         self.a = 1

#     def prepare_params(self, code: PreCaptureFunctionCode):
#         with code.capture_vars():
#             code.raw(f"""
#             int A = $(self.a);
#             """)
        
#     @appctx.observe_function
#     def dev(self):
#         a = tv.zeros([1], tv.float32, 0)
#         code = PreCaptureFunctionCode()
#         self.prepare_params(code)
#         code.raw(f"""
#         tv::printf2(A, "RTX2411214");
#         """)
#         self.inliner.kernel_1d("dev", 1, 0, code)
    